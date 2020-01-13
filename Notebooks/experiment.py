# -------------------------------------------------------------------------------------------------
# Implementation of GNN experiments

#%%
# -------------------------------------------------------------------------------------------------
# General Imports
# -------------------------------------------------------------------------------------------------

# neural network libraries
import dgl
import dgl.function as fn
import torch as th
import torch.nn as nn
import torch.nn.functional as F
from dgl import DGLGraph

# graph
from dgl.data import citation_graph as citegrh
import networkx as nx

# util
import pickle
import copy
import numpy as np
import pandas as pd
import itertools
import os
import time
import datetime
import json
from zipfile import ZipFile
from zipfile import ZIP_DEFLATED as zipDEF
import traceback

# evaluation
import performance as pf

# GNN classes
from GCN import GCN_Net
from GAT import GAT_Net_fast
from LGNN import LGNN_Net

#%%
from multiprocessing import Pool

#%%
# -------------------------------------------------------------------------------------------------
# Data Loading
# -------------------------------------------------------------------------------------------------

data = citegrh.load_cora()
labels = th.LongTensor(data.labels)

# graph
g = DGLGraph(data.graph)

# line graph
lg = g.line_graph(backtracking=False)

# graph with self-loops
g_selfl = data.graph
g_selfl.remove_edges_from(nx.selfloop_edges(g_selfl))
g_selfl = DGLGraph(g_selfl)
g_selfl.add_edges(g_selfl.nodes(), g_selfl.nodes())

# features based on the keywords of the papers
features_keywords = th.FloatTensor(data.features)
features_keywords_lg = th.FloatTensor(
    np.vstack([features_keywords[e[0], :] for e in data.graph.edges])
)  # for the nodes in the line graph, we take the features of the source paper as feature

# features based on the node_id of the papers
features_nodeid = th.eye(g.number_of_nodes())

# features based on the node degree
features_degree_g = g.in_degrees().float().unsqueeze(1)
features_degree_lg = lg.in_degrees().float().unsqueeze(1)

num_classes = len(np.unique(labels))

#%%
# -------------------------------------------------------------------------------------------------
# Loss Functions
# -------------------------------------------------------------------------------------------------


def init_loss_function(labels, func_type, nclasses=None):
    func = None
    if func_type == "nll":
        func = lambda logits, labels, mask: F.nll_loss(logits[mask], labels[mask])
    elif func_type == "inv":
        loss_function = pf.perm_inv_loss(labels)
        func = lambda logits, labels, mask: loss_function.approximate_loss(
            logits, mask, nclasses=nclasses
        )
    return func


#%%
# -------------------------------------------------------------------------------------------------
# Definition of a Network Run
# -------------------------------------------------------------------------------------------------


def perform_run(run):
    starttime = time.time()
    print(f"{datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}: Started run {run['name']}.")
    try:
        # ---------------------------------------------------------------------------------------------
        # Initialization of Network with Parameters
        # ---------------------------------------------------------------------------------------------

        if np.random.rand() > 0.5:
            k = int("k")

        net = run["net"](**run["params"])

        # get the correct features
        if run["feature"] == "keywords":
            net_features = features_keywords
        elif run["feature"] == "node_id":
            net_features = features_nodeid
        elif run["feature"] == "node_degree":
            net_features = features_degree_g

        # add features for line graph in case of LGNN
        if run["net"] == LGNN_Net:
            if run["feature"] == "keywords":
                net_features = (net_features, features_keywords_lg)
            elif run["feature"] == "node_degree":
                net_features = (net_features, features_degree_lg)

        mask_train = run["split"]["train"]
        mask_val = run["split"]["val"]
        mask_test = run["split"]["test"]

        # ---------------------------------------------------------------------------------------------
        # Training
        # ---------------------------------------------------------------------------------------------

        loss_f = init_loss_function(
            labels=labels,
            func_type=run["loss_function"]["func_type"],
            nclasses=run["loss_function"]["nclasses"],
        )
        optimizer = th.optim.Adam(
            net.parameters(), lr=run["learning_rate"], weight_decay=run["weight_decay"]
        )
        net.train()  # Set to training mode

        log = []
        # pd.DataFrame(
        #    columns=["epoch", "best_epoch", "dur", "loss", "rand_train", "rand_val", "rand_test"]
        # )

        current_best = -np.infty  # arbitrarily bad
        current_best_epoch = 0
        current_best_params = None
        no_improvement_for = 0

        for epoch in range(10000):
            t0 = time.time()

            # Compute performance for train, validation and test set
            net.eval()
            prediction = net(net_features)
            train_rand = pf.rand_score(
                labels[mask_train].numpy(),
                np.argmax(prediction[mask_train].detach().numpy(), axis=1),
            )
            validation_rand = pf.rand_score(
                labels[mask_val].numpy(), np.argmax(prediction[mask_val].detach().numpy(), axis=1)
            )
            test_rand = pf.rand_score(
                labels[mask_test].numpy(), np.argmax(prediction[mask_test].detach().numpy(), axis=1)
            )

            # Save current best model
            if validation_rand > current_best:
                current_best = validation_rand
                current_best_epoch = epoch
                current_best_params = copy.deepcopy(net.state_dict())
                no_improvement_for = 0
            else:
                no_improvement_for += 1

            # Apply early stopping
            if (
                epoch > run["early_stopping"]["min"]
                and no_improvement_for > run["early_stopping"]["wait"]
            ):
                break

            net.train()

            # Compute loss for train nodes
            logits = net(net_features)

            loss = loss_f(logits=logits, labels=labels, mask=mask_train)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            dur = time.time() - t0

            if run["verbatim"]:
                print(
                    f"Epoch {epoch:05d} | Loss {loss.detach().item():.4f} | Train.Rand {train_rand:.4f}"
                    + f" | Valid.Rand {validation_rand:.4f} | Test.Rand {test_rand:.4f} | Time(s) {dur:.4f}"
                )

            log.append(
                {
                    "epoch": epoch,
                    "epoch_best": current_best_epoch,
                    "dur": dur,
                    "loss": loss.detach().item(),
                    "rand_train": train_rand,
                    "rand_val": validation_rand,
                    "rand_val_best": current_best,
                    "rand_test": test_rand,
                }
            )

        net.load_state_dict(current_best_params)
        log = pd.DataFrame(log)

        # ---------------------------------------------------------------------------------------------
        # Evaluation
        # ---------------------------------------------------------------------------------------------

        net.eval()  # Set net to evaluation mode
        final_prediction = net(net_features).detach()
        res = pf.compute_performance(labels, final_prediction, splits=run["split"])
        res = {
            f"{score_name}_{split}": score
            for split, scores in res.items()
            for score_name, score in scores.items()
        }

        # ---------------------------------------------------------------------------------------------
        # Storage of Results
        # ---------------------------------------------------------------------------------------------

        # Save run and results
        # Create a flat dictionary which describes the run
        description = copy.copy(run)
        description["datetime"] = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        description["split_percentage_train"] = description["split_percentages"]["train"]
        description["split_percentage_val"] = description["split_percentages"]["val"]
        description["split_percentage_test"] = description["split_percentages"]["test"]
        del description["split_percentages"]
        description["loss_function_type"] = description["loss_function"]["func_type"]
        description["loss_function_nclasses"] = description["loss_function"]["nclasses"]
        del description["loss_function"]
        description["early_stopping_min"] = description["early_stopping"]["min"]
        description["early_stopping_wait"] = description["early_stopping"]["wait"]
        del description["early_stopping"]
        description = dict(**description, **description["params"])
        del description["params"]
        del description["g"]
        del description["split"]
        if "lg" in description:
            del description["lg"]
        description["net"] = description["net"].__name__
        description = dict(**description, **res)

        with open(f'../results/{description["name"]}.json', "w") as f:
            json.dump(description, f, indent=4)

        # Save logs
        if run["save_logs"]:
            log.to_csv(f'../logs/{description["name"]}.csv', index=False)

        # Save model
        if run["save_models"]:
            th.save(net.state_dict(), f'../models/{description["name"]}.pth')

        # ZIP-compress model
        if run["save_models"] and run["zip_models"]:
            with ZipFile(f'../models/{description["name"]}.zip', "w") as zip:
                zip.write(
                    filename=f'../models/{description["name"]}.pth',
                    compress_type=zipDEF,
                    compresslevel=9,
                )

        # Save prediction
        if run["save_predictions"]:
            np.save(f'../predictions/{description["name"]}_pred', final_prediction.numpy())

        print(
            f"{datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}: Completed run {run['name']} in {time.time()-starttime:.4f} seconds."
        )
        return {
            "time": datetime.datetime.now(),
            "duration": time.time() - starttime,
            "name": run["name"],
            "error": None,
        }

    except Exception as e:
        print(
            f"{datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}: Run {run['name']} FAILED after {time.time()-starttime:.4f} seconds."
        )
        if run["verbatim"]:
            print(traceback.format_exc())
        return {
            "time": datetime.datetime.now(),
            "duration": time.time() - starttime,
            "name": run["name"],
            "error": traceback.format_exc(),
        }


###################################################################################################
# -------------------------------------------------------------------------------------------------
# -------------------------------------------------------------------------------------------------
# Multiprocessing Part


if __name__ == "__main__":
    # ---------------------------------------------------------------------------------------------
    # Configuration
    # ---------------------------------------------------------------------------------------------

    with open(f'../experiments/base configuration.json', "r") as f:
            conf = json.load(f)

    

    #%%
    # ---------------------------------------------------------------------------------------------
    # Selection of Training, Validation and Test Splits
    # ---------------------------------------------------------------------------------------------

    permutation_files = os.listdir("../data/permutations")
    permutations = {
        i: pickle.load(open(f"../data/permutations/cora_permutation{i}.pickle", "rb"))
        for i in conf["permutations"]
    }
    print(f"Loaded {len(permutations)} different permutations.")

    def get_split(perm, percentage_train, percentage_val, percentage_test):
        mask_train = np.zeros(len(perm))
        mask_val = np.zeros(len(perm))
        mask_test = np.zeros(len(perm))

        i_train = int(percentage_train * len(perm))
        i_val = i_train + int(percentage_val * len(perm))
        # for the test split, we take the end of the permutation to ensure
        # that it is the same for different train/val percentages
        i_test = len(perm) - int(percentage_test * len(perm))

        mask_train[perm[range(0, i_train)]] = 1
        mask_val[perm[range(i_train, i_val)]] = 1
        mask_test[perm[range(i_test, len(perm))]] = 1

        mask_train = th.BoolTensor(mask_train)
        mask_val = th.BoolTensor(mask_val)
        mask_test = th.BoolTensor(mask_test)

        return {"train": mask_train, "val": mask_val, "test": mask_test}

    splits = [
        {
            "permutation": perm_i,
            "split_percentages": split_per,
            "split": get_split(perm, split_per["train"], split_per["val"], split_per["test"]),
        }
        for perm_i, perm in permutations.items()
        for split_per in conf["split_percentages"]
    ]

    print(f"Created {len(splits)} different splits.")

    def get_net(name):
        if name=="GCN_Net": return GCN_Net
        if name=="GAT_Net_fast": return GAT_Net_fast
        if name=="LGNN_Net": return LGNN_Net
        raise ValueError(f'Network {name} provided in config does not exist.')

    #%%
    # ---------------------------------------------------------------------------------------------
    # Define Runs
    # ---------------------------------------------------------------------------------------------

    runs = []
    run_id = 0

    for split in splits:
        for repetition in range(conf["repetitions"]):
            for learning_rate in conf["learning_rates"]:
                for weight_decay in conf["weight_decays"]:
                    for loss_function in conf["loss_functions"]:
                        for net in conf["nets"]:
                            for feature in net["features"]:
                                params = [{"placeholder": None}]
                                for param_name, param_list in net["structure"].items():
                                    params = [
                                        dict(**p, **{param_name: p_add})
                                        for p in params
                                        for p_add in param_list
                                    ]
                                for param_name, param_list in net["tricks"].items():
                                    params = [
                                        dict(**p, **{param_name: p_add})
                                        for p in params
                                        for p_add in param_list
                                    ]

                                for param_set in params:
                                    runs.append(
                                        {
                                            "name": f"{conf['name']}-{run_id}",
                                            "save_logs": conf["save_logs"],
                                            "save_predictions": conf["save_predictions"],
                                            "save_models": conf["save_models"],
                                            "zip_models": conf["zip_models"],
                                            "verbatim": conf["verbatim"],
                                            "permutation": split["permutation"],
                                            "split_percentages": split["split_percentages"],
                                            "split": split["split"],
                                            "repetition": repetition + 1,
                                            "learning_rate": learning_rate,
                                            "weight_decay": weight_decay,
                                            "loss_function": loss_function,
                                            "net": get_net(net["type"]),
                                            "early_stopping": net["early_stopping"],
                                            "feature": feature,
                                            "params": param_set,
                                        }
                                    )
                                    run_id += 1

    for run in runs:
        del run["params"]["placeholder"]
        if run["net"] == LGNN_Net:
            run["params"]["g"] = g
            run["params"]["lg"] = lg
        else:
            run["params"]["g"] = g_selfl

        if run["feature"] == "keywords":
            run["params"]["in_feats"] = features_keywords.shape[1]
        elif run["feature"] == "node_id":
            run["params"]["in_feats"] = features_nodeid.shape[1]
        else:
            run["params"]["in_feats"] = 1

        run["params"]["out_feats"] = num_classes

    print(f"Registered {len(runs)} different runs.")

    #%%
    # ---------------------------------------------------------------------------------------------
    # Parallel Execution
    # ---------------------------------------------------------------------------------------------
    jobs = runs[0:4]

    progress = []
    batchsize = conf["batchsize"]
    n_processes = conf["n_processes"]
    current_job_i = 0
    print(
        f"\n###################################################################\nSTARTING EXPERIMENTS\n###################################################################"
    )

    while current_job_i < len(jobs):
        print(
            f"\n#-------------------------------------------------------------------\nNEXT BATCH (jobs {current_job_i}-{current_job_i+batchsize-1})\n#-------------------------------------------------------------------"
        )
        with Pool(processes=n_processes) as pool:  # start worker processes
            completed = pool.map(perform_run, jobs[current_job_i : (current_job_i + batchsize)])

        progress.append(pd.DataFrame(completed).sort_values("time"))
        witherror = progress[-1]["error"].notnull().sum()
        print("\n-----------------\nBATCH COMPLETED:")
        print(
            f"{datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}: {len(completed)-witherror} jobs completed, {witherror} jobs failed:"
        )
        print(progress[-1])
        current_job_i += batchsize

        status = pd.concat(progress)
        status.to_csv(f'../results/job overview {conf["name"]}.csv', index=False)
        status["is_error"] = status["error"].notnull()
        progress_summary = pd.concat(
            [
                status.groupby("is_error")["duration"].mean(),
                status.groupby("is_error")["error"].size().rename("count"),
            ],
            axis=1,
        )

        print("\n-----------------\nPROGRESS SUMMARY:")
        print(progress_summary)
        progress_summary.to_csv(f'../results/status summary {conf["name"]}.csv', index=False)

    print(
        f"\n###################################################################\nEXPERIMENTS COMPLETED\n###################################################################"
    )
    print("JOB OVERVIEW:")
    print(pd.concat(progress))
    print("\n-----------------\nPROGRESS SUMMARY:")
    print(progress_summary)

