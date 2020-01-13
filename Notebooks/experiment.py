# ----------------------------------------------------------------------------
# Implementation of GNN experiments

#%%
# ----------------------------------------------------------------------------
# General Imports
# ----------------------------------------------------------------------------

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
import itertools
import os
import time

# evaluation
import performance as pf

# GNN classes
from GCN import GCN_Net
from GAT import GAT_Net_fast
from LGNN import LGNN_Net

#%%
# ----------------------------------------------------------------------------
# Configuration
# ----------------------------------------------------------------------------

conf = {
    "name": "conf1",
    "split_percentages": [
        {"train": 0.04, "val": 0.01, "test": 0.95},
        {"train": 0.08, "val": 0.02, "test": 0.9},
        {"train": 0.16, "val": 0.04, "test": 0.8},
    ],
    "permutations": [1, 2, 3, 4],
    "repetitions": 2,
    "learning_rates": [1e-2],
    "weight_decays": [0, 1e-2],
    "loss_functions": [
        {"func_type": "nll", "nclasses": "all"},
        {"func_type": "inv", "nclasses": 6},
        {"func_type": "inv", "nclasses": 7},
    ],
    "nets": [
        {
            "type": GCN_Net,
            "early_stopping": {"min": 300, "wait": 100},
            "features": ["node_id", "keywords"],
            "structure": {"hidden_size": [50, 100], "hidden_layers": [1, 2]},
            "tricks": {"dropout": [0, 0.2, 0.4], "batchnorm": [False, True]},
        },
        {
            "type": GAT_Net_fast,
            "early_stopping": {"min": 300, "wait": 100},
            "features": ["node_id", "keywords"],
            "structure": {"hidden_size": [50, 100], "hidden_layers": [2, 3], "num_heads": [1, 2]},
            "tricks": {
                "dropout": [0, 0.2, 0.4],
                "batchnorm": [False, True],
                "residual": [False, True],
            },
        },
        {
            "type": LGNN_Net,
            "early_stopping": {"min": 300, "wait": 100},
            "features": ["node_degree", "keywords"],
            "structure": {"hidden_size": [50, 100], "hidden_layers": [1, 2]},
            "tricks": {"dropout": [0, 0.2, 0.4], "batchnorm": [False, True], "radius": [1, 2, 3]},
        },
    ],
}

#%%
# ----------------------------------------------------------------------------
# Data Loading
# ----------------------------------------------------------------------------

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

print(f"Loaded Cora dataset.")

#%%
# ----------------------------------------------------------------------------
# Selection of Training, Validation and Test Splits
# ----------------------------------------------------------------------------

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

#%%
# ----------------------------------------------------------------------------
# Loss Functions
# ----------------------------------------------------------------------------

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
# ----------------------------------------------------------------------------
# Define Runs
# ----------------------------------------------------------------------------

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
                                        "permutation": split["permutation"],
                                        "split_percentages": split["split_percentages"],
                                        "split": split["split"],
                                        "repetition": repetition + 1,
                                        "learning_rate": learning_rate,
                                        "weight_decay": weight_decay,
                                        "loss_function": loss_function,
                                        "net": net["type"],
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
# ----------------------------------------------------------------------------
# Initialize Network with Parameters
# ----------------------------------------------------------------------------

run = runs[2040]
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

#%%
# ----------------------------------------------------------------------------
# Training
# ----------------------------------------------------------------------------

loss_f = init_loss_function(labels=labels, func_type=run["loss_function"]["func_type"], nclasses=run["loss_function"]["nclasses"])
optimizer = th.optim.Adam(net.parameters(), lr=run["learning_rate"], weight_decay=run["weight_decay"])
net.train() # Set to training mode

dur = []
loss_ev = []
current_best = -np.infty #arbitrarily bad
current_best_epoch = 0
current_best_params = None
no_improvement_for = 0

for epoch in range(10000):
    if epoch >=3:
        t0 = time.time()

    # Compute performance for train, validation and test set
    net.eval()
    prediction = net(net_features)
    train_rand=pf.rand_score(labels[mask_train].numpy(),np.argmax(prediction[mask_train].detach().numpy(), axis=1))
    validation_rand=pf.rand_score(labels[mask_val].numpy(),np.argmax(prediction[mask_val].detach().numpy(), axis=1))
    test_rand=pf.rand_score(labels[mask_test].numpy(),np.argmax(prediction[mask_test].detach().numpy(), axis=1))

    # Save current best model
    if train_rand>current_best:
        current_best = train_rand
        current_best_epoch = epoch
        current_best_params = copy.deepcopy(net.state_dict())
        no_improvement_for = 0
    else: no_improvement_for += 1
    
    # Apply early stopping
    if epoch>run["early_stopping"]["min"] and no_improvement_for>run["early_stopping"]["wait"]:
        break

    net.train()

    # Compute loss for train nodes
    logits = net(net_features)

    loss = loss_f(logits=logits,labels=labels,mask=mask_train)
    loss_ev.append(loss.detach().item())
    
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if epoch >=3:
        dur.append(time.time() - t0)
        print(f"Epoch {epoch:05d} | Loss {loss.item():.4f} | Train.Rand {train_rand:.4f} | Valid.Rand {validation_rand:.4f} | Time(s) {np.mean(dur):.4f}")
    else:
        print(f"Epoch {epoch:05d} | Loss {loss.item():.4f} | Train.Rand {train_rand:.4f} | Valid.Rand {validation_rand:.4f} | Time(s) unknown")
        
net.load_state_dict(current_best_params)

