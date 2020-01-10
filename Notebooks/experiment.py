# ----------------------------------------------------------------------------
# Implementation of GNN experiments

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

# evaluation
import performance as pf

# GNN classes
from GCN import GCN_Net
from GAT import GAT_Net

# ----------------------------------------------------------------------------
# Configuration
# ----------------------------------------------------------------------------

conf = {
    "split_percentages": [
        {"train": 0.04, "val": 0.01, "test": 0.95},
        {"train": 0.08, "val": 0.02, "test": 0.9},
        {"train": 0.16, "val": 0.04, "test": 0.8},
    ],
    "permutations": [1, 2, 3, 4],
    "repetitions": 2,
}

# ----------------------------------------------------------------------------
# Data Loading
# ----------------------------------------------------------------------------

data = citegrh.load_cora()
labels = th.LongTensor(data.labels)
g = data.graph

# add self loop to graph
g.remove_edges_from(nx.selfloop_edges(g))
g = DGLGraph(g)
g.add_edges(g.nodes(), g.nodes())

features_abstract = th.FloatTensor(data.features)
features_nodeid = th.eye(g.number_of_nodes())

print(f"Loaded Cora dataset.")

# ----------------------------------------------------------------------------
# Selection of Training, Validation and Test Splits
# ----------------------------------------------------------------------------

permutation_files = os.listdir("../data/permutations")
permutations = {
    i: pickle.load(open("../data/permutations/cora_permutation{i}.pickle", "rb"))
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
    i_test = len(perm) - int(percentage_val * len(perm))

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
        "split": get_split(
            perm, split_per["train"], split_per["val"], split_per["test"]
        ),
    }
    for perm_i, perm in permutations.items()
    for split_per in conf["split_percentages"]
]

print(f"Created {len(splits)} different splits.")

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


# ----------------------------------------------------------------------------
# Networks and Parameters
# ----------------------------------------------------------------------------



