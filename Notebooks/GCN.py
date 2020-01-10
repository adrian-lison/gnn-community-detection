# ----------------------------------------------------------------------------
# Implementation of Graph Convolutional Network (Kipf and Welling)

# ----------------------------------------------------------------------------
# Imports
# ----------------------------------------------------------------------------
import dgl
import dgl.function as fn
import torch as th
import torch.nn as nn
import torch.nn.functional as F
from dgl import DGLGraph

import numpy as np

# ----------------------------------------------------------------------------
# GNN Definition
# ----------------------------------------------------------------------------
class LinearModule(nn.Module):
    """Linear transformation part of the GCN layer"""

    def __init__(self, in_feats, out_feats, activation, batchnorm=False):
        super(LinearModule, self).__init__()
        self.batchnorm = batchnorm
        self.bn = nn.BatchNorm1d(in_feats)
        self.linear = nn.Linear(in_feats, out_feats)
        self.activation = activation  # This is the activation function

    def forward(self, node):
        if self.batchnorm:
            h = self.bn(node.data["h"])
        else:
            h = node.data["h"]
        h = self.linear(h)
        h = self.activation(h)
        return {"h": h}


class GCN(nn.Module):
    """GCN layer"""

    def __init__(self, in_feats, out_feats, activation, batchnorm=False):
        super(GCN, self).__init__()
        self.apply_mod = LinearModule(in_feats, out_feats, activation, batchnorm)

    def forward(self, g, feature):
        g.ndata["h"] = feature
        g.update_all(
            message_func=fn.copy_src(src="h", out="m"), reduce_func=fn.sum(msg="m", out="h")
        )
        g.apply_nodes(func=self.apply_mod)
        return g.ndata.pop("h")


class GCN_Net(nn.Module):
    """Complete network"""

    def __init__(self, g, in_feats, hidden_size, out_feats, dropout, batchnorm):
        super(GCN_Net, self).__init__()
        self.g = g
        self.gcn1 = GCN(in_feats, hidden_size, F.relu, batchnorm)
        self.dropout = nn.Dropout(dropout)
        self.gcn2 = GCN(hidden_size, out_feats, F.relu, batchnorm)

    def forward(self, features):
        h = self.gcn1(self.g, features)
        h = self.dropout(h)
        h = self.gcn2(self.g, h)
        h = F.log_softmax(h, 1)
        return h
