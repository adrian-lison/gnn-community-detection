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

    def __init__(self, in_feats, out_feats, activation):
        super(LinearModule, self).__init__()
        self.linear = nn.Linear(in_feats, out_feats)
        self.activation = activation  # This is the activation function

    def forward(self, node):
        h = node.data["h"]
        h = self.linear(h)
        h = self.activation(h)
        return {"h": h}


class GCN(nn.Module):
    """GCN layer"""

    def __init__(self, in_feats, out_feats, activation, batchnorm=False):
        super(GCN, self).__init__()
        self.apply_mod = LinearModule(in_feats, out_feats, activation)
        self.batchnorm = batchnorm
        self.bn = nn.BatchNorm1d(out_feats)

    def forward(self, g, feature):
        g.ndata["h"] = feature
        g.update_all(
            message_func=fn.copy_src(src="h", out="m"), reduce_func=fn.sum(msg="m", out="h")
        )
        g.apply_nodes(func=self.apply_mod)

        res = g.ndata.pop("h")
        if self.batchnorm:
            res = self.bn(res)
        return res


class GCN_Net(nn.Module):
    """Complete network"""

    def __init__(self, g, in_feats, hidden_size, hidden_layers, out_feats, dropout, batchnorm):
        super(GCN_Net, self).__init__()
        self.g = g
        self.gcn_in = GCN(in_feats, hidden_size, F.relu, batchnorm)
        self.gcn_hidden = [
            GCN(hidden_size, hidden_size, F.relu, batchnorm) for i in range(hidden_layers)
        ]
        self.dropout = nn.Dropout(dropout)
        self.gcn_out = GCN(hidden_size, out_feats, F.relu, batchnorm)

    def forward(self, features):
        h = self.gcn_in(self.g, features)
        for layer in self.gcn_hidden:
            h = self.dropout(h)
            h = layer(self.g, h)
        h = self.dropout(h)
        h = self.gcn_out(self.g, h)
        h = F.log_softmax(h, 1)
        return h
