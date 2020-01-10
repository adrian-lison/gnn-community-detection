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
class GATLayer(nn.Module):
    def __init__(self, g, in_feats, out_feats, batchnorm):
        super(GATLayer, self).__init__()
        self.g = g
        # equation (1)
        self.fc = nn.Linear(in_feats, out_feats, bias=False)
        # equation (2)
        self.attn_fc = nn.Linear(2 * out_feats, 1, bias=False)

        self.batchnorm = batchnorm
        self.bn = nn.BatchNorm1d(out_feats)

    def edge_attention(self, edges):
        # edge UDF for equation (2)
        z2 = th.cat([edges.src["z"], edges.dst["z"]], dim=1)
        a = self.attn_fc(z2)
        return {"e": F.leaky_relu(a)}

    def message_func(self, edges):
        # message UDF for equation (3) & (4)
        return {"z": edges.src["z"], "e": edges.data["e"]}

    def reduce_func(self, nodes):
        # reduce UDF for equation (3) & (4)
        # equation (3)
        alpha = F.softmax(nodes.mailbox["e"], dim=1)
        # equation (4)
        h = th.sum(alpha * nodes.mailbox["z"], dim=1)
        return {"h": h}

    def forward(self, h):
        # equation (1)
        z = self.fc(h)
        self.g.ndata["z"] = z
        # equation (2)
        self.g.apply_edges(self.edge_attention)
        # equation (3) & (4)
        self.g.update_all(self.message_func, self.reduce_func)

        res = self.g.ndata.pop("h")
        if self.batchnorm:
            res = self.bn(res)
        return res


class MultiHeadGATLayer(nn.Module):
    def __init__(self, g, in_feats, out_feats, num_heads, batchnorm, merge="cat"):
        super(MultiHeadGATLayer, self).__init__()
        self.heads = nn.ModuleList()
        for i in range(num_heads):
            self.heads.append(GATLayer(g, in_feats, out_feats, batchnorm))
        self.merge = merge

    def forward(self, h):
        head_outs = [attn_head(h) for attn_head in self.heads]
        if self.merge == "cat":
            # concat on the output feature dimension (dim=1)
            return th.cat(head_outs, dim=1)
        else:
            # merge using average
            return th.mean(th.stack(head_outs))


class GAT_Net(nn.Module):
    def __init__(
        self, g, in_feats, hidden_size, hidden_layers, out_feats, dropout, batchnorm, num_heads
    ):
        super(GAT_Net, self).__init__()
        # The input dimension is hidden_dim*num_heads since
        # multiple head outputs are concatenated together. Also, only
        # one attention head in the output layer.
        self.layer_in = MultiHeadGATLayer(g, in_feats, hidden_size, num_heads, batchnorm)
        self.layer_hidden = [
            MultiHeadGATLayer(g, hidden_size * num_heads, hidden_size, num_heads, batchnorm)
            for i in range(hidden_layers)
        ]
        self.layer_out = MultiHeadGATLayer(g, hidden_size * num_heads, out_feats, 1, batchnorm)

        self.dropout = nn.Dropout(dropout)

    def forward(self, features):
        h = self.layer_in(features)
        h = F.elu(h)
        for layer in self.layer_hidden:
            h = self.dropout(h)
            h = layer(h)
            h = F.elu(h)
        h = self.dropout(h)
        h = self.layer_out(h)
        h = F.log_softmax(h, 1)
        return h
