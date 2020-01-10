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
import scipy.sparse as ss

# ----------------------------------------------------------------------------
# GNN Definition
# ----------------------------------------------------------------------------


def aggregate_radius(radius, g, z):
    """Return a list containing features gathered from multiple hops in a radius."""
    # initializing list to collect message passing result
    z_list = []
    g.ndata["z"] = z
    # pulling message from 1-hop neighbourhood
    g.update_all(fn.copy_src(src="z", out="m"), fn.sum(msg="m", out="z"))
    z_list.append(g.ndata["z"])
    for i in range(radius - 1):
        for j in range(2 ** i):
            # pulling message from 2^j neighborhood
            g.update_all(fn.copy_src(src="z", out="m"), fn.sum(msg="m", out="z"))
        z_list.append(g.ndata["z"])
    return z_list


class LGNNCore(nn.Module):
    def __init__(self, in_feats, out_feats, radius, batchnorm):
        super(LGNNCore, self).__init__()
        self.out_feats = out_feats
        self.radius = radius
        self.batchnorm = batchnorm

        self.linear_prev = nn.Linear(in_feats, out_feats)
        self.linear_deg = nn.Linear(in_feats, out_feats)
        self.linear_radius = nn.ModuleList([nn.Linear(in_feats, out_feats) for i in range(radius)])
        self.linear_fuse = nn.Linear(in_feats, out_feats)
        self.bn = nn.BatchNorm1d(out_feats)

    def forward(self, g, feat_a, feat_b, deg, pm_pd):
        # term "prev"
        prev_proj = self.linear_prev(feat_a)
        # term "deg"
        deg_proj = self.linear_deg(deg * feat_a)

        # term "radius"
        # aggregate 2^j-hop features
        hop2j_list = aggregate_radius(self.radius, g, feat_a)
        # apply linear transformation
        hop2j_list = [linear(x) for linear, x in zip(self.linear_radius, hop2j_list)]
        radius_proj = sum(hop2j_list)

        # term "fuse"
        fuse = self.linear_fuse(th.mm(pm_pd, feat_b))

        # sum them together
        result = prev_proj + deg_proj + radius_proj + fuse

        # skip connection and batch norm
        n = self.out_feats
        result = th.cat([result[:, :n], F.relu(result[:, n:])], 1)
        if self.batchnorm:
            result = self.bn(result)

        return result


class LGNNLayer(nn.Module):
    def __init__(self, in_feats, in_feats_lg, out_feats, radius, batchnorm):
        super(LGNNLayer, self).__init__()
        self.g_layer = LGNNCore(in_feats, out_feats, radius, batchnorm)
        self.lg_layer = LGNNCore(in_feats_lg, out_feats, radius, batchnorm)

    def forward(self, g, lg, h, lg_h, deg_g, deg_lg, pm_pd):
        next_h = self.g_layer(g, h, lg_h, deg_g, pm_pd)
        pm_pd_y = th.transpose(pm_pd, 0, 1)
        next_lg_h = self.lg_layer(lg, lg_h, h, deg_lg, pm_pd_y)
        return next_h, next_lg_h


class LGNN_Net(nn.Module):
    def __init__(self, g, in_feats, hidden_size, out_feats, dropout, batchnorm, lg, radius):
        super(LGNN_Net, self).__init__()
        self.g = g
        self.lg = lg

        # pmpd
        matrix = ss.lil_matrix((g.number_of_nodes(), g.number_of_edges()))
        for s, d in zip(g.edges()[0], g.edges()[1]):
            matrix[s, g.edge_id(s, d)] = -1
            matrix[d, g.edge_id(s, d)] = 1
        inputs_pmpd = ss.coo_matrix(matrix, dtype="int64")
        indices = th.LongTensor([inputs_pmpd.row, inputs_pmpd.col])
        self.pmpd = th.sparse.FloatTensor(
            indices, th.from_numpy(inputs_pmpd.data).float(), inputs_pmpd.shape
        )

        # input
        self.layer1 = LGNNLayer(in_feats, in_feats, hidden_size, radius, batchnorm)

        self.layer2 = LGNNLayer(hidden_size, hidden_size, hidden_size, radius, batchnorm)

        self.layer3 = LGNNLayer(hidden_size, hidden_size, hidden_size, radius, batchnorm)

        # predict classes
        self.linear = nn.Linear(hidden_size, hidden_size, out_feats)

        self.dropout = nn.Dropout(dropout)

        # compute the degrees
        self.deg_g = g.in_degrees().float().unsqueeze(1)
        self.deg_lg = lg.in_degrees().float().unsqueeze(1)

    def forward(self, features):
        # assume that features is a tuple of g_features and lg_features
        (h, lg_h) = features

        h, lg_h = self.layer1(self.g, self.lg, h, lg_h, self.deg_g, self.deg_lg, self.pmpd)

        h = self.dropout(h)
        h, lg_h = self.layer2(self.g, self.lg, h, lg_h, self.deg_g, self.deg_lg, self.pmpd)

        h = self.dropout(h)
        h, lg_h = self.layer3(self.g, self.lg, h, lg_h, self.deg_g, self.deg_lg, self.pmpd)

        return self.linear(h)
