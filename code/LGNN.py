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
from dgl.nn.pytorch.conv import GraphConv as GCNconv

import numpy as np
import scipy.sparse as ss

# ----------------------------------------------------------------------------
# GNN Definition
# ----------------------------------------------------------------------------


class LGNNCore(nn.Module):
    def __init__(self, g, in_feats, out_feats, radius, batchnorm):
        super(LGNNCore, self).__init__()
        self.out_feats = out_feats
        self.radius = radius
        self.batchnorm = batchnorm
        self.g = g

        self.linear_prev = nn.Linear(in_feats, out_feats)
        self.linear_deg = nn.Linear(in_feats, out_feats)
        self.linear_radius = nn.ModuleList([nn.Linear(in_feats, out_feats) for i in range(radius)])
        self.gcnconv = GCNconv(in_feats, out_feats)
        self.linear_fuse = nn.Linear(in_feats, out_feats)
        self.bn = nn.BatchNorm1d(out_feats)

    def aggregate_radius(self, radius, z):
        """Return a list containing features gathered from multiple hops in a radius."""
        """# initializing list to collect message passing result
        z_list = []
        g.ndata["z"] = z
        # pulling message from 1-hop neighbourhood
        g.update_all(message_func=fn.copy_src(src="z", out="m"), reduce_func=fn.sum(msg="m", out="z"))
        z_list.append(g.ndata["z"])
        for i in range(radius - 1):
            for j in range(2 ** i):
                # pulling message from 2^j neighborhood
                g.update_all(fn.copy_src(src="z", out="m"), fn.sum(msg="m", out="z"))
            z_list.append(g.ndata["z"])
        try:
            res = g.ndata.pop("z")
        except Exception as e:
            print(e)"""

        self.g.ndata["z"] = z
        self.g.update_all(
            message_func=fn.copy_src(src="z", out="m"), reduce_func=fn.sum(msg="m", out="z")
        )
        res = self.g.ndata.pop("z")
        return res

    def forward(self, feat_a, feat_b, deg, pm_pd):
        # term "prev"
        prev_proj = self.linear_prev(feat_a)
        # term "deg"
        deg_proj = self.linear_deg(deg * feat_a)

        # term "radius"
        # aggregate 2^j-hop features
        # hop2j_list = self.aggregate_radius(self.radius, feat_a)
        # apply linear transformation
        # hop2j_list = [linear(x) for linear, x in zip(self.linear_radius, hop2j_list)]
        # radius_proj = sum(hop2j_list)
        # radius_proj = self.linear_radius[0](hop2j_list)
        radius_proj = self.gcnconv.forward(self.g, feat_a)

        # term "fuse"
        fuse = self.linear_fuse(th.mm(pm_pd, feat_b))

        # sum them together
        result = prev_proj + deg_proj + fuse + radius_proj

        # skip connection and batch norm
        n = self.out_feats
        result = th.cat([result[:, :n], F.relu(result[:, n:])], 1)
        if self.batchnorm:
            result = self.bn(result)

        return result


class LGNNLayer(nn.Module):
    def __init__(self, g, lg, in_feats, in_feats_lg, out_feats, radius, batchnorm):
        super(LGNNLayer, self).__init__()
        self.g = g
        self.lg = lg
        self.g_layer = LGNNCore(self.g, in_feats, out_feats, radius, batchnorm)
        self.lg_layer = LGNNCore(self.lg, in_feats_lg, out_feats, radius, batchnorm)

    def forward(self, h, lg_h, deg_g, deg_lg, pm_pd):
        next_h = self.g_layer(h, lg_h, deg_g, pm_pd)
        pm_pd_y = th.transpose(pm_pd, 0, 1)
        # next_lg_h = self.lg_layer(lg_h, h, deg_lg, pm_pd_y)
        next_lg_h = th.tensor.rand(lg_h.shape[0], next_h.shape[1])
        return next_h, next_lg_h


class LGNN_Net_old(nn.Module):
    def __init__(
        self, g, in_feats, hidden_size, hidden_layers, out_feats, dropout, batchnorm, lg, radius
    ):
        super(LGNN_Net_old, self).__init__()
        self.g = g
        self.lg = lg

        # pmpd
        matrix = ss.lil_matrix((self.g.number_of_nodes(), self.g.number_of_edges()))
        for s, d in zip(self.g.edges()[0], self.g.edges()[1]):
            matrix[s, self.g.edge_id(s, d)] = -1
            matrix[d, self.g.edge_id(s, d)] = 1
        inputs_pmpd = ss.coo_matrix(matrix, dtype="int64")
        indices = th.LongTensor([inputs_pmpd.row, inputs_pmpd.col])
        self.pmpd = th.sparse.FloatTensor(
            indices, th.from_numpy(inputs_pmpd.data).float(), inputs_pmpd.shape
        )

        # input
        self.layer_in = LGNNLayer(
            self.g, self.lg, in_feats, in_feats, hidden_size, radius, batchnorm
        )

        self.layer_hidden = [
            LGNNLayer(self.g, self.lg, hidden_size, hidden_size, hidden_size, radius, batchnorm)
            for i in range(hidden_layers)
        ]

        # predict classes
        self.layer_out = nn.Linear(hidden_size, out_feats)

        self.dropout = nn.Dropout(dropout)

        # compute the degrees
        self.deg_g = self.g.in_degrees().float().unsqueeze(1)
        self.deg_lg = self.lg.in_degrees().float().unsqueeze(1)

    def forward(self, features):
        # assume that features is a tuple of g_features and lg_features
        (h, lg_h) = features

        h, lg_h = self.layer_in(h, lg_h, self.deg_g, self.deg_lg, self.pmpd)

        for layer in self.layer_hidden:
            h = self.dropout(h)
            h, lg_h = layer(h, lg_h, self.deg_g, self.deg_lg, self.pmpd)

        h = self.dropout(h)
        h = self.layer_out(h)
        h = F.log_softmax(h, 1)
        return h


class LGNNModule(nn.Module):
    def __init__(self, in_feats, out_feats, radius, batchnorm):
        super().__init__()
        self.out_feats = out_feats
        self.radius = radius

        new_linear = lambda: nn.Linear(in_feats, out_feats)
        new_linear_list = lambda: nn.ModuleList([new_linear() for i in range(radius)])

        self.theta_x, self.theta_deg, self.theta_y = new_linear(), new_linear(), new_linear()
        self.theta_list = new_linear_list()

        self.gamma_y, self.gamma_deg, self.gamma_x = new_linear(), new_linear(), new_linear()
        self.gamma_list = new_linear_list()

        self.batchnorm = batchnorm
        self.bn_x = nn.BatchNorm1d(out_feats)
        self.bn_y = nn.BatchNorm1d(out_feats)

    def aggregate(self, a_graph, z):
        z_list = []
        a_graph.set_n_repr({"z": z})
        a_graph.update_all(fn.copy_src(src="z", out="m"), fn.sum(msg="m", out="z"))
        z_list.append(a_graph.get_n_repr()["z"])
        for i in range(self.radius - 1):
            for j in range(2 ** i):
                a_graph.update_all(fn.copy_src(src="z", out="m"), fn.sum(msg="m", out="z"))
            z_list.append(a_graph.get_n_repr()["z"])
        return z_list

    def forward(self, g, lg, x, y, deg_g, deg_lg, pm_pd, last=False):
        pmpd_x = F.embedding(pm_pd, x)

        sum_x = sum(theta(z) for theta, z in zip(self.theta_list, self.aggregate(g, x)))

        g.set_e_repr({"y": y})
        g.update_all(fn.copy_edge(edge="y", out="m"), fn.sum("m", "pmpd_y"))
        pmpd_y = g.pop_n_repr("pmpd_y")

        x = self.theta_x(x) + self.theta_deg(deg_g * x) + sum_x + self.theta_y(pmpd_y)
        n = self.out_feats // 2
        x = th.cat([x[:, :n], F.relu(x[:, n:])], 1)
        if self.batchnorm:
            x = self.bn_x(x)

        if last:
            return x
        else:
            sum_y = sum(gamma(z) for gamma, z in zip(self.gamma_list, self.aggregate(lg, y)))

            y = self.gamma_y(y) + self.gamma_deg(deg_lg * y) + sum_y + self.gamma_x(pmpd_x)
            y = th.cat([y[:, :n], F.relu(y[:, n:])], 1)
            if self.batchnorm:
                y = self.bn_y(y)

            return x, y


class LGNN_Net(nn.Module):
    def __init__(
        self, g, in_feats, hidden_size, hidden_layers, out_feats, dropout, batchnorm, lg, radius
    ):
        super(LGNN_Net, self).__init__()
        self.g = g
        self.lg = lg
        in_feats = [in_feats] + [hidden_size] * hidden_layers
        self.module_list = nn.ModuleList(
            [GNNModule(m, n, radius, batchnorm) for m, n in zip(in_feats[:-1], in_feats[1:])]
        )
        self.linear = nn.Linear(in_feats[-1], out_feats)
        self.dropout = nn.Dropout(dropout)

        self.pmpd = self.g.edges()[0]

        # compute the degrees
        self.deg_g = self.g.in_degrees().float().unsqueeze(1)
        self.deg_lg = self.lg.in_degrees().float().unsqueeze(1)

    def forward(self, features):
        (h, lg_h) = features
        for i, module in enumerate(self.module_list):
            if i == len(self.module_list) - 1:
                h = self.lastmodule(
                    self.g, self.lg, h, lg_h, self.deg_g, self.deg_lg, self.pmpd, last=True
                )  # the last pass should only return h, otherwhise we get a memory leak
                h = self.dropout(h)
            else:
                h, lg_h = module(self.g, self.lg, h, lg_h, self.deg_g, self.deg_lg, self.pmpd)
                h = self.dropout(h)
                lg_h = self.dropout(lg_h)
        return self.linear(h)
