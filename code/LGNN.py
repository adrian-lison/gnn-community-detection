# ----------------------------------------------------------------------------
# Implementation of Line Graph Neural Network (LGNN)

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


class LGNNModule(nn.Module):
    """This is a graph network block of LGNN"""

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
            return x  # do not compute the rest of the lgnn part to avoid memory leak

        sum_y = sum(gamma(z) for gamma, z in zip(self.gamma_list, self.aggregate(lg, y)))

        y = self.gamma_y(y) + self.gamma_deg(deg_lg * y) + sum_y + self.gamma_x(pmpd_x)
        y = th.cat([y[:, :n], F.relu(y[:, n:])], 1)
        if self.batchnorm:
            y = self.bn_y(y)

        return x, y


class LGNN_Net(nn.Module):
    """This is a whole LGNN"""

    def __init__(
        self, g, in_feats, hidden_size, hidden_layers, out_feats, dropout, batchnorm, lg, radius
    ):
        super(LGNN_Net, self).__init__()
        self.g = g
        self.lg = lg
        in_feats = [in_feats] + [hidden_size] * hidden_layers
        self.module_list = nn.ModuleList(
            [LGNNModule(m, n, radius, batchnorm) for m, n in zip(in_feats[:-1], in_feats[1:])]
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
                h = module(
                    self.g, self.lg, h, lg_h, self.deg_g, self.deg_lg, self.pmpd, last=True
                )  # the last pass should only return h, otherwhise we get a memory leak
                h = self.dropout(h)
            else:
                h, lg_h = module(self.g, self.lg, h, lg_h, self.deg_g, self.deg_lg, self.pmpd)
                h = self.dropout(h)
                lg_h = self.dropout(lg_h)
        h = self.linear(h)
        h = F.log_softmax(h, 1)
        return h
