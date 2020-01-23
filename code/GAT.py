# ----------------------------------------------------------------------------
# Implementation of Graph Attention Network

# ----------------------------------------------------------------------------
# Imports
# ----------------------------------------------------------------------------
import dgl
import dgl.function as fn
import torch as th
import torch.nn as nn
import torch.nn.functional as F
from dgl.nn.pytorch import edge_softmax
from dgl.nn.pytorch.utils import Identity

from Sparsemax import Sparsemax

from dgl import DGLGraph

import numpy as np

# ----------------------------------------------------------------------------
# GNN Definition
# ----------------------------------------------------------------------------


class GATConv(nn.Module):
    """This is a graph network block of GAT"""

    def __init__(
        self,
        in_feats,
        out_feats,
        num_heads,
        feat_drop=0.0,
        attn_drop=0.0,
        negative_slope=0.2,
        residual=False,
        activation=None,
        sparsemax=False,
    ):
        super(GATConv, self).__init__()
        self._num_heads = num_heads
        self._in_feats = in_feats
        self._out_feats = out_feats
        self.fc = nn.Linear(in_feats, out_feats * num_heads, bias=False)
        self.attn_l = nn.Parameter(th.FloatTensor(size=(1, num_heads, out_feats)))
        self.attn_r = nn.Parameter(th.FloatTensor(size=(1, num_heads, out_feats)))
        self.feat_drop = nn.Dropout(feat_drop)
        self.attn_drop = nn.Dropout(attn_drop)
        self.leaky_relu = nn.LeakyReLU(negative_slope)

        self.sparsemax = sparsemax
        self.sparsemaxF = Sparsemax(dim=1)

        if residual:
            if in_feats != out_feats:
                self.res_fc = nn.Linear(in_feats, num_heads * out_feats, bias=False)
            else:
                self.res_fc = Identity()
        else:
            self.register_buffer("res_fc", None)
        self.reset_parameters()
        self.activation = activation

    def reset_parameters(self):
        """Reinitialize learnable parameters."""
        gain = nn.init.calculate_gain("relu")
        nn.init.xavier_normal_(self.fc.weight, gain=gain)
        nn.init.xavier_normal_(self.attn_l, gain=gain)
        nn.init.xavier_normal_(self.attn_r, gain=gain)
        if isinstance(self.res_fc, nn.Linear):
            nn.init.xavier_normal_(self.res_fc.weight, gain=gain)

    def relu_udf(self, edges):
        return {"e": self.leaky_relu(edges.data["e"])}

    def sparsemax_udf(self, edges):
        return {"a": self.sparsemaxF(edges.data["e"])}

    def attn_drop_udf(self, edges):
        return {"a": self.attn_drop(edges.data["a"])}

    def forward(self, graph, feat):
        graph = graph.local_var()
        h = self.feat_drop(feat)
        feat = self.fc(h).view(-1, self._num_heads, self._out_feats)
        el = (feat * self.attn_l).sum(dim=-1).unsqueeze(-1)
        er = (feat * self.attn_r).sum(dim=-1).unsqueeze(-1)
        graph.ndata.update({"ft": feat, "el": el, "er": er})
        # compute edge attention
        graph.apply_edges(fn.u_add_v("el", "er", "e"))
        # apply leaky relu
        graph.apply_edges(self.relu_udf)

        # compute softmax/sparsemax
        if self.sparsemax:
            graph.apply_edges(self.sparsemax_udf)
        else:
            graph.edata["a"] = edge_softmax(graph, graph.edata.pop("e"))

        # attention dropout
        graph.apply_edges(self.attn_drop_udf)

        # message passing
        graph.update_all(fn.u_mul_e("ft", "a", "m"), fn.sum("m", "ft"))
        rst = graph.ndata["ft"]
        # residual
        if self.res_fc is not None:
            resval = self.res_fc(h).view(h.shape[0], -1, self._out_feats)
            rst = rst + resval
        # activation
        if self.activation:
            rst = self.activation(rst)
        return rst


class GAT_Net_fast(nn.Module):
    """This is a whole GAT"""

    def __init__(
        self,
        g,
        in_feats,
        hidden_size,
        hidden_layers,
        out_feats,
        dropout,
        batchnorm,
        num_heads,
        residual,
        sparsemax=False,
    ):
        super(GAT_Net_fast, self).__init__()
        self.g = g
        self.hidden_layers = hidden_layers
        self.gat_layers = nn.ModuleList()
        self.activation = F.elu
        negative_slope = 0.2
        feat_drop = dropout
        attn_drop = feat_drop / 2

        self.batchnorm = batchnorm
        self.bn = [nn.BatchNorm1d(hidden_size * num_heads) for i in range(hidden_layers)]

        # input projection (no residual)
        self.gat_layers.append(
            GATConv(
                in_feats,
                hidden_size,
                num_heads,
                feat_drop,
                attn_drop,
                negative_slope,
                residual=False,
                activation=self.activation,
                sparsemax=sparsemax,
            )
        )
        # hidden layers
        for l in range(hidden_layers):
            # due to multi-head, in_feats = hidden_size * num_heads
            self.gat_layers.append(
                GATConv(
                    hidden_size * num_heads,
                    hidden_size,
                    num_heads,
                    feat_drop,
                    attn_drop,
                    negative_slope,
                    residual=residual,
                    activation=self.activation,
                    sparsemax=sparsemax,
                )
            )
        # output projection
        self.gat_layers.append(
            GATConv(
                hidden_size * num_heads,
                out_feats,
                num_heads,
                feat_drop,
                attn_drop,
                negative_slope,
                residual=residual,
                activation=None,
                sparsemax=False,
            )
        )

    def forward(self, features):
        h = features
        h = self.gat_layers[0](self.g, h).flatten(1)
        for l in range(self.hidden_layers):
            h = self.bn[l](h)
            h = self.gat_layers[l + 1](self.g, h).flatten(1)
        # output projection
        logits = self.gat_layers[-1](self.g, h).mean(1)
        logits = F.log_softmax(logits, 1)
        return logits
