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
from dgl.nn.pytorch import edge_softmax
from dgl.utils import Identity

from Sparsemax import Sparsemax

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
        self.num_heads = nn.ModuleList()
        for i in range(num_heads):
            self.num_heads.append(GATLayer(g, in_feats, out_feats, batchnorm))
        self.merge = merge

    def forward(self, h):
        head_outs = [attn_head(h) for attn_head in self.num_heads]
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


class GATConv(nn.Module):
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
        r"""Compute graph attention network layer.

        Parameters
        ----------
        graph : DGLGraph
            The graph.
        feat : torch.Tensor
            The input feature of shape :math:`(N, D_{in})` where :math:`D_{in}`
            is size of input feature, :math:`N` is the number of nodes.

        Returns
        -------
        torch.Tensor
            The output feature of shape :math:`(N, H, D_{out})` where :math:`H`
            is the number of heads, and :math:`D_{out}` is size of output feature.
        """
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
            # due to multi-head, the in_feats = hidden_size * num_heads
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
