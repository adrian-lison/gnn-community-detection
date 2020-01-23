"""A pytorch implementation of SparseMax
Code was taken from https://github.com/OpenNMT/OpenNMT-py"""

from torch.autograd import Function
import torch.nn as nn
import torch as th

from dgl.function import TargetCode
from dgl.base import ALL, is_all
from dgl import backend as F
from dgl import utils


def _make_ix_like(input, dim=0):
    d = input.size(dim)
    rho = th.arange(1, d + 1, device=input.device, dtype=input.dtype)
    view = [1] * input.dim()
    view[0] = -1
    return rho.view(view).transpose(0, dim)


def _threshold_and_support(input, dim=0):
    """Sparsemax building block: compute the threshold
    Args:
        input: any dimension
        dim: dimension along which to apply the sparsemax
    Returns:
        the threshold value
    """

    input_srt, _ = th.sort(input, descending=True, dim=dim)
    input_cumsum = input_srt.cumsum(dim) - 1
    rhos = _make_ix_like(input, dim)
    support = rhos * input_srt > input_cumsum

    support_size = support.sum(dim=dim).unsqueeze(dim)
    tau = input_cumsum.gather(dim, support_size - 1)
    tau /= support_size.to(input.dtype)
    return tau, support_size


class SparsemaxFunction(Function):
    @staticmethod
    def forward(ctx, input, dim=0):
        """sparsemax: normalizing sparse transform (a la softmax)
        Parameters:
            input (Tensor): any shape
            dim: dimension along which to apply sparsemax
        Returns:
            output (Tensor): same shape as input
        """
        ctx.dim = dim
        max_val, _ = input.max(dim=dim, keepdim=True)
        input -= max_val  # same numerical stability trick as for softmax
        tau, supp_size = _threshold_and_support(input, dim=dim)
        output = th.clamp(input - tau, min=0)
        ctx.save_for_backward(supp_size, output)
        return output

    @staticmethod
    def backward(ctx, grad_output):
        supp_size, output = ctx.saved_tensors
        dim = ctx.dim
        grad_input = grad_output.clone()
        grad_input[output == 0] = 0

        v_hat = grad_input.sum(dim=dim) / supp_size.to(output.dtype).squeeze()
        v_hat = v_hat.unsqueeze(dim)
        grad_input = th.where(output != 0, grad_input - v_hat, grad_input)
        return grad_input, None


sparsemax = SparsemaxFunction.apply


class Sparsemax(nn.Module):
    def __init__(self, dim=0):
        self.dim = dim
        super(Sparsemax, self).__init__()

    def forward(self, input):
        return sparsemax(input, self.dim)


#########################
# Here we adapt Sparsemax on edges for GAT

__all__ = ["edge_softmax"]


class EdgeSparsemax(th.autograd.Function):
    """Apply sparsemax over signals of incoming edges."""

    @staticmethod
    def forward(ctx, g, score, eids):

        # remember to save the graph to backward cache before making it
        # a local variable
        if not is_all(eids):
            g = g.edge_subgraph(eids.long())

        n_nodes = g.number_of_nodes()
        n_edges = g.number_of_edges()
        gidx = g._graph.get_immutable_gidx(utils.to_dgl_context(score.device))
        ctx.backward_cache = n_nodes, n_edges, gidx

        # g.update_all(fn.copy_e('s', 'm'), fn.max('m', 'smax'))
        smax = F.copy_reduce("max", gidx, TargetCode.EDGE, score, n_nodes)
        # g.apply_edges(fn.e_sub_v('s', 'smax', 'out'))
        out = F.binary_reduce(
            "none", "sub", gidx, TargetCode.EDGE, TargetCode.DST, score, smax, n_edges
        )

        # g.edata['out'] = th.exp(g.edata['out'])
        out = th.exp(out)
        # g.update_all(fn.copy_e('out', 'm'), fn.sum('m', 'out_sum'))
        out_sum = F.copy_reduce("sum", gidx, TargetCode.EDGE, out, n_nodes)
        # g.apply_edges(fn.e_div_v('out', 'out_sum', 'out'))
        out = F.binary_reduce(
            "none", "div", gidx, TargetCode.EDGE, TargetCode.DST, out, out_sum, n_edges
        )

        ctx.save_for_backward(out)
        return out

    @staticmethod
    def backward(ctx, grad_out):
        """Backward function.
        Pseudo-code:
        .. code:: python
            g, out = ctx.backward_cache
            grad_out = dgl.EData(g, grad_out)
            out = dgl.EData(g, out)
            sds = out * grad_out  # type dgl.EData
            sds_sum = sds.dst_sum()  # type dgl.NData
            grad_score = sds - sds * sds_sum  # multiple expressions
            return grad_score.data
        """
        n_nodes, n_edges, gidx = ctx.backward_cache
        out, = ctx.saved_tensors

        # g.edata['grad_s'] = out * grad_out
        grad_s = out * grad_out
        # g.update_all(fn.copy_e('grad_s', 'm'), fn.sum('m', 'accum'))
        accum = F.copy_reduce("sum", gidx, TargetCode.EDGE, grad_s, n_nodes)
        # g.apply_edges(fn.e_mul_v('out', 'accum', 'out'))
        out = F.binary_reduce(
            "none", "mul", gidx, TargetCode.EDGE, TargetCode.DST, out, accum, n_edges
        )
        # grad_score = g.edata['grad_s'] - g.edata['out']
        grad_score = grad_s - out

        return None, grad_score, None


def edge_sparsemax(graph, logits, eids=ALL):
    return EdgeSparsemax.apply(graph, logits, eids)
