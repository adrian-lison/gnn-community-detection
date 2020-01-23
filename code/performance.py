# ----------------------------------------------------------------------------
# This script provides performance measures for community detection and
# the permutation-invariant loss function and its approximation


# Imports
import numpy as np
from scipy import sparse as sp
from math import log
import pandas as pd
from sklearn import metrics as skmetrics

import torch as th
import torch.nn.functional as F

import itertools

# Loss Functions


class perm_inv_loss:
    def __init__(self, labels):
        self.labels = labels
        self.num_classes = len(labels.unique())
        self.label_perms = {i: None for i in range(2, self.num_classes + 1)}

    def compute_loss(self, logits, mask):
        if self.label_perms[self.num_classes] is None:
            self.label_perms[self.num_classes] = list(
                itertools.permutations(range(self.num_classes))
            )

        loss = th.tensor(np.infty, requires_grad=True)
        for p in self.label_perms:
            loss = th.min(loss, F.nll_loss(logits[mask][:, p], self.labels[mask]))
        return loss

    def approximate_loss(self, logits, mask, nclasses=3):
        if self.label_perms[nclasses] is None:
            self.label_perms[nclasses] = list(itertools.permutations(range(nclasses)))

        # randomly assign labels to new clusters (trying to roughly achieve equal distribution)
        assignments = np.random.choice(
            [i % nclasses for i in range(self.num_classes)], size=self.num_classes, replace=False
        )
        new_labels = th.LongTensor(assignments[self.labels])
        one_hot_assignments = th.BoolTensor(np.eye(np.max(assignments) + 1)[assignments])
        tensors = [th.sum(logits[:, one_hot_assignments[:, i]], dim=1) for i in range(nclasses)]
        new_logits = th.stack(tensors, 1)
        new_label_perms = list(itertools.permutations(np.unique(new_labels)))
        loss = th.tensor(np.infty, requires_grad=True)
        for p in new_label_perms:
            loss = th.min(loss, F.nll_loss(new_logits[mask][:, p], new_labels[mask]))
        return loss


# Scoring Functions


def rand_score(labels, preds):
    return skmetrics.adjusted_rand_score(labels, preds)


def mutual_info_score(labels, preds):
    return skmetrics.adjusted_mutual_info_score(labels, preds, average_method="arithmetic")


def variation_of_information_score(labels, preds):
    def mi(x, y):
        contingency = skmetrics.cluster.contingency_matrix(x, y, sparse=True)
        # print(contingency.todense())
        nzx, nzy, nz_val = sp.find(contingency)
        contingency_sum = contingency.sum()

        pi = np.ravel(contingency.sum(axis=1))
        pj = np.ravel(contingency.sum(axis=0))
        # print(nz_val)
        log_contingency_nm = np.log(nz_val)
        # print(log_contingency_nm)
        contingency_nm = nz_val / contingency_sum
        # print(contingency_nm)

        # Don't need to calculate the full outer product, just for non-zeroes
        outer = pi.take(nzx).astype(np.int64, copy=False) * pj.take(nzy).astype(
            np.int64, copy=False
        )
        # print(outer)
        log_outer = -np.log(outer) + log(pi.sum()) + log(pj.sum())
        # print(log_outer)
        mi = (
            contingency_nm * (log_contingency_nm - log(contingency_sum))
            + contingency_nm * log_outer
        )
        # print(mi)
        return mi.sum()

    return mi(labels, labels) + mi(preds, preds) - 2 * mi(labels, preds)


# Commodity Functions


def compute_performance(labels, logits, splits):
    logits = logits.detach().numpy()
    preds = np.argmax(logits, axis=1)
    labels = labels.numpy()
    pred_sets = {
        "All ": preds,
        "Train": preds[splits["train"]],
        "Val": preds[splits["val"]],
        "Test": preds[splits["test"]],
    }
    label_sets = {
        "All ": labels,
        "Train": labels[splits["train"]],
        "Val": labels[splits["val"]],
        "Test": labels[splits["test"]],
    }
    eval_functions = {
        "Rand-Index": rand_score,
        "Mutual Information": mutual_info_score,
        "Variation of Information": variation_of_information_score,
    }
    scores = {
        subset: {
            name: func(label_sets[subset], pred_sets[subset])
            for name, func in eval_functions.items()
        }
        for subset in pred_sets.keys()
    }
    return scores


def print_performance(labels, logits, splits):
    scores = compute_performance(labels, logits, splits)
    for subset_n, data in scores.items():
        eval_message = f"\n{subset_n}:\n"
        for func, score in data.items():
            eval_message += f" {func}: {score:.4f} |"
        print(eval_message)


def performance_as_df(labels, logits, splits):
    scores = compute_performance(labels, logits, splits)
    return pd.DataFrame(scores)
