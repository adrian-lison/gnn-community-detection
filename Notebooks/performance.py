
# Imports
import numpy as np
from scipy import sparse as sp
from math import log
import pandas as pd
from sklearn import metrics as skmetrics

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
        outer = (pi.take(nzx).astype(np.int64, copy=False)
                 * pj.take(nzy).astype(np.int64, copy=False))
        # print(outer)
        log_outer = -np.log(outer) + log(pi.sum()) + log(pj.sum())
        # print(log_outer)
        mi = (contingency_nm * (log_contingency_nm -
                                log(contingency_sum)) + contingency_nm * log_outer)
        # print(mi)
        return mi.sum()
    return mi(labels, labels) + mi(preds, preds) - 2 * mi(labels, preds)

# Commodity Functions


def compute_performance(labels, logits, mask):
    logits = logits.detach().numpy()
    preds = np.argmax(logits, axis=1)
    labels = labels.numpy()
    mask = mask.numpy().astype(bool)
    pred_sets = {"All ": preds,
                 "Train": preds[mask], "Test": preds[np.invert(mask)]}
    label_sets = {"All ": labels,
                  "Train": labels[mask], "Test": labels[np.invert(mask)]}
    eval_functions = {
        "Rand-Index": rand_score,
        "Mutual Information": mutual_info_score,
        "Variation of Information": variation_of_information_score}
    scores = {subset: {name: func(label_sets[subset], pred_sets[subset])
                       for name, func in eval_functions.items()} for subset in pred_sets.keys()}
    return scores


def print_performance(labels, logits, mask):
    scores = compute_performance(labels, logits, mask)
    for subset_n, data in scores.items():
        eval_message = f"\n{subset_n}:\n"
        for func, score in data.items():
            eval_message += f" {func}: {score:.4f} |"
        print(eval_message)


def performance_as_df(labels, logits, mask):
    scores = compute_performance(labels, logits, mask)
    return pd.DataFrame(scores)
