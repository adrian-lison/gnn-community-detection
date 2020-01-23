# |||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||
# Code blueprint for analysis of dataframes
# |||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||

# -----------------------------------------------------------------------------------------------
# Loading the dataframes from the json result files
# -----------------------------------------------------------------------------------------------
import glob, os, json
import pandas as pd
import numpy as np

# hyperparameter tuning
resultlist = []
resultlist.extend(glob.glob(os.path.join("results/", "hyper1*.json")))
resultlist.extend(glob.glob(os.path.join("results/", "hyper2*.json")))
resultlist.extend(glob.glob(os.path.join("results/", "hyper3*.json")))
df = pd.DataFrame([json.load(open(f, "r")) for f in resultlist])

# comparison of GAT with and without residual connection
resultlist = []
resultlist.extend(glob.glob(os.path.join("results/", "compare_residual*.json")))
df = pd.DataFrame([json.load(open(f, "r")) for f in resultlist])

# comparison of GAT with different loss functions
resultlist = []
resultlist.extend(glob.glob(os.path.join("results/", "compare_loss*.json")))
df = pd.DataFrame([json.load(open(f, "r")) for f in resultlist])

# comparison of tuned GCN, GAT and LGNN on keyword features
resultlist = []
resultlist.extend(glob.glob(os.path.join("results/", "opti_keywords*.json")))
df = pd.DataFrame([json.load(open(f, "r")) for f in resultlist])

# comparison of tuned GCN, GAT and LGNN on node_id features
resultlist = []
resultlist.extend(glob.glob(os.path.join("results/", "opti_node*.json")))
df = pd.DataFrame([json.load(open(f, "r")) for f in resultlist])

# -----------------------------------------------------------------------------------------------
# Analysis
# -----------------------------------------------------------------------------------------------

# hyperparameter tuning
relevant_features = [
    "learning_rate",
    "weight_decay",
    "loss_function_type",
    "loss_function_nclasses",
    "hidden_size",
    "hidden_layers",
    "num_heads",
    "dropout",
    "batchnorm",
    "residual",
    "sparsemax",
    "radius",
]

# Show best results per net and permutation
df.groupby(["feature", "net", "permutation"])[["Rand-Index_Test", "Mutual Information_Test"]].max()

df.groupby(["feature", "net", "permutation"])[
    ["Rand-Index_Test", "Mutual Information_Test"]
].max().groupby(["feature", "net"])[["Rand-Index_Test", "Mutual Information_Test"]].mean()

df_best = df.loc[df.groupby(["net", "feature", "permutation"])["Rand-Index_Test"].idxmax(),][
    ["feature", "permutation", "net"] + relevant_features
]
df_best = df_best.fillna("none")

df_best[["feature", "permutation", "net"] + relevant_features]

# Show which hyperparemters were among the best how often
for feat in relevant_features:
    print(
        "\n"
        + str(
            df_best.groupby(["net", "feature"])[feat].agg(
                lambda x: tuple(dict(pd.Series.value_counts(x)).items())
            )
        )
    )

###############################
# comparison of GAT with and without residual connection

relevant_features = [
    "learning_rate",
    "weight_decay",
    "loss_function_type",
    "loss_function_nclasses",
    "hidden_size",
    "hidden_layers",
    "num_heads",
    "dropout",
    "batchnorm",
    "residual",
]
df_best = df.loc[
    df.groupby(["net", "feature", "permutation", "repetition"])["Rand-Index_Test"].idxmax(),
][["feature", "permutation", "net"] + relevant_features]
df_best = df_best.fillna("none")

df.groupby(["permutation", "residual"])["Rand-Index_Test"].mean()

###############################
# comparison of GAT with different loss functions

relevant_features = ["loss_function_type", "loss_function_nclasses"]
df_best = df.loc[
    df.groupby(["net", "feature", "permutation", "repetition"])["Rand-Index_Test"].idxmax(),
][["feature", "permutation", "net"] + relevant_features]
df_best = df_best.fillna("none")

df.groupby(["net", "feature", "permutation", "loss_function_type", "loss_function_nclasses"])[
    "Rand-Index_Test", "duration"
].mean()

df.groupby(["permutation", "residual"])["Rand-Index_Test"].mean()


###############################
# comparison of tuned GCN, GAT and LGNN on keyword features

df.head(100)[["Rand-Index_Test", "Mutual Information_Test", "Variation of Information_Test"]].corr()


df.groupby(["split_percentage_train","net","learning_rate"])["Rand-Index_Test"].mean()


# -----------------------------------------------------------------------------------------------
# useful resource
# -----------------------------------------------------------------------------------------------

hyperparameters = [
    "learning_rate",
    "weight_decay",
    "loss_function_type",
    "loss_function_nclasses",
    "hidden_size",
    "hidden_layers",
    "num_heads",
    "dropout",
    "batchnorm",
    "residual",
    "early_stopping_min",
    "early_stopping_wait",
]

