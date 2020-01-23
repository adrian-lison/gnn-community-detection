# Graph Neural Networks

The Graph Neural Network (GNN) is a comparatively novel concept which allows neural networks to operate on arbitrary graphs. As irregular problem structures are ubiquitous in real-world domains and can be best represented by graphs, GNNs offer new exciting applications and further generalization potential for machine learning as a whole, but also significant improvement of performance in a number of deep learning domains.

The goal of this seminar work is to provide an introduction into the principles and functioning of Graph Neural Networks. In order to illustrate some of the subtleties of different approaches, the task of community detection will be used as a practical example of a promising application, where node features and graph structure can be merged into a rich network representation.

## Seminar Paper
The outline of this paper is as follows. First, we introduce community detection as a challenging graph clustering task, shortly highlighting existing solution approaches. Then, we present our findings from a high-level literature review to capture the current state of research on Graph Neural Networks as well as trace back landmarks of influential publications in the historical development of the field. We provide a lightweight introduction into neural networks and point out the major innovation of Graph Neural Networks. Moreover, we present suitable taxonomies and frameworks which help to make sense of and distinguish between the different GNN architectures that have been hitherto proposed. Based on a second review, we present three suitable GNNs for community detection and describe a computational experiment where we tailor them to conduct semi-supervised community detection on the well-known Cora citation network. We finally discuss our experimental results and draw conclusions in order to point out potential future research paths in a short outlook.

## Computational Experiment
Our experiment is an instance of semi-supervised, inductive learning, where we train on partially labeled graph data. A small subset of nodes has labels during the training phase and the loss to be minimized is calculated using only this subset, nevertheless all nodes are used by the network for induction. The task is to perform community detection, viz. to predict a distinct label for each node such that nodes within the same community have the same label. Note that the exact class indicated by the label does not matter as long as the aforementioned condition is fulfilled.

We design the experiment to compare the performance of different GNNs as well as benchmark the method against other non-GNN approaches for community detection.
Simple kmeans-clustering on the nodes features (regardless of the graph structure) is performed to obtain a reference value for the feature data. Similarly, we run the deepwalk algorithm as an example of a powerful non-GNN approach to community detection.
The graph neural network architectures tested here are the previously mentioned Graph Convolutional Network (GCN) as proposed by Kipf and Welling, Line Graph Neural Network (LGNN) by Chen et al. and Graph Attention Network (GAT) by Veličković et al.

### Usage
The main class is located in *code/experiment.py*. It can be run from the command line with an additional parameter that provides the path to an experiment configuration file located in the folder *experiments*. The file is then loaded and all experiments are carried out in parallel mode. The exemplary file *experiments/base configuration explained.json* shows explanations on how to configure.

Stored models can be loaded as param_dicts by pytorch.
