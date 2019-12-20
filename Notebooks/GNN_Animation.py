from sklearn.manifold import TSNE
import matplotlib.animation as animation
import matplotlib.pyplot as plt
import networkx as nx


class GNN_Animation:
    @staticmethod
    def _compute_positions(logit_list, method="tsne"):
        if method == "tsne":
            pos_epochs = dict()
            pos_epochs[0] = TSNE(n_components=2, n_iter=250, early_exaggeration=30).fit_transform(
                logit_list[0].numpy())
            for x in range(1, len(logit_list)):
                #print(f"Compute scaling of epoch {x}")
                pos_epochs[x] = TSNE(n_components=2, init=pos_epochs[x-1],
                                     n_iter=250, early_exaggeration=30).fit_transform(logit_list[x].numpy())
        if method == "pca":

        return pos_epochs

    @staticmethod
    def draw_epoch(i, pos_epochs, logit_list, graph, ax):
        # cls1color = '#00FFFF'
        # cls2color = '#FF00FF'
        pos2d = {k: pos for k, pos in enumerate(pos_epochs[i])}
        colors = [l.numpy().argmax() for l in logit_list[i]]
        ax.cla()
        ax.axis('off')
        ax.set_title(f"Epoch: {i}")
        return nx.draw_networkx(graph, pos=pos2d, node_color=colors, cmap=plt.cm.Set1,
                                with_labels=False, node_size=30, ax=ax, width=0.5, alpha=0.8, linewidths=0)

    @staticmethod
    def animate(dglgraph, logit_list):
        pos_epochs = GNN_Animation._compute_positions(logit_list)

        fig = plt.figure(dpi=150, figsize=(7, 7))
        fig.clf()
        ax = fig.subplots()
        ani = animation.FuncAnimation(fig, GNN_Animation.draw_epoch,
                                      frames=len(logit_list),
                                      fargs=(
                                          pos_epochs, logit_list, dglgraph.to_networkx().to_undirected(), ax),
                                      interval=200, repeat_delay=1500)
        ani_html = ani.to_jshtml()
        plt.close()
        return ani, ani_html
