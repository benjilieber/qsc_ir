import matplotlib as mpl
import networkx as nx
import matplotlib.pyplot as plt

class Graph(object):
    def __init__(self):
        self.n = 0
        self.G = []
        self.entropies = []
        self.threshold = 0

    def addEdges(self, i_to_j, n):
        self.n = n
        edges = [(n+i, j) for i, j_list in enumerate(i_to_j) for j in j_list]
        self.G = nx.Graph()
        self.G.add_edges_from(edges)

    def removeNodes(self, entropies, threshold):
        to_remove = [node for node in self.G if node < self.n and entropies[node] < threshold]
        self.G.remove_nodes_from(to_remove)
        # self.G.remove_nodes_from(list(nx.isolates(self.G)))
        self.entropies = entropies
        self.threshold = threshold

    def visualize(self):
        selected_entropies = [entropy for entropy in self.entropies if entropy >= self.threshold]
        if len(selected_entropies) > 1:
            low, *_, high = sorted(selected_entropies)
            norm = mpl.colors.Normalize(vmin=low, vmax=high, clip=True)
            mapper = mpl.cm.ScalarMappable(norm=norm, cmap=mpl.cm.coolwarm)
            color_map = [mapper.to_rgba(self.entropies[node]) if node < self.n else (0, 0, 0, 0) for node in self.G]
        else:
            color_map = [(0, 0, 0, 0) for node in self.G]
        # color_map = [self.entropies[node] if node < self.n else 'green' for node in self.G]
        # color_map = ['blue' if node < self.n else 'green' for node in self.G]
        # color_map = ['blue'] * self.number_of_symbols + ['green'] * (self.G.number_of_nodes() - self.number_of_symbols)
        nx.draw(self.G, node_color=color_map, with_labels=False, node_size=5)
        # # nx.draw_networkx(self.G, node_size=100)
        plt.show()