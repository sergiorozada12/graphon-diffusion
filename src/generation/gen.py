import os
import pickle
import numpy as np
import networkx as nx
import scipy as sp


NUM_GRAPHS = 200
PLANAR_SIZE_RANGE = (20, 60)
TREE_SIZE_RANGE = (20, 60)
SBM_COMMS_RANGE = (2, 2)
SBM_COMMS_SIZE = (10, 30)
SEED = 0
BASE_PATH = "data/"
PATHS = {
    "planar": os.path.join(BASE_PATH, "extrapolation_planar.pkl"),
    "tree": os.path.join(BASE_PATH, "extrapolation_tree.pkl"),
    "sbm": os.path.join(BASE_PATH, "extrapolation_sbm.pkl"),
}


def generate_planar_graphs(num_graphs, min_size, max_size, seed):
    rng = np.random.default_rng(seed)
    graphs = []
    for _ in range(num_graphs):
        n = rng.integers(min_size, max_size + 1)
        points = rng.random((n, 2))
        tri = sp.spatial.Delaunay(points)
        adj = sp.sparse.lil_array((n, n), dtype=np.int32)
        for t in tri.simplices:
            for i in range(3):
                for j in range(i + 1, 3):
                    adj[t[i], t[j]] = 1
                    adj[t[j], t[i]] = 1
        G = nx.from_scipy_sparse_array(adj)
        graphs.append(G)
    return graphs


def generate_tree_graphs(num_graphs, min_size, max_size, seed):
    def custom_random_tree(n, rng):
        prufer = rng.integers(0, n, size=n - 2)
        degree = np.ones(n, dtype=int)
        for node in prufer:
            degree[node] += 1
        G = nx.Graph()
        for i in range(n):
            G.add_node(i)
        for node in prufer:
            leaf = np.flatnonzero(degree == 1)[0]
            G.add_edge(leaf, node)
            degree[leaf] -= 1
            degree[node] -= 1
        u, v = np.flatnonzero(degree == 1)
        G.add_edge(u, v)
        return G

    rng = np.random.default_rng(seed)
    graphs = []
    for _ in range(num_graphs):
        n = rng.integers(min_size, max_size + 1)
        G = custom_random_tree(n, rng)
        graphs.append(G)
    return graphs


def generate_sbm_graphs(num_graphs, min_comms, max_comms, min_comm_size, max_comm_size, seed):
    rng = np.random.default_rng(seed)
    graphs = []
    while len(graphs) < num_graphs:
        num_communities = rng.integers(min_comms, max_comms + 1)
        community_sizes = rng.integers(min_comm_size, max_comm_size + 1, size=num_communities)
        probs = np.full((num_communities, num_communities), 0.005)
        np.fill_diagonal(probs, 0.3)
        G = nx.stochastic_block_model(community_sizes.tolist(), probs, seed=rng)
        if nx.is_connected(G):
            graphs.append(G)
    return graphs


def save_graphs(graphs, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump(graphs, f)


if __name__ == "__main__":
    planar_graphs = generate_planar_graphs(NUM_GRAPHS, *PLANAR_SIZE_RANGE, seed=SEED)
    tree_graphs = generate_tree_graphs(NUM_GRAPHS, *TREE_SIZE_RANGE, seed=SEED + 1)
    sbm_graphs = generate_sbm_graphs(NUM_GRAPHS, *SBM_COMMS_RANGE, *SBM_COMMS_SIZE, seed=SEED + 2)

    save_graphs(planar_graphs, PATHS["planar"])
    save_graphs(tree_graphs, PATHS["tree"])
    save_graphs(sbm_graphs, PATHS["sbm"])
