import numpy as np
import networkx as nx
from scipy.stats import chi2
import json
import os
import wandb
from pytorch_lightning.loggers import WandbLogger
from graph_tool.all import Graph, minimize_blockmodel_dl
from graph_tool.inference import contiguous_map  # add at the top of the file
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score


class EvaluationMetrics:
    def __init__(self, generated_graphs, graph_type="sbm", train_graphs=None):
        self.generated = [g for g in generated_graphs if g.number_of_nodes() > 0]
        self.graph_type = graph_type
        self.train = train_graphs

    def run_all_metrics(self, out_folder="eval", filename=None, wandb_logger=None):
        if filename is None:
            raise ValueError("A filename must be provided.")
        os.makedirs(out_folder, exist_ok=True)

        results = {
            "accuracy": self.compute_accuracy(),
            "fraction_unique": self.compute_fraction_unique(),
            "fraction_unique_precise": self.compute_fraction_unique(precise=True),
            "fraction_isomorphic": self.compute_fraction_isomorphic() if self.train else None,
            "fraction_valid_and_unique": self.compute_fraction_valid_and_unique()
        }

        with open(os.path.join(out_folder, f"{filename}.json"), "w") as f:
            json.dump(results, f, indent=2)

        if wandb_logger and isinstance(wandb_logger, WandbLogger):
            table = wandb.Table(columns=["Metric", "Value"])
            for k, v in results.items():
                table.add_data(k, v)

            wandb_logger.experiment.log({
                "metrics_barplot": wandb.plot.bar(
                    table, "Metric", "Value", title="Evaluation Metrics"
                )
            })

    def compute_accuracy(self, p_intra=0.3, p_inter=0.005):
        if self.graph_type == "tree":
            return self._acc_tree()
        elif self.graph_type == "planar":
            return self._acc_planar()
        elif self.graph_type == "sbm":
            return self._acc_sbm()
        else:
            raise ValueError(f"Unknown graph type: {self.graph_type}")

    def compute_fraction_unique(self, precise=False):
        count_non_unique = 0
        fake_evaluated = []

        for g in self.generated:
            is_unique = True
            for h in fake_evaluated:
                if precise:
                    if nx.faster_could_be_isomorphic(g, h) and nx.is_isomorphic(g, h):
                        is_unique = False
                        break
                else:
                    if nx.faster_could_be_isomorphic(g, h) and nx.could_be_isomorphic(g, h):
                        is_unique = False
                        break
            if is_unique:
                fake_evaluated.append(g)
            else:
                count_non_unique += 1

        return (len(self.generated) - count_non_unique) / len(self.generated)

    def compute_fraction_isomorphic(self):
        if self.train is None:
            raise ValueError("train_graphs must be provided.")
        count = 0
        for fake in self.generated:
            for real in self.train:
                if nx.faster_could_be_isomorphic(fake, real) and nx.is_isomorphic(fake, real):
                    count += 1
                    break
        return count / len(self.generated)

    def compute_fraction_valid_and_unique(self):
        valid_graphs = [
            g for g in self.generated if (
                (self.graph_type == "tree" and nx.is_tree(g)) or
                (self.graph_type == "planar" and nx.is_connected(g) and nx.check_planarity(g)[0]) or
                (self.graph_type == "sbm" and self._is_sbm(g))  # use shared logic
            )
        ]

        count_non_unique = 0
        evaluated = []
        for g in valid_graphs:
            is_unique = True
            for h in evaluated:
                if nx.faster_could_be_isomorphic(g, h) and nx.is_isomorphic(g, h):
                    is_unique = False
                    break
            if is_unique:
                evaluated.append(g)
            else:
                count_non_unique += 1

        return (len(valid_graphs) - count_non_unique) / len(self.generated)

    def _acc_tree(self):
        return sum(nx.is_tree(g) for g in self.generated) / len(self.generated)

    def _acc_planar(self):
        return sum(nx.is_connected(g) and nx.check_planarity(g)[0] for g in self.generated) / len(self.generated)

    def _acc_sbm(self):
        return sum(self._is_sbm(G) for G in self.generated) / len(self.generated)

    def _is_sbm(self, G):
        L = nx.normalized_laplacian_matrix(G).astype(float)
        _, eigvecs = np.linalg.eigh(L.A)
        X = eigvecs[:, 1:3]
        labels = KMeans(n_clusters=2, n_init=10).fit_predict(X)
        score = silhouette_score(X, labels)
        return score > 0.4
