# NOTE: Execute the replicator dynamic on the grand table also visualize it as a graph.
# You may change everything in this file.

from typing import List
import matplotlib.pyplot as plt
import random
import time
from GrandTable import GrandTable
from MatrixSuite import MatrixSuite, FixedMatrixSuite, RandomIntMatrixSuite, RandomFloatMatrixSuite
import Strategies

# Proportions you will have to use:
uniform_with_own_strat = [1/9] * 9
uniform_without_own_strat = [1/8] * 8
non_uniform_with_own_strat = [0.12, 0.08,
                              0.06, 0.15, 0.05, 0.21, 0.06, 0.09, 0.18]
non_uniform_without_own_strat = [
    0.22, 0.19, 0.04, 0.06, 0.13, 0.10, 0.05, 0.21]
color = ['#000000', '#0000FF', '#A52A2A', '#32CD32',
         '#008000', '#FFA500', '#FFC0CB', '#FF0000', '#FFFF00']


class ReplicatorDynamic:
    proportions: List[float]
    proportion_history: dict
    scores: List[float]
    proportion_change: List[float]
    change_interval: int
    change_threshold: float
    grand_table: GrandTable

    def __init__(self, start_proportions: List[float], grand_table: GrandTable):
        self.grand_table = grand_table
        self.proportions = start_proportions
        self.scores = [0 for i in range(len(self.proportions))]
        self.proportion_history = {k: [] for k in range(len(self.proportions))}
        self.proportion_change = []
        self.change_interval = 200
        self.change_threshold = 0.001

    def run(self) -> None:
        self.grand_table.execute()

        while sum(self.proportion_change) >= self.change_threshold or len(self.proportion_change) < self.change_interval:
            self.update_history()
            self.calculate_score()
            new_proportions = self.normalize_proportion(
                self.hadamard_product())
            self.update_change(new_proportions)
            self.proportions = new_proportions

        self.to_graph()

    def to_graph(self) -> None:
        """Visualize the evolution of proportions."""
        # plt.title("Proportions evolution")
        x_axix = list(range(len(self.proportion_history[0])))
        cmap = self.get_cmap(len(self.proportions))
        for i in range(len(self.proportions)):
            plt.plot(
                x_axix, self.proportion_history[i], color=color[i], label=self.grand_table.row_strategies[i].name)
        plt.legend()
        plt.xlabel("Iterations")
        plt.ylabel("Proportion")
        plt.savefig("img/" + str(len(self.proportions)) + " " +
                    self.grand_table.matrix_suite.name + " " + str(time.time()) + ".jpg")
        plt.close()
        # plt.show()

    def get_cmap(self, n, name="hsv"):
        ''' Returns a function that maps each index in 0, 1, ..., n-1 to a distinct
            RGB color; the keyword argument name must be a standard mpl colormap name.
        '''
        return plt.cm.get_cmap(name, n)

    def update_history(self) -> None:
        for i in range(len(self.proportions)):
            self.proportion_history[i].append(self.proportions[i])

    def calculate_score(self) -> None:
        for i in range(len(self.grand_table.grand_table)):
            for j in range(len(self.grand_table.grand_table)):
                self.scores[i] += self.proportions[j] * \
                    self.grand_table.grand_table[i][j]

    def hadamard_product(self) -> List[float]:
        product = []
        for i in range(len(self.proportions)):
            product.append(self.proportions[i] * self.scores[i])
        return product

    def normalize_proportion(self, product: List[float]) -> List[float]:
        norm = [i / sum(product) for i in product]
        return norm

    def update_change(self, new_proportions: List[float]) -> None:
        change = 0
        for i in range(len(self.proportions)):
            change += abs(self.proportions[i] - new_proportions[i])
        self.proportion_change.append(change)
        if len(self.proportion_change) > self.change_interval:
            self.proportion_change.pop(0)
