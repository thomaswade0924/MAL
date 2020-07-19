# Note: From this script the program can be run.
# You may change everything about this file.

from MatrixSuite import FixedMatrixSuite, RandomIntMatrixSuite, RandomFloatMatrixSuite
import Strategies
import Game
from GrandTable import GrandTable
from ReplicatorDynamic import ReplicatorDynamic
import Nash

# Proportions, matrix suites, startegies needed
uniform_with_own_strat = [1/9] * 9
uniform_without_own_strat = [1/8] * 8
non_uniform_with_own_strat = [0.12, 0.08,
                              0.06, 0.15, 0.05, 0.21, 0.06, 0.09, 0.18]
non_uniform_without_own_strat = [
    0.22, 0.19, 0.04, 0.06, 0.13, 0.10, 0.05, 0.21]
matrix_suites = [RandomIntMatrixSuite(), RandomFloatMatrixSuite()]


def run_grand_table(grand_table: GrandTable) -> None:
    grand_table.execute()
    print(grand_table)
    Nash.nash_equilibria(grand_table)


'''
GRAND TABLE AND NASH CODE
'''
with_own_stra = [Strategies.SmartHybrid(), Strategies.SmartBully(), Strategies.Bully(), Strategies.SmartGreedy(), Strategies.EGreedy(), Strategies.Aselect(), Strategies.UCB(), Strategies.Satisficing(),
                 Strategies.Softmax(), Strategies.Fictitious(), Strategies.PRM()]

# Fixed
grand_table = GrandTable(FixedMatrixSuite(), with_own_stra, 9, 1000)
run_grand_table(grand_table)

# Random Int
grand_table = GrandTable(RandomIntMatrixSuite(), with_own_stra, 19, 1000)
run_grand_table(grand_table)

# Random Float
grand_table = GrandTable(RandomFloatMatrixSuite(), with_own_stra, 19, 1000)
run_grand_table(grand_table)


'''
RAPLICATOR DYNAMIC CODE
'''
with_own_stra = [Strategies.SmartHybrid(), Strategies.Aselect(), Strategies.EGreedy(), Strategies.UCB(), Strategies.Satisficing(),
                 Strategies.Softmax(), Strategies.Fictitious(), Strategies.Bully(), Strategies.PRM()]
without_own_stra = [Strategies.Aselect(), Strategies.EGreedy(), Strategies.UCB(), Strategies.Satisficing(),
                    Strategies.Softmax(), Strategies.Fictitious(), Strategies.Bully(), Strategies.PRM()]
with_own_proportions = [uniform_with_own_strat, non_uniform_with_own_strat]
without_own_proportions = [
    uniform_without_own_strat, non_uniform_without_own_strat]

# Random matrix suites
for suite in matrix_suites:
    for proportions in with_own_proportions:
        grand_table = GrandTable(suite, with_own_stra, 19, 1000)
        rd = ReplicatorDynamic(proportions, grand_table)
        rd.run()
    for proportions in without_own_proportions:
        grand_table = GrandTable(suite, without_own_stra, 19, 1000)
        rd = ReplicatorDynamic(proportions, grand_table)
        rd.run()

# Fixed matrix suites
for proportions in with_own_proportions:
    grand_table = GrandTable(FixedMatrixSuite(), with_own_stra, 9, 1000)
    rd = ReplicatorDynamic(proportions, grand_table)
    rd.run()
for proportions in without_own_proportions:
    grand_table = GrandTable(FixedMatrixSuite(), without_own_stra, 9, 1000)
    rd = ReplicatorDynamic(proportions, grand_table)
    rd.run()
