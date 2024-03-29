# Note: This is the logic to execute the gambit enum mixed commandline tool.
#
# In this file you should only have to complete the nash_equilibria method.
# It should convert your GrandTable object to a call to the run_gambit method.
#
# As of writing the run_gambit method has only been tested on Windows,
# if it doesn't work on Mac or Linux let us know,
# as you should NOT have to modify it yourself.
#
# A test for the run_gambit method is included in Main.py

import subprocess
from typing import List

from GrandTable import GrandTable
from Strategies import Strategy

from Utils import flatten, transpose


def nash_equilibria(grand_table: GrandTable) -> None:
    """Go from the GrandTable to a call to run_gambit.
    :param grand_table: The calculated Grand Table"""
    gt = [[0 for _ in range(len(grand_table.grand_table))]
          for _ in range(len(grand_table.grand_table))]
    for i in range(len(grand_table.grand_table)):
        for j in range(len(grand_table.grand_table)):
            gt[i][j] = grand_table.grand_table[i][j]
    run_gambit(grand_table.col_strategies, gt)


def run_gambit(strategies: List[Strategy], table: List[List[float]]) -> None:
    """Run the gambit enum mixed command line tool and pretty print the result.
    :param strategies: List of the strategies used to create the table.
    :param table: Grand Table scores as a 2D matrix.

    To understand how gambit-enummixed should be used, here is an example:
    Let's say we have a table that looks like this.
      |A|B|
    A |1|2|
    B |3|4|

    gambit-enummixed wants this table in NFG, Normal Form Game,
    in this case that would be a file structured in the following way:
    '
    NFG 1 R "" { "1" "2" } { 2 2 }
    1 1
    3 2
    2 3
    4 4
    '
    The first line means:
    NFG: Normal Form Game
    1: normal form version 1, only version that is supported
    R: use rational numbers, obsolete distinction so every file should have an R here.
    "": The name of the game, irrelevant and thus empty.
    { "1" "2" }: The names of the players, we want to have 2 players since it is a 2D table.
    { 2 2 }: The number of actions (strategies in our case) for each player.

    The other lines are the table in normal form.
    First the score of A against A (and the opposing score A against A),
    then the score of B against A (and the opposing score A against B),
    then the score of A against B (and the opposing score B against A),
    finally the score of B against B (and the opposing score B against B).

    The trick is that it first increments the action of player 1,
    and every time it rolls over to the first action, it increments the action of player 2.

    Since newlines don't matter to the tool, we can structure the input like this:
    gambit_input = 'NFG 1 R "" { "1" "2" } { 2 2 } 1 1 3 2 2 3 4 4'
    with some escape characters before the quotation marks.

    Now we want to call gambit-enummixed with this input but it only accepts files.
    So we used the 'echo' command and the pipe operator '|', to pass it the input as if it where a file.
    We also want to suppress any unnecessary output and round to 2 decimal places.
    This is accomplished by adding the '-q' and '-d2' flags.

    The final command looks like this:

    echo 'NFG 1 R "" { "1" "2" } { 2 2 } 1 1 3 2 2 3 4 4' | .\gambit-enummixed -q -d2

    If you run it, the output will be
        NE,0.00,1.00,0.00,1.00
    This means that there is a Nash equilibrium (NE),
    if both players play their first action with a probability of 0,
    and their second action with a probability of 1
    This is better visible if you reformat the output and add action names:
    A: 0.00, 0.00
    B: 1.00, 1.00
    So their is a Nash equilibrium when both players always play action B,
    given the table this is no surprise.

    Note: The extra backslash character in the code is necessary to escape the escape character.
    Also in the output format the 0.00 are replaced with ----
    """

    # Determine padding for pretty printing output.
    padding = max(map(lambda s: len(s.name), strategies)) + 1
    name_format = '{:>' + padding.__repr__() + '}'
    num_format = '{:^6}'
    linesize = padding + 14
    hline = ("=" * linesize) + "|"

    # Make a call to gambit
    nr_of_strats = len(strategies)
    gambit_input = 'NFG 1 R "" { "1" "2" } { ' + \
                   nr_of_strats.__repr__() + " " + nr_of_strats.__repr__() + " } " + \
                   " ".join(map(repr, flatten(zip(flatten(transpose(table)), flatten(table)))))
    result = subprocess.check_output("echo " + gambit_input + "| .\\gambit-enummixed -q -d2", shell=True)

    for line in result.splitlines():
        line = line.__repr__()
        line = line.strip("'")
        line = line.replace('0.00', '----')
        values = line.split(sep=",")
        print(hline)
        for i, strategy in enumerate(strategies):
            name = name_format.format(strategy.name)
            x = num_format.format(values[i + 1])
            y = num_format.format(values[i + 1 + nr_of_strats])
            print(name + ":" + x + "|" + y + "|")
    print(hline)
