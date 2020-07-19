# Note: You may not change methods in the Strategy class, nor their input parameters.
# Information about the entire game is given in the *initialize* method, as it gets the entire MatrixSuite.
# During play the payoff matrix doesn't change so if your strategy needs that information,
#  you can save it to Class attributes of your strategy. (like the *actions* attribute of Aselect)

import abc
import random
from typing import List, Dict, Tuple
import numpy as np
import math

import MatrixSuite
from MatrixSuite import Action, Payoff


class Strategy(metaclass=abc.ABCMeta):
    """Abstract representation of a what a Strategy should minimally implement.

    Class attributes:
        name: A string representing the name of the strategy.
    """
    name: str

    def __repr__(self) -> str:
        """The string representation of a strategy is just it's name.
        So it you call **print()** it will output the name.
        """
        return self.name

    @abc.abstractmethod
    def initialize(self, matrix_suite: MatrixSuite, player: str) -> None:
        """Initialize/reset the strategy with a new game.
        :param matrix_suite: The current MatrixSuite,
        so the strategy can extract the information it needs from the payoff matrix.
        :param player: A string of either 'row' or 'col',
        representing which player the strategy is currently playing as.
        """
        pass

    @abc.abstractmethod
    def get_action(self, round_: int) -> Action:
        """Calculate the action for this round.
        :param round_: The current round number.
        """
        pass

    @abc.abstractmethod
    def update(self, round_: int, action: Action, payoff: Payoff, opp_action: Action) -> None:
        """Update the strategy with the result of this round.
        :param round_: The current round number.
        :param action: The action this strategy played this round.
        :param payoff: The payoff this strategy received this round.
        :param opp_action: The action the opposing strategy played this round.
        """
        pass


# As an example we have implemented the first strategy for you.


class Aselect(Strategy):
    """Implements the Aselect (random play) algorithm."""
    actions: List[Action]

    def __init__(self):
        """The __init__ method will be called when you create an object.
        This should set the name of the strategy and handle any additional parameters.

        Example:
            To create an object of the Aselect class you use, "Aselect()"

            If you want to pass it parameters on creation, e.g. "Aselect(0.1)",

            you need to modify "__init__(self)" to "__init__(self, x: float)",
            and you probably also want to save "x", or whatever you name it,
            to a class attribute, so you can use it in other methods.
            """
        self.name = "Aselect"

    def initialize(self, matrix_suite: MatrixSuite, player: str) -> None:
        """Just save the actions as that's the only thing we need."""
        self.actions = matrix_suite.get_actions(player)

    def get_action(self, round_: int) -> Action:
        """Pick the next action randomly from the possible actions."""
        return random.choice(self.actions)

    def update(self, round_: int, action: Action, payoff: Payoff, opp_action: Action) -> None:
        """Aselect has no update mechanic."""
        pass


class SmartGreedy(Strategy):
    """Implements the smart epsilon greedy algorithm with epsilon=0.1."""
    actions: List[Action]
    payoffs: List[float]
    counts = List[int]
    max_payoff: float

    def __init__(self):
        self.name = "SmartGreedy"

    def initialize(self, matrix_suite: MatrixSuite, player: str) -> None:
        self.actions = matrix_suite.get_actions(player)

        # get max and min payoff for geometric average
        max_payoff = np.amax(np.array(matrix_suite.payoff_matrix))
        min_payoff = np.amin(np.array(matrix_suite.payoff_matrix))

        self.counts = np.ones(len(self.actions))
        # keep last 20 payoffs for each strategy
        self.payoffs = np.full((len(self.actions), 20),
                               (max_payoff - min_payoff) / 2)

    def get_action(self, round_: int) -> Action:
        """Selects random action with probability epsilon=0.1 and best historic action over span of 20 rounds with probability 0.9."""
        if np.random.random() < 0.1:
            return random.choice(self.actions)
        else:
            return np.argmax([np.sum(self.payoffs[action]) for action in self.actions])

    def update(self, round_: int, action: Action, payoff: Payoff, opp_action: Action) -> None:
        self.counts[action] += 1
        # remove oldest payoff from history
        for index in range(len(self.payoffs[action]) - 1):
            self.payoffs[action][index] = self.payoffs[action][index + 1]
        # update payoff history
        self.payoffs[action][len(self.payoffs[action]) - 1] = payoff


class EGreedy(Strategy):
    """Implements the epsilon greedy algorithm with epsilon=0.1."""
    actions: List[Action]
    average_payoff: List[float]
    counts = List[int]

    def __init__(self):
        self.name = "EGreedy"

    def initialize(self, matrix_suite: MatrixSuite, player: str) -> None:
        self.actions = matrix_suite.get_actions(player)

        # get max and min payoff for geometric average
        if player == "row":
            max_payoff = np.amax(np.array(matrix_suite.payoff_matrix[:][:][0]))
            min_payoff = np.amin(np.array(matrix_suite.payoff_matrix[:][:][0]))
        else:
            max_payoff = np.amax(np.array(matrix_suite.payoff_matrix[:][:][1]))
            min_payoff = np.amin(np.array(matrix_suite.payoff_matrix[:][:][1]))
        '''
        max_payoff = np.amax(
            np.array(matrix_suite.payoff_matrix))
        min_payoff = np.amin(
            np.array(matrix_suite.payoff_matrix))
        '''

        self.counts = np.ones(len(self.actions))
        self.average_payoff = np.full(
            len(self.actions), (max_payoff - min_payoff) / 2)

    def get_action(self, round_: int) -> Action:
        """Selects random action with probability epsilon=0.1 and best historic action with probability 0.9."""
        if np.random.random() < 0.1:
            return random.choice(self.actions)
        else:
            return np.argmax([self.average_payoff[action] for action in self.actions])

    def update(self, round_: int, action: Action, payoff: Payoff, opp_action: Action) -> None:
        self.counts[action] += 1
        self.average_payoff[action] += (payoff -
                                        self.average_payoff[action]) / self.counts[action]


class UCB(Strategy):
    """Implements the UCB algorithm."""
    actions: List[Action]
    counts: int
    payoff_history: List[float]

    def __init__(self):
        self.name = "UCB"

    def initialize(self, matrix_suite: MatrixSuite, player: str) -> None:
        self.actions = matrix_suite.get_actions(player)
        payoff_matrix = matrix_suite.payoff_matrix

        # get max and min payoff for geometric average
        max_payoff, min_payoff = 0, 0
        for ra in range(len(payoff_matrix)):
            for ca in range(len(payoff_matrix[0])):
                if player == "row":
                    max_payoff = max(max_payoff, payoff_matrix[ra][ca][0])
                    min_payoff = min(min_payoff, payoff_matrix[ra][ca][0])
                else:
                    max_payoff = max(max_payoff, payoff_matrix[ra][ca][1])
                    min_payoff = min(min_payoff, payoff_matrix[ra][ca][1])

        self.counts = np.ones(len(self.actions))
        self.payoff_history = np.full(
            len(self.actions), (max_payoff + min_payoff) / 2)

    def get_action(self, round_: int) -> Action:
        """Selects action according to UCB equation."""
        average_payoff_history = np.zeros(len(self.actions))
        for action in self.actions:
            average_payoff_history[action] = self.payoff_history[action] / \
                self.counts[action] + \
                np.sqrt((np.log(round_)) / self.counts[action])
        return np.argmax(average_payoff_history)

    def update(self, round_: int, action: Action, payoff: Payoff, opp_action: Action) -> None:
        self.counts[action] += 1
        self.payoff_history[action] += payoff


class Satisficing(Strategy):
    """Implements the Satisficing play algorithm for lambda = 0.1."""
    actions: List[Action]
    action: Action
    aspiration: int

    def __init__(self):
        self.name = "Satisficing"

    def initialize(self, matrix_suite: MatrixSuite, player: str) -> None:
        self.actions = matrix_suite.get_actions(player)
        self.action = random.choice(self.actions)  # randomly select action
        payoff_matrix = matrix_suite.payoff_matrix
        '''
        total_payoff = 0
        for ra in range(len(payoff_matrix)):
            for ca in range(len(payoff_matrix[0])):
                if player == "row":
                    total_payoff += payoff_matrix[ra][ca][0]
                else:
                    total_payoff += payoff_matrix[ra][ca][1]
        self.aspiration = total_payoff / \
            (len(payoff_matrix) * len(payoff_matrix[0]))
        '''
        total_payoff = 0  # calculate average payoff for randomly selected action to use for aspiration initialization
        if player == "row":
            for oa in matrix_suite.get_actions("col"):
                total_payoff += payoff_matrix[self.action][oa][0]
            self.aspiration = total_payoff / len(self.actions)
        else:
            for oa in matrix_suite.get_actions("row"):
                total_payoff += payoff_matrix[oa][self.action][1]
            self.aspiration = total_payoff / len(self.actions)

    def get_action(self, round_: int) -> Action:
        """Selects random action if payoff of past action < aspiration and the past action otherwise."""
        return self.action

    def update(self, round_: int, action: Action, payoff: Payoff, opp_action: Action) -> None:
        if self.aspiration <= payoff:
            self.action = action
        else:
            # randomly select different action
            self.action = random.choice(self.actions)
        self.aspiration = 0.1 * self.aspiration + 0.9 * payoff  # update aspiration


class Softmax(Strategy):
    """Implements the Softmax algorithm for lambda = 0.1 and gamma = 1.0."""
    actions: List[Action]
    qualities: List[float]

    def __init__(self):
        self.name = "Softmax"

    def initialize(self, matrix_suite: MatrixSuite, player: str) -> None:
        self.actions = matrix_suite.get_actions(player)
        payoff_matrix = matrix_suite.payoff_matrix

        self.qualities = np.zeros(len(self.actions))
        for action in self.actions:  # calculate average payoff for each action to use as init for qualities
            total_payoff = 0
            if player == "row":
                for oa in matrix_suite.get_actions("col"):
                    total_payoff += payoff_matrix[action][oa][0]
                self.qualities[action] = total_payoff / \
                    len(matrix_suite.get_actions("col"))
            else:
                for oa in matrix_suite.get_actions("row"):
                    total_payoff += payoff_matrix[oa][action][1]
                self.qualities[action] = total_payoff / \
                    len(matrix_suite.get_actions("row"))

    def get_action(self, round_: int) -> Action:
        """Selects action with probability proportional to its Softmax equation."""
        denominator = 0
        for action in self.actions:
            denominator += math.exp(self.qualities[action])

        p_values = np.zeros(len(self.actions))
        for action in self.actions:
            p_values[action] = (
                math.exp(self.qualities[action])) / denominator

        return np.random.choice(self.actions, p=p_values)

    def update(self, round_: int, action: Action, payoff: Payoff, opp_action: Action) -> None:
        """Updates qualities of each action using learning rate = 0.1."""
        self.qualities[action] = self.qualities[action] + \
            0.1 * (payoff - self.qualities[action])


class Fictitious(Strategy):
    """Implements the fictitious play algorithm."""
    actions: List[Action]
    payoff_matrix: List[int]
    beliefs: List[float]
    player: str

    def __init__(self):
        self.name = "Fictitious"

    def initialize(self, matrix_suite: MatrixSuite, player: str) -> None:
        self.actions = matrix_suite.get_actions(player)
        self.payoff_matrix = matrix_suite.payoff_matrix
        self.player = player

        self.beliefs = np.zeros(len(self.actions))
        for action in self.actions:
            total_payoff = 0.0
            if player == "row":
                for oa in matrix_suite.get_actions("col"):
                    total_payoff += self.payoff_matrix[action][oa][0]
                self.beliefs[action] = total_payoff / \
                    len(matrix_suite.get_actions("col"))
            else:
                for oa in matrix_suite.get_actions("row"):
                    total_payoff += self.payoff_matrix[oa][action][1]
                self.beliefs[action] = total_payoff / \
                    len(matrix_suite.get_actions("row"))

    def get_action(self, round_: int) -> Action:
        """Selects action with highest belief."""
        return np.argmax(self.beliefs)

    def update(self, round_: int, action: Action, payoff: Payoff, opp_action: Action) -> None:
        """Updates belief of played action using payoff."""
        if self.player == "row":
            self.beliefs[action] += self.payoff_matrix[action][opp_action][0]
        else:
            self.beliefs[action] += self.payoff_matrix[opp_action][action][1]


class Bully(Strategy):
    """Implements the Bully algorithm."""
    actions: List[Action]
    max_strategy: List[Action]

    def __init__(self):
        self.name = "Bully"

    def initialize(self, matrix_suite: MatrixSuite, player: str) -> None:
        self.actions = matrix_suite.get_actions(player)
        self.max_strategy = []
        payoff_matrix = matrix_suite.payoff_matrix

        max_strategies_opponent = {action: [] for action in self.actions}
        for action in self.actions:
            max_payoff_opponent = 0
            if player == "row":
                for oa in matrix_suite.get_actions("col"):
                    max_payoff_opponent = max(
                        max_payoff_opponent, payoff_matrix[action][oa][1])
                for oa in matrix_suite.get_actions("col"):
                    if payoff_matrix[action][oa][1] == max_payoff_opponent:
                        max_strategies_opponent[action].append(oa)
            else:
                for oa in matrix_suite.get_actions("row"):
                    max_payoff_opponent = max(
                        max_payoff_opponent, payoff_matrix[oa][action][0])
                for oa in matrix_suite.get_actions("row"):
                    if payoff_matrix[oa][action][0] == max_payoff_opponent:
                        max_strategies_opponent[action].append(oa)

        security_values = [0] * len(self.actions)
        for action in self.actions:
            min_payoff = 10
            if player == "row":
                for oa in max_strategies_opponent[action]:
                    min_payoff = min(
                        min_payoff, payoff_matrix[action][oa][0])
            else:
                for oa in max_strategies_opponent[action]:
                    min_payoff = min(
                        min_payoff, payoff_matrix[oa][action][1])
            security_values[action] = min_payoff

        self.max_strategy.append(np.argmax(security_values))

    def get_action(self, round_: int) -> Action:
        """Always selects best action according to opponents hypothesized behaviour."""
        return random.choice(self.max_strategy)

    def update(self, round_: int, action: Action, payoff: Payoff, opp_action: Action) -> None:
        """ Bully is stateless """
        pass


class PRM(Strategy):
    """"Implements the proportional regret matching algorithm."""
    actions: List[Action]
    regrets: List[float]
    payoff_matrix: List[int]
    player: str

    def __init__(self):
        self.name = "PRM"

    def initialize(self, matrix_suite: MatrixSuite, player: str) -> None:
        self.actions = matrix_suite.get_actions(player)
        self.player = player
        self.payoff_matrix = matrix_suite.payoff_matrix

        # initialize average regret
        if self.player == 'row':
            self.regrets = np.sum(np.array(matrix_suite.payoff_matrix).T[0], axis=0) / len(
                self.actions)
        if self.player == 'col':
            self.regrets = np.array(
                [np.sum(row) for row in np.array(matrix_suite.payoff_matrix).T[1]]) / len(self.actions)

    def get_action(self, round_: int) -> Action:
        """Selects action with probability proportional to its regret."""
        p_value = [0 for i in range(len(self.actions))]
        for action in self.actions:
            if self.regrets[action] > 0:
                p_value[action] = self.regrets[action]
        p_value = [p_value[i] / np.sum(p_value) for i in range(len(p_value))]
        return np.random.choice(self.actions, p=p_value)

    def update(self, round_: int, action: Action, payoff: Payoff, opp_action: Action) -> None:
        """Update regret for each potential action using potential payoff."""
        for potential_action in self.actions:
            if self.player == "row":
                potential_payoff = self.payoff_matrix[potential_action][opp_action][0]
            else:
                potential_payoff = self.payoff_matrix[opp_action][potential_action][1]
            self.regrets[potential_action] += potential_payoff - payoff


class ProbResponse(Strategy):
    """Our own strategy"""
    player: str
    matrix_suite: MatrixSuite
    opponent_actions: List[int]
    best_responses: List[Action]

    def __init__(self):
        self.name = "HardHeaded"

    def initialize(self, matrix_suite: MatrixSuite, player: str) -> None:
        self.player = player
        self.matrix_suite = matrix_suite
        self.best_responses = []
        self.opponent_actions = []
        if player == "row":
            for ca in range(len(matrix_suite.payoff_matrix[0])):
                best_action, best_response = 0, 0
                for ra in range(len(matrix_suite.payoff_matrix)):
                    if matrix_suite.payoff_matrix[ra][ca][0] > best_response:
                        best_action = ra
                self.best_responses.append(best_action)
                self.opponent_actions.append(0)
        else:
            for ra in range(len(matrix_suite.payoff_matrix)):
                best_action, best_response = 0, 0
                for ca in range(len(matrix_suite.payoff_matrix[0])):
                    if matrix_suite.payoff_matrix[ra][ca][1] > best_response:
                        best_action = ca
                self.best_responses.append(best_action)
                self.opponent_actions.append(0)

    def get_action(self, round_: int) -> Action:
        if round_ == 1:
            return self.best_responses[random.randint(0, len(self.best_responses) - 1)]
        rand = sum(self.opponent_actions) * random.random()
        temp_sum = 0
        for i in range(len(self.best_responses)):
            if temp_sum < rand and temp_sum + self.opponent_actions[i] >= rand:
                return self.best_responses[i]
            else:
                temp_sum += self.opponent_actions[i]

    def update(self, round_: int, action: Action, payoff: Payoff, opp_action: Action) -> None:
        self.opponent_actions[opp_action] += 1


class SmartBully(Strategy):
    """Implements the SmartBully algorithm."""
    actions: List[Action]
    past_opponent_actions: List[Action]
    max_actions: List[Action]
    max_actions_payoff: List[int]
    total_rounds: int

    def __init__(self):
        self.name = "SmartBully"

    def initialize(self, matrix_suite: MatrixSuite, player: str) -> None:
        self.actions = matrix_suite.get_actions(player)
        payoff_matrix = np.array(matrix_suite.payoff_matrix)
        if player == 'row':
            self.past_opponent_actions = np.ones(
                len(payoff_matrix[0])).astype(int)
        if player == 'col':
            self.past_opponent_actions = np.ones(
                len(payoff_matrix)).astype(int)
        self.total_rounds = len(self.past_opponent_actions)  # random init

        self.max_actions = np.zeros(
            len(self.past_opponent_actions)).astype(int)
        self.max_actions_payoff = np.zeros(len(self.past_opponent_actions))
        # get best action in response to each opponent action
        for opp_action in range(len(self.past_opponent_actions)):
            max_payoff, max_action = 0, 0  # random init
            for action in self.actions:
                if player == 'row':
                    payoff = payoff_matrix[action][opp_action][0]
                if player == 'col':
                    payoff = payoff_matrix[opp_action][action][1]
                if payoff > max_payoff:
                    max_payoff = payoff
                    max_action = action
            self.max_actions[opp_action] = max_action
            self.max_actions_payoff[opp_action] = max_payoff

    def get_action(self, round_: int) -> Action:
        """Selects action which yields highest expected payoff according to opponents past behaviour."""
        p_values = self.past_opponent_actions / \
            self.total_rounds  # probability of opponent selecting each action
        # expected payoff according to probability of opponent selecting each action
        expected_payoff = p_values * self.max_actions_payoff
        best_action = random.choice(np.argwhere(expected_payoff == np.amax(expected_payoff)))[
            0]  # select randomly if more than one action yields highest expected payoff
        return self.max_actions[best_action]

    def update(self, round_: int, action: Action, payoff: Payoff, opp_action: Action) -> None:
        self.total_rounds += 1
        self.past_opponent_actions[opp_action] += 1


class SmartHybrid(Strategy):
    """Implements the SmartHybrid algorithm."""
    actions: List[Action]
    strategies: List[Strategy]
    strategies_past_payoffs: List[float]
    strategies_past_count: List[int]
    total_rounds: int
    current_strategy: Strategy

    def __init__(self):
        self.name = "SmartHybrid"

    def initialize(self, matrix_suite: MatrixSuite, player: str) -> None:
        self.actions = matrix_suite.get_actions(player)
        self.strategies = [SmartGreedy(), SmartBully()]
        for strategy in self.strategies:
            strategy.initialize(matrix_suite, player)

        if player == 'row':
            payoff_matrix = np.array(matrix_suite.payoff_matrix).T[0]
        if player == 'col':
            payoff_matrix = np.array(matrix_suite.payoff_matrix).T[1]
        # max payoff used for geometric average
        max_payoff = np.amax(payoff_matrix)
        # min payoff used for geometric average
        min_payoff = np.amin(payoff_matrix)

        self.total_rounds = len(self.strategies)
        self.strategies_past_count = np.ones(len(self.strategies)).astype(
            int)  # document how often each strategy was used
        self.strategies_past_payoffs = np.full(len(self.strategies),
                                               (max_payoff - min_payoff) / 2)  # document cumulative payoff per strategy
        self.current_strategy = 0  # random init

    def get_action(self, round_: int) -> Action:
        """Pick strategy which performed the best to date."""
        expected_payoff = self.strategies_past_payoffs / \
            self.strategies_past_count  # expected payoff according to past payoffs
        self.current_strategy = random.choice(np.argwhere(expected_payoff == np.amax(expected_payoff)))[
            0]  # select randomly if more than one action yields highest expected payoff
        return self.strategies[self.current_strategy].get_action(self.total_rounds)

    def update(self, round_: int, action: Action, payoff: Payoff, opp_action: Action) -> None:
        self.strategies[self.current_strategy].update(
            self.total_rounds, action, payoff, opp_action)
        self.total_rounds += 1
        self.strategies_past_count[self.current_strategy] += 1
        self.strategies_past_payoffs[self.current_strategy] += payoff
        # print(self.strategies_past_payoffs, self.strategies_past_count)
