import time
import math
import random
import copy

import sys

# TODO: replace by new agent?
def randomPolicy(state, agentId, simulator, cumulativeReward=0):
    simulator.setState(state)

    while True:
        try:
            action = simulator.action_space.spaces.get(agentId).sample() #random.choice(state.getPossibleActions())
        except IndexError:
            raise Exception("Non-terminal state has no possible actions: " + str(state))
        obs, reward, done, info = simulator.step({agentId:action})
        cumulativeReward=+reward
        if done:
            break


    return cumulativeReward

class treeNode():
    def __init__(self, state, reward, done, parent):
        self.state = copy.deepcopy(state)
        self.isTerminal = copy.deepcopy(done)
        self.isFullyExpanded = copy.deepcopy(self.isTerminal)
        self.parent = parent
        self.numVisits = 0
        self.immediateReward = reward
        self.totalReward = 0
        self.children = {}


class mctsAgent():
    def __init__(self, agentId, environment, timeLimit=None, iterationLimit=None, explorationConstant=1 / math.sqrt(2),
                 rolloutPolicy=randomPolicy):
        if timeLimit != None:
            if iterationLimit != None:
                raise ValueError("Cannot have both a time limit and an iteration limit")
            # time taken for each MCTS search in milliseconds
            self._timeLimit = timeLimit
            self._limitType = 'time'
        else:
            if iterationLimit == None:
                raise ValueError("Must have either a time limit or an iteration limit")
            # number of iterations of the search
            if iterationLimit < 1:
                raise ValueError("Iteration limit must be greater than one")
            self._searchLimit = iterationLimit
            self._limitType = 'iterations'
        self._explorationConstant = explorationConstant
        self._rollout = rolloutPolicy
        self._simulator = copy.deepcopy(environment)
        self._agentId = agentId

    def step(self, observation, reward, done):
        self._simulator.setState(copy.deepcopy(observation))
        self._isRootTerminal = done

        self._root = treeNode(state=self._simulator.getState(), reward=0, done=self._isRootTerminal, parent=None)
        if self._isRootTerminal:
            return {self._agentId: self._simulator.action_space.spaces.get(self._agentId).sample()}

        if self._limitType == 'time':
            timeLimit = time.time() + self._timeLimit / 1000
            while time.time() < timeLimit:
                self._executeRound()
        else:
            for i in range(self._searchLimit):
                self._executeRound()

        bestChild = self._getBestChild(self._root, 0)
        return {self._agentId: self._getAction(self._root, bestChild)}

    def _executeRound(self):
        node, startingReward = self._selectNode(self._root)
        totalReward = self._rollout(node.state, self._agentId, self._simulator, startingReward)
        self._backpropagate(node, totalReward)

    def _selectNode(self, node):
        startingReward = node.immediateReward
        while not node.isTerminal:
            if node.isFullyExpanded:
                node = self._getBestChild(node, self._explorationConstant)
                startingReward += node.immediateReward
            else:
                node = self._expand(node)
                startingReward += node.immediateReward
                break

        return node, startingReward

    def _expand(self, node):
        self._simulator.setState(node.state)
        agentActionSpace = self._simulator.action_space.spaces.get(self._agentId)

        while True:
            action = agentActionSpace.sample()
            if action not in node.children.keys():
                break

        state, reward, done, info = self._simulator.step({self._agentId:action})
        newNode = treeNode(state, reward, done, node)
        node.children[action] = newNode
        if agentActionSpace.n == len(node.children):
            node.isFullyExpanded = True
        return newNode

        raise Exception("Should never reach here")

    def _backpropagate(self, node, reward):
        while node is not None:
            node.numVisits += 1
            node.totalReward += reward
            node = node.parent

    def _getBestChild(self, node, explorationValue):
        bestValue = float("-inf")
        bestNodes = []
        for child in node.children.values():
            nodeValue = child.totalReward / child.numVisits + explorationValue * math.sqrt(2 * math.log(node.numVisits) / child.numVisits)
            if nodeValue > bestValue:
                bestValue = nodeValue
                bestNodes = [child]
            elif nodeValue == bestValue:
                bestNodes.append(child)
        return random.choice(bestNodes)

    def _getAction(self, root, bestChild):
        for action, node in root.children.items():
            if node is bestChild:
                return action
