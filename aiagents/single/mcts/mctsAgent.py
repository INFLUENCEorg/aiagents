import time
import math
import random
import copy
from aiagents.single.RandomAgent import RandomAgent
from aiagents.multi.ComplexAgentComponent import ComplexAgentComponent
from aienvs.runners.Episode import Episode


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
    def __init__(self, agentId, environment, timeLimit=None, iterationLimit=None, explorationConstant=1 / math.sqrt(2), otherAgents=None):
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
        self._simulator = copy.deepcopy(environment)
        self._agentId = agentId
        self._otherAgents = otherAgents

    def step(self, observation, reward, done):
        observedState = copy.deepcopy(observation)
        self._simulator.setState(copy.deepcopy(observedState))

        self._root = treeNode(state=observedState, reward=0, done=done, parent=None)

        if done:
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
        totalReward = self._rollout(node, startingReward)
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
        
        treeAgent = RandomAgent( self._agentId, agentActionSpace )
        # remove Nones from the list
        jointAgent = ComplexAgentComponent( list(filter(None.__ne__, [treeAgent, self._otherAgents])) )
        actions = jointAgent.step(node.state, node.immediateReward, node.isTerminal)
        state, reward, done, info = self._simulator.step(actions)

        newNode = treeNode(state, reward, done, node)
        agentAction = actions.get( self._agentId )

        node.children[agentAction] = newNode

        if agentActionSpace.n == len(node.children):
            node.isFullyExpanded = True
        return newNode

        raise Exception("Should never reach here")

    def _backpropagate(self, node, reward):
        while node is not None:
            node.numVisits += 1
            node.totalReward += reward
            node = node.parent

    def _rollout(self, node, startingReward):
        if node.isTerminal:
            return startingReward

        rolloutAgent =  RandomAgent( self._agentId, self._simulator.action_space.spaces.get(self._agentId) )
        jointAgent = ComplexAgentComponent( list(filter(None.__ne__, [rolloutAgent, self._otherAgents])) )
        firstActions = jointAgent.step( node.state, node.immediateReward, node.isTerminal )
        rolloutEpisode = Episode( jointAgent, self._simulator, firstActions )


        steps, rolloutReward = rolloutEpisode.run()

        return startingReward + rolloutReward
            

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
