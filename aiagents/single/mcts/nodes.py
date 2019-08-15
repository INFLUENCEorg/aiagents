from aienvs.runners.Episode import Episode
import copy
import logging
import math
from aiagents.multi.ComplexAgentComponent import BasicComplexAgent
import random


class TreeNode():

    def __init__(self, parent):
        self.numVisits = 0
        self.totalReward = 0
        self.numExpands = 0
        self.isFullyExpanded = False
        # parent and children kept private. Nice!
        # we need the parent to backpropagate unfortunately
        self._parent = parent
        self._children = {}

        if(parent):
            self.simulator = self._parent.simulator
            self.agentId = self._parent.agentId
            self.treeAgent = self._parent.treeAgent
            self.otherAgents = self._parent.otherAgents
            self.rolloutAgent = self._parent.rolloutAgent
            self.parameters = self._parent.parameters
 
    def backpropagate(self, reward):
        # recursion cleaner -- allows to keep parent private -- and not that much slower (10%) but possibility of optimization here
        self.numVisits += 1
        self.totalReward += reward
        if self._parent is not None:
            self._parent.backpropagate(reward)


class StateNode(TreeNode):

    def __init__(self, state, reward, done, parent: TreeNode):
        super().__init__(parent)
        self._STATE = copy.deepcopy(state)
        self.isTerminal = copy.deepcopy(done)
        self.isFullyExpanded = copy.deepcopy(self.isTerminal)
        self.immediateReward = reward

    def executeRound(self):
        """
        we select the next state node by selecting the best action node then sampling a child state
        """
        selectedNode = self
        reward = 0

        while True:
            reward += selectedNode.immediateReward
            if selectedNode.isFullyExpanded:
                if selectedNode.isTerminal:
                    break
                selectedNode = selectedNode._UCT(self.parameters['explorationConstant']).sampleChild()
            else:
                # calls expand twice, first for the state node then the action node
                selectedNode, rolloutReward = selectedNode.expand()
                reward += rolloutReward
                break

        selectedNode.backpropagate(reward)

    def getBestChild(self):
        """
        returns the best child
        """
        return self._UCT(explorationValue=0)
 
    def getState(self):
        """
        STATE should be immutable for lifetime of the object
        """
        return copy.deepcopy(self._STATE)

    def expand(self):
        """
        expands the state node by the agent action, automatically expands the child node
        """
        agentAction = self.treeAgent.step(self.getState(), self.immediateReward, self.isTerminal)

        key = str(agentAction)
        if key not in self._children.keys():
            self._children[key] = ActionNode(agentAction, self)
        self._children[key].numExpands += 1

        expandedNode = self._children[key].expand()

        if self.simulator.action_space.spaces.get(self.agentId).n <= len(self._children.values()):
            self.isFullyExpanded = all([_childActionNode.isFullyExpanded for _childActionNode in self._children.values()])

        return expandedNode

    def _UCT(self, explorationValue):
        """
        the UCT formula
        """
        bestValue = float("-inf")
        bestNodes = []
        for child in self._children.values():
            childValue = child.totalReward / child.numVisits + explorationValue * math.sqrt(2 * math.log(self.numVisits) / child.numVisits)
            if childValue > bestValue:
                bestValue = childValue
                bestNodes = [child]
            elif childValue == bestValue:
                bestNodes.append(child)

            if(explorationValue == 0):
                logging.info("Action: " + str(child.getAction()) + " child value " + str(childValue) + " numVisits " + str(child.numVisits) + " totalReward " + str(child.totalReward))

        return random.choice(bestNodes)


class RootNode(StateNode):

    def __init__(self, state, reward, done, agentId=None, simulator=None, parameters=None, treeAgent=None, otherAgents=None, rolloutAgent=None):
        # no parent
        super().__init__(state, reward, done, None)
        # initialize the parent variables/pointers
        self.simulator = simulator
        self.agentId = agentId
        self.treeAgent = treeAgent
        self.rolloutAgent = rolloutAgent
        self.otherAgents = otherAgents
        self.parameters = parameters


class ActionNode(TreeNode):

    def __init__(self, action, parent: StateNode):
        super().__init__(parent)
        self._ACTION = copy.deepcopy(action)

    def getAction(self):
        """
        ACTION should be immutable for the lifetime of the object
        """
        return copy.deepcopy(self._ACTION)

    def sampleChild(self):
        """
        Sampling a child by weighted number of expands,
        emulating the environment dynamics
        """
        childStateNodes = list(self._children.values())
        return random.choices(childStateNodes, [childState.numExpands for childState in childStateNodes])[0]

    def expand(self):
        """
        expands a node by sampling from the simulator
        """
        # sample an action, step the simulator
        simulator = self.simulator
        simulator.setState(self._parent.getState())
        actions = {}
        if(self.otherAgents):
            actions = self.otherAgents.step(self._parent.getState(), self._parent.immediateReward, self._parent.isTerminal)
        actions.update(self.getAction())
        state, reward, done, info = simulator.step(actions)
        
        key = str(state)
        if key not in self._children.keys():
            self._children[key] = StateNode(state, reward, done, self)
        self._children[key].numExpands += 1

        if(self.numExpands >= self.parameters['samplingLimit']):
            self.isFullyExpanded = True

        # rollout and backpropagate
        if done:
            rolloutReward = 0
        else:
            rolloutReward = self._rollout(simulator, state, reward, done, self.otherAgents)
        
        return self._children[key], rolloutReward + reward

    def _rollout(self, simulator, state, reward, done, otherAgents):
        # removes Nones from the list
        jointAgent = BasicComplexAgent([self.rolloutAgent] + self.otherAgents) 
        rolloutEpisode = Episode(jointAgent, simulator, state)
        steps, rolloutReward = rolloutEpisode.run()
        return rolloutReward

