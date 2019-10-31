from aienvs.runners.Episode import Episode
import copy
import logging
import math
from aiagents.multi.BasicComplexAgent import BasicComplexAgent
import random


class TreeNode():

    def __init__(self, parent):
        self.numVisits = 0
        self.totalReward = 0
        self.Mn = 0
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
        oldVisits = self.numVisits
        oldTotalReward = self.totalReward
        if oldVisits==0:
            oldMean=0
        else:
            oldMean=oldTotalReward/oldVisits
        self.numVisits += 1
        self.totalReward += reward
        self.Mn += (reward - oldMean)*(reward - self.totalReward/self.numVisits)
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
                actionNode = selectedNode._UCT(self.parameters['explorationConstant'])
                if(actionNode.isFullyExpanded):
                    # sparse sampling
                    selectedNode = actionNode.sampleChild()
                else:
                    # calls expand once and then rollout
                    selectedNode, rolloutReward = actionNode.expand()
                    reward += rolloutReward
                    break
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
        agentActionNo = 0
        N = self.simulator.action_space.get(self.agentId).getSize()
        while True:
            #agentActionNo = agentActionNo % N
            #agentAction = {self.agentId: agentActionNo}
            #agentActionNo += 1
            agentAction = self.treeAgent.step(self.getState(), self.immediateReward, self.isTerminal)

            key = tuple(agentAction.items())
            if key not in self._children.keys():
                self._children[key] = ActionNode(agentAction, self)
                if not self._children[key].isFullyExpanded:
                    expandedNode = self._children[key].expand()
                    break

        
        if self.simulator.action_space.get(self.agentId).getSize() <= len(self._children.values()):
            self.isFullyExpanded = True#all([_childActionNode.isFullyExpanded for _childActionNode in self._children.values()])

        return expandedNode

    def _UCT(self, explorationValue):
        """
        the UCT formula
        """
        bestValue = float("-inf")
        bestNodes = []

        for child in self._children.values():
            childValue = child.totalReward / child.numVisits + max(self.parameters["maxSteps"] - self._STATE.step, 1.) * explorationValue * math.sqrt(2 * math.log(self.numVisits) / child.numVisits)
            if childValue > bestValue:
                bestValue = childValue
                bestNodes = [child]
            elif childValue == bestValue:
                bestNodes.append(child)

            if(explorationValue == 0):
                variance = child.Mn / child.numVisits #no Bessel correction to avoid division by 0
                print("Action: " + str(child.getAction()) + " child value " + str(childValue) + " numVisits " + str(child.numVisits) + " totalReward " + str(child.totalReward) + " variance " + str(variance/child.numVisits)) # variance of a sample mean formula

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
        self.numExpands += 1
        simulator = self.simulator
        simulator.setState(self._parent.getState())

        if(self.otherAgents):
            otherActions = self.otherAgents.step(self._parent.getState(), self._parent.immediateReward, self._parent.isTerminal)
            actions=copy.deepcopy(otherActions)
            otherActionsRep = tuple(otherActions.items())
        else:
            actions = {}
            otherActionsRep = ""

        actions.update(self.getAction())
        state, reward, done, info = simulator.step(actions)
        
        key = (state, otherActionsRep)
        if key not in self._children.keys():
            self._children[key] = StateNode(state, reward, done, self)
        self._children[key].numExpands += 1

        if(self.numVisits >= self.parameters['samplingLimit']):
            self.isFullyExpanded = True

        # rollout and backpropagate
        if done:
            rolloutReward = 0
        else:
            rolloutReward = self._rollout(simulator, state, reward, done)
        
        return self._children[key], rolloutReward + reward

    def _rollout(self, simulator, state, reward, done):
        rolloutEpisode = Episode(self.rolloutAgent, simulator, state)
        steps, rolloutReward = rolloutEpisode.run()
        return rolloutReward

