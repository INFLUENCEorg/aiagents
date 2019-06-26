import time
import random
import math
import copy
import logging
from aiagents.single.RandomAgent import RandomAgent
from aiagents.multi.ComplexAgentComponent import ComplexAgentComponent
from aienvs.runners.Episode import Episode
from aienvs.Environment import Env

class treeNode():
    def __init__(self, parent):
        self.numVisits = 0
        self.totalReward = 0
        self.numExpands = 0
        self.parent = parent
        self.children = {}
        self.isFullyExpanded=False
    
    def backpropagate(self, reward):
        node=self
        while node is not None:
            node.numVisits += 1
            node.totalReward += reward
            node = node.parent

    def getBestChild(self, explorationValue):
        """
        gets best child modulo the exploration constant for UCT
        """
        bestValue = float("-inf")
        bestNodes = []
        for child in self.children.values():
            # UCT
            childValue = child.totalReward / child.numVisits + explorationValue * math.sqrt(2 * math.log(self.numVisits) / child.numVisits)
            if childValue > bestValue:
                bestValue = childValue
                bestNodes = [child]
            elif childValue == bestValue:
                bestNodes.append(child)

            if(explorationValue==0):
                logging.info("Action: " + str(child.getAction()) + " child value " + str(childValue) + " numVisits " + str(child.numVisits) + " totalReward " + str(child.totalReward))

        return random.choice(bestNodes)

    def sampleChild(self):
        """
        Sampling a child by weighted number of expands,
        emulating the environment dynamics
        """
        childStateNodes = list(self.children.values())
        return random.choices( childStateNodes, [childState.numExpands for childState in childStateNodes] )[0]


class stateNode(treeNode):
    def __init__(self, state, reward, done, parent):
        super().__init__(parent)
        self._STATE = copy.deepcopy(state)
        self.isTerminal = copy.deepcopy(done)
        self.isFullyExpanded = copy.deepcopy(self.isTerminal)
        self.immediateReward = reward

    def getState(self):
        """
        STATE should be immutable for lifetime of the object
        """
        return copy.deepcopy(self._STATE)

class actionNode(treeNode):
    def __init__(self, action, parent: stateNode):
        super().__init__(parent)
        self._ACTION = copy.deepcopy(action)

    def getAction(self):
        """
        ACTION should be immutable for the lifetime of the object
        """
        return copy.deepcopy(self._ACTION)


class mctsAgent():
    DEFAULT_PARAMETERS = {'iterationLimit':5000, 'explorationConstant': 1 / math.sqrt(2), 'samplingLimit': 10}

    def __init__(self, agentId, environment: Env, parameters: dict, otherAgents=None):
        """
        TBA
        """
        self._parameters = copy.deepcopy(self.DEFAULT_PARAMETERS)
        self._parameters.update(parameters)

        if 'timeLimit' in self._parameters:
            if 'iterationLimit' in self._parameters:
                raise ValueError("Cannot have both a time limit and an iteration limit")
            # time taken for each MCTS search in milliseconds
            self._limitType = 'time'
        else:
            if 'iterationLimit' not in self._parameters:
                raise ValueError("Must have either a time limit or an iteration limit")
            # number of iterations of the search
            if self._parameters['iterationLimit'] < 1:
                raise ValueError("Iteration limit must be greater than one")
            self._limitType = 'iterations'

        self._simulator = copy.deepcopy(environment)
        self._agentId = agentId

        if(otherAgents):
            self._otherAgents = ComplexAgentComponent(otherAgents)
        else:
            self._otherAgents = None

    def step(self, observation, reward, done):
        observedState = copy.deepcopy(observation)
        self._simulator.setState(copy.deepcopy(observedState))

        self._root = stateNode(state=observedState, reward=0, done=done, parent=None)

        if done:
            return {self._agentId: self._simulator.action_space.spaces.get(self._agentId).sample()}

        if self._limitType == 'time':
            timeLimit = time.time() + self._parameters['timeLimit'] / 1000
            while time.time() < timeLimit:
                self._executeRound()
        else:
            for i in range(self._parameters['iterationLimit']):
                self._executeRound()

        bestChild = self._root.getBestChild(explorationValue = 0)
        action = bestChild.getAction()
        logging.info("Agent id " + self._agentId + " best action " + self._simulator.ACTIONS[action])

        return {self._agentId: action}

    def _executeRound(self):
        node, startingReward = self._selectNode(self._root)
        totalReward = self._rollout(node, startingReward)
        node.backpropagate(totalReward)
    
    def _selectNode(self, stateNode):
        startingReward = stateNode.immediateReward
        while not stateNode.isTerminal:
            if stateNode.isFullyExpanded:
                actionNode = stateNode.getBestChild(self._parameters['explorationConstant'])
                stateNode = actionNode.sampleChild()
                startingReward += stateNode.immediateReward
            else:
                stateNode = self._expand(stateNode)
                startingReward += stateNode.immediateReward
                break

        return stateNode, startingReward

    def _expand(self, node):
        # expands by creating first an action node and then a state node
        # using str() for hashing -- shouldn't be slow but not the cleanest
        # TODO: consider making actions immutable
        startState=node.getState()
        self._simulator.setState(startState)

        agentActionSpace = self._simulator.action_space.spaces.get(self._agentId) 
        treeAgent = RandomAgent( self._agentId, self._simulator )
        # remove Nones from the list
        jointAgent = ComplexAgentComponent( list(filter(None.__ne__, [treeAgent, self._otherAgents])) )
        actions = jointAgent.step( startState, node.immediateReward, node.isTerminal )
        agentAction = actions.get( self._agentId )

        try:
            childActionNode=node.children[str(agentAction)]
        except KeyError:
            # create new action node
            childActionNode=actionNode( agentAction, node )
            node.children[str(agentAction)]=childActionNode

        if( childActionNode.isFullyExpanded ):
            # sparse sampling
            childStateNode = childActionNode.sampleChild()
        else:
            state, reward, done, info = self._simulator.step(actions)
            try:  
                # state can be aggregated
                childStateNode = childActionNode.children[str(state)]
            except KeyError:
                # create new state node
                childStateNode = stateNode(state, reward, done, childActionNode)
                childActionNode.children[str(state)]=childStateNode
            childStateNode.numExpands+=1
            childActionNode.numExpands+=1

        if( childActionNode.numExpands >= self._parameters['samplingLimit'] ):
            childActionNode.isFullyExpanded = True

        if agentActionSpace.n == len(node.children.values()):
            node.isFullyExpanded = all( [_childActionNode.isFullyExpanded for _childActionNode in node.children.values()] )

        return childStateNode

    def _rollout(self, node, startingReward):
        if node.isTerminal:
            return startingReward

        startState = node.getState()
        self._simulator.setState(startState)

        rolloutAgent =  RandomAgent( self._agentId, self._simulator )
        #removes Nones from the list
        jointAgent = ComplexAgentComponent( list(filter(None.__ne__, [rolloutAgent, self._otherAgents])) )
        firstActions = jointAgent.step( startState, node.immediateReward, node.isTerminal )
        rolloutEpisode = Episode( jointAgent, self._simulator, firstActions )

        steps, rolloutReward = rolloutEpisode.run()
        return startingReward + rolloutReward

