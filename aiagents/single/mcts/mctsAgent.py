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
    def __init__(self, state, reward, done, parent):
        self._STATE = copy.deepcopy(state)
        self.isTerminal = copy.deepcopy(done)
        self.isFullyExpanded = copy.deepcopy(self.isTerminal)
        self.parent = parent
        self.numVisits = 0
        self.immediateReward = reward
        self.totalReward = 0
        self.children = {}

    def getState(self):
        """
        STATE should be immutable for lifetime of the object
        """
        return copy.deepcopy(self._STATE)

class actionNode():
    def __init__(self, action, parent: treeNode):
        self._ACTION = copy.deepcopy(action)
        self.parent = parent
        self.children = {}
        self.numVisits = 0
        self.totalReward = 0
        self.numExpands = 0
        self.isFullyExpanded=False

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
        TODO: remove parameter duplication (e.g. self._parameters, self._timeLimit)
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

        self._root = treeNode(state=observedState, reward=0, done=done, parent=None)

        if done:
            return {self._agentId: self._simulator.action_space.spaces.get(self._agentId).sample()}

        if self._limitType == 'time':
            timeLimit = time.time() + self._parameters['timeLimit'] / 1000
            while time.time() < timeLimit:
                self._executeRound()
        else:
            for i in range(self._parameters['iterationLimit']):
                self._executeRound()

        #breakpoint()
        bestChild = self._getBestChild(self._root, 0)
        return {self._agentId: self._getAction(self._root, bestChild)}

    def _executeRound(self):
        node, startingReward = self._selectNode(self._root)
        totalReward = self._rollout(node, startingReward)
        self._backpropagate(node, totalReward)
    
    def _selectNode(self, stateNode):
        startingReward = stateNode.immediateReward
        while not stateNode.isTerminal:
            if stateNode.isFullyExpanded:
                actionNode = self._getBestChild(stateNode, self._parameters['explorationConstant'])
                childStateNodes = list(actionNode.children.values())
                stateNode = random.choices( childStateNodes, [childState.numVisits for childState in childStateNodes] )[0]
                startingReward += stateNode.immediateReward
            else:
                stateNode = self._expand(stateNode)
                startingReward += stateNode.immediateReward
                break

        return stateNode, startingReward

    def _expand(self, node):
        # expands by creating first an action node and then a state node
        startState=node.getState()
        self._simulator.setState(startState)

        agentActionSpace = self._simulator.action_space.spaces.get(self._agentId) 
        treeAgent = RandomAgent( self._agentId, self._simulator )
        # remove Nones from the list
        jointAgent = ComplexAgentComponent( list(filter(None.__ne__, [treeAgent, self._otherAgents])) )
        actions = jointAgent.step( startState, node.immediateReward, node.isTerminal )
        agentAction = actions.get( self._agentId )

        try:
            # using str() for hashing -- shouldn't be slow but not the cleanest
            # TODO: consider making actions immutable
            childActionNode=node.children[str(agentAction)]
        except KeyError:
            childActionNode=actionNode( agentAction, node )
            node.children[str(agentAction)]=childActionNode

        if( childActionNode.isFullyExpanded ):
            childStateNodes = list(childActionNode.children.values())
            childStateNode = random.choices( childStateNodes, [childState.numVisits for childState in childStateNodes] )[0]
        else:
            state, reward, done, info = self._simulator.step(actions)
            # using str() for hashing -- shouldn't be slow but not the cleanest
            # TODO: consider making states immutable
            try:  
                childStateNode = childActionNode.children[str(state)]
            except KeyError:
                childStateNode = treeNode(state, reward, done, childActionNode)
                childActionNode.children[str(state)]=childStateNode
            childActionNode.numExpands+=1

        if( childActionNode.numExpands >= self._parameters['samplingLimit'] ):
            childActionNode.isFullyExpanded = True

        if agentActionSpace.n == len(node.children.values()):
            node.isFullyExpanded = all( [_childActionNode.isFullyExpanded for _childActionNode in node.children.values()] )

        return childStateNode

    def _backpropagate(self, node, reward):
        while node is not None:
            node.numVisits += 1
            node.totalReward += reward
            node = node.parent

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
            

    def _getBestChild(self, node, explorationValue):
        bestValue = float("-inf")
        bestNodes = []
        if(explorationValue==0):
            logging.info("AGENT ID: " + self._agentId)
        for child in node.children.values():
            nodeValue = child.totalReward / child.numVisits + explorationValue * math.sqrt(2 * math.log(node.numVisits) / child.numVisits)
            #if(explorationValue==0 && no):
            #    logging.info("Action: " + str(self._simulator.ACTIONS[action]) + " Node value " + str(nodeValue) + " numVisits " + str(child.numVisits) + " totalReward " + str(child.totalReward))
            if nodeValue > bestValue:
                bestValue = nodeValue
                bestNodes = [child]
            elif nodeValue == bestValue:
                bestNodes.append(child)
        return random.choice(bestNodes)

    def _getAction(self, root, bestChild):
        for node in root.children.values():
            if node is bestChild:
                return node.getAction()
