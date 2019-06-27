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
    def __init__(self, parent=None, agentId=None, simulator=None, expandLimit=None):
        self.numVisits = 0
        self.totalReward = 0
        self.numExpands = 0
        self.parent = parent
        self.children = {}
        self.isFullyExpanded=False

        if(parent):
            self.simulator = parent.simulator
            self.agentId = parent.agentId
            self.expandLimit = parent.expandLimit
        else:
            self.simulator = simulator
            self.agentId = agentId
            self.expandLimit=expandLimit
    
    def backpropagate(self, reward):
        node=self
        while node is not None:
            node.numVisits += 1
            node.totalReward += reward
            node = node.parent

    def getBestChild(self):
        """
        returns the best child
        """
        return self.selectNode(explorationValue=0)

    def selectNode(self, explorationValue):
        """
        selects child node based on the UCT formula
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
    def __init__(self, state, reward, done, parent, simulator=None, agentId=None, expandLimit=None):
        super().__init__(parent, agentId, simulator, expandLimit)
        self._STATE = copy.deepcopy(state)
        self.isTerminal = copy.deepcopy(done)
        self.isFullyExpanded = copy.deepcopy(self.isTerminal)
        self.immediateReward = reward

    def getState(self):
        """
        STATE should be immutable for lifetime of the object
        """
        return copy.deepcopy(self._STATE)

    def expand(self, myAgent, explorationValue=0):
        """
        expands the state node by the agent action child
        """
        if( self.isFullyExpanded ):
            # sparse sampling
            # return self.selectNode(explorationValue)
            raise( "Should never reach here" )

        agentAction = myAgent.step( self.getState(), self.immediateReward, self.isTerminal )

        try:
            childActionNode=self.children[str(agentAction)]
        except KeyError:
            # create new action node
            childActionNode=actionNode( agentAction, self )
            self.children[str(agentAction)]=childActionNode


        childActionNode.numExpands+=1

        if self.simulator.action_space.spaces.get(self.agentId).n <= len(self.children.values()):
            #self.isFullyExpanded = True
            self.isFullyExpanded = all( [_childActionNode.isFullyExpanded for _childActionNode in self.children.values()] )

        return childActionNode


class actionNode(treeNode):
    def __init__(self, action, parent: stateNode):
        super().__init__(parent)
        self._ACTION = copy.deepcopy(action)

    def getAction(self):
        """
        ACTION should be immutable for the lifetime of the object
        """
        return copy.deepcopy(self._ACTION)

    def selectNode(self, explorationValue=0):
        """
        we cannot really select a state node, we can just sample from it
        """
        return self.sampleChild()

    def expand(self, expandLimit, otherAgents):
        """
        expands a node by sampling from the simulator
        """
        #if( self.isFullyExpanded ):
        #    # sparse sampling
            #return self.selectNode()
        #    breakpoint()
        #    raise Exception("Should never reach here")

        self.simulator.setState(self.parent.getState())
        actions = otherAgents.step(self.parent.getState(), self.parent.immediateReward, self.parent.isTerminal)
        actions.update(self.getAction())
        
        state, reward, done, info = self.simulator.step(actions)
        try:  
            # state can be aggregated
            childStateNode = self.children[str(state)]
        except KeyError:
            # create new state node
            #childExpandLimit = self.simulator.action_space.spaces.get(self.agentId).n
            childStateNode = stateNode(state, reward, done, self)
            self.children[str(state)]=childStateNode

        childStateNode.numExpands+=1

        if( self.numExpands >= expandLimit ):
            self.isFullyExpanded = True

        return childStateNode


class mctsAgent():
    DEFAULT_PARAMETERS = {'iterationLimit':10000, 'explorationConstant': 1 / math.sqrt(2), 'samplingLimit': 20}

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
        self.agentId = agentId

        if(otherAgents):
            self._otherAgents = ComplexAgentComponent(copy.deepcopy(otherAgents))
        else:
            self._otherAgents = None

        self._treeAgent = RandomAgent( self.agentId, self._simulator )

    def step(self, observation, reward, done):
        observedState = copy.deepcopy(observation)
        self._simulator.setState(copy.deepcopy(observedState))
        self._root = stateNode(state=observedState, reward=0., done=done, simulator=self._simulator, agentId=self.agentId, parent=None, expandLimit=self._parameters['samplingLimit'])

        if done:
            return {self.agentId: self._simulator.action_space.spaces.get(self.agentId).sample()}

        if self._limitType == 'time':
            timeLimit = time.time() + self._parameters['timeLimit'] / 1000
            while time.time() < timeLimit:
                self._executeRound()
        else:
            for i in range(self._parameters['iterationLimit']):
                self._executeRound()

        bestChild = self._root.getBestChild()
        action = bestChild.getAction()
        logging.info("Action "+ str(action))

        return action

    def _executeRound(self):
        node, startingReward = self._selectNode(self._root)
        totalReward = self._rollout(node, startingReward)
        node.backpropagate(totalReward)
    
    def _selectNode(self, stateNode):
        startingReward = stateNode.immediateReward
        while not stateNode.isTerminal:
            if stateNode.isFullyExpanded:
                actionNode = stateNode.selectNode(self._parameters['explorationConstant'])
                stateNode = actionNode.selectNode(self._parameters['explorationConstant'])
                startingReward += stateNode.immediateReward
            else:
                stateNode = self._expand(stateNode)
                startingReward += stateNode.immediateReward
                break

        return stateNode, startingReward

    def _expand(self, parentStateNode):
        # expands by creating first an action node and then a state node
        # using str() for hashing -- shouldn't be slow but not the cleanest
        # TODO: consider making actions immutable
        childActionNode = parentStateNode.expand( self._treeAgent, self._parameters['explorationConstant'] )
        childStateNode = childActionNode.expand( self._parameters['samplingLimit'], self._otherAgents )

        return childStateNode

    def _rollout(self, node, startingReward):
        if node.isTerminal:
            return startingReward

        startState = node.getState()
        self._simulator.setState(startState)
        rolloutAgent =  RandomAgent( self.agentId, self._simulator )
        #removes Nones from the list
        jointAgent = ComplexAgentComponent( list(filter(None.__ne__, [rolloutAgent, self._otherAgents])) )
        firstActions = jointAgent.step( startState, node.immediateReward, node.isTerminal )
        rolloutEpisode = Episode( jointAgent, self._simulator, firstActions )

        steps, rolloutReward = rolloutEpisode.run()
        totalReward = startingReward + rolloutReward

        return totalReward
