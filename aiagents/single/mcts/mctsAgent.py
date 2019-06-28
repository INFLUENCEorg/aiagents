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
    def __init__(self, parent=None, agentId=None, simulator=None, parameters=None, treeAgent=None, otherAgents=None, rolloutAgent=None):
        self.numVisits = 0
        self.totalReward = 0
        self.numExpands = 0
        self.parent = parent
        self.children = {}
        self.isFullyExpanded=False

        if(parent):
            self.simulator = parent.simulator
            self.agentId = parent.agentId
            self.treeAgent = parent.treeAgent
            self.otherAgents = parent.otherAgents
            self.rolloutAgent = parent.rolloutAgent
            self.parameters = parent.parameters
        else:
            self.simulator = simulator
            self.agentId = agentId
            self.treeAgent = treeAgent
            self.rolloutAgent = rolloutAgent
            self.otherAgents = otherAgents
            self.parameters = parameters

class stateNode(treeNode):
    def __init__(self, state, reward, done, parent, simulator=None, agentId=None, parameters=None, treeAgent=None, otherAgents=None, rolloutAgent=None):
        super().__init__(parent, agentId, simulator, parameters, treeAgent, otherAgents)
        self._STATE = copy.deepcopy(state)
        self.isTerminal = copy.deepcopy(done)
        self.isFullyExpanded = copy.deepcopy(self.isTerminal)
        self.immediateReward = reward

    def executeRound(self):
        """
        we select the next state node by selecting the best action node then sampling a child state
        """
        selectedNode = self
        reward=0
        
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
        return self._UCT(explorationValue=0)
 
    def _UCT(self, explorationValue):
        """
        the UCT formula
        """
        bestValue = float("-inf")
        bestNodes = []
        for child in self.children.values():
            childValue = child.totalReward / child.numVisits + explorationValue * math.sqrt(2 * math.log(self.numVisits) / child.numVisits)
            if childValue > bestValue:
                bestValue = childValue
                bestNodes = [child]
            elif childValue == bestValue:
                bestNodes.append(child)

            if(explorationValue==0):
                logging.info("Action: " + str(child.getAction()) + " child value " + str(childValue) + " numVisits " + str(child.numVisits) + " totalReward " + str(child.totalReward))

        return random.choice(bestNodes)


    def getState(self):
        """
        STATE should be immutable for lifetime of the object
        """
        return copy.deepcopy(self._STATE)

    def expand(self):
        """
        expands the state node by the agent action, automatically expands the child node
        """
        agentAction = self.treeAgent.step( self.getState(), self.immediateReward, self.isTerminal )

        key = str(agentAction)
        if key not in self.children.keys():
            self.children[key]=actionNode( agentAction, self )
        self.children[key].numExpands+=1

        if self.simulator.action_space.spaces.get(self.agentId).n <= len(self.children.values()):
            self.isFullyExpanded = all( [_childActionNode.isFullyExpanded for _childActionNode in self.children.values()] )

        return self.children[key].expand()


class actionNode(treeNode):
    def __init__(self, action, parent: stateNode):
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
        childStateNodes = list(self.children.values())
        return random.choices( childStateNodes, [childState.numExpands for childState in childStateNodes] )[0]

    def expand(self):
        """
        expands a node by sampling from the simulator
        """
        # sample an action, step the simulator
        simulator = self.simulator
        simulator.setState(self.parent.getState())
        actions = self.otherAgents.step(self.parent.getState(), self.parent.immediateReward, self.parent.isTerminal)
        actions.update(self.getAction())
        state, reward, done, info = simulator.step(actions)
        
        key = str(state)
        if key not in self.children.keys():
            self.children[key] = stateNode(state, reward, done, self)
        self.children[key].numExpands+=1

        if( self.numExpands >= self.parameters['samplingLimit'] ):
            self.isFullyExpanded = True

        # rollout and backpropagate
        if done:
            rolloutReward = 0
        else:
            rolloutReward = self._rollout( simulator, state, reward, done, self.otherAgents )
        
        return self.children[key], rolloutReward + reward

    def _rollout(self, simulator, state, reward, done, otherAgents):
        rolloutAgent =  RandomAgent( self.agentId, simulator )
        #removes Nones from the list
        jointAgent = ComplexAgentComponent( list(filter(None.__ne__, [rolloutAgent, self.otherAgents])) )
        firstActions = jointAgent.step( state, reward, done )
        rolloutEpisode = Episode( jointAgent, simulator, firstActions )
        steps, rolloutReward = rolloutEpisode.run()
        return rolloutReward

class mctsAgent():
    DEFAULT_PARAMETERS = {'iterationLimit':10000, 'treeParameters': {'explorationConstant': 1 / math.sqrt(2), 'samplingLimit': 20}}

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
        self._rolloutAgent = RandomAgent( self.agentId, self._simulator )

    def step(self, observation, reward, done):
        root = stateNode(state=observation, reward=0., done=done, simulator=self._simulator, agentId=self.agentId, parent=None, parameters=self._parameters['treeParameters'],
                treeAgent=self._treeAgent, otherAgents=self._otherAgents, rolloutAgent = self._rolloutAgent)

        if done:
            return {self.agentId: self._simulator.action_space.spaces.get(self.agentId).sample()}

        if self._limitType == 'time':
            timeLimit = time.time() + self._parameters['timeLimit'] / 1000
            while time.time() < timeLimit:
                root.executeRound()
        else:
            for i in range(self._parameters['iterationLimit']):
                root.executeRound()

        action = root.getBestChild().getAction()
        logging.info("Action "+ str(action))

        return action

