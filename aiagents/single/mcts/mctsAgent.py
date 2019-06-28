import time
import copy
import logging
from aiagents.single.RandomAgent import RandomAgent
from aiagents.single.mcts.nodes import RootNode
from aiagents.multi.ComplexAgentComponent import ComplexAgentComponent
from aienvs.Environment import Env
import math

class MctsAgent():
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
    
        # TODO: create factories and construct these objects
        self._treeAgent = RandomAgent( self.agentId, self._simulator )
        self._rolloutAgent = RandomAgent( self.agentId, self._simulator )

    def step(self, observation, reward, done):
        if done:
            # whatever action is ok
            return self._treeAgent.step(observation, reward, done)

        root = RootNode(state=observation, reward=0., done=done, simulator=self._simulator, 
                agentId=self.agentId, parameters=self._parameters['treeParameters'],
                treeAgent=self._treeAgent, otherAgents=self._otherAgents, rolloutAgent = self._rolloutAgent)

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

