import time
import copy
import logging
from aiagents.single.RandomAgent import RandomAgent
from aiagents.single.mcts.nodes import RootNode
from aiagents.multi.ComplexAgentComponent import ComplexAgentComponent
from aienvs.Environment import Env
import math

class MctsAgent():
    DEFAULT_PARAMETERS = {'treeParameters': {'explorationConstant': 1 / math.sqrt(2), 'samplingLimit': 20}}

    def __init__(self, agentId, environment: Env, parameters: dict, otherAgents=None, simulator=None, treeAgent=None, rolloutAgent=None): 
        """
        TBA
        """
        self._parameters = copy.deepcopy(self.DEFAULT_PARAMETERS)
        self._parameters.update(parameters)
        self.agentId = agentId

        if 'timeLimit' in self._parameters:
            if 'iterationLimit' in self._parameters:
                raise ValueError("Cannot have both a time limit and an iteration limit")
            self._limitType = 'time'
        else:
            if 'iterationLimit' not in self._parameters:
                DEFAULT_LIMIT=1000
                logging.error("Must have either a time limit or an iteration limit. Using default iteration limit: " + str(DEFAULT_LIMIT))
                self._parameters['iterationLimit']=DEFAULT_LIMIT
            # number of iterations of the search
            if self._parameters['iterationLimit'] < 1:
                raise ValueError("Iteration limit must be greater than one")
            self._limitType = 'iterations'

        if simulator is None:
            self._simulator = copy.deepcopy(environment)
        else:
            self._simulator = simulator

        if treeAgent is None:
            self._treeAgent = RandomAgent( self.agentId, self._simulator )
        else:
            self._treeAgent = treeAgent

        if rolloutAgent is None:
            self._rolloutAgent = RandomAgent( self.agentId, self._simulator )
        else:
            self._rolloutAgent
            
        self._otherAgents = otherAgents
       
    def step(self, observation, reward, done):
        if done:
            # whatever action is ok
            return self._treeAgent.step(observation, reward, done)

        root = RootNode(state=observation, reward=0., done=done, simulator=self._simulator, 
                agentId=self.agentId, parameters=self._parameters['treeParameters'],
                treeAgent=self._treeAgent, otherAgents=self._otherAgents, rolloutAgent = self._rolloutAgent)

        if self._limitType == 'time':
            timeLimit = time.time() + self._parameters['timeLimit']
            while time.time() < timeLimit:
                root.executeRound()
        else:
            for i in range(self._parameters['iterationLimit']):
                root.executeRound()

        action = root.getBestChild().getAction()
        logging.info("Action "+ str(action))

        return action

