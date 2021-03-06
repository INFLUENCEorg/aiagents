import time
import copy
import logging
from aiagents.single.RandomAgent import RandomAgent
from aiagents.single.mcts.nodes import RootNode
from aiagents.multi.BasicComplexAgent import BasicComplexAgent
from aienvs.Environment import Env
from aienvs.gym.ModifiedGymEnv import ModifiedGymEnv
from aienvs.gym.DecoratedSpace import DecoratedSpace
import math
from aiagents.single.AtomicAgent import AtomicAgent
from aiagents.AgentFactory import createAgent
from dict_recursive_update import recursive_update
from aienvs.FactoryFloor.DiyFactoryFloorAdapter import DiyFactoryFloorAdapter
from gym.spaces import Dict
import aienvs.EnvironmentFactory as EnvironmentFactory


class MctsAgent(AtomicAgent):
    DEFAULT_PARAMETERS = {'treeParameters': {
        'explorationConstant': 1 / math.sqrt(2),
        'samplingLimit': 20,
        'maxSteps': 0}}

    def __init__(self, agentId, actionspace: Dict, observationspace, parameters: dict): 
        """
        @param parameters dict that must contain keys 'otherAgents', 'treeAgent' and 'rolloutAgent'
        'otherAgents' must map to a (possibly empty) list of dict objects for a call to createAgents
        'treeAgent' and 'rolloutAgent' must map to a dict object for a call to createAgent.
        The dict must also contain a 'simulator' key containing a copy of the env parameters,
        so that the agent can create a duplicate environment. The simulator dict must contain
        a key 'fullname' containing the full name of the environment
        for the class loader (se EnvironmentsFactory).
        """
        super().__init__(agentId, actionspace, observationspace, parameters)
        if not ('treeAgent' in parameters and 'rolloutAgent' in parameters):
            raise "parameters does not contain 'treeAgent', 'rolloutAgent':" + str(parameters)
        self._parameters = copy.deepcopy(self.DEFAULT_PARAMETERS)
        self._parameters = recursive_update(self._parameters, parameters)

        if 'timeLimit' in self._parameters:
            if 'iterationLimit' in self._parameters:
                raise ValueError("Cannot have both a time limit and an iteration limit")
            self._limitType = 'time'
        else:
            if 'iterationLimit' not in self._parameters:
                DEFAULT_LIMIT = 1000
                logging.error("Must have either a time limit or an iteration limit. Using default iteration limit: " + str(DEFAULT_LIMIT))
                self._parameters['iterationLimit'] = DEFAULT_LIMIT
            # number of iterations of the search
            if self._parameters['iterationLimit'] < 1:
                raise ValueError("Iteration limit must be greater than one")
            self._limitType = 'iterations'
        
        # start the simulator environment
        envparams = self._parameters['simulator']
        e = EnvironmentFactory.createEnvironment(envparams['fullname'], envparams)
        self._simulator = ModifiedGymEnv(e, DecoratedSpace.create(copy.deepcopy(e.action_space)))

        # diyBonus logic: to refactor -- include in a simulator factory / only for FactoryFloor env
        diyBonus = self._parameters.get("diyBonus")
        if diyBonus is not None:
            self._simulator = DiyFactoryFloorAdapter(self._simulator, diyBonus, self.agentId)

        self._treeAgent = createAgent(self._simulator.action_space, self._simulator.observation_space, parameters['treeAgent'])

        if 'otherAgents' in parameters:
            rolloutAgentDict = copy.deepcopy(parameters['otherAgents'])
            rolloutAgentList = rolloutAgentDict['subAgentList']
            rolloutAgentList.append(parameters['rolloutAgent'])
            rolloutAgentDict['subAgentList'] = rolloutAgentList
            self._rolloutAgent = createAgent(self._simulator.action_space, self._simulator.observation_space, rolloutAgentDict)
            self._otherAgents = createAgent(self._simulator.action_space, self._simulator.observation_space, parameters['otherAgents'])
        else:
            self._otherAgents = None
            self._rolloutAgent = createAgent(self._simulator.action_space, self._simulator.observation_space, parameters['rolloutAgent'])

    def step(self, observation, reward, done):
        if done:
            # in current "Episode" implementation should never reach here
            raise Exception("Should never reach here")

        root = RootNode(state=observation, reward=0., done=done, simulator=self._simulator,
                agentId=self.agentId, parameters=self._parameters['treeParameters'],
                treeAgent=self._treeAgent, otherAgents=self._otherAgents, rolloutAgent=self._rolloutAgent)

        if self._limitType == 'time':
            timeLimit = time.time() + self._parameters['timeLimit']
            while time.time() < timeLimit:
                root.executeRound()
        else:
            for i in range(self._parameters['iterationLimit']):
                root.executeRound()
        
        action = root.getBestChild().getAction()
        logging.info("Action " + str(action))

        return action

