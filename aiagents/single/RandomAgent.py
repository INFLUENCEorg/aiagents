from aiagents.single.AtomicAgent import AtomicAgent
from aienvs.Environment import Env
from aienvs.gym.DecoratedSpace import DecoratedSpace
import logging
import random
from gym.spaces import Dict


class RandomAgent(AtomicAgent):
    """
    A simple agent component represents a single agent
    This implementation selects random actions
    """

    def __init__(self, agentId:str, actionspace:Dict, observationspace, parameters:dict=None):
        super().__init__(agentId, actionspace, observationspace, parameters)
        full_action_space = DecoratedSpace.create(actionspace)
        self.action_space = full_action_space.get(agentId)

    def step(self, state, reward=None, done=None):
        """
        Selects just a single random action, wraps in a single element agentId:actionId dictionary
        """
        actions = dict()
        action = random.randint(0, self.action_space.getSize() - 1)
        actions.update({self._agentId: action })

        logging.debug("Id / action:" + str(actions))

        return actions
