from aiagents.single.AtomicAgent import AtomicAgent
from aienvs.Environment import Env
import logging
import random


class RandomAgent(AtomicAgent):
    """
    A simple agent component represents a single agent
    This implementation selects random actions
    """

    def step(self, state, reward=None, done=None):
        """
        Selects just a single random action, wraps in a single element agentId:actionId dictionary
        """
        actions = dict()
        action = random.randint(0, self._environment.action_space.spaces.get(self._agentId).n-1)
        actions.update({self._agentId: action })

        logging.debug("Id / action:" + str(actions))

        return actions

