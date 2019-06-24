from aiagents.single.AtomicAgent import AtomicAgent
from aienvs.Environment import Env
import logging


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
        actions.update({self._agentId: self._environment.action_space.spaces.get(self._agentId).sample()})

        logging.debug("Id / action:" + str(actions))

        return actions

