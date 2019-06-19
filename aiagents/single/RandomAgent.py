from aiagents.AgentComponent import AgentComponent
from aiagents.FixedActionsSpace import FixedActionsSpace
import logging


class RandomAgent(AgentComponent):
    """
    A simple agent component represents a single agent
    This implementation selects random actions
    """

    def __init__(self, agentId, actionSpace:FixedActionsSpace):
        self._agentId = agentId
        self._actionSpace = actionSpace

    def step(self, state, reward=None, done=None):
        """
        Selects just a single random action, wraps in a single element agentId:actionId dictionary
        """
        actions = dict()
        actions.update({self._agentId: self._actionSpace.sample()})

        logging.debug("Id / action:" + str(actions))

        return actions

