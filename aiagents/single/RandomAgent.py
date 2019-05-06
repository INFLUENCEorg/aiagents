from aiagents.AgentComponent import AgentComponent
from gym.spaces import Discrete
import logging

class RandomAgent(AgentComponent):
    """
    A simple agent component represents a single agent
    This implementation selects random actions
    """
    def __init__(self, agentId, actionSpace:Discrete):
        self._agentId = agentId
        self._actionSpace = actionSpace

    def observe(self, state):
        """
        We assume that we can observe the whole state
        """
        self._observation=state

    def select_actions(self):
        """
        Selects just a single random action, wraps in a single element agentId:actionId dictionary
        """
        actions = dict()
        actions.update({self._agentId: self._actionSpace.sample()})

        logging.debug("Id / action:" + str(actions))

        return actions

