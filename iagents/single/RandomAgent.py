from sumoai.AgentComponent import AgentComponent
import random

class RandomAgent(AgentComponent):
    """
    A simple agent component represents a single agent
    This implementation selects random actions
    """
    def __init__(self, agentId, actionScope, verbose=False):
        self._agentId=agentId
        self._actionScope=actionScope
        self._verbose=verbose

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
        actions.update({self._agentId: random.choice(self._actionScope)})

        if( self._verbose ):
            print("Id / action:" + str(actions))

        return actions

