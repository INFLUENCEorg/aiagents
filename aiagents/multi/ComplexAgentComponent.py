from aiagents.AgentComponent import AgentComponent
import logging

class ComplexAgentComponent(AgentComponent):
    """
    A complex agent component has a list of subcomponents that it needs to iterate through
    This simple class provides a method for brute force iteration of subcomponents
    The subcomponents should be disjoint otherwise actions get overwritten 
    """

    def __init__(self, agentComponentList):
        self._agentSubcomponents=agentComponentList

    def observe(self, state, reward=None, done=None):
        """
        Loops over agent components, all agent components observe the state
        """
        for agentComponent in self._agentSubcomponents:
            agentComponent.observe(state, reward, done)

    def select_actions(self):
        """
        Loops over agent components, all agents components select their action,
        which is aggregated in one dictionary
        """
        actions = dict()
        for agentComponent in self._agentSubcomponents:
            actions.update(agentComponent.select_actions())

        logging.debug("Aggregate actions:" + str(actions))

        return actions

