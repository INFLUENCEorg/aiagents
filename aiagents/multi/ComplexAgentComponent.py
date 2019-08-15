from aiagents.multi.ComplexAgent import ComplexAgent
import logging

class BasicComplexAgent(ComplexAgent):
    """
    A basic complex agent has a list of subcomponents that it needs to iterate through
    This simple class provides a method for brute force iteration of subcomponents
    The subcomponents should be disjoint otherwise actions get overwritten 
    """
    def step(self, state, reward=None, done=None):
        """
        Loops over agent components, all agent components step and actions are aggregated
        """
        actions = dict()

        for agentComponent in self._agentSubcomponents:
            agentActions = agentComponent.step(state, reward, done)
            actions.update(agentActions)
 
        logging.debug("Aggregate actions:" + str(actions))
        return actions

