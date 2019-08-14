
from aiagents.single.AtomicAgent import AtomicAgent
import copy
import logging
from aienvs.Environment import Env
from aiagents.AgentFactory import createAgent
from aienvs.utils import getParameters


class ComplexAgentComponent(AtomicAgent):
    """
    A complex agent component has a list of subagents that it needs to iterate through
    This simple class provides a method for brute force iteration of subcomponents
    The subcomponents should be disjoint otherwise actions get overwritten 
    """

    def __init__(self, agentId:str, environment:Env, parameters:dict=None):
        """
        parameters contains a "agents" key. Its value is a dictionary A.
        The keys of dict A are agent IDs, the value is another dict D.
        The dict D contains 
        * 'parameters' containing a dict of additional parameters
        that will be added to our parameters
        * 'classname' containing the fully specified class name for the
        agent, so that we can call AgentFactory.createAgent
        * the sub agentID will be a concat of our own and the new agent ID.
        """
        super().__init__(agentId, environment, parameters)
        self._subAgents = []

        if not ('agents' in parameters and isinstance(parameters['agents'], dict)) :
                raise Exception("key 'agents' containing dict is required in parameters " + str(parameters))

        subsettings = parameters['agents']
        for aid in subsettings:
            if not (isinstance(aid, str)) :
                raise Exception("agent id" + str(aid) + " in agents dict must be str in parameters " + str(parameters))
            agentsettings = subsettings[aid]
            if not (isinstance(agentsettings, dict)) :
                raise Exception("agent " + aid + " must have dict value: " + str(parameters))

            if not ('classname' in agentsettings and isinstance(agentsettings['classname'], str)) :
                raise Exception("key 'classname' containing string is required in parameters " + str(parameters))
            if not ('parameters' in agentsettings and isinstance(agentsettings['parameters'], dict)) :
                raise Exception("key 'parameters' containing dict is required in parameters " + str(parameters))
            try:
                self._subAgents.append(self._newAgent(aid, agentsettings['classname'], agentsettings['parameters']))
            except Exception as e:
                raise Exception("Failed to create sub-agent " + aid + " using " + str(parameters)) from e

    def step(self, state, reward=None, done=None):
        """
        Loops over agent components, all agent components step and actions are aggregated
        """
        actions = dict()

        for agentComponent in self._subAgents:
            agentActions = agentComponent.step(state, reward, done)
            # FIXME simplistic merge? At least warn if keys conflict?
            actions.update(agentActions)
 
        logging.debug("Aggregate actions:" + str(actions))
        return actions

    def getSubAgents(self) -> list:
        return self._subAgents

    def _newAgent(self, newid:str, classname: str, extraparams:dict):
        """
        @param newid the id for the agent
        @param classname: the full class name for the agent
        @param extraparams: extra params to be merged with (our params without the 'agents' key).
        
        """
        newid = self.getAgentId() + "_" + newid
        newparams = copy.deepcopy(self.getParameters())
        del newparams['agents'] 
        newparams.update(extraparams)
        return createAgent(classname, newid, self.getEnvironment(), newparams)
