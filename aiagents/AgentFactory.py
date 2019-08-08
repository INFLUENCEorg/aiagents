from aienvs.Environment import Env
from aiagents import AgentComponent


def createAgent(fullname:str, agentId:str, environment:Env, parameters:dict) -> AgentComponent:
    '''
    Create an agent from a given full path name
    @param fullname the full.path.name to the agent to create, eg
    "aiagents.single.RandomAgent.RandomAgent"
    @param agentId see AgentComponent __init__
    @param environment see AgentComponent __init__
    @param parameters see AgentComponent __init__
    @return an initialized AgentComponent
    '''
    klass = classForName(fullname)
    # FIXME check klass instantiates AgentComponent, like this
    #         if not isinstance(klass, AgentComponent):
    #             raise Exception("Class " + fullname + " does not extend AgentComponent")

    # For unknown reasons, we need to add the 'self' to this call?
    obj = klass(agentId, environment, parameters)
    return obj


def classForName(kls:str):
    """
    @param kls the string full path to the class to load. 
    Eg "aiagents.single.RandomAgent.RandomAgent".
    The class to load has to be on the classpath.
    @return a class object. You can make instances of this class object 
    by calling it with the constructor arguments.
    """
    parts = kls.split('.')
    module = ".".join(parts[:-1])
    m = __import__(module)
    for comp in parts[1:]:
        m = getattr(m, comp)            
    return m

