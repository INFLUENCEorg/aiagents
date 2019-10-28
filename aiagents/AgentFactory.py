from aienvs.Environment import Env
from aiagents.AgentComponent import AgentComponent
import logging
import glob
import os


def createAgent(environment:Env, parameters:dict) -> AgentComponent:
    '''
    Create an agent from a given full path name
    @param environment see AgentComponent __init__
    @param parameters this parameter is a dictionary. It must contain these keys
    * 'class': the class name (str) of the agent to be loaded.The class will be resolved by resolve from the aiagents directory.
    * 'parameters': the subparameters (dict) to be passed into the agent.
    * 'either 'id' or 'subAgentList' which fulfills two purposes:
     1. It is used to guess the type of full.class.name that will be created,
     so that the correct constructor can be called.
     2. It can be used as parameter to recursively call createAgent
     It works as follows
     * If 'id':'name' is a key-value(str), it is assumed to be an AtomicAgent
        and the agent will get 'name' as id.
     * if 'subAgentList':[subparam1,...] is a key-value, it is assumed 
       a ComplexAgent. createAgent will be (recursively) called 
       for each subparam to create a list of subAgents.
       These subAgents are then passed into the ComplexAgent constructor.
    @return an initialized AgentComponent
    '''
    logging.debug(parameters)
    if not 'class' in parameters:
        raise ValueError("Parameters must have key 'class' but got " + str(parameters))
    if (not 'parameters' in parameters) or not isinstance(parameters['parameters'], dict) :
        raise ValueError("Parameters must have key 'parameters' containing a dict but got " + str(parameters))
    if not ('id' in parameters or 'subAgentList' in parameters):
        raise ValueError("Parameters must contain either key 'id' or key 'subAgentList but got " + str(parameters))
    if ('subAgentList' in parameters and not isinstance(parameters['subAgentList'], list)):
        raise ValueError("the subAgentList parameter must contain a list, but got " + str(parameters))
    classname = parameters['class']
    # classname = resolve(parameters['class'], 'aiagents')
    class_parameters = parameters['parameters']
    try:
        klass = classForNameTyped(classname, AgentComponent)
    except Exception as error:
        raise ValueError("Can't load " + str(classname) + " from " + str(parameters)) from error
    if 'id' in parameters:
        obj = klass(parameters['id'], environment, class_parameters)
    elif 'subAgentList' in parameters:
        subAgentList = []
        for subAgentParameters in parameters['subAgentList']:
            subAgentList.append(createAgent(environment, subAgentParameters))
            try:
                obj = klass(subAgentList, class_parameters)
            except Exception as error:
                raise ValueError(classname + " failed on __init__:") from error
    else:
        raise Exception("parameters " + str(parameters) + " does not contain key 'id' or 'subAgentList'")

    return obj


def resolve(shortname:str, basepath: str):
    """
    @shortname the name of the class needed eg 'RandomParty'
    @basepath the directory where to start the search, eg 'aiagents'
    @returns a full class name 'aiagents.single.RandomAgent.RandomAgent'.
    The procedure is to search recursively through all subdirs of the given basepath
    until a file with given shortname is found. It is assumed that this
    file contains the requested class as well
    """
    items = glob.glob(basepath + "/**/" + shortname + ".py", recursive=True)
    if len(items) == 0:
        raise Exception("No candidate find to resolve " + shortname)
    if len(items) != 1:
        raise Exception("Resolve failed of " + shortname + ": multiple candidates " + str(items))
    return items[0][:-3].replace('/', '.') + "." + shortname


def classForNameTyped(klsname:str, expectedkls):
    """
    @param klsname the string full path to the class to load. 
    Eg "aiagents.single.RandomAgent.RandomAgent".
    The class to load has to be on the classpath.
    @param expectedkls the expected class, eg AgentComponent
    @return a class object that is subclass of expectedkls. You can make instances of this class object 
    by calling it with the constructor arguments.
    @raise exception if klsname does not contain expected class or subclass of it. 
    """
    klass = classForName(klsname)
    if not issubclass(klass, expectedkls):
        raise Exception("Class " + klsname + " does not extend " + str(expectedkls))
    return klass


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

