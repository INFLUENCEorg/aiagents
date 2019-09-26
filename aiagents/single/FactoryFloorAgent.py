from aiagents.single.AtomicAgent import AtomicAgent
from aienvs.FactoryFloor.FactoryFloor import FactoryFloor
from aienvs.FactoryFloor.FactoryFloorState import FactoryFloorState
from aienvs.FactoryFloor.FactoryGraph import FactoryGraph
import logging
from numpy import array, ndarray, fromstring
import networkx 
from gym import spaces


class FactoryFloorAgent(AtomicAgent):
    """
    A Factory Floor Agent that tries to go to the nearest task
    """

    def __init__(self, agentId, environment: FactoryFloor, parameters):
        """
        invert the actions
        """
        super().__init__(agentId, environment, parameters)
        # inverting the key action pairs for meaningful navigation
        self._ACTIONS = dict(zip(environment.ACTIONS.values(), environment.ACTIONS.keys()))
        self._graph = FactoryGraph(environment.getMap())
        self._mapping = { "[0 -1]":self._ACTIONS.get("UP"),
                         '[ 0 -1]':self._ACTIONS.get("UP"),
                         "[0 1]":self._ACTIONS.get("DOWN"),
                         '[ 0 1]':self._ACTIONS.get("DOWN"),
                         "[-1 0]":self._ACTIONS.get("LEFT"),
                         '[-1  0]':self._ACTIONS.get("LEFT"),
                         "[1 0]":self._ACTIONS.get("RIGHT"),
                         '[1  0]':self._ACTIONS.get("RIGHT")
                         }

    def step(self, state: FactoryFloorState, reward=None, done=None) -> spaces.Dict:
        """
        Selects just a single random action, wraps in a single element agentId:actionId dictionary
        """

        # we need a robot dictionary in the state
        for robot in state.robots:
            if robot._id == self._agentId:
                robotpos = robot.getPosition()
                break

        if not state.tasks:
            return {self._agentId: self._ACTIONS.get("ACT")}

        bestDistance = 100
        for task in state.tasks:
            taskpos = task.getPosition()
            distance = sum(abs(robotpos - taskpos))
            if(distance < bestDistance):
                targetTask = task
                bestDistance = distance
        
        if (targetTask.getPosition() == robot.getPosition()).all():
            action = {self._agentId: self._ACTIONS.get("ACT")}
        else:
            path = networkx.shortest_path(self._graph, source=str(robotpos), target=str(targetTask.getPosition()))
            delta = self._toarray(path[1]) - self._toarray(path[0])
            action = {self._agentId:self._mapping.get(str(delta))}
        
        logging.debug(action)
        return action
    
    def _toarray(self, alist:str):
        """
        @param alist: string of form [2 3 5]: 
        ints separated by whitespaces and possibly 
        enclosed in square brackets.
        @return: numpy array with 
        """
        return fromstring(alist.replace('[', '').replace(']', ''), dtype=int, sep=' ')
