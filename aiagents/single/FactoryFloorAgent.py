from aiagents.single.AtomicAgent import AtomicAgent
from aienvs.FactoryFloor.FactoryFloor import FactoryFloor
from aienvs.FactoryFloor.FactoryFloorState import FactoryFloorState
from aienvs.FactoryFloor.FactoryGraph import FactoryGraph
import logging
from numpy import array, ndarray, fromstring
from gym.spaces import Dict
import networkx
import math
import operator


class FactoryFloorAgent(AtomicAgent):
    """
    A Factory Floor Agent that tries to go to the nearest task
    """

    ENVACTIONS = {
        0: "ACT",
        1: "UP",
        2: "DOWN",
        3: "LEFT",
        4: "RIGHT"
    }

    def __init__(self, agentId, actionspace:Dict, observationspace, parameters):
        """
        invert the actions
        """
        super().__init__(agentId, actionspace, observationspace, parameters)
        # inverting the key action pairs for meaningful navigation
        self._ACTIONS = dict(zip(FactoryFloorAgent.ENVACTIONS.values(), FactoryFloorAgent.ENVACTIONS.keys()))
        self._graph = None
        
        self._mapping = { (0, -1):self._ACTIONS.get("UP"),
                          (0, 1):self._ACTIONS.get("DOWN"),
                          (-1, 0):self._ACTIONS.get("LEFT"),
                          (1, 0):self._ACTIONS.get("RIGHT")
                         }
    
    def getPathDict(self, state:FactoryFloorState):
        """
        @return pathdict with all connected paths on the map
        """
        if self._graph == None:
            self._graph = FactoryGraph(state.getMap())
            self._pathDict = dict(networkx.all_pairs_dijkstra_path(self._graph))
        return self._pathDict
    
    def step(self, state: FactoryFloorState, reward=None, done=None) -> Dict:
        # we can exit early
        if not state.tasks:
            return {self._agentId: self._ACTIONS.get("ACT")}
        self.getPathDict(state)

        robotpos = self._getCurrentPosition(state)
        socialOrder = self._computeSocialOrder(state, robotpos)

        positionEvaluation = evaluateAllPositions(state, robotpos, self._pathDict)
        # more robots than tasks our robot stays put
        if(len(positionEvaluation) <= socialOrder):
            return {self._agentId: self._ACTIONS.get("ACT")}

        positionToReach = setTargetPosition(positionEvaluation, socialOrder)
        path = getPath(robotpos, positionToReach, self._pathDict)
        action = self._getAction(path)

        logging.debug(action)
        return action

    def _getAction(self, path):
        if getDistance(path) == 0:
            action = {self._agentId: self._ACTIONS.get("ACT")}
        else:
            delta = tuple(array(path[1]) - array(path[0]))
            action = {self._agentId:self._mapping.get(delta)}

        return action

    def _getCurrentPosition(self, state):
        return state.robots.get(self._agentId).getPosition()

    def _computeSocialOrder(self, state, robotpos):
        # social law based on agent ids lexicographic order
        socialOrder = 0
        for robot in state.robots.values():
            if (robot.getPosition() == robotpos).all() and (robot.getId() < self._agentId):
                socialOrder += 1
        
        return socialOrder

    def _toarray(self, alist:str):
        """
        @param alist: string of form [2 3 5]: 
        ints separated by whitespaces and possibly 
        enclosed in square brackets.
        @return: numpy array with 
        """
        return fromstring(alist.replace('[', '').replace(']', ''), dtype=int, sep=' ')


def getPath(source, destination, pathDict):
    return pathDict[tuple(source)][tuple(destination)]


def getDistance(path):
    return len(path) - 1


def setTargetPosition(positionEvaluationDictionary, socialOrder):
    # sort the positions
    sortedEvaluation = sorted(positionEvaluationDictionary.items(),
            key=operator.itemgetter(1), reverse=True)
    return sortedEvaluation[socialOrder][0]


def evaluateAllPositions(state, robotpos, pathDict):
    positionEvaluation = {}
    for task in state.tasks:
        taskpos = tuple(task.getPosition())
        try:
            currentEvaluation = positionEvaluation[taskpos]
        except KeyError:
            currentEvaluation = 0

        path = getPath(robotpos, taskpos, pathDict)
        distance = getDistance(path)
        if(distance == 0):
            positionEvaluation.update({taskpos: math.inf})
        else:
            newEvaluation = currentEvaluation + 1. / getDistance(path)
            positionEvaluation.update({taskpos: newEvaluation})

    return positionEvaluation
