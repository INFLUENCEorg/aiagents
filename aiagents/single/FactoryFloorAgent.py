from aiagents.single.AtomicAgent import AtomicAgent
from aienvs.FactoryFloor.FactoryFloor import FactoryFloor
from aienvs.FactoryFloor.FactoryFloorState import FactoryFloorState
from aienvs.FactoryFloor.FactoryGraph import FactoryGraph
import logging
from numpy import array, ndarray, fromstring
import networkx
from gym import spaces
import math
import operator

def getPath(source, destination, pathDict):
    return pathDict[str(source)][str(destination)]

def getDistance(path):
    return len(path)-1

def setTargetPosition(positionEvaluationDictionary, socialOrder):
    # sort the positions
    sortedEvaluation = sorted(positionEvaluationDictionary.items(), 
            key=operator.itemgetter(1), reverse=True)
    return sortedEvaluation[socialOrder][0]

def evaluateAllPositions(state, robotpos, pathDict):
    positionEvaluation = {}
    for task in state.tasks:
        taskpos = str(task.getPosition())
        try:
            currentEvaluation = positionEvaluation[taskpos]
        except KeyError:
            currentEvaluation = 0

        path = getPath(robotpos, taskpos, pathDict)
        distance=getDistance(path)
        if(distance==0):
            positionEvaluation.update({taskpos: math.inf})
        else:
            newEvaluation = currentEvaluation + 1./getDistance(path)
            positionEvaluation.update({taskpos: newEvaluation})

    return positionEvaluation

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
        self.pathDict = dict(networkx.all_pairs_dijkstra_path(self._graph))
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
        """
        # we can exit early
        if not state.tasks:
            return {self._agentId: self._ACTIONS.get("ACT")}

        robotpos = self._getCurrentPosition(state)
        socialOrder = self._computeSocialOrder(state, robotpos)

        positionEvaluation = evaluateAllPositions(state, robotpos, self.pathDict)
        # more robots than tasks our robot stays put
        if( len(positionEvaluation) <= socialOrder ):
            return {self._agentId: self._ACTIONS.get("ACT")}

        positionToReach = setTargetPosition(positionEvaluation, socialOrder)
        path = getPath(robotpos, positionToReach, self.pathDict)
        action = self.getAction(path)

        logging.debug(action)
        return action

    def getAction(self, path):
        if getDistance(path) == 0:
            action = {self._agentId: self._ACTIONS.get("ACT")}
        else:
            delta = self._toarray(path[1]) - self._toarray(path[0])
            action = {self._agentId:self._mapping.get(str(delta))}

        return action

    def _getCurrentPosition(self, state):
        return state.robots.get(self._agentId).getPosition()

    def _computeSocialOrder(self, state, robotpos):
        # social law based on agent ids lexicographic order
        socialOrder=0
        for robot in state.robots.values():
            if (robot.getPosition()==robotpos).all() and (robot.getId() < self._agentId):
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
