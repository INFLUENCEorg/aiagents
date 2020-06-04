import logging
import operator
from typing import List
from gym.spaces import Dict

from aienvs.FactoryFloor.FactoryFloorState import FactoryFloorState

from aiagents.multi.ComplexAgent import ComplexAgent
from aiagents.single.FactoryFloorAgent import FactoryFloorAgent, getPath, getDistance, evaluateAllPositions


class FactoryFloorIterativeGreedy(ComplexAgent):
    """
    Iterative Greedy as in Claes'17 paper
    """

    def __init__(self, ffAgentList:List[FactoryFloorAgent], actionspace:Dict, observationspace, parameters=None):
        super().__init__(ffAgentList, actionspace, observationspace, parameters)
        self._ffAgentDict = {}
        for ffAgent in ffAgentList:
            self._ffAgentDict[ffAgent.agentId] = ffAgent

    def step(self, state:FactoryFloorState, reward=None, done=None):
        """
        """
        # create a dictionary evaluating robot-action pairs
        pathDict = self._agentSubcomponents[0].pathDict
        evaluationDict = {}
        for robotId in self._ffAgentDict.keys():
            robotpos = state.robots[robotId].getPosition()
            robotEvaluation = evaluateAllPositions(state, robotpos, pathDict)
            for taskpos in robotEvaluation.keys():
                evaluationDict.update({(robotId, taskpos): robotEvaluation[taskpos]})

        # create a list 
        sortedRobotPosEval = sorted(evaluationDict.items(), key=operator.itemgetter(1), reverse=True)
        
        actions = {}
        # all agents by default assumed to follow no path, unless later specified
        for ffAgentId, ffAgent in self._ffAgentDict.items():
            stayInPlacePath = [str(state.robots[ffAgentId].getPosition())]
            action = ffAgent.getAction(stayInPlacePath)
            actions.update(action)

        while len(sortedRobotPosEval) > 0:
            bestRobot = sortedRobotPosEval[0][0][0]
            correspondingPosition = sortedRobotPosEval[0][0][1]
            robotPath = getPath(state.robots[bestRobot].getPosition(), correspondingPosition, pathDict)
            action = self._ffAgentDict[bestRobot].getAction(robotPath)
            actions.update(action)
            
            newPosEval = [] 
            for item in sortedRobotPosEval:
                if ((item[0][0] != bestRobot) and (item[0][1] != correspondingPosition)):
                    newPosEval.append(item)

            sortedRobotPosEval = newPosEval
        
        logging.debug("Aggregate actions:" + str(actions))
        return actions

