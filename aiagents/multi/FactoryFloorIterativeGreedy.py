from aiagents.multi.ComplexAgent import ComplexAgent
from aiagents.single.FactoryFloorAgent import getPath, getDistance, evaluateAllPositions
from aienvs.FactoryFloor.FactoryFloorState import FactoryFloorState
import logging
import operator

class FactoryFloorIterativeGreedy(ComplexAgent):
    """
    Iterative Greedy as in Claes'17 paper
    """
    def __init__(self, ffAgentList, parameters=None):
        self.pathDict=ffAgentList[0].pathDict
        self._ffAgentDict = {}
        for ffAgent in ffAgentList:
            self._ffAgentDict[ffAgent.agentId]=ffAgent

    def step(self, state:FactoryFloorState, reward=None, done=None):
        """
        """
        #create a dictionary evaluating robot-action pairs
        evaluationDict = {}
        for robotId in self._ffAgentDict.keys():
            robotpos = state.robots[robotId].getPosition()
            robotEvaluation = evaluateAllPositions(state,robotpos,self.pathDict)
            for taskpos in robotEvaluation.keys():
                evaluationDict.update({(robotId,taskpos): robotEvaluation[taskpos]})

        #create a list 
        sortedRobotPosEval = sorted(evaluationDict.items(), key=operator.itemgetter(1), reverse=True)
        
        actions={}
        # all agents by default assumed to follow no path, unless later specified
        for ffAgentId, ffAgent in self._ffAgentDict.items():
            stayInPlacePath=[str(state.robots[ffAgentId].getPosition())]
            action=ffAgent.getAction(stayInPlacePath)
            actions.update(action)

        while len(sortedRobotPosEval) > 0:
            bestRobot=sortedRobotPosEval[0][0][0]
            correspondingPosition=sortedRobotPosEval[0][0][1]
            robotPath=getPath(state.robots[bestRobot].getPosition(), correspondingPosition, self.pathDict)
            action=self._ffAgentDict[bestRobot].getAction(robotPath)
            actions.update(action)
            
            newPosEval = [] 
            for item in sortedRobotPosEval:
                if ((item[0][0] != bestRobot) and (item[0][1] != correspondingPosition)):
                    newPosEval.append(item)

            sortedRobotPosEval = newPosEval
        
        logging.debug("Aggregate actions:" + str(actions))
        return actions

