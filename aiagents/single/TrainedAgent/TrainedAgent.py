from aiagents.single.AtomicAgent import AtomicAgent
from keras.models import load_model
# TODO: make it extensible to other environments
from aienvs.FactoryFloor.FactoryFloorState import FactoryFloorState, encodeStateAsArray, toTuple
from aienvs.FactoryFloor.FactoryFloor import FactoryFloor
from aiagents.AgentFactory import createAgent
import numpy as np
import yaml
import logging

#logger = logging.getLogger()
#logger.setLevel(logging.DEBUG)

class TrainedAgent(AtomicAgent):
    def __init__(self, agentId, environment: FactoryFloor, parameters: dict): 
        """
        TBA
        """
        try:
            self._model = load_model(parameters["modelFile"])
            self._subAgent = None
        except OSError:
            # for first generation we will start with a "backup class"
            backupClassDictionary = {}
            backupClassDictionary["id"]=agentId
            backupClassDictionary["class"]=parameters["backupClass"]
            backupClassDictionary["parameters"]=parameters["backupClassParameters"]
            self._subAgent = createAgent(environment, backupClassDictionary)
            logging.error("Agent " + str(agentId) + " using backup agent")

        super().__init__(agentId, environment, parameters)
       
    def step(self, observation: FactoryFloorState, reward=None, done=None):
        if self._subAgent is not None:
            return self._subAgent.step(observation, reward, done)

        image = encodeStateAsArray(state=observation)
        image_aug = np.expand_dims(image, axis=0)
        # softmax probabilities
        probabilities = self._model.predict(image_aug, batch_size=1)
        action = np.argmax(probabilities)
        logging.debug("OBSERVATION " + str(observation))
        logging.debug("PROBABILITIES " +str (probabilities))
        logging.debug("PREDICTION ACTION " + str(action) + " FOR AGENT " + str(self._agentId))

        return {self._agentId: action}


class TabularAgent(AtomicAgent):
    def __init__(self, agentId, environment: FactoryFloor, parameters: dict): 
        """
        TBA
        """
     #   try:
     #       self._model = parameters["model"] # model is a dictionary states to actions
     #   except KeyError:
        try:
            with open(parameters["modelFile"], 'r') as stream:
                try:
                    self._model = yaml.load(stream)
                except yaml.YAMLError as exc:
                    breakpoint()
                    logging.error(exc)
        except FileNotFoundError:
            print("No agent files available yet")
            self._model={}

        backupClassDictionary = {}
        backupClassDictionary["id"]=agentId
        backupClassDictionary["class"]=parameters["backupClass"]
        backupClassDictionary["parameters"]=parameters["backupClassParameters"]
        self._subAgent = createAgent(environment, backupClassDictionary)

        super().__init__(agentId, environment, parameters)

        self.predicts=0
        self.noPredicts=0
       
    def step(self, observation: FactoryFloorState, reward=None, done=None):
        state = toTuple(encodeStateAsArray(state=observation))
        try: 
            action = self._model[state]
            self.predicts += 1
        except KeyError:
            self.noPredicts += 1
            return self._subAgent.step(observation, reward, done)
        except AttributeError:
            print("ATTR ERROR")
            breakpoint()
        
        return {self._agentId: action}

    def __del__(self):
        try:
            print("Ratio of predicts = " + str(self.predicts / (self.predicts+self.noPredicts) ) )
        except ZeroDivisionError:
            print("Ratio of predicts = 0")


