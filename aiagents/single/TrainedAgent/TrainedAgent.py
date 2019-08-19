from aiagents.single.AtomicAgent import AtomicAgent
from keras.models import load_model
# TODO: make it extensible to other environments
from aienvs.FactoryFloor.FactoryFloorState import FactoryFloorState, encodeStateAsArray
from aienvs.FactoryFloor.FactoryFloor import FactoryFloor
import numpy as np
import logging

#logger = logging.getLogger()
#logger.setLevel(logging.DEBUG)

class TrainedAgent(AtomicAgent):
    def __init__(self, agentId, environment: FactoryFloor, parameters: dict): 
        """
        TBA
        """
        super().__init__(agentId, environment, parameters)
        self._model = load_model(parameters["modelFile"])
        self._observationWidth=environment.observation_space.nvec[1]
        self._observationHeight=environment.observation_space.nvec[2]
       
    def step(self, observation: FactoryFloorState, reward=None, done=None):
        image = encodeStateAsArray(state=observation, width=self._observationWidth, height=self._observationHeight, currentRobotId=self._agentId)
        image_aug = np.expand_dims(image, axis=0)
        # softmax probabilities
        probabilities = self._model.predict(image_aug, batch_size=1)
        action = np.argmax(probabilities)
        logging.debug("OBSERVATION " + str(observation))
        logging.debug("PROBABILITIES " +str (probabilities))
        logging.debug("PREDICTION ACTION " + str(action) + " FOR AGENT " + str(self._agentId))

        return {self._agentId: action}

