from aiagents.single.AtomicAgent import AtomicAgent
from keras.models import load_model
# TODO: make it extensible to other environments
from aienvs.FactoryFloor.FactoryFloorState import FactoryFloorState, encodeStateAsArray
import numpy as np

class TrainedAgent(AtomicAgent):
    def __init__(self, agentId, environment: FactoryFloor, parameters: dict): 
        """
        TBA
        """
        super.__init__(agentId, environment, parameters)
        self._model = load_model(parameters["modelFile"])
       
    def step(self, observation: FactoryFloorState, reward=None, done=None):
        image = encodeStateAsArray(observation)
        image_aug = np.expand_dims(image, axis=0)
        # softmax probabilities
        probabilities = self._model.predict(image_aug, batch_size=1)
        action = np.argmax(probabilities)

        return action

