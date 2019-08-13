from aiagents.single.TrainedAgent.Preprocessor import FactoryFloorPreprocessor
from aiagents.single.TrainedAgent.Predictor import Predictor

class TrainedAgent(AtomicAgent):
    DEFAULT_PARAMETERS = {}

    def __init__(self, agentId, environment: Env, parameters: dict): 
        """
        TBA
        """
        self._parameters = copy.deepcopy(self.DEFAULT_PARAMETERS)
        self._parameters.update(parameters)
        self.agentId = agentId

        self._preprocessor = FactoryFloorPreprocessor(agentId)
        self._predictor = Predictor(self._parameters['predictor']) 

    def train(self, data):
        processed_data = self._preprocessor.format(data)
        self._predictor.train(processed_data)
       
    def step(self, observation, reward, done):
        processed_observation = self._preprocessor.format(observation)
        action = self._predictor.forward(processed_observation)
        return action

