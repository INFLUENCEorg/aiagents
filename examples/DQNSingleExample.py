# to suppress all the annoying warnings from tensorflow
import warnings
warnings.filterwarnings("ignore")
import tensorflow as tf
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
from aienvs.Sumo.GridSumoEnv import GridSumoEnv
from aiagents.single.DQN.DQNAgent import DQNAgent

# build up the sumo rl environment
shape = (1,1)
env = GridSumoEnv({
    'shape': shape,
    'lane_length': 100,
    'car_tm': 100,
    'car_pr': 0.2,
    'resolutionInPixelsPerMeterX': 0.25,
    'resolutionInPixelsPerMeterY': 0.25,
    'gui': False
})

# environment information
print("information about the environment:")
agent_id, action_space = list(env.action_space.spaces.items())[0]
print("agent id:", agent_id)
print("action space:", action_space)
print("observation space:", env.observation_space)

env.reset()
s, r, done, _ = env.step({agent_id: 0})
print("\nan observation that the agent would get:")

# build up a DQN agent
agent = DQNAgent(
    agentId=agent_id,
    actionspace=action_space,
    observationspace=env.observation_space,
    parameters={
        'num_frames': 4, # the number of current and previous frames we stack as the input to the network
        'gamma': 0.99, # discount factor
        'learning_rate': 2.5e-4,
        'batch_size': 32,
        'train_frequency': 1,
        'epsilon': 0.1, # for epsilon-greedy policy during training
        'double_dqn': True,
        'encoder_type': "large", # the size of the encoder used by the DQN network
        'memory_size': 5000,
        'freeze_interval': 5000
    }
)

# Train the DQN agent in the environment

# turn on the training mode
agent.train()
# train for 1000 episodes
num_episodes = 500
training_returns = []
for i_episode in range(num_episodes):
    obs = env.reset()
    done = False
    reward = None
    Return = 0.0
    while True:
        action = agent.step(observation=obs, reward=reward, done=done)
        obs, reward, done, _ = env.step(action)
        Return += reward
        if done:
            # send the final transition to the agent
            agent.step(observation=obs, reward=reward, done=done)
            break
    print("Episode {} return: {}".format(i_episode, Return))
    training_returns.append(Return)
