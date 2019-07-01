import sys
sys.path
from dqnmodel import DeepQNetwork, Agent
from utils import plotLearning
import numpy as np
import gym
import gym_gridworlds
import os


def preprocess(observation):
    return np.mean(observation[30:, :], axis=2).reshape(180, 160, 1)
    # cropping the image, therefore considering only after 30th row of the array.
    # Taking mean along the axis=2 which is the axis representing the colour of
    # of the frames, so we reduce it from RBG to one containing the mean, with only
    # single channel.

def stack_frames(stacked_frames, frame, buffer_size):
    if stacked_frames is None:
        stacked_frames = np.zeros((buffer_size, *frame.shape))
        # here frame.shape has the shape of observation after pre-processing (180, 160, 1).
        for idx, _ in enumerate(stacked_frames):
            stacked_frames[idx, :] = frame
            # stacked_frames[idx, :].shape = (180, 160, 1)
    else:
        stacked_frames[0:buffer_size-1, :] = stacked_frames[1:, :]
        # Removing the first frame in order to store the current new frame
        stacked_frames[buffer_size-1, :] = frame
        # Adding the current frame to the stacked_frames.

    stacked_frames = stacked_frames.reshape(1, *frame.shape[0:2], buffer_size)
    # stacked_frames.shape=(1, 180, 160, 4) after reshaping.
    # frame.shape=(2, 160, 1)

    return stacked_frames


if __name__ == '__main__':

    env = gym.make('Gridworld-v0')
    load_checkpoint = os.path.isfile('tmp/q_eval/deepqnet.ckpt')
    agent = Agent(gamma=0.99, epsilon=1.0, alpha=0.00025, input_dims=(180,160,4),
                  n_actions=4, mem_size=1000, batch_size=32)
    if load_checkpoint:
        agent.load_models()
    filename = 'breakout-alpha0p000025-gamma0p99-only-one-fc.png'
    scores = []
    eps_history = []
    episode_loss = []
    numGames = 5000
    stack_size = 4
    score = 0
    # uncomment the line below to record every episode.
    #env = wrappers.Monitor(env, "tmp/breakout-0",
    #                         video_callable=lambda episode_id: True, force=True)

    print("Loading up the agent's memory with random gameplay")

    while agent.mem_cntr < 1000:
        done = False
        observation = env.reset()
        # observation.shape = (210, 160, 3)
        observation = preprocess(observation)
        # now the shape is (180, 160, 1)
        stacked_frames = None
        observation = stack_frames(stacked_frames, observation, stack_size)
        # The shape changes to (1, 180, 160, 4)
        while not done:
            action = np.random.choice([0, 1, 2])
            # Randomly choose a single element from the array above and
            # assign it to the variable action.
            action += 1
            # we add one to the action and pass it to the env.step()
            # so if 0th action is chosen, adding one makes it one and then passing
            # 1 into the env.step(1) selects first action, which is actually the 0th action.

            observation_, reward, done, info = env.step(action)
            # observation_.shape = (210, 160, 3)
            # info is {'ale.lives':5}

            observation_ = stack_frames(stacked_frames,
                                        preprocess(observation_), stack_size)
            # observation_.shape =(1, 180, 160,4)
            action -= 1
            # Then we subtract 1 back since we added it to the action.
            agent.store_transition(observation, action,
                                   reward, observation_, int(done))
            # Then this is passed to the store.transition() function which first creates an
            # array of zeros(called actions) like ([0, 0, 0]) with len equal to the number
            # of actions.  This action is passed as an index to this array of zeros and
            # that particular element is replaced by one. So if 0th action is selected,
            # the ac
            # [1, 0, 0]). This is hot encoding the action

            observation = observation_
            # observation.shape = (1, 180, 160, 4)
    print("Done with random game play. Game on.")

    for i in range(numGames):
       # if(i==1):
      #      pdb.set_trace()
        done = False
        if i % 2 == 0 and i > 0:
            #avg_score = np.mean(scores[max(0, i-10):(i+1)])
            print('episode: ', i, 'score: ', score,
                  #' average score %.3f' % avg_score,
                  'epsilon %.3f' % agent.epsilon)

        else:
            print('episode: ', i, 'score: ', score)

        observation = env.reset()
        observation = preprocess(observation)
        stacked_frames = None
        # we change it to none inside the loop so that we can have current 4 stacked of frames.

        observation = stack_frames(stacked_frames, observation, stack_size)

        score = 0

        while not done:
            action = agent.choose_action(observation)
            action += 1
            observation_, reward, done, info = env.step(action)
            observation_ = stack_frames(stacked_frames,
                                        preprocess(observation_), stack_size)
            score += reward
            action -= 1
            agent.store_transition(observation, action,
                                   reward, observation_, int(done))
            observation = observation_
            agent.learn()

        eps_history.append(agent.epsilon)
        scores.append(score)
    x = [i+1 for i in range(numGames)]
    plotLearning(x, scores, eps_history, filename)
