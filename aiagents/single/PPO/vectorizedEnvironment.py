from aiagents.single.PPO.worker import Worker
import multiprocessing as mp
import numpy as np

class VectorizedEnvironment(object):
    """
    Creates multiple instances of an environment to run in parallel.
    Each of them contains a separate worker (actor) all of them following
    the same policy
    """

    def __init__(self, env, parameters):
        if parameters['num_workers'] < mp.cpu_count():
            self.num_workers = parameters['num_workers']
        else:
            self.num_workers = mp.cpu_count()
        self.workers = [Worker(env, parameters, i) for i in range(self.num_workers)]
        self.parameters = parameters

    def reset(self):
        """
        Resets each of the environment instances
        """
        for worker in self.workers:
            worker.child.send(('reset', None))
        output = {'obs': [], 'prev_action': []}
        for worker in self.workers:
            obs = worker.child.recv()
            stacked_obs = np.zeros((self.parameters['frame_height'],
                                    self.parameters['frame_width'],
                                    self.parameters['num_frames']))
            breakpoint()
            stacked_obs[:, :, 0] = obs[:, :, 0]
            output['obs'].append(stacked_obs)
            output['prev_action'].append(-1)
        return output

    def step(self, actions, prev_stacked_obs):
        """
        Takes an action in each of the enviroment instances
        """
        for worker, action in zip(self.workers, actions):
            worker.child.send(('step', action))
        output = {'obs': [], 'reward': [], 'done': [], 'prev_action': [],
                  'info': []}
        i = 0
        for worker in self.workers:
            obs, reward, done, info = worker.child.recv()
            new_stacked_obs = np.zeros((self.parameters['frame_height'],
                                        self.parameters['frame_width'],
                                        self.parameters['num_frames']))
            new_stacked_obs[:, :, 0] = obs[:, :, 0]
            new_stacked_obs[:, :, 1:] = prev_stacked_obs[i][:, :, :-1]
            output['obs'].append(new_stacked_obs)
            output['reward'].append(reward)
            output['done'].append(done)
            output['info'].append(info)
            i += 1
        output['prev_action'] = actions
        return output

    @property
    def action_space(self):
        """
        Returns the dimensions of the environment's action space
        """
        self.workers[0].child.send(('action_space', None))
        action_space = self.workers[0].child.recv()
        return action_space

    def close(self):
        """
        Closes each of the threads in the multiprocess
        """
        for worker in self.workers:
            worker.child.send(('close', None))
