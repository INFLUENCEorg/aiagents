from aiagents.single.PPO.controller import Controller
from aiagents.single.PPO.PPOmodel import PPOmodel
from aiagents.single.PPO.replay_memory import SerialSampling
import numpy as np
from aiagents.AgentComponent import AgentComponent


class PPOAgent(Controller, AgentComponent):

    def observe(self, observation, reward, done):
        new_state = np.zeros((1, self.parameters['frame_height'],
                              self.parameters['frame_width'],
                              self.parameters['num_frames']))
        frame = observation
        prev_state = np.copy(new_state)
        new_state[0, :, :, 0] = frame
        new_state[0, :, :, 1:] = prev_state[:, :, :, :-1]

        next_step_output = dict({"obs": new_state, "reward": reward, "done": done, "prev_action": self._prev_action})
        if( self._step_output["obs"] is None ):
            self._step_output = next_step_output

        self.increment_step()
        get_actions_output = self.get_actions(self._step_output)

        if self.parameters['mode'] == 'train':
            # Store the transition in the replay memory.
            self.add_to_memory(self._step_output,
                                next_step_output,
                                get_actions_output, self.t)
            self.bootstrap(next_step_output)
            if self.step % self.train_frequency == 0 and self.full_memory():
                self.update()
            if self.step % self.save_frequency == 0:
                # Tensorflow only stores a limited number of networks.
                self.save_graph(step)
            self.write_summary()

        self._step_output = next_step_output

    def select_actions(self):
        action=self.get_actions(self._step_output)
        self._prev_action=action.get('action')[0]
        return {self._controller_id: action['action'][0][0]}

    def __init__(self, parameters, observation_space, action_map, controller_id=0, logger=None):
        self.num_actions = {}
        #TODO: change to self._step_output = dict({"obs": observation_space.sample(), "reward": None, "done": None, "prev_action": None})
        self._prev_state = None
        self._observation_space = observation_space
        self._step_output = dict({"obs": None, "reward": None, "done": None, "prev_action": None})
        self._prev_action = [-1]
        self.parameters = parameters
        self.num_actions = action_map.n
        self.train_frequency = self.parameters['train_frequency']
        self.save_frequency = self.parameters['save_frequency']
        self._controller_id = controller_id
        self.model = {}
        self.replay_memory = {}
        self.cumulative_rewards = {}
        self.stats = {}
        self.model = PPOmodel(self.parameters, self.num_actions)
        self.replay_memory = SerialSampling(self.parameters, self.num_actions)
        self.cumulative_rewards = 0
        self.stats = {"cumulative_rewards": [],
                        "episode_length": [],
                        "value": [],
                        "learning_rate": [],
                        "entropy": [],
                        "policy_loss": [],
                        "value_loss": []}
        super().__init__(self.parameters, action_map)
        self.t = 0
        if self.parameters['influence']:
            self.seq_len = self.parameters['inf_seq_len']
        elif self.parameters['recurrent']:
            self.seq_len = self.parameters['seq_len']
        else:
            self.seq_len = 1

    def add_to_memory(self, step_output, next_step_output, get_actions_output,
                      episode_step):
        """
        Append the last transition to the replay memories of the factors.
        """
        # The given actions come from the experimentor and are actions,
        # but we need action indices.
        # TODO: this does not work as expected with factors. We would like to
        # append the values of each factor to the corresponding key in the dictionary
        self.replay_memory['obs'].append(step_output['obs'])
        self.replay_memory['rewards'].append(next_step_output['reward'])
        self.replay_memory['dones'].append(next_step_output['done'])
        self.replay_memory['actions'].append(get_actions_output['action'])
        self.replay_memory['values'].append(get_actions_output['value'][0])
        self.replay_memory['action_probs'].append(get_actions_output['action_probs'])
        # This mask is added so we can ignore experiences added when zero
        # padding incomplete sequences
        self.replay_memory['masks'].append(1)
        self.cumulative_rewards += next_step_output['reward']
        self.stats['value'].append(get_actions_output['value'][0])
        self.stats['entropy'].append(get_actions_output['entropy'])
        self.stats['learning_rate'].append(get_actions_output['learning_rate'])
        # Note: States out is used when updating the network to feed the
        # initial state of a sequence. In PPO this internal state will not
        # differ that much from the current one. However for DQN we might
        # rather set the initial state as zeros like in Jinke's
        # implementation
        if self.parameters['recurrent']:
            self.replay_memory['states_in'].append(get_actions_output['state_in'])
            self.replay_memory['prev_actions'].append(step_output['prev_action'])
        if self.parameters['influence']:
            self.replay_memory['inf_states_in'].append(get_actions_output['inf_state_in'])
            self.replay_memory['inf_prev_actions'].append(step_output['prev_action'])

        if next_step_output['done']:
            if self.parameters['recurrent'] or self.parameters['influence']:
                self.model.reset_state_in()
            self.stats['cumulative_rewards'].append(self.cumulative_rewards)
            self.stats['episode_length'].append(episode_step)
            self.cumulative_rewards = 0
            # zero padding incomplete sequences
            remainder = len(self.replay_memory['masks']) % self.seq_len
            if remainder != 0:
                missing = self.seq_len - remainder
                self.replay_memory.zero_padding(missing)
                self.t += missing

    def bootstrap(self, next_step_output):
        """
        """
        # TODO: consider the case where the episode is over because the maximum
        # number of steps in an episode has been reached.
        self.t += 1
        if self.t >= self.parameters['time_horizon']:
            evaluate_value_output = self.model.evaluate_value(
                                        next_step_output['obs'],
                                        next_step_output['prev_action'])
            next_value = evaluate_value_output['value']
            batch = self.replay_memory.get_last_entries(self.t, ['rewards', 'values', 'dones'])
            advantages = self.compute_advantages(np.reshape(batch['rewards'],
                                                            (1, -1)),
                                                 np.reshape(batch['values'],
                                                            (1, -1)),
                                                 np.reshape(batch['dones'],
                                                            (1, -1)),
                                                 next_value,
                                                 self.parameters['gamma'],
                                                 self.parameters['lambda'])
            self.replay_memory['advantages'].extend(advantages)
            returns = advantages + batch['values']
            self.replay_memory['returns'].extend(returns)
            self.t = 0

    def update(self):
        """
        Sample a batch from the replay memory (if it is completely filled) and
        use it to update the models.
        """
        import time
        start = time.time()
        print(len(self.replay_memory[0]['returns']))
        print(len(self.replay_memory[0]['masks']))
        policy_loss = 0
        value_loss = 0
        n_sequences = self.parameters['batch_size'] // self.seq_len
        n_batches = self.parameters['memory_size'] // self.parameters['batch_size']
        for e in range(self.parameters['num_epoch']):
            self.replay_memory.shuffle()
            for b in range(n_batches):
                batch = self.replay_memory.sample(b, n_sequences)
                update_model_output = self.model.update_model(batch)
                policy_loss += update_model_output['policy_loss']
                value_loss += update_model_output['value_loss']
        self.replay_memory.empty()
        self.stats['policy_loss'].append(np.mean(policy_loss))
        self.stats['value_loss'].append(np.mean(value_loss))
        end = time.time()
        print(end - start)

    def compute_advantages(self, rewards, values, dones, last_value, gamma, lambd):
        """
        """
        last_advantage = 0
        advantages = np.zeros(self.parameters['time_horizon'], dtype=np.float32)
        for t in reversed(range(self.parameters['time_horizon'])):
            mask = 1.0 - dones[:, t]
            last_value = last_value*mask
            last_advantage = last_advantage*mask
            delta = rewards[:, t] + gamma*last_value - values[:, t]
            last_advantage = delta + gamma*lambd*last_advantage
            advantages[t] = last_advantage
            last_value = values[:, t]
        return advantages
