import os
import tensorflow as tf
import numpy as np

class Controller(object):
    """
    Controller object that interacts with all the agents in the system,
    runs the coordination algorithms (in the multi-agent case), gets
    agent's actions, passes the environment signals to the agents.
    """

    def __init__(self, parameters:dict, action_map, logger=None):
        """
        @param parameters a dictionary with all kinds of options for the run.
        @param action_map  a dict with 
        and a list of allowed actions as values 
        @param logger 
        """
        # Initialize all factor objects here
        self.parameters = parameters
        self.logger = logger

        self.num_actions = {}
        self.step = {}
        tf.reset_default_graph()
        self.num_actions = action_map.n
        self.step = 0
        summary_path = 'summaries/' + self.parameters['name'] + '_' + \
                        self.parameters['algorithm']
        if not os.path.exists(summary_path):
            os.makedirs(summary_path)
        self.summary_writer = tf.summary.FileWriter(summary_path)

    def get_actions(self, step_output, ip=None):
        """
        Get each factor's action based on its local observation. Append the given
        state to the factor's replay memory.
        """
        evaluate_policy_output = {}
        evaluate_policy_output.update(self.model.evaluate_policy(step_output['obs'], step_output['prev_action']))
        return evaluate_policy_output

    def coordination(self, Q):
        """
        Runs coordination algorithm and returns the corresponding
        actions. Not using epsilon (for now) since it is used
        in evaluation only (because Transfer Planning).
        """
        # TODO: add coordination algorithms
        global_action = []
        i_index = self.epsilon_greedy([Q], 1.0)
        global_action.append(i_index)
        print(("Global action: {}".format(global_action)))
        return global_action

    def get_last_transition_info(self):
        """
        Get each factor's TD-error given the transition information.
        """
        abs_td = []
        batch = self.replay_memory.get_latest_entry()
        td_error = self.model.td_error(batch)
        abs_td.append(np.abs(td_error))

        return abs_td

    def update(self):
        """
        Sample a batch from the replay memory (if it is completely filled) and
        use it to update the models.
        """
        raise NotImplementedError

    def full_memory(self):
        """
        Check if the replay memories are filled.
        """
        return self.replay_memory.full()

    def reset(self):
        """
        Reset the Q-functions. This is necessary for Recurrent Neural Networks.
        """
        if not self.replay_memory.full():
            print(("Replay Memory filled for {}%".format(self.replay_memory.fullness())))

        self.model.reset()

    def save_graph(self, time_step):
        """
        Store all the networks and replay memories.
        """
        # Create factor path if it does not exist.
        factor_path = os.path.join('models', self.parameters['name'])
        if not os.path.exists(factor_path):
            os.makedirs(factor_path)
        self.model.save_graph(time_step)

    # TODO: replay memory
    def store_memory(self, path):
        # Create factor path if it does not exist.
        factor_path = os.path.join(os.environ['APPROXIMATOR_HOME'], path)
        if not os.path.exists(factor_path):
            os.makedirs(factor_path)
        # Store the replay memory
        self.replay_memory.store(factor_path)

    def increment_step(self):
        self.model.increment_step()
        self.step = self.model.get_current_step()

    def write_summary(self):
        """
        Saves training statistics to Tensorboard.
        """
        if self.step % self.parameters['summary_frequency'] == 0 and \
           self.parameters['tensorboard']:

            summary = tf.Summary()
            for key in self.stats.keys():
                if len(self.stats[key]) > 0:
                    stat_mean = float(np.mean(self.stats[key]))
                    #self.logger.log_scalar(key, stat_mean, self.step)
                    summary.value.add(tag='{}'.format(key), simple_value=stat_mean)
                    self.stats[key] = []
            self.summary_writer.add_summary(summary, self.step)
            self.summary_writer.flush()
