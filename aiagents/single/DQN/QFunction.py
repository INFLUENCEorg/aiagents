import tensorflow as tf
import os

class Q_function(object):
    """
    A super class that describes basic functionalities for a Q-network.
    """
    def __init__(self, num_actions, parameters):
        """
        Store parameter values
        """
        print("Building network...")
        self.graph = tf.Graph()
        self.num_actions = num_actions
        self.num_frames = parameters["num_frames"]
        self.b_s = parameters["batch_size"]
        self.lr = parameters["learning_rate"]
        self.freeze_interval = parameters["freeze_interval"]
        self.gamma = parameters["gamma"]
        self.init_func = self.initializer('xu', 0.0)
        self.f_height = parameters["frame_height"]
        self.f_width = parameters["frame_width"]
        self.target_input = {}
        self.target_assign_op = {}
        with self.graph.as_default():
            with tf.variable_scope('parameters'):
                # Not every operation works on the same batch size.
                self.batch_size = tf.placeholder(tf.int32, name='batch_size')
            with tf.variable_scope('optimizer'):
                self.learning_rate = tf.placeholder(tf.float32, name='learning_rate')
                self.updates = tf.train.AdamOptimizer(self.learning_rate)
        self.sess = tf.Session(graph=self.graph)

    def initializer(self, initialization_type, init_param):
        """
        Builds an initializer object based on initialization_type
        and init_param.

        Keyword arguments:
            initialization_type -- a string specifying the
                                    initializer to use.
            init_param -- a string specifying the initializer
                            parameter - standard deviation in
                            case of normal, gain in case of He.

        Returns: an object
        """
        with self.graph.as_default():
            if type(init_param) != float:
                raise TypeError(("Initialization parameter"
                                    " {} is {}, should be "
                                    "float.").format(init_param,
                                                     type(init_param)))
            if initialization_type == 'u':
                init_func = tf.random_uniform_initializer(-init_param, init_param)
            elif initialization_type == 'n':
                init_func = tf.random_normal_initializer(stddev=init_param)
            elif initialization_type == 'xu':
                init_func = tf.contrib.layers.xavier_initializer()
            elif initialization_type == 'xn':
                init_func = tf.contrib.layers.xavier_initializer(uniform=False)
            else:
                raise ValueError(("Option '{}' unknown for setting "
                                    "initialization function. "
                                    "Please supply one of ['u',"
                                    " 'n', 'hu', 'hn', 'gu', "
                                    "'gn']").format(initialization_type))
        return init_func

    def loss_function(self, q_vals, next_q_vals, rewards, actions, double_q_vals=None):
        """
        This method calculates the loss used for training.
        """
        with self.graph.as_default():
            with tf.name_scope('loss'):
                """
                Calculate the target value(s)
                """
                if double_q_vals is not None:
                    # Select maximizing action using online network
                    max_index = tf.argmax(double_q_vals, axis=1, output_type=tf.int32)
                    indices = tf.stack([tf.range(0,self.batch_size), max_index], axis=-1)
                    # Evaluate Q using target network
                    next_q_acted = tf.gather_nd(next_q_vals, indices)
                else:
                    # Select the maximum value of the next_q_vals: max_a Q(s_t+1,a)
                    next_q_acted = tf.reduce_max(next_q_vals, axis=1)
                # y = r + gamma * max Q(s_t+1)
                target = tf.add_n([rewards, tf.scalar_mul(self.gamma, next_q_acted)], name='target_values')
                """
                Retrieve the Q-value(s) of the given actions
                """
                # Q(s_t,a_t)
                indices = tf.stack([tf.range(0,self.batch_size), actions], axis=-1)
                q_acted = tf.gather_nd(q_vals, indices)
                """
                Calculate the loss: squared TD-error
                """
                # This is the TD-error: y - Q(s_t,a_t)
                diff = tf.subtract(target, q_acted, name='TD_errors')
                # reduce-mean averages the negative and positive td-errors
                td_loss = tf.square(diff, name='squared_TD_errors')
                loss = tf.reduce_mean(td_loss)
                # Squared_TD_errors is the mean-squared-loss we want to minimize in training

        return loss, diff

    def train(self, states, actions, rewards, next_states):
        """
        Train one batch.

        Keyword arguments:
        states -- mini batch of states
        actions -- mini batch of actions
        rewards -- mini batch of rewards
        next_states -- mini batch of next states

        Returns: average squared loss
        """
        raise NotImplemented

    def get_q_values(self, state):
        """
            Compute the Q-values for the given state.

            Keyword arguments:
                state -- the state to compute Q-values for (num_frames, width, height) nparray
        """
        raise NotImplemented

    def td_error(self, states, actions, rewards, next_states):
        """
            Computes the TD error for the given batch of transitions.

            Keyword arguments:
            states -- mini batch of states
            actions -- mini batch of actions
            rewards -- mini batch of rewards
            next_states -- mini batch of next states

            Returns: a vector
        """
        raise NotImplemented

    def inference(self, x, name):
        """
            This is the forward pass through the network given an input state.
            The implementation of this function is different per network type.
            Args:
                x:      4D float Tensor
                name:   String containing the name of this network (e.g. action or target)
            Returns:
                q_vals: 2D float Tensor
                var:    a dictionary containing the variables of the network
        """
        raise NotImplemented

    def reset_q_hat(self, action_vars, target_vars):
        """
            Resets the target network by copying all weight values from
            the action network Q(s_t,a) to the target network Q(s_t+1, a).
        """
        op_holder = []
        for name in list(action_vars.keys()):
            op_holder.append(target_vars[name].assign(action_vars[name]))

        return op_holder

    def reset(self):
        """
            Resets parameters that need to be resetted at the end of an episode.
        """
        pass

    def store(self, path, time_step):
        """
            Retrieve network weights for action and target networks
            and store these in separate files.
        """
        with self.graph.as_default():
            file_name = os.path.join(path, "network")
            print("Saving networks...")
            self.saver.save(self.sess, file_name, time_step)
            print("Saved!")

    def load(self, path, nr_of_saves, test_it=-1):
        """
            Load network weights for action and target networks
            and load these into separate networks.
        """
        with self.graph.as_default():
            print("Loading networks...")
            checkpoint_dir = os.path.join(os.environ['APPROXIMATOR_HOME'], path, "network-"+str(test_it))
            self.saver = tf.train.Saver(max_to_keep=nr_of_saves+1)
            try:
                self.saver.restore(self.sess, checkpoint_dir)
                print("Loaded: {}".format(checkpoint_dir))
            except Exception:
                if test_it <= 0:
                    # Initialize the variables
                    self.sess.run(tf.global_variables_initializer())
                    print("Failed! Initializing the network variables...")
                else:
                    raise
