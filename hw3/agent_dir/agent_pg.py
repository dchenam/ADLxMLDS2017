from agent_dir.agent import Agent
import gym
import os
import numpy as np
import pickle

import tensorflow as tf

class Policy:
    def __init__(self, hidden_layer_size, learning_rate, n_actions, checkpoints_dir):
        self.observations = tf.placeholder(tf.float32, [None, 6400])
        self.sampled_actions = tf.placeholder(tf.float32, [None, n_actions])
        self.advantage = tf.placeholder(tf.float32, [None, 1])

        # h = tf.layers.dense(
        #     self.observations,
        #     units=hidden_layer_size,
        #     activation=tf.nn.relu,
        #     kernel_initializer=tf.contrib.layers.xavier_initializer())
        #
        # self.action_probability = tf.layers.dense(
        #     h,
        #     units=n_actions,
        #     activation=tf.nn.softmax,
        #     kernel_initializer=tf.contrib.layers.xavier_initializer())

        self.tf_model = {}
        with tf.variable_scope('layer_one', reuse=False):
            xavier_l1 = tf.truncated_normal_initializer(mean=0, stddev=1. / np.sqrt(6400), dtype=tf.float32)
            self.tf_model['W1'] = tf.get_variable("W1", [6400, hidden_layer_size], initializer=xavier_l1)
        with tf.variable_scope('layer_two', reuse=False):
            xavier_l2 = tf.truncated_normal_initializer(mean=0, stddev=1. / np.sqrt(hidden_layer_size), dtype=tf.float32)
            self.tf_model['W2'] = tf.get_variable("W2", [hidden_layer_size, n_actions], initializer=xavier_l2)

        # tf_discounted_epr = self.tf_discount_rewards(self.advantage)
        # tf_mean, tf_variance = tf.nn.moments(tf_discounted_epr, [0], shift=None, name="reward_moments")
        # tf_discounted_epr -= tf_mean
        # tf_discounted_epr /= tf.sqrt(tf_variance + 1e-6)

        self.action_prob = self.forward_pass(self.observations)
        loss = tf.nn.l2_loss(self.sampled_actions - self.action_prob)
        optimizer = tf.train.RMSPropOptimizer(learning_rate, decay=0.99)
        tf_grads = optimizer.compute_gradients(loss, var_list=tf.trainable_variables(), grad_loss=self.advantage)
        self.train_op = optimizer.apply_gradients(tf_grads)

        # tf_grads = optimizer.compute_gradients(self.loss, var_list=tf.trainable_variables(), grad_loss=self.advantage)
        # self.loss = tf.losses.log_loss(
        #     labels=self.sampled_actions,
        #     predictions=self.up_probability,
        #     weights=self.advantage)
        #
        # #self.train_op = optimizer.apply_gradients(tf_grads)
        # self.train_op = optimizer.minimize(self.loss)
        self.sess = tf.InteractiveSession()
        tf.global_variables_initializer().run()
        self.saver = tf.train.Saver(tf.global_variables())
        self.checkpoint_path = checkpoints_dir
        self.checkpoint_file = os.path.join(checkpoints_dir,
                                            'policy_network.ckpt')

    def tf_discount_rewards(self, tf_r):  # tf_r ~ [game_steps,1]
        discount_f = lambda a, v: a * 0.99 + v;
        tf_r_reverse = tf.scan(discount_f, tf.reverse(tf_r, [True, False]))
        tf_discounted_r = tf.reverse(tf_r_reverse, [True, False])
        return tf_discounted_r

    def forward_pass(self, observations):
        # action_prob = self.sess.run(
        #     self.action_probability,
        #     feed_dict={self.observations: observations.reshape([1, -1])})
        # return action_prob
        h = tf.matmul(observations, self.tf_model['W1'])
        h = tf.nn.relu(h)
        logp = tf.matmul(h, self.tf_model['W2'])
        p = tf.nn.softmax(logp)
        return p

    def train(self, states, actions, rewards):
        #print("Training with %d (state, action, reward) tuples" %
        #      len(state_action_reward_tuples))

        #states, actions, rewards = zip(*state_action_reward_tuples)
        states = np.vstack(states)
        actions = np.vstack(actions)
        rewards = np.vstack(rewards)

        feed_dict = {
            self.observations: states,
            self.sampled_actions: actions,
            self.advantage: rewards
        }
        self.sess.run(self.train_op, feed_dict)

#From Andrej's Code
def prepro(I):
    """ prepro 210x160x3 into 6400 """
    I = I[35:195]
    I = I[::2, ::2, 0]
    I[I == 144] = 0
    I[I == 109] = 0
    I[I != 0 ] = 1
    return I.astype(np.float).ravel()

class Agent_PG(Agent):
    def __init__(self, env, args):
        """
        Initialize every things you need here.
        For example: building your model
        """
        super(Agent_PG,self).__init__(env)

        # hyperparameters
        self.hidden_units = 200  # number of hidden layer neurons
        self.batch_size = 1  # every how many episodes to do a param update?
        self.learning_rate = 1e-3
        self.gamma = 0.99  # discount factor for reward
        self.decay_rate = 0.99  # decay factor for RMSProp leaky sum of grad^2
        self.n_actions = 3

        #Game space
        self.env = env
        self.i_episode = 0
        self.prev_state = None
        self.running_reward = None
        self.states, self.actions, self.rewards = [], [], []
        self.reward_history = []


        if not os.path.exists('./checkpoints'):
            os.makedirs('./checkpoints')
        self.policy = Policy(hidden_layer_size=self.hidden_units, learning_rate=self.learning_rate,
                             n_actions=self.n_actions, checkpoints_dir='./checkpoints/')
        if args.resume or args.test_pg:
            self.load_checkpoint()

    def init_game_setting(self):
        """
        Testing function will call this function at the begining of new game
        Put anything you want to initialize if necessary
        """
        pass

    def discount_rewards(self, rewards):
        R = 0
        discount_rewards = []
        # Discount Rewards
        for r in rewards[::-1]:
            if r != 0: R = 0  # reset the sum, after someone scores a point
            R = r + self.gamma * R
            discount_rewards.insert(0, R)
        # Normalize Rewards
        discount_rewards -= np.mean(discount_rewards)
        discount_rewards /= np.std(discount_rewards) + np.finfo(np.float32).eps
        return discount_rewards.tolist()

    def parameter_update(self):
        self.rewards = self.discount_rewards(self.rewards)
        #batch_state_action_reward_tuples = list(zip(states, actions, rewards))
        self.policy.train(self.states, self.actions, self.rewards)
        del self.states[:]
        del self.actions[:]
        del self.rewards[:]

    def train(self):
        while True:
            #Reset Episode Parameters
            done = False
            reward_sum = 0
            round_n = 1
            state = self.env.reset()
            while not done:
                #Compute State Delta
                state = prepro(state)
                state_delta = state - self.prev_state if self.prev_state is not None else np.zeros_like(state)
                self.prev_state = state

                #Sample Stochastic Policy
                feed = {self.policy.observations: state_delta.reshape([1, -1])}
                action_prob = self.policy.sess.run(self.policy.action_prob, feed)[0]
                # action_prob = self.policy.forward_pass(state_delta)[0]
                action = np.random.choice(self.n_actions, p=action_prob)
                label = np.zeros_like(action_prob)
                label[action] = 1

                # Step Environment and Update Reward
                state, reward, done, _ = self.env.step(action + 1)
                reward_sum += reward

                # Record Game History
                self.states.append(state_delta)
                self.actions.append(label)
                self.rewards.append(reward)
                round_n += 1

            # Logging
            self.running_reward = reward_sum if self.running_reward is None else self.running_reward * 0.99 + reward_sum * 0.01
            self.reward_history.append(self.running_reward)
            if self.i_episode % 10 == 0:
                print(
                    'ep {}: reward: {}, mean reward: {:3f}'.format(self.i_episode, reward_sum, self.running_reward))
            else:
                print('\tep {}: finished after {} rounds, reward: {}'.format(self.i_episode, round_n, reward_sum))


            # Update Parameter
            if self.i_episode % self.batch_size == 0:
                #print('ep %d: policy network parameters updating...' % (self.i_episode))
                self.parameter_update()
            # Save model in every 50 episode
            if self.i_episode % 50 == 0:
                print('ep %d: model saving...' % (self.i_episode))
                self.save_checkpoint()
            self.i_episode += 1

    def load_checkpoint(self):
        print("Loading checkpoint...")
        ckpt = tf.train.get_checkpoint_state(self.policy.checkpoint_path)
        load_path = ckpt.model_checkpoint_path
        self.policy.saver.restore(self.policy.sess, load_path)
        self.policy.saver = tf.train.Saver(tf.global_variables())
        self.reward_history = pickle.load(open('./reward_history.p', 'rb'))
        self.i_episode = int(load_path.split('-')[-1])

    def save_checkpoint(self):
        print("Saving checkpoint...")
        self.policy.saver.save(self.policy.sess, self.policy.checkpoint_file, global_step=self.i_episode)
        pickle.dump(self.reward_history, open('./reward_history.p', 'wb'))

    def make_action(self, observation, test=True):
        """
        Return predicted action of your agent

        Input:
            observation: np.array
                current RGB screen of game, shape: (210, 160, 3)

        Return:
            action: int
                the predicted action from trained model
        """
        # Compute State Delta
        state = prepro(observation)
        state_delta = state - self.prev_state if self.prev_state is not None else np.zeros_like(state)
        self.prev_state = state

        # Sample Stochastic Policy
        action_prob = self.policy.forward_pass(state_delta)[0]
        print(action_prob)
        action = np.random.choice(self.n_actions, p=action_prob)
        return action + 1

