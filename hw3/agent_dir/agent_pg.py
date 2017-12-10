from agent_dir.agent import Agent
import gym
import os
import numpy as np
import pickle

import tensorflow as tf

class Policy:
    def __init__(self, learning_rate, decay_rate, n_actions, checkpoints_dir):
        self.observations = tf.placeholder(tf.float32, [None, 80, 80, 1])
        self.sampled_actions = tf.placeholder(tf.int32, [None])
        self.advantage = tf.placeholder(tf.float32, [None])

        self.conv1 = tf.layers.conv2d(
            self.observations,
            filters=16,
            kernel_size=(8, 8),
            strides=(4, 4),
            activation=tf.nn.relu)

        self.conv2 = tf.layers.conv2d(
            self.conv1,
            filters=32,
            kernel_size=(4, 4),
            strides=(2, 2),
            activation=tf.nn.relu)

        self.fc1 = tf.contrib.layers.flatten(self.conv2)

        self.fc1 = tf.layers.dense(
            self.fc1,
            units=128,
            activation=tf.nn.relu,
            kernel_initializer=tf.random_normal_initializer())

        self.action_logit = tf.layers.dense(
            self.fc1,
            units=n_actions)

        self.action = tf.reshape(tf.multinomial(self.action_logit, 1),[-1])
        self.log_prob = -tf.nn.sparse_softmax_cross_entropy_with_logits(
            labels=self.sampled_actions, logits=self.action_logit)
        self.loss = -tf.reduce_mean(self.log_prob * self.advantage)

        # loss = -tf.losses.softmax_cross_entropy(
        #     onehot_labels=self.sampled_actions,
        #     logits=self.action_logit,
        #     weights=self.advantage)

        # self.log_prob = tf.nn.log_softmax(self.action_logit)
        # batch_size = tf.shape(self.observations)[0]
        # self.action_log_prob = self.fancy_slice_2d(self.log_prob, tf.range(batch_size), self.sampled_actions)
        # loss = -tf.reduce_mean(self.advantage * self.action_log_prob)

        optimizer = tf.train.AdamOptimizer(learning_rate)
        self.train_op = optimizer.minimize(self.loss)

        self.sess = tf.InteractiveSession()
        tf.global_variables_initializer().run()
        self.saver = tf.train.Saver(tf.global_variables())
        self.checkpoint_path = checkpoints_dir
        self.checkpoint_file = os.path.join(checkpoints_dir,
                                            'policy_network.ckpt')

    def fancy_slice_2d(self, X, inds0, inds1):
        inds0 = tf.cast(inds0, tf.int64)
        inds1 = tf.cast(inds1, tf.int64)
        shape = tf.cast(tf.shape(X), tf.int64)
        ncols = shape[1]
        Xflat = tf.reshape(X, [-1])
        return tf.gather(Xflat, inds0 * ncols + inds1)

    def sample(self, observations):
        action = self.sess.run(
            self.action,
            feed_dict={self.observations: np.reshape(observations, (-1, 80, 80, 1))})
        return action.tolist()[0]

    def train(self, states, actions, rewards):
        states = np.vstack(states)
        actions = np.vstack(actions)
        rewards = np.vstack(rewards)

        feed_dict = {
            self.observations: np.reshape(states, (-1, 80, 80, 1)),
            self.sampled_actions: actions.squeeze(1),
            self.advantage: rewards.squeeze(1)
        }
        self.sess.run(self.train_op, feed_dict)

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
        super(Agent_PG,self).__init__(env)

        # hyperparameters
        self.batch_size = 1
        self.learning_rate = 1e-4
        self.gamma = 0.99
        self.decay_rate = 0.99
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

        self.policy = Policy(learning_rate=self.learning_rate, decay_rate = self.decay_rate,
                             n_actions=self.n_actions, checkpoints_dir='./checkpoints/')
        if args.resume or args.test_pg:
            self.load_checkpoint()

    def init_game_setting(self):
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
                # state_delta = state - self.prev_state if self.prev_state is not None else state
                # self.prev_state = state

                #Sample Stochastic Policy
                action = self.policy.sample(state)
                # label = np.array([0, 0, 0])
                # label[action] = 1

                self.states.append(state)

                # Step Environment and Update Reward
                state, reward, done, _ = self.env.step(action + 1)
                reward_sum += reward

                # Record Game History
                self.actions.append(action)
                self.rewards.append(reward)
                round_n += 1

            # Logging
            self.running_reward = reward_sum if self.running_reward is None else self.running_reward * 0.99 + reward_sum * 0.01
            self.reward_history.append(reward_sum)
            if self.i_episode % 10 == 0:
                print(
                    'ep {}: reward: {}, mean reward: {:3f}'.format(self.i_episode, reward_sum, self.running_reward))
            else:
                print('\tep {}: finished after {} rounds, reward: {}'.format(self.i_episode, round_n, reward_sum))

            # Update Parameter
            if self.i_episode % self.batch_size == 0:
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
        # Compute State Delta
        state = prepro(observation)
        state_delta = state - self.prev_state if self.prev_state is not None else np.zeros_like(state)
        self.prev_state = state

        # Sample Stochastic Policy
        action = self.policy.sample(state_delta)
        return action + 1

