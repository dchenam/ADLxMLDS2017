from agent_dir.agent import Agent
import gym
import os
import numpy as np
from itertools import count

import tensorflow as tf

class Policy:
    def __init__(self, hidden_layer_size, learning_rate, checkpoints_dir):
        self.learning_rate = learning_rate

        self.sess = tf.InteractiveSession()

        self.observations = tf.placeholder(tf.float32, [None, 6400])
        # +1 for up, -1 for down
        self.sampled_actions = tf.placeholder(tf.float32, [None, 1])
        self.advantage = tf.placeholder(
            tf.float32, [None, 1], name='advantage')

        h = tf.layers.dense(
            self.observations,
            units=hidden_layer_size,
            activation=tf.nn.relu,
            kernel_initializer=tf.contrib.layers.xavier_initializer())

        self.up_probability = tf.layers.dense(
            h,
            units=1,
            activation=tf.sigmoid,
            kernel_initializer=tf.contrib.layers.xavier_initializer())

        self.loss = tf.losses.log_loss(
            labels=self.sampled_actions,
            predictions=self.up_probability,
            weights=self.advantage)

        optimizer = tf.train.AdamOptimizer(self.learning_rate)
        self.train_op = optimizer.minimize(self.loss)
        tf.global_variables_initializer().run()

        self.saver = tf.train.Saver()
        self.checkpoint_file = os.path.join(checkpoints_dir,
                                            'policy_network.ckpt')

    def load_checkpoint(self):
        print("Loading checkpoint...")
        self.saver.restore(self.sess, self.checkpoint_file)

    def save_checkpoint(self):
        print("Saving checkpoint...")
        self.saver.save(self.sess, self.checkpoint_file)

    def forward_pass(self, observations):
        up_probability = self.sess.run(
            self.up_probability,
            feed_dict={self.observations: observations.reshape([1, -1])})
        return up_probability

    def train(self, state_action_reward_tuples):
        print("Training with %d (state, action, reward) tuples" %
              len(state_action_reward_tuples))

        states, actions, rewards = zip(*state_action_reward_tuples)
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
        self.batch_size = 5  # every how many episodes to do a param update?
        self.learning_rate = 5e-4
        self.gamma = 0.99  # discount factor for reward
        self.decay_rate = 0.99  # decay factor for RMSProp leaky sum of grad^2
        self.epochs = 10000
        self.env = env
        if not os.path.exists('./checkpoints'):
            os.makedirs('./checkpoints')
        self.policy = Policy(hidden_layer_size=self.hidden_units, learning_rate=self.learning_rate, checkpoints_dir='./checkpoints/')
        if args.resume or args.test_pg:
            self.policy.load_checkpoint()

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
        return discount_rewards

    def finish_episode(self, states, actions, rewards, batch_rewards):
        batch_state_action_reward_tuples = list(zip(states, actions, batch_rewards))
        self.policy.train(batch_state_action_reward_tuples)
        del states[:]
        del actions[:]
        del rewards[:]
        del batch_rewards[:]

    def train(self):
        """
        Implement your training algorithm here
        """
        running_reward = None
        #batch_state_action_reward_tuples = []
        states, actions, rewards, batch_rewards = [], [], [], []
        for i_episode in count(1):
            reward_sum = 0
            observation = self.env.reset()
            prev_state = prepro(observation)  # To compute difference frame
            action = self.env.action_space.sample()
            observation, _, _, _ = self.env.step(action)
            state = prepro(observation)
            for t in range(self.epochs):
                state_delta = state - prev_state
                prev_state = state
                up_probability = self.policy.forward_pass(state_delta)[0]
                if np.random.uniform() < up_probability:
                    action = 1
                else:
                    action = 2
                #Gym 2 is UP 3 is DOWN
                observation, reward, done, _ = self.env.step(action + 1)
                state = prepro(observation)
                reward_sum += reward
                states.append(state_delta)
                actions.append(action - 1)
                rewards.append(reward)
                #batch_state_action_reward_tuples.append((state_delta, (action - 1), reward))
                if done:
                    discount_rewards = self.discount_rewards(rewards)
                    batch_rewards += discount_rewards.tolist()
                    # tracking log
                    running_reward = reward_sum if running_reward is None else running_reward * 0.99 + reward_sum * 0.01
                    print('resetting env. episode reward total was %f. running mean: %f' % (reward_sum, running_reward))
                    break

                if reward != 0:
                    print('ep %d: game finished, reward: %f, steps: %d' % (i_episode, reward, t) + ('' if reward == -1 else ' !!!!!!!'))

            # use policy gradient update model weights
            if i_episode % self.batch_size == 0:
                print('ep %d: policy network parameters updating...' % (i_episode))
                self.finish_episode(states, actions, rewards, batch_rewards)

            # Save model in every 50 episode
            if i_episode % 50 == 0:
                print('ep %d: model saving...' % (i_episode))
                self.policy.save_checkpoint()

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
        state = prepro(observation)
        up_probability = self.policy.forward_pass(state)[0]
        if np.random.uniform() < up_probability:
            action = 2
        else:
            action = 3
        return action.data

