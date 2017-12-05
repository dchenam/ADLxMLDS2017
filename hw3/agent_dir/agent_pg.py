from agent_dir.agent import Agent
import gym
import os
import numpy as np
from itertools import count

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.autograd as autograd
from torch.autograd import Variable

def prepro(I):
    """ prepro 210x160x3 into 6400 """
    I = I[35:195]
    I = I[::2, ::2, 0]
    I[I == 144] = 0
    I[I == 109] = 0
    I[I != 0 ] = 1
    return I.astype(np.float).ravel()

class Policy(nn.Module):
    def __init__(self):
        super(Policy,self).__init__()
        self.affine1 = nn.Linear(6400, 200)
        self.affine2 = nn.Linear(200, 3) # action 1 = stop, action 2 = move up, action 3 = move down
        self.saved_actions = []
        self.rewards = []
    def forward(self, x):
        x = F.relu(self.affine1(x))
        action_scores = self.affine2(x)
        return F.softmax(action_scores)

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
        self.learning_rate = 1e-4
        self.gamma = 0.99  # discount factor for reward
        self.decay_rate = 0.99  # decay factor for RMSProp leaky sum of grad^2
        self.epochs = 200
        self.env = env
        self.policy = Policy().cuda()
        self.optimizer = optim.RMSprop(self.policy.parameters(), lr=self.learning_rate, weight_decay=self.decay_rate)
        if args.test_pg:
            if os.path.isfile('pg_params.pkl'):
                print('loading trained model')
                self.policy.load_state_dict(torch.load('pg_params.pkl'))

    def init_game_setting(self):
        """

        Testing function will call this function at the begining of new game
        Put anything you want to initialize if necessary

        """
        pass

    def select_action(self, state):
        state = torch.from_numpy(state).float().unsqueeze(0)
        probs = self.policy(Variable(state).cuda())
        action = probs.multinomial()
        self.policy.saved_actions.append(action)
        return action.data

    def finish_episode(self):
        R = 0
        rewards = []
        #Discount Rewards
        for r in self.policy.rewards[::-1]:
            R = r + self.gamma * R
            rewards.insert(0, R)
        rewards = torch.Tensor(rewards)
        #Normalize Rewards
        rewards = (rewards - rewards.mean()) / (rewards.std() + np.finfo(np.float32).eps)
        print(rewards)

        #Approximate Gradient
        for action, r in zip(self.policy.saved_actions, rewards):
            action.reinforce(r)
        self.optimizer.zero_grad()
        autograd.backward(self.policy.saved_actions, [None for _ in self.policy.saved_actions])
        self.optimizer.step()
        #Clear Memory
        del self.policy.rewards[:]
        del self.policy.saved_actions[:]

    def train(self):
        """
        Implement your training algorithm here
        """
        running_reward = None
        reward_sum = 0
        for i_episode in count(1):
            state = self.env.reset()
            for t in range(self.epochs):
                state = prepro(state)
                action = self.select_action(state)[0, 0]
                action = action + 1
                state, reward, done, _ = self.env.step(action)
                reward_sum += reward
                self.policy.rewards.append(reward)
                if done:
                    # tracking log
                    running_reward = reward_sum if running_reward is None else running_reward * 0.99 + reward_sum * 0.01
                    print('resetting env. episode reward total was %f. running mean: %f' % (reward_sum, running_reward))
                    reward_sum = 0
                    break

                if reward != 0:
                    print('ep %d: game finished, reward: %f' % (i_episode, reward) + ('' if reward == -1 else ' !!!!!!!'))

            # use policy gradient update model weights
            if i_episode % self.batch_size == 0:
                print('ep %d: policy network parameters updating...' % (i_episode))
                self.finish_episode()

            # Save model in every 50 episode
            if i_episode % 50 == 0:
                print('ep %d: model saving...' % (i_episode))
                torch.save(self.policy.state_dict(), 'pg_params.pkl')

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
        probs = self.policy(Variable(observation).cuda())
        action = probs.multinomial()
        self.policy.saved_actions.append(action)
        return action.data

