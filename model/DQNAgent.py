import os
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
import random
import gym
import numpy as np
from collections import deque
from model import Model

class DQNAgent:

    def __init__(self, enviroment):
        # self.amb = gym.make('CartPole-v1')
        self.state_size = enviroment.observation_space.shape[0]
        self.action_size = enviroment.action_space.n
        # self.EPISODES = 500
        self.memory = deque(maxlen=2000)
        self.gamma = 0.95 
        self.epsilon = 1.0
        # self.epsilon_min = 0.001
        # self.epsilon_decay = 0.999
        self.batch_size = 64
        self.train_start = 1000
        self.model = Model(input_shape=(self.state_size,), action_space = self.action_size)

    def replay (self):
        if len(self.memory) < self.train_start:
            return
        
        experience = random.sample(self.memory, min(len(self.memory), self.batch_size))
        state = np.zeros((self.batch_size, self.state_size))
        next_state = np.zeros((self.batch_size, self.state_size))
        action, reward, end = [],[],[]

        for i in range(self.batch_size):
            state[i] = experience[i][0]
            action.append(experience[i][1])
            reward.append(experience[i][2])
            next_state[i] = experience[i][3]
            end.append(experience[i][4])

        target = self.model.predict(state)    
        target_prox = self.model.predict(next_state)

        for i in range(self.batch_size):
            if end[i]:
                target[i][action[i]] = reward[i]
            else:    
                target[i][action[i]] = reward[i] + self.gamma * (np.amax(target_prox[i]))
                
        self.model.fit(state, target, batch_size=self.batch_size, verbose=0)

    def act(self, state):
        if np.random.random() <= self.epsilon:
            return random.randrange(self.action_size)
        else:
            return np.argmax(self.model.predict(state)) 

    def load(self, name):
        self.model = load_model(name)
    
    def save(self, name):
        self.model.save(name)
    
    def remember(self, state, action, reward, next_state, end):
        self.memory.append((state, action, reward, next_state, end))