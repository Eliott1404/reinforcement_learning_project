import numpy as np
import pandas as pd
from datetime import datetime

from TestEnv import HydroElectric_Test

class TabularAgent():
    
    def __init__(self, discount_rate, bin_sizes):
        
        '''
        Params:
        
        discount_rate = discount rate used for future rewards
        bin_size = number of bins used for discretizing the state space
        
        '''
        
        #Set the discount rate
        self.discount_rate = discount_rate
        
        #The algoritm has 5 discrete actions
        self.action_space = np.array([-0.2, -0.1, 0, 0.1, 0.2])
        
        #Set the bin size
        self.bin_sizes = bin_sizes
        self.bins = []
        
    def make_bins(self, train_data):
        df = pd.read_excel(train_data)
        prices = df.iloc[1:, 1:]
        min_price = prices.min().min()
        max_price = prices.max().max()

        #Create bins for observed features
        self.bin_price = np.linspace(min_price,max_price,self.bin_sizes[0])
        self.bin_price[0] = -np.inf
        self.bin_price[-1] = np.inf

        self.bin_damlevel = np.linspace(0, 1, self.bin_sizes[1])
    
        #Put all bins together
        self.bins = [self.bin_price, self.bin_damlevel]

    def discretize_state(self, observation):
        state = [observation[1], observation[0], observation[4], observation[3]]
        discretized_state = []
        for i, bin in enumerate(self.bins):
            discretized_state.append(int(np.digitize(state[i], bin))-1)
        if state[2] > 1 and state[2] < 9:
            discretized_state.append(0)
        else:
            discretized_state.append(1)
        discretized_state.append(int(state[3]))         
        return discretized_state
    
    def act(self, observation):
        discretized_state = self.discretize_state(observation)

        #Pick random action
        if np.random.uniform() < self.epsilon:
            digitized_action = np.random.randint(0, len(self.action_space))
                    
        #Pick a greedy action              
        else:
            digitized_action = np.argmax(self.Qtable[discretized_state[0], discretized_state[1], discretized_state[2], discretized_state[3],:])
        return digitized_action
    
    def create_Q_table(self):
        #Initialize all values in the Q-table to zero    
        dims = self.bin_sizes
        self.Qtable = np.zeros((dims[0], dims[1], dims[2], dims[3], len(self.action_space)))

    def train(self, epochs, learning_rate, epsilon, path):
        
        '''
        Params:
        
        simulations = number of episodes of a game to run
        learning_rate = learning rate for the update eqaution
        epsilon = epsilon value for epsilon-greedy algorithm
        '''
        
        #Initialize variables that keep track of the rewards
        self.train_rewards = []
        
        #Call the Q table function to create an initialized Q table
        self.create_Q_table()
        
        #Set epsilon rate, epsilon decay and learning rate
        self.epsilon = epsilon
        self.learning_rate = learning_rate
        
        for epoch in range(epochs):
            #Initialize the environment
            env = HydroElectric_Test(path_to_test_data=path)
            observation = env.observation()
            state = self.discretize_state(observation)
        
            #Set the rewards to 0
            total_reward = 0

            for i in range(1096*24 -1): # Loop through 2 years -> 730 days * 24 hours
                # The observation is the tuple: [volume, price, hour_of_day, day_of_week, day_of_year, month_of_year, year]
                digitized_action = self.act(observation)
                action = self.action_space[digitized_action]
                next_observation, reward, terminated, truncated, info = env.step(action)
                total_reward += reward

                done = terminated or truncated
                observation = next_observation

                next_state = self.discretize_state(observation)
                
                #Target value 
                Q_target = (reward + self.discount_rate*np.max(self.Qtable[next_state[0], next_state[1], next_state[2], next_state[3]]))

                #Calculate the Temporal difference error (delta)
                delta = self.learning_rate * (Q_target - self.Qtable[state[0], state[1], state[2], state[3], digitized_action])
                
                #Update the Q-value
                self.Qtable[state[0], state[1], state[2], state[3], digitized_action] = self.Qtable[state[0], state[1], state[2], state[3], digitized_action] + delta
                
                #Update the reward and the hyperparameters
                total_reward += reward
                state = next_state
            env.close()

            print(f'Epoch {epoch}: Total reward = {total_reward}')            
