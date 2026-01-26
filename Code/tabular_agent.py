import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from datetime import datetime

import copy
import math


from TestEnv import HydroElectric_Test

class TabularAgent():
    
    def __init__(self, discount_rate):
        
        '''
        Params:
        
        discount_rate = discount rate used for future rewards
        bin_size = number of bins used for discretizing the state space
        
        '''
        
        #Set the discount rate
        self.discount_rate = discount_rate
        
        #The algoritm has 5 discrete actions
        self.action_space = np.array([-0.5, -0.25, 0, 0.25, 0.5])
        
        #Make lookup tables for bins
        self.bins_dam_levels = np.array([9999, 19999, 30000, 69999, 80000, 90000])
        self.bins_rsi = np.array([15,30,50,70,85])

        #Keep a list of previous prices for RSI calculation
        self.prices = []
        
    # def compute_rsi(self, prices, period=14):
    #     """
    #     returns: RSI value in [0, 100]
    #     """
    #     prices = np.asarray(prices[-:])

    #     if len(prices) < period + 1:
    #         return 50

    #     deltas = np.diff(prices[-(period + 1):])

    #     gains = np.clip(deltas, 0, None)
    #     losses = np.clip(-deltas, 0, None)

    #     avg_gain = np.mean(gains)
    #     avg_loss = np.mean(losses)

    #     if avg_loss == 0:
    #         return 100.0  # price only went up

    #     rs = avg_gain / avg_loss
    #     rsi = 100 - (100 / (1 + rs))

    #     return rsi
    def compute_rsi(self, prices, period=14):
        """
        returns: RSI value in [0, 100]
        """
        prices = np.asarray(prices)

        if len(prices) < period + 1:
            return 50.0

        deltas = np.diff(prices[-(period + 1):])

        gains = np.clip(deltas, 0, None)
        losses = np.clip(-deltas, 0, None)

        avg_gain = np.mean(gains)
        avg_loss = np.mean(losses)

        if avg_loss == 0:
            return 100.0

        rs = avg_gain / avg_loss
        return 100 - (100 / (1 + rs))
    
    def discretize_state(self, observation):
        dam_level = observation[0]
        digitized_dam_level = np.digitize(dam_level, self.bins_dam_levels)

        rsi = self.compute_rsi(copy.deepcopy(self.prices))
        digitized_rsi = np.digitize(rsi, self.bins_rsi)

        digitized_weekday = int(observation[3])

        return [digitized_dam_level, digitized_rsi, digitized_weekday]
    
    def update_price_window(self, observation):
        self.prices.append(observation[1])
        if len(self.prices) > (24*7):
            self.prices.pop(0)
    
    def act(self, observation):
        discretized_state = self.discretize_state(observation)

        #Pick random action
        if np.random.uniform() < self.epsilon:
            digitized_action = np.random.randint(0, len(self.action_space))
                    
        #Pick a greedy action              
        else:
            digitized_action = np.argmax(self.Qtable[discretized_state[0], discretized_state[1], discretized_state[2]])
        return digitized_action

    def create_Q_table(self):
        #Initialize all values in the Q-table to zero    
        dims = [7, 6, 7]
        self.Qtable = np.zeros((dims[0], dims[1], dims[2], len(self.action_space))) 

    # def shape_reward(self, reward, observation, action_value):
    #     # action_value is actual action: -0.1, 0, +0.1
    #     p = observation[1]
    #     if len(self.prices) >= 5:
    #         mu = float(np.mean(self.prices[-15:]))
    #     else:
    #         mu = p

    #     spread = p - mu

    #     # price-driven shaping (small coefficient so it doesn't dominate env reward)
    #     k = 5  # keep small!
    #     if action_value < 0:        # generate / sell
    #         price_shape = +k * spread
    #     elif action_value > 0:      # pump / buy
    #         price_shape = -k * spread
    #     else:
    #         price_shape = -k * 0.1 * abs(spread)  # tiny nudge to act only when signal strong

    #     # reservoir safety (optional)
    #     vol = observation[0]
    #     if vol < 30000:
    #         safety = -0.2
    #     else:
    #         safety = 0.0

    #     return reward + price_shape + safety


    def shape_reward(self, observation, action):
        if action < 0:
            # if observation[0] == 0:
            reward_price = action * (np.mean(self.prices)-0.9*observation[1])
        elif action > 0:
            reward_price = action * (np.mean(self.prices)-1.25*observation[1])
        else:
            reward_price = 0

        if observation[0] == 0:
            reward_energy = -10
        elif observation[0] == 100000:
            reward_energy = -10
        elif observation[0] < 10000:
            reward_energy = -2
        elif observation[0] > 90000:
            reward_energy = -1
        else:
            reward_energy = 0

        shaped_reward = reward_price + reward_energy
        return shaped_reward

    def train(self, epochs, learning_rate, path):
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
        eps_start = 1
        eps_end = 0.05
        eps_decay = 100000
        step = 0

        self.learning_rate = learning_rate
        epochs = []
        cumulative_regular = []
        cumulative_shaped = [] 
        
        for epoch in range(epochs):
            action_counts = np.zeros(len(self.action_space), dtype=int)
            level_counts = np.zeros(len(self.bins_dam_levels)+1, dtype=int)
            # rsi_counts = np.zeros(len(self.bins_rsi)+1, dtype=int)
            # weekday_counts = np.zeros(7, dtype=int)

            #Initialize the environment
            env = HydroElectric_Test(path_to_test_data=path)
            self.prices = []
            observation = env.observation()
            self.update_price_window(observation)
            state = self.discretize_state(observation)
        
            #Set the rewards to 0
            total_reward = 0
            total_shaped = 0

            for i in range(1096*24 -1): # Loop through 2 years -> 730 days * 24 hours
                self.epsilon = eps_end + (eps_start - eps_end) * math.exp(-step / eps_decay)
                
                # The observation is the tuple: [volume, price, hour_of_day, day_of_week, day_of_year, month_of_year, year]
                digitized_action = self.act(observation)
                action = self.action_space[digitized_action]
                next_observation, reward, terminated, truncated, info = env.step(action)
                shaped_reward = self.shape_reward(next_observation, action)

                done = terminated or truncated
                observation = next_observation

                self.update_price_window(next_observation)
                next_state = self.discretize_state(next_observation)

                #Track Q table counts
                action_counts[digitized_action] += 1
                level_counts[next_state[0]] += 1
                # rsi_counts[next_state[1]] += 1
                # weekday_counts[next_state[2]] += 1
                
                #Target value 
                Q_target = (shaped_reward + self.discount_rate*np.max(self.Qtable[next_state[0], next_state[1], next_state[2]]))
                # if done:
                #     Q_target = shaped_reward
                # else:
                #     Q_target = shaped_reward + self.discount_rate * np.max(self.Qtable[next_state[0], next_state[1]])


                #Calculate the Temporal difference error (delta)
                delta = self.learning_rate * (Q_target - self.Qtable[state[0], state[1], state[2], digitized_action])
                
                #Update the Q-value
                self.Qtable[state[0], state[1], state[2], digitized_action] = self.Qtable[state[0], state[1], state[2], digitized_action] + delta
                
                #Update the reward and the hyperparameters
                total_reward += reward
                total_shaped += shaped_reward
                state = next_state
                step += 1

                if done:
                    break
            env.close()

            # print statements for training evaluation
            print(f'Epoch {epoch}: Total reward = {total_reward}, Shaped reward = {total_shaped}') 
            print(f'Actions: {action_counts}')
            print(f'Capacity: {level_counts}')
            # print(f'RSI: {rsi_counts}')
            # print(f'Weekdays: {weekday_counts}')
            print(self.epsilon)

            # keep lists for training plot
            epochs.append(epoch)
            cumulative_regular.append(total_reward)
            cumulative_shaped.append(total_shaped)
            
        # Plot the cumulative reward over time
        plt.plot(cumulative_reward[(-24*7):])
        plt.xlabel('Time (Hours)')
        plt.show()
        
         
