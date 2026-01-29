import math

import matplotlib.pyplot as plt
import numpy as np

from TestEnv import HydroElectric_Test

class TabularAgent():
    def __init__(self, discount_factor):
        '''
        Params:
        
        discount_factor = discount factor used for future rewards
        bin_size = number of bins used for discretizing the state space
        
        '''
        
        #Set the discount rate
        self.discount_rate = discount_factor
        self.learning_rate = 0
        self.epsilon = 0
        
        #The algoritm has 5 discrete actions
        self.action_space = np.array([-0.8, -0.4, 0, 0.4, 0.8])
        
        #Make lookup tables for bins
        self.bins_dam_levels = np.array([9999, 30000, 69999, 90000])
        self.bins_rsi = np.array([15,30,50,70,85])
        self.bins_houts = np.array([7, 14, 21])

        #Keep a list of previous prices for RSI calculation
        self.prices = []        
        
    def compute_rsi(self, period=14):
        """
        returns: RSI value in [0, 100]
        """
        prices = np.asarray(self.prices)

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
        #digitize dam level
        dam_level = observation[0]
        digitized_dam_level = np.digitize(dam_level, self.bins_dam_levels)

        #digitize rsi
        rsi = self.compute_rsi()
        digitized_rsi = np.digitize(rsi, self.bins_rsi)

        #digitize weekday
        digitized_weekday = int(observation[3])

        #digitize part of day
        if observation[2] < 8:
            digitized_part_of_day = 0
        else:
            digitized_part_of_day = 1

        return [digitized_dam_level, digitized_rsi, digitized_weekday, digitized_part_of_day]
    
    def update_price_window(self, observation):
        self.prices.append(observation[1])
        if len(self.prices) > (24):
            self.prices.pop(0)
    
    def act(self, observation):
        discretized_state = self.discretize_state(observation)

        #Pick random action
        if np.random.uniform() < self.epsilon:
            digitized_action = np.random.randint(0, len(self.action_space))
                    
        #Pick a greedy action              
        else:
            digitized_action = np.argmax(self.Qtable[discretized_state[0], discretized_state[1], discretized_state[2], discretized_state[3]])
        return digitized_action

    def create_Q_table(self):
        #Initialize all values in the Q-table to zero    
        dims = [5, 6, 7, 4]
        self.Qtable = np.zeros((dims[0], dims[1], dims[2], dims[3], len(self.action_space)))
        self.visits = np.zeros_like(self.Qtable, dtype=np.int32)

    # def shape_reward(self, observation, action):
    #     mean_price_week = np.mean(self.prices)
    #     mean_price_day = np.mean(self.prices[-24:])
    #     if action == -0.8:
    #         reward_week = action * (mean_price_week-0.87*observation[1])
    #         reward_day = action * (mean_price_day-0.87*observation[1])
    #     elif action == -0.4:
    #         reward_week = action * (mean_price_week-0.9*observation[1])
    #         reward_day = action * (mean_price_day-0.9*observation[1])
    #     elif action == 0.8:
    #         reward_week = action * (mean_price_week-1.28*observation[1])
    #         reward_day = action * (mean_price_day-1.28*observation[1])
    #     elif action == 0.4:
    #         reward_week = action * (mean_price_week-1.25*observation[1])
    #         reward_day = action * (mean_price_day-1.25*observation[1])

    #     else:
    #         reward_week = 0
    #         reward_day = 0

    #     if observation[0] == 0:
    #         limit_penalty = -1
    #     elif observation[0] == 100000:
    #         limit_penalty = -1
    #     else:
    #         limit_penalty = 0

    #     shaped_reward = 0.8 * reward_week + 0.2 * reward_day + limit_penalty
    #     return shaped_reward

    def shape_reward(self, reward, observation, next_observation):
        # mean_price_week = np.mean(self.prices)
        mean_price_day = np.mean(self.prices[-24:])

        m = (next_observation[0] - observation[0]) * 1000
        g = 9.81
        h = 30

        #Daily reward     
        daily_reward = reward  + (m*g*h / 3.6e9) * mean_price_day

        #Penalty for having no capacity at high price
        if observation[0] == 0 and observation[1] * 0.8 < mean_price_day:
            penalty = -3
        elif observation[0] == 100000 and observation[1] * 1.35 > mean_price_day:
            penalty = -3
        else:
            penalty = 0

        return daily_reward + penalty

    def train(self, epochs, path):
        '''
        Params:
        
        simulations = number of epochs to run
        learning_rate = learning rate for the update eqaution
        epsilon = epsilon value for epsilon-greedy algorithm
        '''
        
        #Call the Q table function to create an initialized Q table
        self.create_Q_table()
        
        #Configurate adaptive epsilon
        eps_start = 1
        eps_end = 0.05
        eps_decay = 500000
        step = 0

        #Initialize lists to track cumulative rewards
        cumulative_regular = []
        cumulative_shaped = [] 
        
        for epoch in range(epochs):
            action_counts = np.zeros(len(self.action_space), dtype=int)
            level_counts = np.zeros(len(self.bins_dam_levels)+1, dtype=int)
            # rsi_counts = np.zeros(len(self.bins_rsi)+1, dtype=int)
            # weekday_counts = np.zeros(7, dtype=int)
            hour_counts = np.zeros(2)

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
                # shaped_reward = self.shape_reward(next_observation, action)
                shaped_reward = self.shape_reward(reward, observation, next_observation)

                done = terminated or truncated
                observation = next_observation

                self.update_price_window(next_observation)
                next_state = self.discretize_state(next_observation)

                #Store state/action in smaller terms
                s0, s1, s2, s3 = state
                ns0, ns1, ns2, ns3 = next_state
                a = digitized_action

                #Adapt learning rate based on number of visits
                self.visits[s0,s1,s2,s3,a] += 1
                n_visits = self.visits[s0,s1,s2,s3,a]

                alpha = max(0.01, 1/math.sqrt(n_visits))

                # #Target value 
                # Q_target = (shaped_reward + self.discount_rate*np.max(self.Qtable[next_state[0], next_state[1], next_state[2]]))
                if done:
                    Q_target = shaped_reward
                else:
                    Q_target = shaped_reward + self.discount_rate * np.max(self.Qtable[ns0,ns1,ns2,ns3])


                #Calculate the Temporal difference error (delta)
                delta = alpha * (Q_target - self.Qtable[s0,s1,s2,s3,a])
                
                #Update the Q-value
                self.Qtable[s0,s1,s2,s3,digitized_action] = self.Qtable[s0,s1,s2,s3,a] + delta

                #Track Q table counts
                action_counts[a] += 1
                level_counts[ns0] += 1
                # rsi_counts[ns1] += 1
                # weekday_counts[ns2] += 1
                hour_counts[ns3] += 1
                
                #Update the reward and the hyperparameters
                total_reward += reward
                total_shaped += shaped_reward
                state = next_state
                step += 1

                if done:
                    env.close()
                    break
                

            # print statements for training evaluation
            print(f'Epoch {epoch}: Total reward = {total_reward}, Shaped reward = {total_shaped}') 
            print(f'Actions: {action_counts}')
            print(f'Capacity: {level_counts}')
            # print(f'RSI: {rsi_counts}')
            # print(f'Weekdays: {weekday_counts}')
            print(self.epsilon)

            # keep lists for training plot
            cumulative_regular.append(total_reward)
            cumulative_shaped.append(total_shaped)
            
        # Plot the cumulative reward over time
        plt.plot(cumulative_regular)
        plt.plot(cumulative_shaped)
        plt.xlabel('Number of epochs')
        plt.show()
        
         
