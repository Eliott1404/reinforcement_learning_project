from TestEnv import HydroElectric_Test
import argparse
import matplotlib.pyplot as plt

from quantile_baseline import determine_quantiles, quantile_action
from tabular_agent import TabularAgent

# Hyperparameters
lr = 0.05
discount_f = 0.95
bin_sizes = [8,6,2,7] #Price, Dam level, Season, Weekday

# Parse arguments: Train file, test file and agent type
parser = argparse.ArgumentParser()
parser.add_argument('--train_file', type=str, default='Data/train.xlsx')
parser.add_argument('--test_file', type=str, default='Data/validate.xlsx')
parser.add_argument('--agent', type=str, default='tabular')
args = parser.parse_args()

# Save train and test path
train_file = args.train_file
test_file = args.test_file

# Create training environment
train_env = HydroElectric_Test(path_to_test_data=train_file)
train_reward = []
train_cumulative_reward = []
train_dam_level = []
observation = train_env.observation()

# Initialize and train agent
if args.agent == 'tabular':
    RL_agent = TabularAgent(discount_f, bin_sizes)
    RL_agent.make_bins(train_file)
    RL_agent.train(20, lr, 0.5, train_file)

else:
    # Change to DeepAgent or smth
    RL_agent = TabularAgent(observation, bin_sizes)

env = HydroElectric_Test(path_to_test_data=args.test_file)
total_reward = []
cumulative_reward = []
dam_level = []

observation = env.observation()

#Determine quantiles if running quantile baseline
quantiles = determine_quantiles()


for i in range(730*24 -1): # Loop through 2 years -> 730 days * 24 hours
    # # Choose a random action between -1 (full capacity sell) and 1 (full capacity pump)
    
    # action = env.continuous_action_space.sample()
    action = quantile_action(observation, quantiles, 0.6)
    
    # Or choose an action based on the observation using your RL agent!:
    if args.agent == 'tabular':
        action = RL_agent.act(observation)
    # The observation is the tuple: [volume, price, hour_of_day, day_of_week, day_of_year, month_of_year, year]
    next_observation, reward, terminated, truncated, info = env.step(action)
    total_reward.append(reward)
    cumulative_reward.append(sum(total_reward))
    dam_level.append(observation[0])

    done = terminated or truncated
    observation = next_observation

    if done:
        print('Total reward: ', sum(total_reward))
        # Plot the cumulative reward over time
        # plt.plot(cumulative_reward)
        plt.plot(dam_level)
        plt.xlabel('Time (Hours)')
        plt.show()
