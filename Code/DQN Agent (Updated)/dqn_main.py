import numpy as np
import torch
import matplotlib.pyplot as plt
import gymnasium as gym
from tqdm import tqdm
import os
import pandas as pd

# Import your custom environment and agent
from TestEnv import HydroElectric_Test
from dqn_agent import HydroDDQNAgent
from advanced_agent import AdvancedHydroAgent

# -------------------------------------------------------------------------
# CONFIGURATION
# -------------------------------------------------------------------------
# UPDATE THESE PATHS TO MATCH YOUR FILE NAMES
TRAIN_PATH = "/Volumes/T7/Python_Projects/Project RL/Data/train.xlsx"      
VAL_PATH = "/Volumes/T7/Python_Projects/Project RL/Data/validate.xlsx" 

# Hyperparameters
MAX_EPISODES = 100       # Total training episodes (1000)
VAL_INTERVAL = 5         # Run validation every N episodes (20)
BATCH_SIZE = 64
LR = 1e-4
GAMMA = 0.99
EPS_START = 1.0
EPS_END = 0.05
EPS_DECAY = 0.995
WARMUP_STEPS = 200       # Steps to fill buffer using Baseline Agent (2000)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# -------------------------------------------------------------------------
# ENVIRONMENT WRAPPER (Fixes missing reset in TestEnv.py)
# -------------------------------------------------------------------------
class FixedHydroEnv(HydroElectric_Test):
    """
    Inherits from your TestEnv but adds a proper reset method 
    required for Gym training loops.
    """
    def __init__(self, path_to_test_data):
        super().__init__(path_to_test_data)
        
    def reset(self, seed=None, options=None):
        super().reset(seed=seed) if hasattr(super(), 'reset') else None
        
        # Reset internal counters as defined in TestEnv.__init__
        self.counter = 0
        self.hour = 1
        self.day = 1
        self.volume = self.max_volume / 2
        
        # Reset state
        self.state = self.observation()
        
        # Return standard gym format: (obs, info)
        return self.state, {}

# -------------------------------------------------------------------------
# HELPER FUNCTIONS
# -------------------------------------------------------------------------
def run_warmup(env, agent, baseline_agent, steps):
    """Fills replay buffer using the Baseline Agent on the Training Set."""
    print(f"--- Warming up Replay Buffer ({steps} steps) ---")
    state, _ = env.reset()
    processed_state = agent.process_state(state)
    
    for _ in tqdm(range(steps)):
        # Baseline Action
        baseline_act_val = baseline_agent.act(state)[0]
        
        # Map to Discrete (0: Sell, 1: Hold, 2: Buy)
        if baseline_act_val < -0.1: action_idx = 0
        elif baseline_act_val > 0.1: action_idx = 2
        else: action_idx = 1
        
        # Step
        next_state, reward, term, trunc, _ = env.step(baseline_act_val)
        done = term or trunc
        
        processed_next_state = agent.process_state(next_state)
        
        # Store
        agent.memory.add(processed_state, action_idx, reward, processed_next_state, done)
        
        state = next_state
        processed_state = processed_next_state
        
        if done:
            state, _ = env.reset()
            agent.fe.reset_history() 
            baseline_agent.price_history = []
            processed_state = agent.process_state(state)

def evaluate(agent, env):
    """Runs one full episode on the Validation Set in Greedy Mode."""
    state, _ = env.reset()
    agent.fe.reset_history() # Reset feature history for validation
    processed_state = agent.process_state(state)
    
    total_reward = 0
    done = False
    
    while not done:
        # Greedy Action (epsilon=0)
        action_idx = agent.select_action(processed_state, epsilon=0.0)
        env_action = agent.map_action_to_env(action_idx)
        
        next_state, reward, term, trunc, _ = env.step(env_action)
        done = term or trunc
        
        processed_state = agent.process_state(next_state)
        total_reward += reward
        
    return total_reward

# -------------------------------------------------------------------------
# MAIN TRAINING LOOP
# -------------------------------------------------------------------------
def train():
    # 1. Check Files
    if not os.path.exists(TRAIN_PATH) or not os.path.exists(VAL_PATH):
        print(f"Error: Could not find data files.\nLooking for: {TRAIN_PATH} and {VAL_PATH}")
        return

    # 2. Initialize Environments
    print("Loading Training Environment...")
    train_env = FixedHydroEnv(path_to_test_data=TRAIN_PATH)
    print("Loading Validation Environment...")
    val_env = FixedHydroEnv(path_to_test_data=VAL_PATH)
    
    # 3. Initialize Agents
    agent = HydroDDQNAgent(train_env, learning_rate=LR, gamma=GAMMA, device=device)
    baseline = AdvancedHydroAgent() 
    
    # 4. Warm Start (on Train Env)
    run_warmup(train_env, agent, baseline, WARMUP_STEPS)
    
    # 5. Training
    train_rewards = []
    val_rewards = []
    epsilon = EPS_START
    
    print("\n--- Starting DDQN Training ---")
    # NEW: Define how often to learn (Standard DQN practice is every 4 steps)
    LEARN_FREQUENCY = 4 

    print("\n--- Starting DDQN Training ---")
    for episode in range(MAX_EPISODES):
        
        # -- Training Episode --
        state, _ = train_env.reset()
        agent.fe.reset_history() 
        processed_state = agent.process_state(state)
        
        episode_reward = 0
        done = False
        step_count = 0 # Track steps within the episode
        
        # Get total steps for the progress bar (Total hours in your dataset)
        total_steps = len(train_env.price_values.flatten())
        
        # Create a progress bar for THIS episode
        with tqdm(total=total_steps, desc=f"Episode {episode+1}", unit="step") as pbar:
            while not done:
                # 1. Select Action
                action_idx = agent.select_action(processed_state, epsilon)
                env_action = agent.map_action_to_env(action_idx)
                
                # 2. Step Environment
                next_state, reward, term, trunc, _ = train_env.step(env_action)
                done = term or trunc
                
                # 3. Process & Store
                processed_next_state = agent.process_state(next_state)
                agent.memory.add(processed_state, action_idx, reward, processed_next_state, done)
                
                # 4. Learn (Only every 4 steps to speed up)
                if step_count % LEARN_FREQUENCY == 0:
                    agent.learn(BATCH_SIZE)
                
                # Update loop variables
                state = next_state
                processed_state = processed_next_state
                episode_reward += reward
                step_count += 1
                
                # Update Progress Bar
                pbar.update(1)
                pbar.set_postfix({'Reward': f'{episode_reward:.2f}'})

        # Epsilon Decay
        epsilon = max(EPS_END, epsilon * EPS_DECAY)
        train_rewards.append(episode_reward)
        
        # -- Validation Step --
        if (episode + 1) % VAL_INTERVAL == 0:
            val_score = evaluate(agent, val_env)
            val_rewards.append(val_score)
            
            avg_train = np.mean(train_rewards[-VAL_INTERVAL:])
            print(f"Ep {episode+1:4d} | Train Avg: {avg_train:8.2f} | Val Score: {val_score:8.2f} | Eps: {epsilon:.2f}")
        
            # Save Best Model based on Validation
            if val_score >= max(val_rewards):
                torch.save(agent.online_net.state_dict(), "ddqn_hydro_best.pth")

    # 6. Plotting
    plt.figure(figsize=(12, 5))
    plt.plot(train_rewards, label="Training Reward", alpha=0.5)
    # Plot validation dots
    val_x = list(range(VAL_INTERVAL-1, MAX_EPISODES, VAL_INTERVAL))
    plt.plot(val_x, val_rewards, 'r-o', label="Validation Reward", linewidth=2)
    
    plt.title("Hydro DDQN: Training vs Validation")
    plt.xlabel("Episode")
    plt.ylabel("Profit")
    plt.legend()
    plt.grid(True)
    plt.savefig("training_result.png")
    plt.show()

if __name__ == "__main__":
    train()