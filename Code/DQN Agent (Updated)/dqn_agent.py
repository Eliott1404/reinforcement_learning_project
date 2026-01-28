import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import random
from collections import deque
import pandas as pd

# -------------------------------------------------------------------------
# 1. Feature Engineering (The Baseline Logic)
# -------------------------------------------------------------------------
class AdvancedFeatureEngineer:
    """
    Incorporates logic from the AdvancedHydroAgent to augment the state 
    fed into the Neural Network.
    """
    def __init__(self, rsi_window=14, fft_window=168, max_vol=100000.0):
        self.rsi_window = rsi_window
        self.fft_window = fft_window
        self.fft_cutoff = 10
        self.max_volume = max_vol
        self.price_history = []

    def reset_history(self):
        self.price_history = []

    def get_rsi(self):
        if len(self.price_history) < self.rsi_window + 1: return 50.0
        # Use only the tail needed for calculation to be fast
        window = pd.Series(self.price_history[-self.rsi_window-1:])
        delta = window.diff().dropna()
        gain = (delta.where(delta > 0, 0)).mean()
        loss = (-delta.where(delta < 0, 0)).mean()
        if loss == 0: return 100.0
        rs = gain / loss
        return 100 - (100 / (1 + rs))

    def get_fft_trend(self):
        if len(self.price_history) < self.fft_window: 
            return self.price_history[-1] if len(self.price_history) > 0 else 0
        y = np.array(self.price_history[-self.fft_window:])
        fft_coeffs = np.fft.rfft(y)
        fft_coeffs[self.fft_cutoff:] = 0 
        return np.fft.irfft(fft_coeffs, n=len(y))[-1]

    def get_seasonal_deviation(self, vol, month):
        # Target % based on AdvancedAgent logic
        target = 0.0
        if month in [10, 11, 12, 1, 2, 3]: target = 0.90
        elif month in [4, 5]: target = 0.20
        else: target = 0.0 # Summer
        
        current_fill = vol / self.max_volume
        # Positive = Too much water, Negative = Too little
        return current_fill - target

    def process(self, observation):
        """
        Takes raw env observation, updates history, and returns Augmented State.
        Raw Obs: [vol, price, hour, dow, doy, month, year]
        """
        vol, price, hour, dow, doy, month, year = observation
        self.price_history.append(price)

        # 1. Calculate Baseline Features
        rsi = self.get_rsi() / 100.0 # Normalize 0-1
        trend = self.get_fft_trend()
        trend_diff = (price - trend) / (trend + 1e-5) # % diff from trend
        season_dev = self.get_seasonal_deviation(vol, month)

        # 2. Return Augmented State
        # We keep raw vars (normalized roughly) + Expert Features
        # Obs: [Vol%, Price, RSI, TrendDiff, SeasonDev, Hour/24, Month/12]
        aug_state = np.array([
            vol / self.max_volume,
            price, # Raw price is hard to normalize without knowing max, usually fine for NN or use BatchNorm
            rsi,
            trend_diff,
            season_dev,
            hour / 24.0,
            month / 12.0
        ], dtype=np.float32)
        
        return aug_state

# -------------------------------------------------------------------------
# 2. Neural Network
# -------------------------------------------------------------------------
class DQN(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(DQN, self).__init__()
        self.fc1 = nn.Linear(input_dim, 128)
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, 64)
        self.fc4 = nn.Linear(64, output_dim)
        
    def forward(self, x):
        # LeakyReLU is often better for RL stability than Tanh
        x = F.leaky_relu(self.fc1(x))
        x = F.leaky_relu(self.fc2(x))
        x = F.leaky_relu(self.fc3(x))
        return self.fc4(x)

# -------------------------------------------------------------------------
# 3. Experience Replay
# -------------------------------------------------------------------------
class ExperienceReplay:
    def __init__(self, buffer_size, device):
        self.buffer = deque(maxlen=buffer_size)
        self.device = device

    def add(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        
        return (
            torch.tensor(np.array(states), dtype=torch.float32, device=self.device),
            torch.tensor(actions, dtype=torch.int64, device=self.device).unsqueeze(1),
            torch.tensor(rewards, dtype=torch.float32, device=self.device).unsqueeze(1),
            torch.tensor(np.array(next_states), dtype=torch.float32, device=self.device),
            torch.tensor(dones, dtype=torch.float32, device=self.device).unsqueeze(1)
        )
    
    def __len__(self):
        return len(self.buffer)

# -------------------------------------------------------------------------
# 4. Double DQN Agent
# -------------------------------------------------------------------------
class HydroDDQNAgent:
    def __init__(self, env, learning_rate=1e-4, gamma=0.99, buffer_size=50000, 
                 tau=0.005, device='cpu'):
        self.env = env
        self.gamma = gamma
        self.tau = tau
        self.device = device
        
        # Initialize Feature Engineer
        self.fe = AdvancedFeatureEngineer()
        
        # Determine input size by processing one dummy state
        dummy_obs = np.zeros(7) # Based on TestEnv observation shape
        aug_state = self.fe.process(dummy_obs)
        self.input_dim = len(aug_state)
        self.output_dim = 3 # Actions: Sell (-1), Hold (0), Buy (1)

        # Networks
        self.online_net = DQN(self.input_dim, self.output_dim).to(device)
        self.target_net = DQN(self.input_dim, self.output_dim).to(device)
        self.target_net.load_state_dict(self.online_net.state_dict())
        
        self.optimizer = torch.optim.Adam(self.online_net.parameters(), lr=learning_rate)
        self.memory = ExperienceReplay(buffer_size, device)
        
        # Reset FE history for fresh start
        self.fe.reset_history()

    def process_state(self, observation):
        # Wrapper to use FE
        return self.fe.process(observation)

    def map_action_to_env(self, action_idx):
        # Map Neural Net output (0, 1, 2) -> Env Action (-1, 0, 1)
        # 0 -> -1 (Sell)
        # 1 ->  0 (Hold)
        # 2 ->  1 (Buy)
        mapping = {0: -1.0, 1: 0.0, 2: 1.0}
        return mapping[action_idx]
    
    def map_env_to_action(self, env_action):
        # Inverse mapping for pre-training (if needed)
        # -1 -> 0, 0 -> 1, 1 -> 2
        mapping = {-1.0: 0, 0.0: 1, 1.0: 2}
        return mapping[env_action]

    def select_action(self, state, epsilon=0.0):
        if random.random() < epsilon:
            return random.randint(0, self.output_dim - 1)
        
        with torch.no_grad():
            state_t = torch.tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0)
            q_values = self.online_net(state_t)
            return q_values.argmax(dim=1).item()

    def learn(self, batch_size):
        if len(self.memory) < batch_size:
            return
        
        states, actions, rewards, next_states, dones = self.memory.sample(batch_size)
        
        # Double DQN Logic
        # 1. Select best action using Online Net
        with torch.no_grad():
            next_state_actions = self.online_net(next_states).argmax(dim=1, keepdim=True)
            # 2. Evaluate that action using Target Net
            next_q_values = self.target_net(next_states).gather(1, next_state_actions)
            
        target_q = rewards + (self.gamma * next_q_values * (1 - dones))
        
        current_q = self.online_net(states).gather(1, actions)
        
        loss = F.mse_loss(current_q, target_q)
        
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.online_net.parameters(), 1.0) # Gradient clipping for stability
        self.optimizer.step()
        
        # Soft Update Target Network
        for target_param, online_param in zip(self.target_net.parameters(), self.online_net.parameters()):
            target_param.data.copy_(self.tau * online_param.data + (1.0 - self.tau) * target_param.data)

        return loss.item()