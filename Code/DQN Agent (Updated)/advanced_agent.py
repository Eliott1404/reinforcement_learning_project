import numpy as np
import pandas as pd

class AdvancedHydroAgent:
    def __init__(self, 
                 rsi_window=14, 
                 fft_window=168, 
                 vol_sensitivity=10.0, 
                 seasonal_factor=5.0):
        """
        Args:
            rsi_window (int): Lookback for RSI.
            fft_window (int): Lookback for FFT Trend.
            vol_sensitivity (float): How strongly Dam Level affects decisions. 
                                     Higher = more aggressive correction when empty/full.
            seasonal_factor (float): How strongly Month affects the target volume.
        """
        self.rsi_window = rsi_window
        self.fft_window = fft_window
        self.fft_cutoff = 10  # Keep low frequency trend
        
        # Risk Management Parameters
        self.vol_sensitivity = vol_sensitivity
        self.seasonal_factor = seasonal_factor
        
        self.price_history = []
        
        # Max Volume from TestEnv (100k m3)
        self.max_volume = 100000.0

    def get_rsi(self, prices):
        if len(prices) < self.rsi_window + 1: return 50.0
        window = pd.Series(prices[-self.rsi_window-1:])
        delta = window.diff().dropna()
        gain = (delta.where(delta > 0, 0)).mean()
        loss = (-delta.where(delta < 0, 0)).mean()
        if loss == 0: return 100.0
        rs = gain / loss
        return 100 - (100 / (1 + rs))

    def get_fft_trend(self, prices):
        if len(prices) < self.fft_window: return prices[-1]
        y = np.array(prices[-self.fft_window:])
        fft_coeffs = np.fft.rfft(y)
        fft_coeffs[self.fft_cutoff:] = 0 # Low-pass filter
        return np.fft.irfft(fft_coeffs, n=len(y))[-1]

    def get_seasonal_target(self, month):
        """
        Returns the 'Target' fill % for the dam based on the season.
        Strategy: Build inventory in Autumn (Oct-Dec) for Winter peaks.
                  Run lower in Summer.
        """
        # Simple Seasonality Map (Month -> Target %)
        if month in [10, 11, 12, 1, 2, 3]: # Winter/Peak Season
            return 0.90 # Target 80% full to capture massive spikes
        elif month in [4, 5]: # Spring
            return 0.20
        else: # Summer (June-Sept)
            return 0.0 # Target 40% (Prices low, keep room for pumping)

    def act(self, observation):
        """
        Observation: [volume, price, hour, day_of_week, day_of_year, month, year]
        """
        vol, price, hour, dow, doy, month, year = observation
        
        self.price_history.append(price)
        
        # 1. Technical Indicators
        rsi = self.get_rsi(self.price_history)
        trend = self.get_fft_trend(self.price_history)
        
        # 2. Risk & Volume Management
        # Calculate how far we are from our 'Seasonal Target'
        current_fill = vol / self.max_volume
        target_fill = self.get_seasonal_target(month)
        
        # deviation > 0 means we have TOO MUCH water -> Be eager to sell
        # deviation < 0 means we have TOO LITTLE water -> Be eager to buy
        fill_deviation = current_fill - target_fill 
        
        # 3. Dynamic Threshold Adjustment
        # Base RSI Thresholds
        buy_rsi_limit = 40 
        sell_rsi_limit = 60
        
        # Apply Risk Factors
        # If we have too much water (deviation > 0), we RAISE buy limit (harder to buy)
        # and LOWER sell limit (easier to sell).
        
        # Example: Vol Sensitivity = 20. Deviation = +0.2 (20% extra water).
        # Shift = 20 * 0.2 = 4 points.
        # New Sell Limit = 70 - 4 = 66 (Sell earlier!)
        
        threshold_shift = self.vol_sensitivity * fill_deviation
        
        final_buy_limit = buy_rsi_limit - threshold_shift
        final_sell_limit = sell_rsi_limit - threshold_shift
        
        # 4. Final Decision
        action = 0.0
        
        # Buy Signal: Price below trend AND RSI is "low enough"
        if price < trend and rsi < final_buy_limit:
            action = 1.0 # Pump
            
        # Sell Signal: Price above trend AND RSI is "high enough"
        elif price > trend and rsi > final_sell_limit:
            action = -1.0 # Release
            
        return np.array([action], dtype=np.float32)