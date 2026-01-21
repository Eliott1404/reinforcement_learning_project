# Baseline: thresholds driven by RSI + price derivatives

from TestEnv import HydroElectric_Test
import argparse
import matplotlib.pyplot as plt
import numpy as np
from collections import deque


def rsi_from_prices(prices, period=14):
    if len(prices) < period + 1:
        return 50.0
    diffs = np.diff(np.array(prices)[-(period + 1):])
    gains = np.clip(diffs, 0, None).mean()
    losses = (-np.clip(diffs, None, 0)).mean()
    if losses == 0:
        return 100.0
    rs = gains / losses
    return 100.0 - (100.0 / (1.0 + rs))


def run_file(excel_file, rsi_period, rsi_low, rsi_high, deriv_win, deriv_z, plot=False):
    env = HydroElectric_Test(path_to_test_data=excel_file)

    prices = deque(maxlen=max(deriv_win, rsi_period) + 2)
    deltas = deque(maxlen=deriv_win)

    total_reward = 0.0
    cumulative = []
    dam_level = []  # <--- NEW

    obs = env.observation()
    prev_price = None

    for _ in range(730 * 24 - 1):
        dam_level.append(float(obs[0]))  # <--- NEW (store current volume)

        price = float(obs[1])

        if prev_price is not None:
            deltas.append(price - prev_price)
        prev_price = price

        prices.append(price)
        rsi = rsi_from_prices(prices, period=rsi_period)

        if len(deltas) >= 5:
            d = deltas[-1]
            mu = float(np.mean(deltas))
            sd = float(np.std(deltas)) + 1e-9
            dz = (d - mu) / sd
        else:
            dz = 0.0

        if rsi >= rsi_high or dz >= deriv_z:
            action = np.array([-1.0], dtype=np.float32)  # release
        elif rsi <= rsi_low or dz <= -deriv_z:
            action = np.array([1.0], dtype=np.float32)   # pump
        else:
            action = np.array([0.0], dtype=np.float32)   # hold

        obs, reward, terminated, truncated, _ = env.step(action)
        total_reward += float(reward)

        if plot:
            cumulative.append(total_reward)

        if terminated or truncated:
            break

    if plot:
        # Plot dam level like your friend
        plt.plot(dam_level)
        plt.xlim(0, 1000)
        plt.xlabel("Time (Hours)")
        plt.ylabel("Dam level (volume)")
        plt.show()

        # If you ALSO want cumulative reward, uncomment this:
        # plt.figure()
        # plt.plot(cumulative)
        # plt.xlabel("Time (Hours)")
        # plt.ylabel("Cumulative reward")
        # plt.show()

    return total_reward


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_file", type=str, default="../Data/train.xlsx")
    parser.add_argument("--validate_file", type=str, default="../Data/validate.xlsx")
    parser.add_argument("--rsi_period", type=int, default=14)
    parser.add_argument("--rsi_low", type=float, default=30.0)
    parser.add_argument("--rsi_high", type=float, default=70.0)
    parser.add_argument("--deriv_win", type=int, default=24)
    args = parser.parse_args()

    # ---- TRAIN: tune only deriv_z (keep it simple) ----
    deriv_z_candidates = [1.5, 2.0, 2.5]
    best_z, best_train = None, -1e18
    for z in deriv_z_candidates:
        score = run_file(args.train_file, args.rsi_period, args.rsi_low, args.rsi_high, args.deriv_win, z, plot=False)
        if score > best_train:
            best_train, best_z = score, z
    print(f"[TRAIN] best deriv_z={best_z}  reward={best_train}")

    # ---- VALIDATE: evaluate once with tuned deriv_z ----
    val_score = run_file(args.validate_file, args.rsi_period, args.rsi_low, args.rsi_high, args.deriv_win, best_z, plot=True)
    print(f"[VALIDATE] reward={val_score}  (using deriv_z={best_z})")


if __name__ == "__main__":
    main()
