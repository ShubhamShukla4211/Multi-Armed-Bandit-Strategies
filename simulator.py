import numpy as np
import matplotlib.pyplot as plt
from bandit import MultiArmedBandit
from strategies import EpsilonGreedy, UCB, ThompsonSampling

def run_simulation(strategy_class, strategy_args, steps=1000, n_arms=10):
    bandit = MultiArmedBandit(n_arms)
    strategy = strategy_class(n_arms, **strategy_args)
    rewards = []
    optimal_arm = bandit.best_arm()
    optimal_count = 0

    for t in range(steps):
        arm = strategy.select_arm()
        reward = bandit.pull(arm)
        strategy.update(arm, reward)
        rewards.append(reward)
        if arm == optimal_arm:
            optimal_count += 1

    return rewards, optimal_count / steps

def plot_results(results, labels):
    for rewards, label in zip(results, labels):
        cumulative_avg = np.cumsum(rewards) / (np.arange(len(rewards)) + 1)
        plt.plot(cumulative_avg, label=label)
    plt.xlabel('Steps')
    plt.ylabel('Average Reward')
    plt.legend()
    plt.show()
