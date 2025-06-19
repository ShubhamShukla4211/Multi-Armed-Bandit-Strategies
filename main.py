from simulator import run_simulation, plot_results
from strategies import EpsilonGreedy, UCB, ThompsonSampling

def main():
    steps = 5000
    n_arms = 10

    eps_rewards, eps_opt = run_simulation(EpsilonGreedy, {'epsilon': 0.1}, steps, n_arms)
    ucb_rewards, ucb_opt = run_simulation(UCB, {}, steps, n_arms)
    ts_rewards, ts_opt = run_simulation(ThompsonSampling, {}, steps, n_arms)

    print(f"Epsilon-Greedy optimal arm selection rate: {eps_opt:.3f}")
    print(f"UCB optimal arm selection rate: {ucb_opt:.3f}")
    print(f"Thompson Sampling optimal arm selection rate: {ts_opt:.3f}")

    plot_results([eps_rewards, ucb_rewards, ts_rewards],
                 ['Epsilon-Greedy', 'UCB', 'Thompson Sampling'])

if __name__ == "__main__":
    main()
