
from algorithms.Egreedy import Egreedy
from algorithms.Greedy import Greedy
from algorithms.UpperConfidenceBound import UpperConfidenceBound
from classes.MultiArmBandit import MultiArmBandit
from helpers.ExperimentRunner import ExperimentRunner
import argparse
import matplotlib.pyplot as plt
import numpy as np


def main():
    np.random.seed(42)

    parser = argparse.ArgumentParser(description='Multi-Armed Bandit')
    parser.add_argument('--num-arms', type=int, default=10, metavar='N',
                        help='Number of arms of MAB.')
    parser.add_argument('--num-runs', type=int, default=10, metavar='N',
                        help='Number of runs to repeat and average for each algorithm.')
    args = parser.parse_args()
    mab = MultiArmBandit(args.num_arms)
    experiment_runner = ExperimentRunner()

    # Part 1 - A plot of reward over time (averaged over 100 runs each) on the
    # same axes, for 𝜖-greedy with 𝜖 = 0.1, greedy with 𝑄1 = 5, and
    # UCB with 𝑐 = 2
    eps = 0.1
    eps_greedy = Egreedy(eps=eps, mab=mab)
    experiment_runner.runExperiments_part1(alg=eps_greedy,
                                           max_steps=1000, num_runs=100, alg_name=f'e-greedy with e={eps}')

    q1 = 5
    greedy_optimistic_ini = Greedy(Q1=q1, mab=mab)
    experiment_runner.runExperiments_part1(alg=greedy_optimistic_ini,
                                           max_steps=1000, num_runs=100, alg_name=f'Greedy with Q1={q1}')

    c = 2
    upper_bound_conf = UpperConfidenceBound(c=c, mab=mab)
    experiment_runner.runExperiments_part1(alg=upper_bound_conf,
                                           max_steps=1000, num_runs=100, alg_name=f'UCB with c={c}')

    experiment_runner.plot_part1()
    # Part 2 - A summary comparison plot of rewards over first 1000 steps for
    # the three algorithms with different values of the hyperparameters

    # E - greedy
    eps_greedy_range = [1/128, 1/64, 1/32, 1/16, 1/4, 1/2]
    eps_greedy_algs = []
    for eps in eps_greedy_range:
        eps_greedy = Egreedy(eps=eps, mab=mab)
        eps_greedy_algs.append(eps_greedy)
    eps_greedy_results = experiment_runner.runExperiments_part2(algs=eps_greedy_algs, exp_range=eps_greedy_range,
                                                                max_steps=1000,  alg_name='e-greedy')

    # Greedy with 𝑄1
    q1_range = [1/32, 1/16, 1/8, 1/4, 1/2, 1, 2]
    greedy_algs = []
    for q1 in q1_range:
        greedy_optimistic_ini = Greedy(Q1=q1, mab=mab)
        greedy_algs.append(greedy_optimistic_ini)
    greedy_results = experiment_runner.runExperiments_part2(algs=greedy_algs, exp_range=q1_range,
                                                            max_steps=1000,  alg_name='greedy')
    # UCB
    c_range = [1/16, 1/8, 1/4, 1/2, 1, 2, 4]
    upper_bound_conf_algs = []
    for c in c_range:
        upper_bound_conf = UpperConfidenceBound(c=c, mab=mab)
        upper_bound_conf_algs.append(upper_bound_conf)
    ucb_results = experiment_runner.runExperiments_part2(algs=upper_bound_conf_algs, exp_range=c_range,
                                                         max_steps=1000, alg_name='ucb')
    # plot
    experiment_runner.plot_part2(
        eps_greedy_results, greedy_results, ucb_results)
    plt.show()


if __name__ == '__main__':
    main()
