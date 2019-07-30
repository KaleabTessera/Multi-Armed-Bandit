
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

    eps = 0.1
    eps_greedy = Egreedy(eps=eps, mab=mab)
    experiment_runner.runExperiments(alg=eps_greedy,
                                     max_steps=1000, num_runs=100, alg_name=f'e-greedy with e={eps}')

    # mab = MultiArmBandit(args.num_arms)
    q1 = 5
    greedy_optimistic_ini = Greedy(Q1=q1, mab=mab)
    experiment_runner.runExperiments(alg=greedy_optimistic_ini,
                                     max_steps=1000, num_runs=100, alg_name=f'Greedy with Q1={q1}')

    c = 2
    upper_bound_conf = UpperConfidenceBound(c=q1, mab=mab)
    experiment_runner.runExperiments(alg=upper_bound_conf,
                                     max_steps=1000, num_runs=100, alg_name=f'UCB with c={c}')
    experiment_runner.plot()


if __name__ == '__main__':
    main()
