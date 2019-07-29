
from algorithms.Egreedy import Egreedy
from algorithms.Greedy import Greedy
from classes.MultiArmBandit import MultiArmBandit
from helpers.ExperimentRunner import ExperimentRunner
import argparse
import matplotlib.pyplot as plt


def main():
    parser = argparse.ArgumentParser(description='Multi-Armed Bandit')
    parser.add_argument('--num-arms', type=int, default=10, metavar='N',
                        help='Number of arms of MAB.')
    parser.add_argument('--num-runs', type=int, default=10, metavar='N',
                        help='Number of runs to repeat and average for each algorithm.')
    args = parser.parse_args()
    mab = MultiArmBandit(args.num_arms)

    eps = 0.1
    eps_greedy = Egreedy(eps=eps, mab=mab)
    experiment_runner = ExperimentRunner()
    experiment_runner.runExperiments(alg=eps_greedy,
                                     max_steps=1000, num_runs=100, alg_name=f'e-greedy with e={eps}')

    q1 = 5
    greedy_optimistic_ini = Greedy(Q1=q1, mab=mab)
    experiment_runner.runExperiments(alg=greedy_optimistic_ini,
                                     max_steps=1000, num_runs=100, alg_name=f'Greed with Q1={q1}')
    experiment_runner.plot()
    plt.show()


if __name__ == '__main__':
    main()
