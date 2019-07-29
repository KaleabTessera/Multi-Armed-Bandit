
from algorithms.Egreedy import Egreedy
from classes.MultiArmBandit import MultiArmBandit
from helpers.ExperimentRunner import ExperimentRunner
import argparse


def main():
    parser = argparse.ArgumentParser(description='Multi-Armed Bandit')
    parser.add_argument('--num-arms', type=int, default=10, metavar='N',
                        help='Number of arms of MAB.')
    parser.add_argument('--num-runs', type=int, default=10, metavar='N',
                        help='Number of runs to repeat and average for each algorithm.')
    args = parser.parse_args()
    mab = MultiArmBandit(args.num_arms)

    print("Eps Greedy")
    eps_greedy = Egreedy(eps=0.1, mab=mab)
    experiment_e_greedy = ExperimentRunner(eps_greedy)
    experiment_e_greedy.runExperiments(max_steps=1000, num_runs=100)


if __name__ == '__main__':
    main()
