
from algorithms.Egreedy import Egreedy
from objects.MultiArmBandit import MultiArmBandit
import argparse


def main():
    parser = argparse.ArgumentParser(description='PyTorch MNIST Example')
    parser.add_argument('--num-arms', type=int, default=10, metavar='N',
                        help='Number of arms of MAB.')
    args = parser.parse_args()
    mab = MultiArmBandit(args.num_arms)
    # mab.print_actions()
    # index_arm = int(
    #     input(f'Please choose an arm to pull (0-{args.num_arms}):'))
    # mab.pull_arm(index_arm)

    print("Eps Greedy")
    eps_greedy = Egreedy(mab)
    eps_greedy.run(0.2)


if __name__ == '__main__':
    main()
