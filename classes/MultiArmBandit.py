import numpy as np
import math


class MultiArmBandit:
    def __init__(self, num_arms):
        self.num_arms = num_arms
        self.possible_actions = np.zeros(num_arms)
        self.r_t = np.zeros(num_arms)

        # Ini values
        # TODO check var /sd
        for i in np.arange(num_arms):
            q_a = np.random.normal(loc=0, scale=math.sqrt(1))
            self.r_t[i] = np.random.normal(
                loc=q_a, scale=math.sqrt(1))
        print(self.r_t)

    def pull_arm(self, index_arm):
        print(f'Earned reward : {self.r_t[index_arm]}')
        return self.r_t[index_arm]

    def print_rewards(self):
        print(f'Previous Reward - self.possible_actions')

    def print_actions(self):
        print(f'Possible actions - {self.possible_actions}')
