import numpy as np
import math


class MultiArmBandit:
    def __init__(self, num_arms, debug=False):
        self.num_arms = num_arms
        self.possible_actions = np.zeros(num_arms)
        self.q_a = np.zeros(num_arms)

        # Ini mean values
        for i in np.arange(num_arms):
            self.q_a[i] = np.random.normal(loc=0, scale=math.sqrt(1))

    def pull_arm(self, index_arm, debug=False):
        r_t = np.random.normal(loc=self.q_a[index_arm], scale=math.sqrt(1))
        return r_t

    def print_actions(self):
        print(f'Possible actions - {self.possible_actions}')
