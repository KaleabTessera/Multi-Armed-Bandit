import numpy as np
import math


class MultiArmBandit:
    def __init__(self, num_arms, debug=False):
        self.num_arms = num_arms
        self.possible_actions = np.zeros(num_arms)
        self.r_t = np.zeros(num_arms)

        # Ini values
        for i in np.arange(num_arms):
            q_a = np.random.normal(loc=0, scale=math.sqrt(1))
            self.r_t[i] = np.random.normal(
                loc=q_a, scale=math.sqrt(1))
        if(debug):
            print(self.r_t)

    def pull_arm(self, index_arm, debug=False):
        if(debug):
            print(f'Earned reward : {self.r_t[index_arm]}')
        return self.r_t[index_arm]

    def print_actions(self):
        print(f'Possible actions - {self.possible_actions}')
