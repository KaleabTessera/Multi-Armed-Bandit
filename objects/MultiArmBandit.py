import numpy as np
import math


class MultiArmBandit:
    def __init__(self, num_arms):
        self.num_arms = num_arms
        self.possible_actions = np.zeros(num_arms)
        self.last_actions = np.zeros(num_arms)
        self.r_t = np.zeros(num_arms)
        self.q_t = np.zeros(num_arms)

        # Ini values
        # TODO check var /sd
        for i in np.arange(num_arms):
            q_t = np.random.normal(loc=0, scale=math.sqrt(1))
            self.r_t[i] = np.random.normal(
                loc=q_t, scale=math.sqrt(1))

    # index_arm - starting at 0
    def pull_arm(self, index_arm):
        self.last_actions[index_arm] = 1
        print(f'Selected action - {self.last_actions}')

        q_t = np.random.normal(loc=0, scale=math.sqrt(1))
        self.q_t[index_arm] = q_t
        self.r_t[index_arm] = np.random.normal(
            loc=q_t, scale=math.sqrt(1))

        print(f'Reward received - {self.r_t[index_arm]}')
        return self.r_t[index_arm]

    def print_rewards(self):
        print(f'Previous Reward - self.possible_actions')

    def print_actions(self):
        print(f'Possible actions - {self.possible_actions}')
