import numpy as np


class Greedy:
    def __init__(self, Q1, mab):
        self.mab = mab
        self.num_arms = mab.num_arms
        self.Q_a = np.full((self.num_arms), Q1)
        self.N_a = np.zeros(self.num_arms)
        self.total_reward = 0

    def run(self, max_steps, debug=False):
        # Store reward at each t
        self.reward_t = np.zeros(max_steps)
        num_steps = 0
        while(num_steps < max_steps):
            arm_to_pull = self.select_action()
            self.total_reward += self.mab.pull_arm(
                arm_to_pull)
            self.reward_t[num_steps] = self.total_reward
            if(debug):
                print(f'Current reward : {self.total_reward}')
            self.N_a[arm_to_pull] += 1
            self.Q_a[arm_to_pull] = self.Q_a[arm_to_pull] + \
                (1 / self.N_a[arm_to_pull]) * \
                (self.total_reward - self.Q_a[arm_to_pull])
            num_steps += 1
        return self.reward_t, self.total_reward

    def select_action(self):
        # Exploit
        # Arm with highest reward
        arm_to_pull = np.argmax(self.mab.r_t)
        return arm_to_pull

    # Clear vars
    def clear(self):
        self.Q_a = np.zeros(self.num_arms)
        self.N_a = np.zeros(self.num_arms)
        self.total_reward = 0
