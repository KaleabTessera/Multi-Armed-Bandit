import numpy as np


class Egreedy:
    def __init__(self, eps, mab):
        self.mab = mab
        self.num_arms = mab.num_arms
        self.Q_a = np.zeros(self.num_arms)
        self.N_a = np.zeros(self.num_arms)
        self.total_reward = 0
        self.eps = eps

    def run(self, max_steps, debug=False):
        # Store reward at each t
        self.reward_t = np.zeros(max_steps)
        num_steps = 1
        while(num_steps < max_steps):
            arm_to_pull = self.select_action(self.eps)
            current_reward = self.mab.pull_arm(
                arm_to_pull)
            self.total_reward += current_reward
            self.reward_t[num_steps] = current_reward
            if(debug):
                print(f'Current reward : {self.total_reward}')
            self.N_a[arm_to_pull] += 1
            self.Q_a[arm_to_pull] = self.Q_a[arm_to_pull] + \
                (1 / self.N_a[arm_to_pull]) * \
                (current_reward - self.Q_a[arm_to_pull])
            num_steps += 1
        return self.reward_t, self.total_reward

    def select_action(self, eps):
        s = np.random.uniform()
        # Exploit
        if(s > eps):
            # Arm with highest reward
            arm_to_pull = np.argmax(self.Q_a)
        # Explore
        else:
            # Random arm choosen
            arm_to_pull = np.random.randint(
                low=0, high=self.mab.num_arms)
        return arm_to_pull

    # Clear vars
    def clear(self):
        self.Q_a = np.zeros(self.num_arms)
        self.N_a = np.zeros(self.num_arms)
        self.total_reward = 0
