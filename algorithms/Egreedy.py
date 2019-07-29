import numpy as np


class Egreedy:
    def __init__(self, mab):
        self.mab = mab

    def run(self, eps):
        while(True):
            arm_to_pull = self.select_action(eps)
            print(arm_to_pull)

    def select_action(self, eps):
        s = np.random.uniform()
        # Exploit
        if(s > eps):
            return np.argmax(self.mab.r_t)
        # Explore
        else:
            return np.random.random_integers(low=0, high=self.mab.num_arms)
