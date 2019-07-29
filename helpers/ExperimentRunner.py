import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


class ExperimentRunner:
    def __init__(self, alg):
        self.alg = alg

    def runExperiments(self, max_steps=1000, num_runs=100):
        results_t_list = pd.DataFrame()
        for i in np.arange(num_runs):
            results_t, total_results = self.alg.run(max_steps)
            print(results_t.shape)
            results_t_list = pd.concat(
                [results_t_list, pd.DataFrame(results_t)], axis=1)
            self.alg.clear()
        mean_per_timestep = results_t_list.T.mean(axis=0)
        print(mean_per_timestep.shape)
        mean_per_timestep.plot()
        plt.show()
