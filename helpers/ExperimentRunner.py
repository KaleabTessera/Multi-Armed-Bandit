import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


class ExperimentRunner:
    def __init__(self):
        self.all_results = pd.DataFrame()

    def runExperiments(self, alg, max_steps=1000, num_runs=100, alg_name=""):
        self.num_runs = num_runs
        results_t_list = pd.DataFrame()
        for i in np.arange(num_runs):
            results_t, total_results = alg.run(max_steps)
            results_t_list = pd.concat(
                [results_t_list, pd.DataFrame(results_t)], axis=1)
            alg.clear()
        mean_per_timestep = results_t_list.mean(axis=1)
        mean_per_timestep = mean_per_timestep.reset_index().T.drop('index')
        mean_per_timestep['alg_name'] = alg_name
        mean_per_timestep = mean_per_timestep.set_index('alg_name')
        self.all_results = pd.concat(
            [self.all_results, pd.DataFrame(mean_per_timestep)], axis=0)

    def plot(self):
        ax = self.all_results.T.plot(
            title="Average Rewards over time for different MAB algorithms")
        ax.set_ylabel(f'Average reward over {self.num_runs} runs')
        ax.set_xlabel(f'Steps')
        plt.show()
