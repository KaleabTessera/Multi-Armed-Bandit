import numpy as np
import pandas as pd


class ExperimentRunner:
    def __init__(self, alg):
        self.alg = alg

    def runExperiments(self, max_steps=1000, num_runs=100, alg_name=""):
        results_t_list = pd.DataFrame()
        for i in np.arange(num_runs):
            results_t, total_results = self.alg.run(max_steps)
            results_t_list = pd.concat(
                [results_t_list, pd.DataFrame(results_t)], axis=1)
            self.alg.clear()
        mean_per_timestep = results_t_list.mean(axis=1)
        ax = mean_per_timestep.plot(
            label=f'{alg_name}', title="Average reward over time")
        ax.set_ylabel(f'Average reward over {num_runs} runs')
        ax.set_xlabel(f'Steps')
