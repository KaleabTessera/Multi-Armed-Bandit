import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


class ExperimentRunner:
    def __init__(self):
        self.all_results_part1 = pd.DataFrame()
        self.all_results_part2 = pd.DataFrame()

    def runExperiments_part1(self, alg, max_steps=1000, num_runs=100, alg_name=""):
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
        self.all_results_part1 = pd.concat(
            [self.all_results_part1, pd.DataFrame(mean_per_timestep)], axis=0)

    def plot_part1(self):
        ax = self.all_results_part1.T.plot(
            title="Average Rewards over time for different MAB algorithms")
        ax.set_ylabel(f'Average reward over {self.num_runs} runs')
        ax.set_xlabel(f'Steps')
        plt.show()

    def runExperiments_part2(self, algs, max_steps=1000, alg_name=""):
        self.max_steps = max_steps
        results_list = []
        all_mean_value_per_alg = []
        for alg_impl in algs:
            results_t, total_results = alg_impl.run(max_steps)
            mean_per_alg = results_t.mean(axis=0)
            all_mean_value_per_alg.append(mean_per_alg)
        all_mean_value_per_alg_df = pd.DataFrame(all_mean_value_per_alg)
        all_mean_value_per_alg_df['alg_name'] = alg_name
        self.all_results_part2 = pd.concat(
            [self.all_results_part2, pd.DataFrame(all_mean_value_per_alg)], axis=1)

        # self.all_results_part2.plot()
        # plt.show()

    def plot_part2(self):
        ax = self.all_results_part2.plot(
            title="Summary comparison of Algorithms")
        ax.set_ylabel(f'Average reward over {self.max_steps} runs')
        ax.set_xlabel(f'Hyparameters (E-greedy : e , Greedy : Q0/Q1, UCB:c)')
        plt.show()
