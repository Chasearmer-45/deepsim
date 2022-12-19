from train_model import NeuralNetwork

from basico import load_model, run_time_course
import tellurium as te
import torch
import matplotlib
import matplotlib.pyplot as plt
matplotlib.use('TkAgg')
import seaborn as sns
import pandas as pd
import numpy as np

from collections import namedtuple
import copy
import math
import random
import sys
import shutil
import time
import os

# define time step
step_sizes = [1, 5, 10, 20]
num_simulations = 2000
num_simulation_steps = 101  # choosing 101 steps allows the NN to evaluate step 100

# set plot style
sns.set_style("ticks")

for step_size in step_sizes:

    print(f'Starting step size {step_size}')

    # Create the evaluation plot directory, if necessary
    evaluation_plots_dir = "evaluation_plots/"
    if not os.path.exists(evaluation_plots_dir):
        os.mkdir(evaluation_plots_dir)

    # Create the step-specific evaluation plot directory, if necessary
    step_dir = f"{evaluation_plots_dir}step_{step_size}"
    if not os.path.exists(step_dir):
        os.mkdir(step_dir)

    # Clear the contents of the comparison plot directory if it exists, and create a new empty directory
    comparison_plots_dir = f"{step_dir}/comparison_plots"
    if os.path.exists(comparison_plots_dir):
        shutil.rmtree(comparison_plots_dir)
    os.mkdir(comparison_plots_dir)

    # Clear the contents of the distance plot directory if it exists, and create a new empty directory
    distance_plots_dir = f"{step_dir}/distance_plots"
    if os.path.exists(distance_plots_dir):
        shutil.rmtree(distance_plots_dir)
    os.mkdir(distance_plots_dir)

    # load the neural network model
    model = NeuralNetwork()
    model.load_state_dict(torch.load(f"model_step_{step_size}.p"))
    model.eval()

    # define the systems model
    r = te.loada('''
        $x0 -> s1;  (Vm1/Km1)*(x0 - s1/Keq1)/(1 + x0/Km1 + s1/Km2)
        s1 -> s2;   (Vm2/Km3)*(s1 - s2/Keq2)/(1 + s1/Km3 + s2/Km4)
        s2 -> s3;   (Vm3/Km5)*(s2 - s3/Keq3)/(1 + s2/Km5 + s3/Km6)
        s3 -> s4;   (Vm4/Km7)*(s3 - s4/Keq4)/(1 + s3/Km7 + s4/Km8)
        s4 -> s5;   (Vm5/Km9)*(s4 - s5/Keq5)/(1 + s4/Km9 + s5/Km10)
        s5 -> $x1;  (Vm6/Km11)*(s5 - x1/Keq6)/(1 + s5/Km11 + x1/Km12)
    
        Vm1 = 0; Vm2 = 0; Vm3 = 0; Vm4 = 0; Vm5 = 0; Vm6 = 0
        x0 = 0; x1 = 0
        Km1 = 0; Km2 = 0; Km3 = 0; Km4 = 0; Km5 = 0; Km6 = 0
        Km7 = 0; Km8 = 0; Km9 = 0; Km10 = 0; Km11 = 0; Km12 = 0
        Keq1 = 0
        Keq2 = 0
        Keq3 = 0
        Keq4 = 0
        Keq5 = 0
        Keq6 = 0
    ''')

    # define the dataset
    dataset = pd.DataFrame({
        "s1": list(),
        "s2": list(),
        "s3": list(),
        "s4": list(),
        "s5": list(),
        "x0": list(),
        "x1": list(),
        "vm1": list(),
        "vm2": list(),
        "vm3": list(),
        "vm4": list(),
        "vm5": list(),
        "vm6": list(),
        "km1": list(),
        "km2": list(),
        "km3": list(),
        "km4": list(),
        "km5": list(),
        "km6": list(),
        "km7": list(),
        "km8": list(),
        "km9": list(),
        "km10": list(),
        "km11": list(),
        "km12": list(),
        "keq1": list(),
        "keq2": list(),
        "keq3": list(),
        "keq4": list(),
        "keq5": list(),
        "keq6": list(),
        "y1": list(),
        "y2": list(),
        "y3": list(),
        "y4": list(),
        "y5": list(),
    })


    # for each simulation
    differences = list()
    percent_differences = list()
    simulation_times = list()
    Sim_Time = namedtuple('Sim_Time', 'neural tellurium copasi')
    for c in range(num_simulations):

        # set sink, source, and substrate parameters
        for substrate in ['x0', 'x1', 's1', 's2', 's3', 's4', 's5']:
            setattr(r, substrate, random.randrange(1000, 50000) / 1000)  # from 1 to 50 (0.001 resolution)

        # set Vm parameters
        for Vm in ['Vm1', 'Vm2', 'Vm3', 'Vm4', 'Vm5', 'Vm6']:
            setattr(r, Vm, random.randrange(1, 500) / 10)  # from 0.1 to 50 (0.1 resolution)

        # set Km parameters
        for Km in ['Km1', 'Km2', 'Km3', 'Km4', 'Km5', 'Km6', 'Km7', 'Km8', 'Km9', 'Km10', 'Km11', 'Km12']:
            setattr(r, Km, random.randrange(10, 5000) / 100)  # from 0.1 to 50 (0.01 resolution)

        # set Keq parameters
        for Keq in ['Keq1', 'Keq2', 'Keq3', 'Keq4', 'Keq5', 'Keq6']:
            setattr(r, Keq, random.randrange(10, 500) / 10)  # from 1 to 50 (0.1 resolution)

        starting_substrate_counts = copy.deepcopy([r.s1, r.s2, r.s3, r.s4, r.s5])
        params = [
            r.x0, r.x1,
            r.Vm1, r.Vm2, r.Vm3, r.Vm4, r.Vm5, r.Vm6,
            r.Km1, r.Km2, r.Km3, r.Km4, r.Km5, r.Km6, r.Km7, r.Km8, r.Km9, r.Km10, r.Km11, r.Km12,
            r.Keq1, r.Keq2, r.Keq3, r.Keq4, r.Keq5, r.Keq6
        ]

        # save sbml model
        te.saveToFile('model.xml', r.getCurrentSBML())

        # run tellurium simulation, record time
        sim_start_time_te = time.time()
        result = r.simulate(0, num_simulation_steps, num_simulation_steps)
        tellurium_time = time.time() - sim_start_time_te

        # run copasi model, record time
        copasi_model = load_model('model.xml')
        sim_start_time_co = time.time()
        df = run_time_course(model=copasi_model, duration=num_simulation_steps, step_size=1)
        copasi_time = time.time() - sim_start_time_co

        # delete model file
        os.remove('model.xml')

        # create dictionary for logging simulation results, and add the time zero datapoint
        simulations = pd.DataFrame({
            "S1": list(),
            "S2": list(),
            "S3": list(),
            "S4": list(),
            "S5": list(),
            "S1_pred": list(),
            "S2_pred": list(),
            "S3_pred": list(),
            "S4_pred": list(),
            "S5_pred": list(),
        })
        simulations.loc[len(simulations.index)] = [*result[0][1:],
                                                   *starting_substrate_counts]  # log true and pred vals

        # setup vars for NN simulation
        num_points = math.floor((num_simulation_steps - 1) / step_size)     # number of simulation steps for the NN
        neural_time = 0                                                     # a running summation to calculate time
        nn_substrate_counts = starting_substrate_counts
        model_input = torch.tensor(nn_substrate_counts + params).float()       # convert input into a torch tensor

        # run NN simulation
        for t in range(num_points):

            # run neural network and record time
            neural_start_time = time.time()
            nn_substrate_counts = model(model_input).tolist()
            neural_time += (time.time() - neural_start_time)

            # reformat values
            model_input = torch.tensor(nn_substrate_counts + params).float()

            # record tellurium and neural simulation results
            tellurium_substrate_counts = result[(t * step_size) + step_size][1:]  # get te values at the current step
            simulations.loc[len(simulations.index)] = [*tellurium_substrate_counts, *nn_substrate_counts]  # log true and pred vals

        # save sim times
        sim_times = Sim_Time(neural_time, tellurium_time, copasi_time)
        simulation_times.append(sim_times)

        #
        substrate_changes = [end - start for end, start in zip(tellurium_substrate_counts, starting_substrate_counts)]

        # append differences and percent differences
        diffs = [abs(true - pred) for true, pred in zip(tellurium_substrate_counts, nn_substrate_counts)]
        perc_diffs = [abs((true - pred) / change)*100 for true, pred, change in
                      zip(tellurium_substrate_counts, nn_substrate_counts, substrate_changes)]
        differences += diffs
        percent_differences += perc_diffs

        # save 20 plots total
        if c % (num_simulations / 20) == 0:
            # plot the tellurium and NN simulation results
            fig, ax = plt.subplots()
            ax.plot(range(num_points + 1), simulations['S1'], color='red', label='True S1')
            ax.plot(range(num_points + 1), simulations['S1_pred'], color='red', linestyle='dashed', label='Pred S1')
            ax.plot(range(num_points + 1), simulations['S2'], color='blue', label='True S2')
            ax.plot(range(num_points + 1), simulations['S2_pred'], color='blue', linestyle='dashed', label='Pred S2')
            ax.plot(range(num_points + 1), simulations['S3'], color='green', label='True S3')
            ax.plot(range(num_points + 1), simulations['S3_pred'], color='green', linestyle='dashed', label='Pred S3')
            ax.plot(range(num_points + 1), simulations['S4'], color='orange', label='True S4')
            ax.plot(range(num_points + 1), simulations['S4_pred'], color='orange', linestyle='dashed', label='Pred S4')
            ax.plot(range(num_points + 1), simulations['S5'], color='black', label='True S5')
            ax.plot(range(num_points + 1), simulations['S5_pred'], color='black', linestyle='dashed', label='Pred S5')

            ax.set_xlabel = 'Time'
            ax.set_ylabel = 'Substrate Count'
            ax.legend()
            plt.savefig(f"evaluation_plots/step_{step_size}/comparison_plots/comparison_plot_{c}.png")
            plt.close(fig)

            # plot the difference over time between the tellurium and NN simulation for each substrate
            fig, ax = plt.subplots()
            ax.plot(range(num_points + 1), simulations['S1'] - simulations['S1_pred'], color='red', label='True S1')
            ax.plot(range(num_points + 1), simulations['S2'] - simulations['S2_pred'], color='blue', label='True S2')
            ax.plot(range(num_points + 1), simulations['S3'] - simulations['S3_pred'], color='green', label='True S3')
            ax.plot(range(num_points + 1), simulations['S4'] - simulations['S4_pred'], color='orange', label='True S4')
            ax.plot(range(num_points + 1), simulations['S5'] - simulations['S5_pred'], color='black', label='True S5')

            ax.set_xlabel = 'Time'
            ax.set_ylabel = 'Substrate Count'
            ax.legend()
            plt.savefig(f"evaluation_plots/step_{step_size}/distance_plots/distance_plot_{c}.png")
            plt.close(fig)

        # print progress
        if c % (num_simulations / 10) == 0:
            print(f"{100 * (c / num_simulations)}% Complete")

    # calculate time differences and multipliers between the neural network and tellurium / copasi
    time_diffs_tellurium = [sim.neural - sim.tellurium for sim in simulation_times]
    time_multiplier_tellurium = [sim.neural / sim.tellurium for sim in simulation_times]
    time_diffs_copasi = [sim.neural - sim.copasi for sim in simulation_times]
    time_multiplier_copasi = [sim.neural / sim.copasi for sim in simulation_times]

    # print summary statistics
    print(f"Average/Median Difference: {round(np.average(differences), 4)}/{round(np.median(differences), 4)}")
    print(f"Average/Median Percent Difference: {round(np.average(percent_differences), 4)}/{round(np.median(percent_differences), 4)}")
    print(f"Average/Median Time Difference (Tellurium): {round(np.average(time_diffs_tellurium), 4)}/{round(np.median(time_diffs_tellurium), 4)}")
    print(f"Average/Median Time Multiplier (Tellurium): {round(np.average(time_multiplier_tellurium), 4)}/{round(np.median(time_multiplier_tellurium), 4)}")
    print(f"Average/Median Time Difference (copasi): {round(np.average(time_diffs_copasi), 4)}/{round(np.median(time_diffs_copasi), 4)}")
    print(f"Average/Median Time Multiplier (copasi): {round(np.average(time_multiplier_copasi), 4)}/{round(np.median(time_multiplier_copasi), 4)}")
    print()

    # write stats to summary file
    summary_file = open(f"evaluation_plots/step_{step_size}/summary.txt", 'w')
    summary_file.write(f"Average/Median Difference: {round(np.average(differences), 4)}/{round(np.median(differences), 4)}\n")
    summary_file.write(f"Average/Median Percent Difference: {round(np.average(percent_differences), 4)}/{round(np.median(percent_differences), 4)}\n")
    summary_file.write(f"Average/Median Time Difference (Tellurium): {round(np.average(time_diffs_tellurium), 4)}/{round(np.median(time_diffs_tellurium), 4)}\n")
    summary_file.write(f"Average/Median Time Multiplier (Tellurium): {round(np.average(time_multiplier_tellurium), 4)}/{round(np.median(time_multiplier_tellurium), 4)}\n")
    summary_file.write(f"Average/Median Time Difference (copasi): {round(np.average(time_diffs_copasi), 4)}/{round(np.median(time_diffs_copasi), 4)}\n")
    summary_file.write(f"Average/Median Time Multiplier (copasi): {round(np.average(time_multiplier_copasi), 4)}/{round(np.median(time_multiplier_copasi), 4)}\n")
    summary_file.close()

    # plot a histogram of substrate differences between tellurium and NN
    ax = sns.histplot(differences)
    ax.set(xlabel='Difference', title='Tellurium and Neural Network Differences')
    max_value = np.percentile(differences, 97.5)
    ax.set_xlim(-5, max_value)
    plt.axvline(np.average(differences), color='green', linestyle='--', label='Average')
    plt.axvline(np.median(differences), color='red', linestyle='--', label='Median')
    plt.legend()
    plt.savefig(f"evaluation_plots/step_{step_size}/difference_histogram.png")
    plt.clf()

    # plot a histogram of substrate percent differences between tellurium and NN
    ax = sns.histplot(percent_differences)
    max_value = np.percentile(percent_differences, 97.5)
    ax.set_xlim(-5, max_value)
    ax.set(xlabel='Percent Difference', title='Tellurium and Neural Network Percent Differences')
    plt.axvline(np.average(percent_differences), color='green', linestyle='--', label='Average')
    plt.axvline(np.median(percent_differences), color='red', linestyle='--', label='Median')
    plt.legend()
    plt.savefig(f"evaluation_plots/step_{step_size}/percent_difference_histogram.png")
    plt.clf()

    # plot a histogram of time differences between tellurium and NN
    ax = sns.histplot(time_diffs_tellurium)
    ax.set(xlabel='Difference', title='Tellurium and Neural Network Time Differences')
    plt.savefig(f"evaluation_plots/step_{step_size}/time_difference_histogram_tellurium.png")
    plt.clf()

    # plot a histogram of time multipliers between tellurium and NN
    ax = sns.histplot(time_multiplier_tellurium)
    ax.set(xlabel='Difference', title='Tellurium and Neural Network Time Multiplier')
    plt.savefig(f"evaluation_plots/step_{step_size}/time_percent_difference_histogram_tellurium.png")
    plt.clf()

    # plot a histogram of time differences between copasi and NN
    ax = sns.histplot(time_diffs_copasi)
    ax.set(xlabel='Difference', title='copasi and Neural Network Time Differences')
    plt.savefig(f"evaluation_plots/step_{step_size}/time_difference_histogram_copasi.png")
    plt.clf()

    # plot a histogramm of time multipliers between copasi and NN
    ax = sns.histplot(time_multiplier_copasi)
    ax.set(xlabel='Difference', title='copasi and Neural Network Time Multiplier')
    plt.savefig(f"evaluation_plots/step_{step_size}/time_percent_difference_histogram_copasi.png")
    plt.clf()

    # plot a histogram of the NN sim times
    ax = sns.histplot([sim.neural for sim in simulation_times], color='blue')
    ax.set(xlabel='Time', title='Neural Network Simulation Times')
    plt.savefig(f"evaluation_plots/step_{step_size}/neural_network_simulation_times.png")
    plt.clf()

    # plot a histogram of the tellurium sim times
    ax = sns.histplot([sim.tellurium for sim in simulation_times], color='orange')
    ax.set(xlabel='Time', title='Tellurium Simulation Times')
    plt.savefig(f"evaluation_plots/step_{step_size}/tellurium_simulation_times.png")
    plt.clf()

    # plot a histogram of the copasi sim times
    ax = sns.histplot([sim.copasi for sim in simulation_times], color='green')
    ax.set(xlabel='Time', title='copasi Simulation Times')
    plt.savefig(f"evaluation_plots/step_{step_size}/copasi_simulation_times.png")
    plt.clf()

    # plot all sim times
    sns.set_style("darkgrid")
    sim_time_df = pd.DataFrame(
        {
            'Neural Network': [sim.neural for sim in simulation_times],
            'Tellurium': [sim.tellurium for sim in simulation_times],
            'copasi': [sim.copasi for sim in simulation_times]
        }
    )
    histplot = sns.histplot(data=sim_time_df, bins=50, kde=False)
    histplot.set_xlabel('Time')
    histplot.set_title('Simulation Times')
    plt.savefig(f"evaluation_plots/step_{step_size}/simulation_times.png")
    plt.clf()
    plt.close('all')
    sns.set_style("ticks")

