import tellurium as te
import pandas as pd

import copy
import random
import sys


def completion_check(result, index):
    """
    Checks if the simulation has run to completion at the index row of the result table.
    Completion is meant to encompass both convergence and repeated values from oscillation.
    Therefore, a simulation is considered completed at the index if all of the values at the index row are
    similar enough by some threshold (difference of < 1%) to another row which has been seen previously.
    In order to calculate the percent difference, we need a normalization factor with which to calculate percent
    changes. This is because differential equation (such as S1->S2; K1*S1, K1=0.1) will always change by the
    values of K in unbalanced networks. The normalization factor we use is the total amount between all states
    divided by the number of states; otherwise referred to as the average amount per state.
    The purpose of the completion check is to ensure that our dataset does not become overly saturated with the
    identity transformations of converged systems, where the input and output datapoints are equivalent.

    :param result: Array containing all of the simulation data (each row is the amount for each state) over time
    :param index: Index row to determine completion for.
    :return: True or False; True if Completed
    """
    query_row = result[index]
    query_row = query_row[1:]  # exclude time value

    # normalization factor is used to determine when the simulation is complete
    normalization_factor = sum(query_row) / len(query_row)

    for r in range(index):
        other_row = result[r][1:]  # exclude time value

        # check if all values differ by less than 1%
        for v1, v2 in zip(query_row, other_row):

            # skip initial value of zero because division
            if v1 == 0:
                continue

            # if any value differ by more than 1%, break from this loop and continue
            percent_change = abs((v2 - v1) / normalization_factor)
            if percent_change > 0.01:
                break

            # if the break did not trigger, that means all changes are less than 1% and we should return True
            return True

    return False

# define step size (how many simulation steps ahead the model should predict [default: 1])
step_sizes = [1, 5, 10, 20]

# define number of simulations and steps per simulation
num_simulations = 10000
num_OOD_simulations = 2000
num_simulation_steps = 100

for step_size in step_sizes:

    print(f'Starting step size {step_size}')

    # define the model
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

    OOD_dataset = copy.deepcopy(dataset)

    # for each simulation
    for c in range(num_simulations + num_OOD_simulations):

        # set sink, source, and substrate parameters
        for substrate in ['x0', 'x1', 's1', 's2', 's3', 's4', 's5']:
            setattr(r, substrate, random.uniform(1, 50))  # from 1 to 50

        # set Vm parameters
        for Vm in ['Vm1', 'Vm2', 'Vm3', 'Vm4', 'Vm5', 'Vm6']:
            setattr(r, Vm, random.randrange(1, 500)/10)  # from 0.1 to 50 (0.1 resolution)

        # set Km parameters
        for Km in ['Km1', 'Km2', 'Km3', 'Km4', 'Km5', 'Km6', 'Km7', 'Km8', 'Km9', 'Km10', 'Km11', 'Km12']:
            setattr(r, Km, random.randrange(10, 5000)/100)  # from 0.1 to 50 (0.01 resolution)

        # set Keq parameters
        for Keq in ['Keq1', 'Keq2', 'Keq3', 'Keq4', 'Keq5', 'Keq6']:
            setattr(r, Keq, random.randrange(10, 500)/10)  # from 1 to 50 (0.1 resolution)

        # run simulation
        result = r.simulate(0, num_simulation_steps, num_simulation_steps)

        # loop through each time step in the simulation
        for i in range(len(result) - step_size):

            # if the simulation step has reached completion, break
            if completion_check(result, i):
                break

            # get input and output data (substrate values)
            input_row = list(result[i])
            output_row = list(result[i + step_size])

            # remove time column
            input_row = input_row[1:]
            output_row = output_row[1:]

            # concatenate the input, params, and output to create the datapoin
            params = [
                r.x0, r.x1,
                r.Vm1, r.Vm2, r.Vm3, r.Vm4, r.Vm5, r.Vm6,
                r.Km1, r.Km2, r.Km3, r.Km4, r.Km5, r.Km6, r.Km7, r.Km8, r.Km9, r.Km10, r.Km11, r.Km12,
                r.Keq1, r.Keq2, r.Keq3, r.Keq4, r.Keq5, r.Keq6
            ]
            datapoint = input_row + params + output_row

            # add datapoint to the dataset; once the number of simulations has been reached, add instead to OOD dataset
            if c < num_simulations:
                dataset.loc[len(dataset.index)] = datapoint
            else:
                OOD_dataset.loc[len(OOD_dataset.index)] = datapoint

        # print progress
        if c % ((num_simulations + num_OOD_simulations) / 10) == 0:
            print(f"{100*(c / (num_simulations + num_OOD_simulations))}% Complete")

    # write dataset to csv
    dataset.to_csv(f"tellurium_dataset_step_{step_size}.csv", index=False)
    OOD_dataset.to_csv(f"tellurium_OOD_dataset_step_{step_size}.csv", index=False)