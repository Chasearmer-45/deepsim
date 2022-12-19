# deepsim
tellurium_simulations.py allows the user to build datasets with given step sizes. Modify the list of step sizes
and the simulation parameters (number of simulations, number of simulation steps) to build the desired
dataset. This script also builds an OOD dataset which contains data from simulations not included in the training
dataset.

train_model.py allows the user to train a neural network on a given step size dataset. Specify the step size in the
WandB initialization found at the bottom of the file. Model weights are saved after training is completed.

evaluate_model.py allows the user to compare the neural network's simulations with a set of 'ground-truth' tellurium
simulations. Modify the list of step sizes and simulattion parameters (number of simulations,
number of simulation steps) to evaluate with the desired context. This file generates a directory (if necessary)
called 'evaluation_plots' which contains a list of plots for each evaluated step size.
