# deepsim
tellurium_simulations.py allows the user to build datasets with given step sizes. Modify the list of step sizes
and the simulation parameters (number of simulations, number of simulation steps) to build the desired
dataset. This script also builds an OOD dataset which contains data from simulations not included in the training
dataset.

train_model.py allows the user to train a neural network on a given step size dataset. Specify the step size in the
WandB initialization found at the bottom of the file. Model weights are saved after training is completed. Uses Weights & Biases for tracking and visualizing model training.

evaluate_model.py allows the user to compare the neural network's simulations with a set of 'ground-truth' tellurium
simulations. Modify the list of step sizes and simulation parameters (number of simulations,
number of simulation steps) to evaluate with the desired context. This file generates a directory
called 'evaluation_plots' which contains a list of generated plots for each evaluated step size.

## tellurium_simulations
![Screen Shot 2022-12-19 at 1 40 09 PM](https://user-images.githubusercontent.com/97203079/208529345-deb4d196-2c65-4b8f-956f-b952d4de5496.png)
![Screen Shot 2022-12-19 at 1 42 12 PM](https://user-images.githubusercontent.com/97203079/208529932-b0d093bd-68a0-41c8-9dc0-ff1037c7cb32.png)

## train_model
![Screen Shot 2022-12-19 at 1 41 49 PM](https://user-images.githubusercontent.com/97203079/208529913-604d5a33-919e-4ec1-84fd-3b0bdb11555b.png)

## evaluate_model
![Screen Shot 2022-12-19 at 1 47 09 PM](https://user-images.githubusercontent.com/97203079/208530513-e2957682-8761-4c52-8d5f-27b1a9fb0232.png)
![Screen Shot 2022-12-19 at 1 42 49 PM](https://user-images.githubusercontent.com/97203079/208529946-926a8797-9135-4ae0-a114-caa46563dce4.png)
![Screen Shot 2022-12-19 at 1 42 57 PM](https://user-images.githubusercontent.com/97203079/208529953-6896d7ae-2cec-408f-a22f-b758ca98526b.png)

# getting started
**Tellurium Installation Instructions**  
https://github.com/sys-bio/tellurium

**Weights and Biases**  
https://wandb.ai/site  
Sign up on the website above. It should give you instructions for:  
  1. How to pip install WandB  
  2. How to login to your account inside your terminal/code. You only need to login once and it will be permanently connected.  
Once you’ve made your account you should quickly run through the Quickstart Guide linked below just to get a feel for how it works and how you’ll visualize/access the graphs. You can run all of the code in the provided notebook by simply clicking the Play Buttons on the left-most side of each code cell.  
Quickstart Guide: https://colab.research.google.com/github/wandb/examples/blob/master/colabs/intro/Intro_to_Weights_%26_Biases.ipynb#scrollTo=k-gn_8z167SU
