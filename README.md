# Turbine Remaining Useful Time (RUL) with Reinforcement Learning (RL)

This project aims to create a RL-agent that learns to stop a machine before its RUL has expired. The agent will train and test on the C-MAPSS dataset.

The final version of this project leverages [Stable-Baselines3](https://stable-baselines3.readthedocs.io/en/master/), but there's also a branch `keras-rl2` in which we try to implement it with [Keras RL](https://keras-rl.readthedocs.io/en/latest/agents/overview/).

### Overview
`turbine-rul-rl.ipynb`
- Here the train and test data are loaded, preprocessed and linked to an environment. Further, the models are created, trained and tested with the defined environments. The trained models and test results are saved into the `pretrained_models` folder.

`plot_results.ipynb`
-  This file displayes the test and train results for the trained and tested models saved by running `turbine-rul-rl.ipynb`.

`enviroment1.py`
- Defines the first tested environment. Gives +1 as reward for every timestep and gives a large penalty if the machine crashes.

`enviroment2.py`
- Defines the second tested environment, it only differs from the first environment in its rewards and penalties. This environment additionally includes +100 if the machine stops within a defined *sweetspot*.

`enviroment3.py`
- Defines the third tested environment, this also only differs from the other environments in its rewards and penalties. The envrionment includes both a defined *sweetspot* and a defined *badspot*. If the agent stops within the *badspot* it recieves -100 in penalty, if it stops within the *sweetspot* it receives a reward of +100.

### Disclaimer

The preprocessing and some of the utilities are taken from our professor's work, especially [this lecture](https://github.com/lompabo/aiiti-course-2021-05) about RUL on turbines.

## Installation

Requirements:
- python 3.9

```bash
# Create a virtual env to manage the project's dependencies
$ python -m venv venv

# Install the project's dependencies with pip
$ pip install -r requirements.txt
```

## Development
```bash
# Open the project locally in Jupyter Notes
$ jupyter notebook --ip 127.0.0.1
```