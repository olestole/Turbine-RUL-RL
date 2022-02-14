# Turbine Remaining Useful Time (RUL) with Reinforcement Learning (RL)

This project aims to create a RL-agent that learns to stop a machine before its RUL has expired. The agent will train and test on the C-MAPSS dataset.

The final version of this project leverages [Stable-Baselines3](https://stable-baselines3.readthedocs.io/en/master/), but there's also a branch `keras-rl2` in which we try to implement it with [Keras RL](https://keras-rl.readthedocs.io/en/latest/agents/overview/).

### Overview
`turbine-rul-rl.ipynb`
- Should be ran first in order to preprocess and create the data needed. The data is then experimented with in `plot_results.ipynb`.

`plot_results.ipynb`
- Using the data created by `turbine-rul-rl.ipynb`. Discuss and experiment.

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