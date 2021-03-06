#!/usr/bin/env python
# -*- coding: utf-8 -*-

from matplotlib import pyplot as plt
import pandas as pd
import numpy as np

from stable_baselines3 import DQN, PPO
from stable_baselines3.common.monitor import load_results

# Configuration
figsize = (9, 3)


def load_data(data_folder, test=False):
    # Read the CSV files
    if test:
        fnames = ["test_FD001", "test_FD002", "test_FD003", "test_FD004"]
    else:
        fnames = ["train_FD001", "train_FD002", "train_FD003", "train_FD004"]
    cols = ["machine", "cycle", "p1", "p2", "p3"] + [f"s{i}" for i in range(1, 22)]
    datalist = []
    nmcn = 0
    for fstem in fnames:
        # Read data
        data = pd.read_csv(f"{data_folder}/{fstem}.txt", sep=" ", header=None)
        # Drop the last two columns (parsing errors)
        data.drop(columns=[26, 27], inplace=True)
        # Replace column names
        data.columns = cols
        # Add the data source
        data["src"] = fstem
        # Shift the machine numbers
        data["machine"] += nmcn
        nmcn += len(data["machine"].unique())
        # Generate RUL data
        cnts = data.groupby("machine")[["cycle"]].count()
        cnts.columns = ["ftime"]
        data = data.join(cnts, on="machine")
        data["rul"] = data["ftime"] - data["cycle"]
        data.drop(columns=["ftime"], inplace=True)
        # Store in the list
        datalist.append(data)
    # Concatenate
    data = pd.concat(datalist)
    # Put the 'src' field at the beginning and keep 'rul' at the end
    data = data[["src"] + cols + ["rul"]]
    # data.columns = cols
    return data


def plot_dataframe(
    data, labels=None, vmin=-1.96, vmax=1.96, figsize=figsize, autoclose=True, s=4
):
    if autoclose:
        plt.close("all")
    plt.figure(figsize=figsize)
    plt.imshow(data.T.iloc[:, :], aspect="auto", cmap="RdBu", vmin=vmin, vmax=vmax)
    if labels is not None:
        # nonzero = data.index[labels != 0]
        ncol = len(data.columns)
        lvl = -0.05 * ncol
        # plt.scatter(nonzero, lvl*np.ones(len(nonzero)),
        #         s=s, color='tab:orange')
        plt.scatter(
            labels.index,
            np.ones(len(labels)) * lvl,
            s=s,
            color=plt.get_cmap("tab10")(np.mod(labels, 10)),
        )
    plt.tight_layout()


def split_by_field(data, field):
    res = {}
    for fval, gdata in data.groupby(field):
        res[fval] = gdata
    return res


def partition_by_machine(data, tr_machines):
    # Separate
    tr_machines = set(tr_machines)
    tr_list, ts_list = [], []
    for mcn, gdata in data.groupby("machine"):
        if mcn in tr_machines:
            tr_list.append(gdata)
        else:
            ts_list.append(gdata)
    # Collate again
    tr_data = pd.concat(tr_list)
    ts_data = pd.concat(ts_list)
    return tr_data, ts_data


def plot_training_history(history, figsize=figsize, autoclose=True):
    if autoclose:
        plt.close("all")
    plt.figure(figsize=figsize)
    plt.plot(history.history["loss"], label="loss")
    if "val_loss" in history.history.keys():
        plt.plot(history.history["val_loss"], label="val. loss")
        plt.legend()
    plt.tight_layout()


def plot_rul(
    pred=None,
    target=None,
    stddev=None,
    q1_3=None,
    same_scale=True,
    figsize=figsize,
    autoclose=True,
):
    if autoclose:
        plt.close("all")
    plt.figure(figsize=figsize)
    if target is not None:
        plt.plot(range(len(target)), target, label="target", color="tab:orange")
    if pred is not None:
        if same_scale or target is None:
            ax = plt.gca()
        else:
            ax = plt.gca().twinx()
        ax.plot(range(len(pred)), pred, label="pred", color="tab:blue")
        if stddev is not None:
            ax.fill_between(
                range(len(pred)),
                pred - stddev,
                pred + stddev,
                alpha=0.3,
                color="tab:blue",
                label="+/- std",
            )
        if q1_3 is not None:
            ax.fill_between(
                range(len(pred)),
                q1_3[0],
                q1_3[1],
                alpha=0.3,
                color="tab:blue",
                label="1st/3rd quartile",
            )
    plt.legend()
    plt.tight_layout()


def save_model(model, path="./pretrained_models/model_1"):
    model.save(path)


def load_model(path, env=None, model_type=1):
    # If the model is supposed to be trained on, env need to be set. If it's only going to be used for prediction, it can be None
    if model_type == 1:
        model = DQN.load(path, env=env)
    elif model_type == 2:
        model = PPO.load(path, env=env)
    else:
        raise Exception("Not correct model type")
    return model

def evaluate(model, environment, num_episodes=100):
    """
    Evaluate a RL agent
    :param model: (BaseRLModel object) the RL Agent
    :param num_episodes: (int) number of episodes to evaluate it
    :return: (float) Mean reward for the last num_episodes
    """
    # This function will only work for a single Environment
    env = environment
    all_episode_rewards = []
    all_nr_of_actions = []
    for i in range(num_episodes):
        episode_rewards = []
        nr_of_actions = 0
        done = False
        obs = env.reset()
        while not done:
            # _states are only useful when using LSTM policies
            action, _states = model.predict(obs)
            # here, action, rewards and dones are arrays
            # because we are using vectorized env
            obs, reward, done, info = env.step(action)
            episode_rewards.append(reward)
            nr_of_actions += 1
        
        all_episode_rewards.append(sum(episode_rewards))
        all_nr_of_actions.append(nr_of_actions)
    
    mean_episode_reward = np.mean(all_episode_rewards)
    print("Mean reward:", mean_episode_reward, "Num episodes:", num_episodes)

    return mean_episode_reward, all_episode_rewards, all_nr_of_actions

def create_results_dataframe(machine_max_rul, all_episode_rewards, all_nr_of_actions, ):
    # data_sv_test -> The dataframe with the test-data
    # dt_in_test -> The columns to be used as input-data

    # Create a dataframe with the results
    d = {
        'machine': np.arange(1, len(machine_max_rul) + 1),
        'reward': np.array(all_episode_rewards),
        'num_actions': np.array(all_nr_of_actions),
        'max_RUL': np.array(machine_max_rul)
    }

    df = pd.DataFrame(data=d, index=None)

    df["slack"] = (df["max_RUL"] - df["num_actions"])
    df['reward_estimate'] = round(df['reward'] / df['max_RUL'], 2)
    
    return df

def get_crash_points(df):
    return df[(df['max_RUL'] == df["num_actions"]) & (df['reward'] < 0)]

def get_perfect_points(df):
    return df[(df['max_RUL'] == df["num_actions"]) & (df['reward'] > 0)]

def plot_slack(df, stop = 100, normalized = True):
    average_slack = df['slack'].mean()
    
    crash_points = get_crash_points(df)
    perfect_points = get_perfect_points(df)
    normalized_slack = df['slack']/df["max_RUL"]
    normalized_average_slack = normalized_slack.mean()    

    if normalized:
        plt.title(f"Average slack: {round(normalized_average_slack, 2)}")
        plt.plot(normalized_slack[:stop])
        plt.hlines(normalized_average_slack, 0, stop, color='orange', linestyles='dashed')
    else:
        plt.title(f"Average slack: {round(average_slack, 2)}")
        plt.plot(df["slack"][:stop])
        plt.hlines(average_slack, 0, stop, color='orange', linestyles='dashed')
        
    plt.plot(crash_points['slack'][crash_points['slack'].index < stop], '.', color='red')
    plt.plot(perfect_points['slack'][perfect_points['slack'].index < stop], '.', color='green')
    plt.legend(['Slack', 'Average slack', 'Crashed machine', 'Perfectly stoped machines'])
    plt.show()
    return normalized_average_slack

def save_dataframe(df, log_dir):
    df.to_csv(f'{log_dir}results.csv')

def plot_train_reward(log_dir):
    env_monitor = load_results(log_dir)
    rewards = env_monitor['r']
    plt.title('Train reward')
    plt.plot(rewards)
    plt.show()

def plot_test_reward(df):
    plt.title('Test reward')
    plt.plot(df['reward'])
    plt.show()

def plot_crash_points_max_rul(df):
    crash_points = get_crash_points(df)
    max_rul_crash_points = crash_points.copy()['max_RUL'].reset_index().drop('index', axis=1)
    avg_maxrul = max_rul_crash_points.mean()

    plt.plot(max_rul_crash_points)
    plt.hlines(avg_maxrul, 0, max_rul_crash_points.index.max(), color='orange', linestyles='dashed')
    plt.title(f"Max RUL of the crashing machines\nAverage max RUL: {round(avg_maxrul.values[0], 2)}")
    plt.show()

def plot_average_reward(df, stop=100, extra_reward=None):
    if extra_reward:
        df['reward_estimate'] = round(df['reward'] / (df['max_RUL'] + extra_reward), 2)
    df.loc[df['reward_estimate'] < 0, 'reward_estimate'] = 0
    average_reward_estimate = df['reward_estimate'].mean()
    df["slack"] = (df["max_RUL"] - df["num_actions"])
    
    crash_points = get_crash_points(df)
    perfect_points = get_perfect_points(df)


    plt.title(f"Average reward estimate: {round(average_reward_estimate, 2)}")
    plt.plot(df['reward_estimate'][:stop])
    plt.hlines(average_reward_estimate, 0, stop, color='orange', linestyles='dashed')
    plt.plot(crash_points['slack'][crash_points['slack'].index < stop], '.', color='red')
    plt.plot(perfect_points['reward_estimate'][perfect_points['reward_estimate'].index < stop], '.', color='green')

    plt.legend(['Reward estimate', 'Average reward estimate', 'Crashed machine', 'Perfectly stoped machines'])
    plt.show()

def environment_results(df):
    crash_points = get_crash_points(df)
    perfect_points = get_perfect_points(df)

    percent_of_crashed_machines = round(len(crash_points['slack'])/df['machine'].max(), 4)
    print('percent_of_crashed_machines:', percent_of_crashed_machines)
    percent_of_perfect_machines = round(len(perfect_points)/df['machine'].max(), 4)
    print('percent_of_perfect_machines:', percent_of_perfect_machines)
    stop_within_sweetspot = round(len(df[(df['slack'] < 20) & (df['reward_estimate'] > 0)])/df['machine'].max(), 4)
    print('stop_within_sweetspot:', stop_within_sweetspot)
    return percent_of_crashed_machines, percent_of_perfect_machines, stop_within_sweetspot
