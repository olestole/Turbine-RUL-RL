#!/usr/bin/env python
# -*- coding: utf-8 -*-

from matplotlib import pyplot as plt
import pandas as pd
import numpy as np
from sklearn.neighbors import KernelDensity
from sklearn.model_selection import GridSearchCV
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras import callbacks

import tensorflow as tf
import tensorflow_probability as tfp
from tensorflow.keras.layers import Dense
from tensorflow.keras.regularizers import l2
from tensorflow.keras import backend as k

# Configuration
figsize = (9, 3)

def load_data(data_folder, test=False):
    # Read the CSV files
    if test:
        fnames = ['test_FD001', 'test_FD002', 'test_FD003', 'test_FD004']
    else:
        fnames = ['train_FD001', 'train_FD002', 'train_FD003', 'train_FD004']
    cols = ['machine', 'cycle', 'p1', 'p2', 'p3'] + [f's{i}' for i in range(1, 22)]
    datalist = []
    nmcn = 0
    for fstem in fnames:
        # Read data
        data = pd.read_csv(f'{data_folder}/{fstem}.txt', sep=' ', header=None)
        # Drop the last two columns (parsing errors)
        data.drop(columns=[26, 27], inplace=True)
        # Replace column names
        data.columns = cols
        # Add the data source
        data['src'] = fstem
        # Shift the machine numbers
        data['machine'] += nmcn
        nmcn += len(data['machine'].unique())
        # Generate RUL data
        cnts = data.groupby('machine')[['cycle']].count()
        cnts.columns = ['ftime']
        data = data.join(cnts, on='machine')
        data['rul'] = data['ftime'] - data['cycle']
        data.drop(columns=['ftime'], inplace=True)
        # Store in the list
        datalist.append(data)
    # Concatenate
    data = pd.concat(datalist)
    # Put the 'src' field at the beginning and keep 'rul' at the end
    data = data[['src'] + cols + ['rul']]
    # data.columns = cols
    return data


def plot_dataframe(data, labels=None, vmin=-1.96, vmax=1.96,
        figsize=figsize, autoclose=True, s=4):
    if autoclose: plt.close('all')
    plt.figure(figsize=figsize)
    plt.imshow(data.T.iloc[:, :], aspect='auto',
            cmap='RdBu', vmin=vmin, vmax=vmax)
    if labels is not None:
        # nonzero = data.index[labels != 0]
        ncol = len(data.columns)
        lvl = - 0.05 * ncol
        # plt.scatter(nonzero, lvl*np.ones(len(nonzero)),
        #         s=s, color='tab:orange')
        plt.scatter(labels.index, np.ones(len(labels)) * lvl,
                s=s,
                color=plt.get_cmap('tab10')(np.mod(labels, 10)))
    plt.tight_layout()


def split_by_field(data, field):
    res = {}
    for fval, gdata in data.groupby(field):
        res[fval] = gdata
    return res


def plot_series(series, labels=None,
        figsize=figsize, autoclose=True, s=4):
    if autoclose:
        plt.close('all')
    plt.figure(figsize=figsize)
    plt.plot(series.index, series, label='data')
    if labels is not None:
        # nonzero = series.index[labels != 0]
        smin, smax = np.min(series),  np.max(series)
        lvl = smin - 0.05 * (smax-smin)
        # plt.scatter(nonzero, np.ones(len(nonzero)) * lvl,
        #         s=s,
        #         color='tab:orange')
        plt.scatter(labels.index, np.ones(len(labels)) * lvl,
                s=s,
                color=plt.get_cmap('tab10')(labels))
    plt.tight_layout()


def partition_by_machine(data, tr_machines):
    # Separate
    tr_machines = set(tr_machines)
    tr_list, ts_list = [], []
    for mcn, gdata in data.groupby('machine'):
        if mcn in tr_machines:
            tr_list.append(gdata)
        else:
            ts_list.append(gdata)
    # Collate again
    tr_data = pd.concat(tr_list)
    ts_data = pd.concat(ts_list)
    return tr_data, ts_data


def sliding_window_2D(data, wlen, stride=1):
    # Get shifted tables
    m = len(data)
    lt = [data.iloc[i:m-wlen+i+1:stride, :].values for i in range(wlen)]
    # Reshape to add a new axis
    s = lt[0].shape
    for i in range(wlen):
        lt[i] = lt[i].reshape(s[0], 1, s[1])
    # Concatenate
    wdata = np.concatenate(lt, axis=1)
    return wdata


def sliding_window_by_machine(data, wlen, cols, stride=1):
    l_w, l_m, l_r = [], [], []
    for mcn, gdata in data.groupby('machine'):
        # Apply a sliding window
        tmp_w = sliding_window_2D(gdata[cols], wlen, stride)
        # Build the machine vector
        tmp_m = gdata['machine'].iloc[wlen-1::stride]
        # Build the RUL vector
        tmp_r = gdata['rul'].iloc[wlen-1::stride]
        # Store everything
        l_w.append(tmp_w)
        l_m.append(tmp_m)
        l_r.append(tmp_r)
    res_w = np.concatenate(l_w)
    res_m = np.concatenate(l_m)
    res_r = np.concatenate(l_r)
    return res_w, res_m, res_r


def plot_training_history(history, 
        figsize=figsize, autoclose=True):
    if autoclose:
        plt.close('all')
    plt.figure(figsize=figsize)
    plt.plot(history.history['loss'], label='loss')
    if 'val_loss' in history.history.keys():
        plt.plot(history.history['val_loss'], label='val. loss')
        plt.legend()
    plt.tight_layout()


def plot_rul(pred=None, target=None,
        stddev=None,
        q1_3=None,
        same_scale=True,
        figsize=figsize, autoclose=True):
    if autoclose:
        plt.close('all')
    plt.figure(figsize=figsize)
    if target is not None:
        plt.plot(range(len(target)), target, label='target',
                color='tab:orange')
    if pred is not None:
        if same_scale or target is None:
            ax = plt.gca()
        else:
            ax = plt.gca().twinx()
        ax.plot(range(len(pred)), pred, label='pred',
                color='tab:blue')
        if stddev is not None:
            ax.fill_between(range(len(pred)),
                    pred-stddev, pred+stddev,
                    alpha=0.3, color='tab:blue', label='+/- std')
        if q1_3 is not None:
            ax.fill_between(range(len(pred)),
                    q1_3[0], q1_3[1],
                    alpha=0.3, color='tab:blue', label='1st/3rd quartile')
    plt.legend()
    plt.tight_layout()


class RULCostModel:
    def __init__(self, maintenance_cost, safe_interval=0):
        self.maintenance_cost = maintenance_cost
        self.safe_interval = safe_interval

    def cost(self, machine, pred, thr, return_margin=False):
        # Merge machine and prediction data
        tmp = np.array([machine, pred]).T
        tmp = pd.DataFrame(data=tmp,
                           columns=['machine', 'pred'])
        # Cost computation
        cost = 0
        nfails = 0
        slack = 0
        for mcn, gtmp in tmp.groupby('machine'):
            idx = np.nonzero(gtmp['pred'].values < thr)[0]
            if len(idx) == 0:
                cost += self.maintenance_cost
                nfails += 1
            else:
                cost -= max(0, idx[0] - self.safe_interval)
                slack += len(gtmp) - idx[0]
        if not return_margin:
            return cost
        else:
            return cost, nfails, slack


def opt_threshold_and_plot(machine, pred, th_range, cmodel,
        plot=True, figsize=figsize, autoclose=True):
    # Compute the optimal threshold
    costs = [cmodel.cost(machine, pred, thr) for thr in th_range]
    opt_th = th_range[np.argmin(costs)]
    # Plot
    if plot:
        if autoclose:
            plt.close('all')
        plt.figure(figsize=figsize)
        plt.plot(th_range, costs)
        plt.tight_layout()
    # Return the threshold
    return opt_th


def plot_pred_scatter(y_pred, y_true, figsize=figsize, autoclose=True):
    if autoclose:
        plt.close('all')
    plt.figure(figsize=figsize)
    plt.scatter(y_pred, y_true, marker='.', alpha=0.1)
    xl, xu = plt.xlim()
    yl, yu = plt.ylim()
    l, u = min(xl, yl), max(xu, yu)
    plt.plot([l, u], [l, u], ':', c='0.3')
    plt.xlim(l, u)
    plt.ylim(l, u)
    plt.xlabel('prediction')
    plt.ylabel('target')
    plt.tight_layout()


class MLPplusNormal(keras.Model):
    def __init__(self, input_shape, hidden):
        super(MLPplusNormal, self).__init__()
        # Build the estimation model for mu and sigma
        model_in = keras.Input(shape=input_shape, dtype='float32')
        x = model_in
        for h in hidden:
            x = layers.Dense(h, activation='relu')(x)
        mu = layers.Dense(1, activation='linear')(x)
        logsigma = layers.Dense(1, activation='linear')(x)
        self.model = keras.Model(model_in, [mu, logsigma])
        # Choose what to track at training time
        self.loss_tracker = keras.metrics.Mean(name="loss")

    @property
    def metrics(self):
        return [self.loss_tracker]

    def call(self, data, training=True):
        if training:
            x, y = data
        else:
            x = data
        mu, logsigma = self.model(x)
        dist = tfp.distributions.Normal(mu, k.exp(logsigma))
        print(dist)
        raise Exception('HAHAHA')
        
    def train_step(self, data):
        with tf.GradientTape() as tape:
            log_densities = self(data)
            loss = -tf.reduce_mean(log_densities)
            # loss = self.log_loss(data)
        g = tape.gradient(loss, self.trainable_variables)
        self.optimizer.apply_gradients(zip(g, self.trainable_variables))
        self.loss_tracker.update_state(loss)
        return {"loss": self.loss_tracker.result()}