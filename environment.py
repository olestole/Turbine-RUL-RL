from gym import Env
from gym.spaces import Discrete, Box
import numpy as np
import random

class RULEnvironment(Env):
    def __init__(self, dataframe, dt_in):
        self.dataframe = dataframe
        self.episode_nr = 0
        self.dt_in = dt_in

        # Two actions: stopp or continue
        # Actions: 1=stop, 0 = continue
        self.action_space = Discrete(2)
        self.observation_space = Box(np.array(dataframe[dt_in].min()), np.array(dataframe[dt_in].max()))
        self._set_variables()


    def _set_variables(self):
        # Dataframe contains machines from 1-709
        self.episode_nr += 1
        self.episode = self.dataframe.loc[self.dataframe['machine'] == self.episode_nr]
        self.index = 1
        self.state = self.episode[self.dt_in].loc[self.episode['cycle'] == self.index]
        self.RUL = self.episode.loc[self.episode['cycle'] == self.index]['rul'].item() # acctual remaining useful life.

    def step(self, action):
        self.RUL += action - 1
        done = False

        if action == 1:
            # Action is stop
            done = True
        
        if self.RUL <= 0 and action == 0:
            reward = -1000 # Denne må bestemmes i forhold til hva som er max RUL, og hva han skrev i oppg
            done = True
        else:
            reward = 1 # Burde man få høyere reward om man stopper akkurat på null?
            self.index += 1
            self.state = self.episode[self.dt_in].loc[self.episode['cycle'] == self.index]
        
        info = {} # Placeholder not sure why

        return self.state, reward, done, info
    def render(self):
        # For printing states etc
        pass
    
    def reset(self):
        self._set_variables()
        return self.state

