from gym import Env
from gym.spaces import Discrete, Box
import numpy as np
import random

class RULEnvironment_3(Env):
    def __init__(self, dataframe, dt_in, sweet_spot=0.3, bad_spot=0.3):
        self.dataframe = dataframe
        self.episode_nr = 0
        self.dt_in = dt_in
        self.max_episodes = dataframe['machine'].max()

        # Two actions: stopp or continue
        # Actions: 1=stop, 0 = continue
        self.action_space = Discrete(2)
        self.observation_space = Box(np.full((24), -4.0), np.full((24), 4.0))
        # self._set_variables()

        self.penalty = dataframe['rul'].max() * -1
        # Sweetspot tells us that the last percentage is nice
        self.sweet_spot = sweet_spot
        # Badspot tells us that the first percentage is bad
        self.bad_spot = bad_spot
        


    def _set_variables(self):
        # If we want to run the episodes over again:
        if self.episode_nr == self.max_episodes-1:
            self.episode_nr = 0
        # Dataframe contains machines from 1-709
        self.episode_nr += 1
        self.episode = self.dataframe.loc[self.dataframe['machine'] == self.episode_nr]
        self.index = 1
        self.state = np.array(self.episode[self.dt_in].loc[self.episode['cycle'] == self.index]).flatten()
        self.RUL = self.episode.loc[self.episode['cycle'] == self.index]['rul'].item() # acctual remaining useful life.
        self.max_RUL = self.episode['rul'].max()

    def step(self, action):
        done = False
        self.RUL += action - 1

        if action == 1:
            # Action is stop
            if self.RUL < round(self.sweet_spot*self.max_RUL):
                reward = 200
            elif self.RUL > self.max_RUL - round(self.bad_spot*self.max_RUL):
                reward = -200
            else:
                reward = 1
            done = True
        elif self.RUL <= 0 and action == 0:
            reward = self.penalty # Denne må bestemmes i forhold til hva som er max RUL, og hva han skrev i oppg
            done = True
        else:
            reward = 1 # Burde man få høyere reward om man stopper akkurat på null?
            self.index += 1
            self.state = np.array(self.episode[self.dt_in].loc[self.episode['cycle'] == self.index]).flatten()

        info = {"reward": reward} # Placeholder not sure why

        return self.state, reward, done, info

    def render(self):
        # For printing states etc
        print('e:', self.episode_nr, 'i', self.index)
        # print(self.episode.loc[self.episode['cycle'] == self.index]['rul'].item())
    
    def reset(self):
        self._set_variables()
        return self.state

