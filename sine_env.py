import matplotlib.pyplot as plt
import gymnasium as gym
import numpy as np
from gymnasium.vector.utils import spaces
import pandas as pd

class SineEnv(gym.Env):
    def __init__(self, config=None):
        super(SineEnv, self).__init__()
        self.config = config
        self.action_space = spaces.Discrete(2)  # Two discrete actions: 0 and 1; Up and Down
        self.observation_space = spaces.Box(low=-1.0, high=1.0, shape=(3,))  # Set the minimum and maximum values of the observation space to be more realistic.
        self.reset()

    def step(self, action):
        price_direction = self.get_price_direction()
        reward = 1.0 if (action == price_direction) else -1.5
        self.balance = self.balance + reward
        observation = self.get_observation()
        self.day += 1
        done = self.day >= len(self.df.price_changes) or self.balance <= 0
        return observation, reward, done, False, {}

    def get_price_direction(self):
        if self.df.price_changes[self.day] > 0.0:
            return 1  # up
        else:
            return 0  # down

    def get_observation(self):
        # TODO Fix this so it works if no df
        return np.array((
            self.df.price_changes[self.day - 3],
            self.df.price_changes[self.day - 2],
            self.df.price_changes[self.day - 1],
        ), np.float32)

    # Called after every episode;  Should return first observation
    def reset(self, seed=None, options=None, **kwargs):
        super().reset(seed=seed)
        self.df = self.create_sinewave(self.config, seed=seed)
        self.day = self.config['lookback']  # start day 1 if looking back 1 day
        self.balance = self.start_balance = 10.0
        return self.get_observation(), {}

    def show(self):
        plt.figure(figsize=(10, 1))
        plt.plot(self.df.index, self.df['prices'], label='Sine wave')
        plt.show()

    @staticmethod
    def create_sinewave(config, seed=None):
        np.random.seed(seed)
        frequency = np.random.uniform(low=config['frequency_min'], high=config['frequency_max'])
        count = config['count']
        amplitude = np.random.uniform(low=config['amplitude_min'], high=config['amplitude_max'])
        x = np.linspace(0, frequency * np.pi, count)
        y = np.sin(x)  # Generate the sine wave
        y = (y + 1) * (amplitude - config['amplitude_min']) / 2 + config['amplitude_min'] # Rescale the sine wave
        df = pd.DataFrame()
        df['prices'] = y
        df['price_changes'] = df['prices'].pct_change().fillna(0)
        return df                                                


        
