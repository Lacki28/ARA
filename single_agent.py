from sklearn.linear_model import LinearRegression
from stable_baselines3 import PPO
from sympy.physics.units import current

import wandb

import gymnasium as gym
from gymnasium import spaces

import numpy as np

import torch


class DummyEnv(gym.Env):
    def __init__(self):
        super().__init__()
        # low = np.array([
        #     0,
        #     0.01,
        #     0,
        #     0], dtype=np.float32)
        # high = np.array([
        #     np.inf,  # How much in % of the resources we are currently using
        #     np.inf,  # How much above the threshold we are
        #     1,  # lower threshold
        #     1  # upper threshold
        # ], dtype=np.float32)
        low = np.array([0, 0, 0, 0], dtype=np.float32)
        high = np.array([1, np.inf, 1, 1], dtype=np.float32)

        # Observation space: [current_s_lat, s_lat_target, current_n_lat, n_lat_target,total_current]
        self.observation_space = spaces.Box(low=low, high=high, dtype=np.float32)
        self.action_space = gym.spaces.Discrete(3)

    def reset(self, *, seed=None, options=None):
        return np.zeros(4, dtype=np.float32), {}

    def step(self, action):
        return np.zeros(4, dtype=np.float32), 0.0, False, False, {}


class OldEnv(gym.Env):
    def __init__(self):
        super().__init__()
        low = np.array([
            0,
            0.01,
            0,
            0], dtype=np.float32)
        high = np.array([
            np.inf,  # How much in % of the resources we are currently using
            np.inf,  # How much above the threshold we are
            1,  # lower threshold
            1  # upper threshold
        ], dtype=np.float32)
        # low = np.array([0, 0, 0, 0], dtype=np.float32)
        # high = np.array([1, np.inf, 1, 1], dtype=np.float32)

        # Observation space: [current_s_lat, s_lat_target, current_n_lat, n_lat_target,total_current]
        self.observation_space = spaces.Box(low=low, high=high, dtype=np.float32)
        self.action_space = gym.spaces.Discrete(3)

    def reset(self, *, seed=None, options=None):
        return np.zeros(4, dtype=np.float32), {}

    def step(self, action):
        return np.zeros(4, dtype=np.float32), 0.0, False, False, {}

class Agent:
    def __init__(self, rewards, state, state_key, action_space, action_update, name, strategy):
        super().__init__()
        if strategy == "untrained":
            self.model = PPO.load("./train_RA/train_for_latency/models/untrained_model_32_0.0001_512") #../models/general_model_32_0.0001_512")
            self.model.set_env(DummyEnv())
        elif strategy == "old":
            self.model = PPO.load("../models/general_model_32_0.0001_512")
            self.model.set_env(OldEnv())
        elif strategy == "no_noise":
            self.model = PPO.load("./train_RA/train_for_latency/models/no_noise_latency_model_32_0.0001_512") #../models/general_model_32_0.0001_512")
            self.model.set_env(DummyEnv())
        elif strategy =="proactive":
            self.model = LinearRegression()
        else:
            self.model = PPO.load("./train_RA/train_for_latency/models/new_latency_model_32_0.0001_512")
            # self.model = PPO.load("./train_RA/train_for_latency/models/latency_model_32_0.0001_512") #../models/general_model_32_0.0001_512")
            self.model.set_env(DummyEnv())
        self.model.n_steps = 2
        # self.model.rollout_buffer.buffer_size = 2
        # self.model = PPO.load("../models/general_model_32_0.0001_512")
        # important for state information
        self.upper_limit = rewards['upper']
        self.lower_limit = rewards['lower']
        self.current_state = state
        print(f" FIRST state: {state}")
        # this is needed to map the action output to actual values
        self.action_function = action_update['func']
        self.current_resources = (action_space['upper'] / 3)+0.01
        self.name = name
        self.state_name = state_key
        self.action_upper_limit = action_space['upper']
        self.action_lower_limit = action_space['lower']
        self.last_action = 1
        self.step=0
        self.strategy=strategy
        self.states_list=np.array(state)

    def normalize(self, value, lower, upper):
        return (value - lower) / (upper - lower)

    def denormalize(self, norm_value, lower, upper):
        return norm_value * (upper - lower) + lower


    def update_limits(self, SLO_limits):
        SLO_limit=SLO_limits[self.state_name]
        self.upper_limit = SLO_limit['upper']
        self.lower_limit = SLO_limit['lower']

    def get_state_vector(self):
        state = [self.normalize(self.current_resources, self.action_lower_limit, self.action_upper_limit),
                 self.normalize(self.current_state,  0,self.upper_limit),
                 self.normalize(self.lower_limit, 0,self.upper_limit),
                 self.normalize(self.upper_limit, 0,self.upper_limit)]
        return state

    def reactive_strategy(self, state):
        current_value=state[1]
        lower=state[2]
        upper=state[3]
        if current_value<lower:
            return 2
        if current_value>upper:
            return 0
        return 1


    def perform_action(self):
        state = self.get_state_vector()
        self.states_list=np.append(self.states_list, state[1])
        if self.strategy == "reactive":
            self.last_action=self.reactive_strategy(state)
        elif self.strategy == "untrained":
            self.last_action, _ = self.model.predict(state, deterministic=False)
        elif self.strategy =="proactive":
            print(f"state {self.states_list}")
            if len(self.states_list)>1:
                print("HERE")
                X = self.states_list[:-1].reshape(-1, 1)   # all values except last
                y = self.states_list[1:] # all values except first
                self.model.fit(X, y)
                # window = 10
                # recent = self.states_list[-window:]
                # # Build training pairs from the window
                # X = recent[:-1].reshape(-1, 1)
                # y = recent[1:]
                #
                # self.model.fit(X, y)

                last_value = self.states_list[-1]
                next_value = self.model.predict([[last_value]])[0]
                new_state=state.copy()
                wandb.log({f"{self.name} prediction": self.denormalize(next_value,0,self.upper_limit)})
                new_state[1]=next_value
                self.last_action=self.reactive_strategy(new_state)
            else:
                self.last_action = self.reactive_strategy(state)
        else:
            self.last_action, _ = self.model.predict(state, deterministic=True)
        self.current_resources = self.action_function(self.last_action, self.current_resources)
        print(f"{self.name}: action {self.last_action} - {state}")
        print(f"{self.action_lower_limit}<{self.action_upper_limit}")
        print(f"{self.current_resources}-{self.normalize(self.current_resources, self.action_lower_limit, self.action_upper_limit)}")
        wandb.log({f"{self.name} action": self.last_action})
        wandb.log({f"{self.name} resources": self.current_resources})
        # wandb.log({f"{self.name} state": self.current_state})
        self.step+=1

    def add_trace(self, rewards, new_state):
        state = self.get_state_vector()
        reward=rewards[self.state_name]
        wandb.log({f"{self.name} reward": reward})
        # if self.strategy != "reactive" and self.strategy != "proactive":
        #     add_to_buffer(state, self.model, self.last_action, reward)
        self.current_state = new_state[self.state_name]
        # if self.step>=2 and self.strategy != "reactive" and self.strategy != "proactive":
        #      self.train()

    def train(self):
        print("Training PPO...")
        self.model.rollout_buffer.compute_returns_and_advantage(last_values=torch.tensor([0.0]).to(self.model.device),
                                                                dones=False)
        self.model.train()
        print("Training complete.")
        self.model.rollout_buffer.reset()


def add_to_buffer(state, model, action, reward):
    # if action == 1:
    #     reward += 0.5
    state_tensor = torch.tensor(state).float().unsqueeze(0).to(model.device)
    with torch.no_grad():
        distribution = model.policy.get_distribution(state_tensor)
        value = model.policy.predict_values(state_tensor)
        distr_prob = distribution.log_prob(torch.tensor(action).to(model.device))
        print(f"{state} - {distr_prob}")
        logprobs = torch.stack([
            distribution.log_prob(torch.tensor(0)),
            distribution.log_prob(torch.tensor(1)),
            distribution.log_prob(torch.tensor(2))
        ])
        probs = logprobs.exp()  # convert log-probs → probs
        probs = probs / probs.sum()  # normalize to sum to 1
        print(probs)  # see how the probabilities change over time

    model.rollout_buffer.add(state, action, reward, False, value, distr_prob)
