from sklearn.linear_model import LinearRegression
from stable_baselines3 import PPO
import torch.nn as nn
import wandb

import numpy as np

import torch


def safe_tensor(x, dtype=None):
    if isinstance(x, torch.Tensor):
        # clone + detach to avoid autograd issues
        return x.clone().detach()
    return torch.tensor(x, dtype=dtype)

class ReplayBuffer:
    def __init__(self):
        self.obs = []
        self.actions = []
        self.rewards = []
        self.dones = []
        self.logprobs = []
        self.values = []

    def add(self, obs, action, done, logprob, value):
        self.obs.append(safe_tensor(obs, dtype=torch.float32))
        self.actions.append(safe_tensor(action, dtype=torch.int64))
        self.dones.append(safe_tensor(done, dtype=torch.float32))
        self.logprobs.append(safe_tensor(logprob, dtype=torch.float32))
        self.values.append(safe_tensor(value, dtype=torch.float32))

    def add_reward(self, reward):
        self.rewards.append(safe_tensor(reward, dtype=torch.float32))

    def clear(self):
        self.__init__()


class Agent:
    def __init__(self, rewards, state, state_key, action_space, action_update, name, strategy, batch_size, epochs,
                 save_model):
        super().__init__()
        if strategy == "untrained":
            self.model = PPO.load(
                "./train_RA/train_for_latency/models/untrained_model_32_0.0001_512")  # ../models/general_model_32_0.0001_512")
            self.policy = self.model.policy
        elif save_model:
            self.model = PPO.load(
                f"./train_RA/train_for_latency/models/{name}")  # ../models/general_model_32_0.0001_512")
            self.policy = self.model.policy
        elif strategy == "old":
            self.model = PPO.load("../models/general_model_32_0.0001_512")
        elif strategy == "no_noise":
            self.model = PPO.load(
                "./train_RA/train_for_latency/models/no_noise_latency_model_32_0.0001_512")  # ../models/general_model_32_0.0001_512")
        elif strategy == "proactive":
            self.model = LinearRegression()
        elif strategy == "SA_PPO":
            self.model = PPO.load("./models/sa_new_latency_model_32_0.0001_512")
            self.policy = self.model.policy
        else:
            self.model = PPO.load("./train_RA/train_for_latency/models/new_latency_model_32_0.0001_512")
            self.policy = self.model.policy
        if "PPO" in strategy:
            self.optimizer = torch.optim.Adam(self.policy.parameters(), lr=3e-4)
        self.model.n_steps = 2
        self.upper_limit = rewards['upper']
        self.lower_limit = rewards['lower']
        self.current_state = state
        self.buffer = ReplayBuffer()
        print(f" FIRST state: {state}")
        # this is needed to map the action output to actual values
        self.action_function = action_update['func']
        self.current_resources = (action_space['upper'] / 3) + 0.01
        self.name = name
        self.state_name = state_key
        self.action_upper_limit = action_space['upper']
        self.action_lower_limit = action_space['lower']
        self.last_action = 1
        self.step = 0
        self.strategy = strategy
        self.states_list = np.array(state)
        self.batch_size = batch_size
        self.epochs = epochs
        self.save_model = save_model

    def normalize(self, value, lower, upper):
        return (value - lower) / (upper - lower)

    def denormalize(self, norm_value, lower, upper):
        return norm_value * (upper - lower) + lower

    def update_limits(self, SLO_limits):
        SLO_limit = SLO_limits[self.state_name]
        self.upper_limit = SLO_limit['upper']
        self.lower_limit = SLO_limit['lower']

    def get_state_vector(self):
        state = [self.normalize(self.current_resources, self.action_lower_limit, self.action_upper_limit),
                 self.normalize(self.current_state, 0, self.upper_limit),
                 self.normalize(self.lower_limit, 0, self.upper_limit),
                 self.normalize(self.upper_limit, 0, self.upper_limit)]
        return state

    def reactive_strategy(self, state):
        current_value = state[1]
        lower = state[2]
        upper = state[3]
        if current_value < lower:
            return 2
        if current_value > upper:
            return 0
        return 1

    def perform_action(self):
        state = self.get_state_vector()
        self.states_list = np.append(self.states_list, state[1])
        if self.strategy == "reactive":
            self.last_action = self.reactive_strategy(state)
        elif self.strategy == "proactive":
            print(f"state {self.states_list}")
            if len(self.states_list) > 1:
                print("HERE")
                X = self.states_list[:-1].reshape(-1, 1)  # all values except last
                y = self.states_list[1:]  # all values except first
                self.model.fit(X, y)
                last_value = self.states_list[-1]
                next_value = self.model.predict([[last_value]])[0]
                new_state = state.copy()
                wandb.log({f"{self.name} prediction": self.denormalize(next_value, 0, self.upper_limit)})
                new_state[1] = next_value
                self.last_action = self.reactive_strategy(new_state)
            else:
                self.last_action = self.reactive_strategy(state)
        else:
            obs = torch.tensor(state, dtype=torch.float32, device="cpu")
            with torch.no_grad():
                dist = self.policy.get_distribution(obs.unsqueeze(0))
                value = self.policy.predict_values(obs.unsqueeze(0)).squeeze(-1)
                self.last_action = dist.sample()  # This is stochastic!!
                print(self.last_action)
                self.last_action, _ = self.model.predict(obs, deterministic=True)
                print(self.last_action)
                logprob = dist.log_prob(torch.tensor(self.last_action))
                self.buffer.add(state, self.last_action, 0, logprob, value)
        self.current_resources = self.action_function(self.last_action, self.current_resources)
        print(f"{self.name}: action {self.last_action} - {state}")
        print(f"{self.action_lower_limit}<{self.action_upper_limit}")
        print(
            f"{self.current_resources}-{self.normalize(self.current_resources, self.action_lower_limit, self.action_upper_limit)}")
        wandb.log({f"{self.name} action": self.last_action})
        wandb.log({f"{self.name} resources": self.current_resources})
        self.step += 1

    def perform_global_action(self,state, index):
        obs = torch.tensor(state, dtype=torch.float32, device="cpu")
        with torch.no_grad():
            dist = self.policy.get_distribution(obs.unsqueeze(0))
            value = self.policy.predict_values(obs.unsqueeze(0)).squeeze(-1)
            self.last_action = dist.sample()  # This is stochastic!!
            print(self.last_action)
            last_action, _ = self.model.predict(obs, deterministic=True)
            print(self.last_action)
            logprob = dist.log_prob(torch.tensor(self.last_action))
            self.buffer.add(state, self.last_action, 0, logprob, value)
            self.last_action = last_action[index]
        self.current_resources = self.action_function(self.last_action, self.current_resources)
        print(f"{self.name}: action {self.last_action} - {state}")
        print(f"{self.action_lower_limit}<{self.action_upper_limit}")
        print(
            f"{self.current_resources}-{self.normalize(self.current_resources, self.action_lower_limit, self.action_upper_limit)}")
        wandb.log({f"{self.name} action": self.last_action})
        wandb.log({f"{self.name} resources": self.current_resources})
        self.step += 1
    def add_trace(self, rewards, new_state):
        state = self.get_state_vector()
        reward = rewards[self.state_name]
        wandb.log({f"{self.name} reward": reward})
        self.buffer.add_reward(reward)

        self.current_state = new_state[self.state_name]
        # if self.step % self.batch_size == 0 and self.strategy != "reactive" and self.strategy != "proactive":
        #     self.train()

    def train(self):
        print("Training PPO...")
        batch_size = self.batch_size
        with open("demofile_new.txt", "a+") as f:
            f.write(f"{self.buffer.obs}\n")
            f.write(f"{self.buffer.actions}\n")
            f.write(f"{self.buffer.logprobs}\n")
            f.write(f"{self.buffer.values}\n")
            f.write(f"{self.buffer.rewards}\n")
            f.write(f"{self.buffer.dones}\n")
        obs = torch.stack(self.buffer.obs)
        actions = torch.stack(self.buffer.actions)
        old_logprobs = torch.stack(self.buffer.logprobs)
        values = torch.stack(self.buffer.values)
        advantages, returns = compute_gae(self.buffer)
        n = len(obs)
        clip_range = 0.2
        with open("logits_new.txt", "a+") as f:
            f.write(f"-------------------------------------\n")
        for _ in range(self.epochs):
            perm = torch.randperm(n)
            for start in range(0, n, batch_size):
                mb_idx = perm[start:start + batch_size]

                mb_obs = obs[mb_idx]
                mb_actions = actions[mb_idx]
                mb_old_logprobs = old_logprobs[mb_idx]
                mb_adv = advantages[mb_idx]
                mb_returns = returns[mb_idx]

                dist = self.policy.get_distribution(mb_obs)
                logprobs = dist.log_prob(mb_actions)
                entropy = dist.entropy().mean()
                logits = dist.distribution.logits
                probs = torch.softmax(logits, dim=-1)
                rounded = probs.round(decimals=3)
                with open("logits_new.txt", "a+") as f:
                    f.write(f"{rounded}\n")
                values_pred = self.policy.predict_values(mb_obs).squeeze(-1)
                ratio = torch.exp(logprobs - mb_old_logprobs)
                surr1 = ratio * mb_adv
                surr2 = torch.clamp(ratio, 1 - clip_range, 1 + clip_range) * mb_adv

                policy_loss = -torch.min(surr1, surr2).mean()
                value_loss = (mb_returns - values_pred).pow(2).mean()

                loss = policy_loss + 0.5 * value_loss - 0.01 * entropy

                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.policy.parameters(), 0.5)
                self.optimizer.step()
            self.buffer.clear()
            if self.save_model:
                self.model.save(f"./train_RA/train_for_latency/models/{self.name}")

def compute_gae(buffer, gamma=0.99, lam=0.95):
    advantages = []
    returns = []
    gae = 0
    next_value = 0

    for t in reversed(range(len(buffer.rewards))):
        reward = buffer.rewards[t]
        done = buffer.dones[t]
        value = buffer.values[t]

        delta = reward + gamma * next_value * (1 - done) - value
        gae = delta + gamma * lam * (1 - done) * gae

        advantages.insert(0, gae)
        returns.insert(0, gae + value)

        next_value = value

    advantages = torch.tensor(advantages, dtype=torch.float32, device="cpu")
    returns = torch.tensor(returns, dtype=torch.float32, device="cpu")

    advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

    return advantages, returns
