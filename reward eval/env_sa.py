import gymnasium as gym
from gymnasium import spaces
import numpy as np
import random
import wandb

def set_seed(seed):
    np.random.seed(seed)
    random.seed(seed)


set_seed(42)


class Normalizer:
    def __init__(self, new_min=0, new_max=4000):
        self.new_min = new_min
        self.new_max = new_max

    def transform(self, data):
        return (data - self.new_min) / (self.new_max - self.new_min)

    def detransform(self, x_norm):
        return x_norm * (self.new_max - self.new_min) + self.new_min


class General_Env(gym.Env):
    def __init__(self, nr_step):
        super().__init__()

        self.nr_step = nr_step


        # Latency ranges
        self.compute_latency_range = (50, 1450)
        self.network_latency_range = (50, 1450)

        # Resource percentages for compute + network
        self.current_percent = [0.5, 0.5]  # [compute, network]

        # SLO bounds
        self.target_efficiency_limit = (100, 2700)
        self.lower_limit = None
        self.upper_limit = None

        # Normalizer (updated after reset)
        self.normalizer = Normalizer(0, 1500) #-> normalize auf 3000/2

        # Observation space (8 values)
        low = np.array([
            0, 0, 0, 0,      # compute
            0, 0, 0, 0       # network
        ], dtype=np.float32)

        high = np.array([
            1, np.inf, 1, 1,  # compute
            1, np.inf, 1, 1   # network
        ], dtype=np.float32)
        self.observation_space = spaces.Box(low=low, high=high, dtype=np.float32)
        self.action_space = spaces.MultiDiscrete([3, 3])
        self.total_reward = 0


    def reset(self, *, seed=None):
        super().reset(seed=seed)
        self.total_reward = 0
        self.stepss=0
        # Random SLO bounds
        low, high = self.target_efficiency_limit
        self.lower_limit = random.randint(low, high)
        self.upper_limit = self.lower_limit + 200

        self.normalizer = Normalizer(0, int(self.upper_limit/2))

        self.current_percent = [
            random.uniform(0.2, 0.8),  # compute
            random.uniform(0.2, 0.8)   # network
        ]
        self.step_size_c = random.uniform(0.05, 0.1)
        self.step_size_n = random.uniform(0.05, 0.1)
        compute_latency = (1 - self.current_percent[0]) * self.compute_latency_range[1]
        network_latency = (1 - self.current_percent[1]) * self.network_latency_range[1]

        self.state = np.array([
            self.current_percent[0],
            self.normalizer.transform(compute_latency),
            (self.lower_limit / self.upper_limit)/2,
            1.0,

            self.current_percent[1],
            self.normalizer.transform(network_latency),
            (self.lower_limit / self.upper_limit)/2,
            1.0
        ], dtype=np.float32)

        return self.state.copy(), {}


    def decrease(self, pct, step):
        return min(pct + step, 0.99), pct + step >= 1.0

    def increase(self, pct, step):
        return max(0.01, pct - step), pct - step <= 0.0


    def step(self, action):
        assert self.action_space.contains(action)

        a_compute = int(action[0])
        a_network = int(action[1])

        invalid_c = False
        invalid_n = False

        # --- Compute agent ---
        if a_compute == 0:
            self.current_percent[0], invalid_c = self.decrease(self.current_percent[0], self.step_size_c)
        elif a_compute == 2:
            self.current_percent[0], invalid_c = self.increase(self.current_percent[0], self.step_size_c)

        compute_latency = (1 - self.current_percent[0]) * self.compute_latency_range[1]
        tr_compute = self.normalizer.transform(compute_latency)

        # --- Network agent ---
        if a_network == 0:
            self.current_percent[1], invalid_n = self.decrease(self.current_percent[1], self.step_size_n)
        elif a_network == 2:
            self.current_percent[1], invalid_n = self.increase(self.current_percent[1], self.step_size_n)

        network_latency = (1 - self.current_percent[1]) * self.network_latency_range[1]
        tr_network = self.normalizer.transform(network_latency)

        self.state = np.array([
            self.current_percent[0],
            tr_compute,
            (self.lower_limit / self.upper_limit)/2,
            1.0,

            self.current_percent[1],
            tr_network,
            (self.lower_limit / self.upper_limit)/2,
            1.0
        ], dtype=np.float32)

        reward_c, done_c = self.calculate_reward(tr_compute, a_compute, "compute")
        reward_n, done_n = self.calculate_reward(tr_network, a_network, "network")

        reward = reward_c + reward_n
        done = done_c or done_n

        if invalid_c:
            reward -= 2
        if invalid_n:
            reward -= 2
        self.stepss+=1
        self.total_reward += reward
        if self.stepss>=self.nr_step-1:
            wandb.log({f"total_reward_final": self.total_reward})

        return self.state.copy(), float(reward), done, False, {}


    def calculate_reward(self, total_lat, action, name):
        lower = self.normalizer.transform(self.lower_limit/2)
        upper = self.normalizer.transform(self.upper_limit/2)

        done = False

        if total_lat > upper:
            if action == 0:
                reward = +1.0
            elif action == 1:
                reward = -0.5
            else:
                reward = -1.0

        elif total_lat < lower:
            if action == 2:
                reward = +1.0
            elif action == 1:
                reward = -0.5
            else:
                reward = -1.0

        else:
            if action == 1:
                reward = +2.0
            else:
                reward = -1.0
        wandb.log({f"reward": self.total_reward})
        wandb.log({f"action": action})

        return reward, done
