import gymnasium as gym
from gymnasium import spaces
import numpy as np
import wandb
import random

def set_seed(seed):
    np.random.seed(seed)
    random.seed(seed)


set_seed(42)


# to generalize to new scenarios, we normalize the state space
class Normalizer:
    def __init__(self, new_min=0, new_max=4000):
        self.new_min = new_min
        self.new_max = new_max

    def transform(self, data):
        norm_x = (data - self.new_min) / (self.new_max - self.new_min)
        return norm_x

    def detransform(self, x_norm):
        return x_norm * (self.new_max - self.new_min) + self.new_min

class General_Env(gym.Env):
    def __init__(self, nr_step):
        super().__init__()
        self.normal_value = 3000  # upper limit for scaling
        self.normalizer = Normalizer(0, self.normal_value)
        self.current_service_resource_percent = 0.5

        self.target_efficiency_limit = (100, 2700)
        self.lower_limit = None
        self.upper_limit = None
        self.nr_step = nr_step
        low = np.array([0, 0, 0, 0], dtype=np.float32)
        high = np.array([1, np.inf, 1, 1], dtype=np.float32)
        # Observation space: [current_s_lat, s_lat_target, current_n_lat, n_lat_target,total_current]
        self.total_reward = 0
        self.observation_space = spaces.Box(low=low, high=high, dtype=np.float32)
        # Action space: [decrease, same, increase] for service_latency and network_latency
        self.action_space = spaces.Discrete(3)
        wandb.init(project=f'Train RL')

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        print("RESET")
        self.total_reward = 0
        self.state = self.observation_space.sample()
        low, high = self.target_efficiency_limit
        self.lower_limit = random.randint(low, high)
        self.upper_limit = self.lower_limit + 200  # random.randint(low, high)
        self.state[2] = self.normalizer.transform(self.lower_limit)
        self.state[3] = self.normalizer.transform(self.upper_limit)
        self.current_service_resource_percent = self.state[0]
        self.state[1] = self.normalizer.transform(((1 - self.current_service_resource_percent) * self.normal_value))
        self.step_size_c = random.uniform(0.05, 0.15) #changed to 15
        with open("step_size_c.txt", "a+") as f:
            f.write(f"{self.step_size_c}")
        if self.state[1]<self.state[2]:
            with open("start.txt", "a+") as f:
                f.write(f"<\n")
        if self.state[1]>self.state[3]:
            with open("start.txt", "a+") as f:
                f.write(f">\n")
        with open("train_file.txt", "a+") as f:
            f.write(f"{self.state[2]}<{self.state[1]}<{self.state[3]}\n")
        return self.state.copy(), {}

    # if i have more resources, i decrease latency
    def decrease_latency(self, last_percent, step_size):
        return min(last_percent + step_size, 0.99), last_percent + step_size >= 1

    # if i have less resources, i increase latency
    def increase_latency(self, last_percent, step_size):
        return max(0.01, last_percent - step_size), last_percent - step_size <= 0

# We adjust the target in logarithmic steps
    def log_steps(self, last_latency, min_range, max_range, nr_steps):
        steps = np.logspace(0, 1, num=nr_steps, base=10)
        steps = (steps - steps.min()) / (steps.max() - steps.min())
        steps = steps * (max_range - min_range) + min_range
        idx = (np.abs(steps - last_latency)).argmin()
        return steps, idx

    def ratio_steps(self, last_latency):
        return last_latency * (1 / self.upper_limit)

    def step(self, action):
        assert self.action_space.contains(action)
        old_service_percent, tr_old_latency,   r_lower_SLO,tr_upper_SLO = self.state
        invalid_action_s = False
        invalid_action_n = False
        if action == 0:
            self.current_service_resource_percent, invalid_action_s = self.decrease_latency(
                self.current_service_resource_percent, self.step_size_c)
        elif action == 2:
            self.current_service_resource_percent, invalid_action_s = self.increase_latency(
                self.current_service_resource_percent, self.step_size_c)
        current_service_latency = (1 - self.current_service_resource_percent) * self.normal_value
        before=current_service_latency
        current_service_latency+=random.uniform(-0.01*current_service_latency, 0.01*current_service_latency) #5% schwankungen - simulate dynamic env
        after=current_service_latency
        current_service_latency = np.clip(current_service_latency, 0, 1*self.normal_value)
        with open("lol.txt", "a+") as f:
            f.write(
                f"{before}-{current_service_latency}-{after}\n"
            )
        tr_current_total_lat = self.normalizer.transform(current_service_latency)
        # print(tr_current_total_lat)
        self.state = np.array(
            [self.current_service_resource_percent,
             tr_current_total_lat,
             self.normalizer.transform(self.lower_limit),
             self.normalizer.transform(self.upper_limit)], dtype=np.float32)
        reward, done = self.calculate_reward(tr_old_latency, tr_current_total_lat, action, self.state)
        if invalid_action_s:
            reward -= 2
        if invalid_action_n:
            reward -= 2
        return self.state.copy(), float(reward), done, False, {}

    def calculate_reward(self, old_lat, total_lat, action, state):
        total = float(f"{total_lat:.3f}")

        lower = self.normalizer.transform(self.lower_limit)
        upper = self.normalizer.transform(self.upper_limit)

        done = False

        if total > upper:
            if action == 0:  # decrease
                reward = +1.0
            elif action == 1:  # do nothing
                reward = -0.5
            else:  # increase
                reward = -1.0

        elif total < lower:
            if action == 2:  # increase
                reward = +1.0
            elif action == 1:
                reward = -0.5
            else:  # decrease
                reward = -1.0

        else:
            if action == 1:
                reward = +2.0
                # done = True
            else:
                reward = -1.0
        with open("data.txt", "a+") as f:
            f.write(
                f"{self.state}|{action}|{reward}\n"
            )
        self.total_reward += reward
        wandb.log({f"reward": self.total_reward})
        wandb.log({f"action": action})
        return reward, done

    def finish(self):
        print("done")

