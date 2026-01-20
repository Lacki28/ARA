import gymnasium as gym
from gymnasium import spaces
import numpy as np


class BudgetAllocationEnv(gym.Env):
    def __init__(
        self,
        n_parties: int = 2,
        episode_length: int = 50,
        return_mean: float = 0.0,
        return_std: float = 1.0,
        seed: int | None = None,
        render_mode: str | None = None,
    ):
        super().__init__()

        assert n_parties >= 2, "Need at least 2 parties."
        self.n_parties = n_parties
        self.episode_length = episode_length
        self.return_mean = return_mean
        self.return_std = return_std
        self.render_mode = render_mode

        # Action: unconstrained vector; we’ll renormalize to sum to 1.
        # You can also enforce [0,1] bounds if you like.
        self.action_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(self.n_parties,),
            dtype=np.float32,
        )

        # Observation: per-party returns (or features) at current step.
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(self.n_parties,),
            dtype=np.float32,
        )

        self._rng = np.random.default_rng(seed)
        self._t = 0
        self._current_returns = None

    def _sample_returns(self) -> np.ndarray:
        """Sample per-party returns for the current step."""
        return self._rng.normal(
            loc=self.return_mean,
            scale=self.return_std,
            size=(self.n_parties,),
        ).astype(np.float32)

    def reset(self, *, seed: int | None = None, options: dict | None = None):
        super().reset(seed=seed)
        if seed is not None:
            self._rng = np.random.default_rng(seed)

        self._t = 0
        self._current_returns = self._sample_returns()

        observation = self._current_returns.copy()
        info = {}
        return observation, info

    def step(self, action: np.ndarray):
        # Ensure correct shape
        action = np.asarray(action, dtype=np.float32).reshape(self.n_parties)

        # Enforce non-negativity and renormalize to sum to 1
        action = np.clip(action, 0.0, np.inf)
        if action.sum() <= 0:
            # If all zeros or negative, default to uniform allocation
            allocation = np.ones(self.n_parties, dtype=np.float32) / self.n_parties
        else:
            allocation = action / action.sum()

        # Compute reward: dot(allocation, current_returns)
        reward = float(np.dot(allocation, self._current_returns))

        # Advance time
        self._t += 1
        terminated = self._t >= self.episode_length
        truncated = False  # you can add time limits or other truncation logic

        # Sample next returns as next observation
        self._current_returns = self._sample_returns()
        observation = self._current_returns.copy()

        info = {
            "allocation": allocation,
        }

        return observation, reward, terminated, truncated, info

    def render(self):
        if self.render_mode == "human":
            print(f"t={self._t}, returns={self._current_returns}")

    def close(self):
        pass
