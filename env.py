"""Gymnasium trading environment + a helper to roll a trained model through a period."""
from collections import deque

import gymnasium as gym
import numpy as np
from gymnasium import spaces

import config as C
from common import portfolio_step


def softmax(z):
    z = z - z.max()
    e = np.exp(z)
    return e / e.sum()


class PortfolioEnv(gym.Env):
    """
    Observation : last `window` days of the chosen features (flattened) + current weights
    Action      : n_assets + 1 numbers in [-1, 1]  -> softmax -> long-only weights incl. cash
    Reward      : see C.REWARD ("pen", "dsr" or "ret")
    Timing      : observe close of day t, hold new weights during day t+1.
    """
    metadata = {"render_modes": []}

    def __init__(self, X, R, rf_d, bounds, window=C.WINDOW, cost=C.COST,
                 reward_type=C.REWARD, random_start=True, ep_len=C.EP_LEN):
        super().__init__()
        self.X = np.asarray(X, dtype=np.float32)          # [T, n_cols] standardised features
        self.R = np.asarray(R, dtype=np.float64)          # [T, n_assets] simple returns
        self.n_assets = self.R.shape[1]
        self.rf_d = rf_d
        self.s, self.e = bounds
        self.window = window
        self.cost = cost
        self.reward_type = reward_type
        self.random_start = random_start
        self.ep_len = ep_len

        obs_dim = window * self.X.shape[1] + self.n_assets + 1
        self.observation_space = spaces.Box(-np.inf, np.inf, shape=(obs_dim,), dtype=np.float32)
        self.action_space = spaces.Box(-1.0, 1.0, shape=(self.n_assets + 1,), dtype=np.float32)

        self.reward_scale = 1.0 if reward_type == "dsr" else 100.0

    # ------------------------------------------------------------------
    def _obs(self):
        win = self.X[self.t - self.window + 1: self.t + 1].ravel()
        return np.concatenate([win, self.w.astype(np.float32)]).astype(np.float32)

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        if self.random_start:
            hi = max(self.s, self.e - self.ep_len)
            self.t = int(self.s + self.np_random.integers(0, hi - self.s + 1))
            self.t_stop = min(self.e, self.t + self.ep_len)
        else:
            self.t = self.s
            self.t_stop = self.e
        self.w = np.zeros(self.n_assets + 1)
        self.w[-1] = 1.0                                  # start in cash
        self.hist = deque(maxlen=C.PEN_WINDOW)
        self.A = 0.0                                      # DSR moments
        self.B = 0.0
        return self._obs(), {}

    def _reward(self, net):
        if self.reward_type == "ret":
            return float(np.log1p(net))
        if self.reward_type == "pen":
            self.hist.append(net)
            std = float(np.std(self.hist)) if len(self.hist) > 1 else 0.0
            return float(net - C.PEN_LAMBDA * std)
        if self.reward_type == "dsr":
            eta = 0.01
            dA, dB = net - self.A, net * net - self.B
            var = self.B - self.A ** 2
            d = 0.0 if var <= 1e-12 else (self.B * dA - 0.5 * self.A * dB) / var ** 1.5
            self.A += eta * dA
            self.B += eta * dB
            return float(d)
        raise ValueError(self.reward_type)

    def step(self, action):
        a = np.clip(np.asarray(action, dtype=np.float64), -1.0, 1.0)
        w_target = softmax(C.ACTION_TEMP * a)
        net, turnover, self.w = portfolio_step(self.w, w_target, self.R[self.t + 1],
                                               self.rf_d, self.cost)
        reward = self.reward_scale * self._reward(net)
        self.t += 1
        terminated = self.t >= self.t_stop and not self.random_start
        truncated = self.t >= self.t_stop and self.random_start
        info = {"net_ret": net, "turnover": turnover}
        return self._obs(), reward, terminated, truncated, info


def run_episode(model, env, index):
    """Roll a trained model through the whole period of `env` (random_start=False).

    Returns (dates, net_returns, turnover).
    """
    obs, _ = env.reset()
    net, turn = [], []
    done = False
    while not done:
        action, _ = model.predict(obs, deterministic=True)
        obs, _, term, trunc, info = env.step(action)
        net.append(info["net_ret"])
        turn.append(info["turnover"])
        done = term or trunc
    dates = index[env.s + 1: env.s + 1 + len(net)]
    return dates, np.array(net), np.array(turn)
