"""Step 3 - train PPO agents (variants A/B/C/D x several seeds) and evaluate them.

    python 03_train.py                         # everything (about 30-60 min on a laptop CPU)
    python 03_train.py --variant D --seeds 0   # one quick run
    python 03_train.py --timesteps 20000       # smoke test

For each run the best checkpoint is chosen on the VALIDATION period only; the test
period is touched once, at the end. Finished runs are skipped unless --force is given.
"""
import argparse
import time

import numpy as np
import pandas as pd
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback

import config as C
from common import (ensure_dirs, load_processed, metrics, rf_daily, save_returns,
                    split_bounds)
from env import PortfolioEnv, run_episode


def build_arrays(variant, feats, rets):
    """Standardise the chosen feature columns using TRAIN rows only."""
    tickers = list(rets.columns)
    cols = [(t, f) for t in tickers for f in C.BASE_FEATURES + C.VARIANTS[variant]]
    Xdf = feats[cols]
    b = split_bounds(feats.index)
    train_part = Xdf.iloc[: b["train"][1] + 1]
    mu, sd = train_part.mean(), train_part.std().replace(0.0, 1.0)
    X = ((Xdf - mu) / sd).clip(-5, 5).values.astype(np.float32)
    return X, rets.values, b


class ValCallback(BaseCallback):
    """Every `freq` steps, test the policy on validation data and keep the best one."""

    def __init__(self, val_env, index, freq, path):
        super().__init__()
        self.val_env, self.index, self.freq, self.path = val_env, index, freq, path
        self.best = -np.inf

    def check(self):
        _, net, _ = run_episode(self.model, self.val_env, self.index)
        s = metrics(net)["sharpe"]
        if s > self.best:
            self.best = s
            self.model.save(self.path)
        return s

    def _on_step(self):
        if self.n_calls % self.freq == 0:
            self.check()
        return True


def run_one(variant, seed, args, feats, rets):
    name = f"{variant}_s{seed}"
    test_csv = C.RESULTS_DIR / "returns" / "test" / f"{name}.csv"
    if test_csv.exists() and not args.force:
        print(f"[skip] {name} already done")
        return None

    X, R, b = build_arrays(variant, feats, rets)
    rf_d = rf_daily()
    mk = lambda split, rnd: PortfolioEnv(X, R, rf_d, b[split], reward_type=args.reward,
                                         random_start=rnd)
    train_env, val_env, test_env = mk("train", True), mk("val", False), mk("test", False)

    best_path = str(C.MODELS_DIR / f"{name}_best")
    cb = ValCallback(val_env, feats.index, min(C.EVAL_FREQ, args.timesteps), best_path)
    model = PPO("MlpPolicy", train_env, seed=seed, device="cpu", verbose=0,
                learning_rate=3e-4, n_steps=1024, batch_size=128, gamma=0.99,
                ent_coef=0.005, policy_kwargs=dict(net_arch=[64, 64]))

    t0 = time.time()
    model.learn(total_timesteps=args.timesteps, callback=cb)
    cb.check()                                              # also evaluate the final policy
    best = PPO.load(best_path, device="cpu")

    out = {}
    for split, env in (("val", val_env), ("test", test_env)):
        dates, net, turn = run_episode(best, env, feats.index)
        save_returns(C.RESULTS_DIR / "returns" / split / f"{name}.csv", dates, net, turn)
        out[split] = metrics(net)
    print(f"[done] {name}: val Sharpe {out['val']['sharpe']:.2f} | "
          f"test Sharpe {out['test']['sharpe']:.2f}, maxDD {out['test']['max_drawdown']:.1%} "
          f"({time.time() - t0:.0f}s)")
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--variant", default="all", help="A, B, C, D or all")
    ap.add_argument("--seeds", type=int, nargs="*", default=C.SEEDS)
    ap.add_argument("--timesteps", type=int, default=C.TIMESTEPS)
    ap.add_argument("--reward", default=C.REWARD, choices=["pen", "dsr", "ret"])
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()

    ensure_dirs()
    feats, rets, _ = load_processed()
    variants = list(C.VARIANTS) if args.variant == "all" else [args.variant.upper()]
    for v in variants:
        for s in args.seeds:
            run_one(v, s, args, feats, rets)
    print("Training finished. Next: python 04_baselines.py")
