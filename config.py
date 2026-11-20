"""Central settings for the volatility-aware RL portfolio project (Indian market)."""
from pathlib import Path

ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"
RESULTS_DIR = ROOT / "results"
MODELS_DIR = ROOT / "models"

# ---------------------------------------------------------------- data
TICKERS = [
    "RELIANCE.NS", "TCS.NS", "HDFCBANK.NS", "INFY.NS", "ITC.NS",
    "HINDUNILVR.NS", "LT.NS", "SBIN.NS", "GOLDBEES.NS",
]
BENCH_TICKER = "^NSEI"          # NIFTY 50 index, used only as a benchmark
START = "2010-01-01"
END = "2026-11-17"              # yfinance treats END as exclusive

# chronological split (never shuffled)
TRAIN_END = "2019-12-31"
VAL_END = "2021-12-31"          # test = everything after this date

# ---------------------------------------------------------------- environment
WINDOW = 30                     # days of history the agent sees
COST = 0.001                    # 0.1% per unit of traded risky-asset value
RF_ANNUAL = 0.06                # cash / risk-free return (approx. Indian T-bills)
EP_LEN = 252                    # training episode length (trading days)
ACTION_TEMP = 5.0               # softmax sharpness for turning actions into weights

# reward: "pen" = return - lambda * rolling std, "dsr" = differential Sharpe, "ret" = log return
REWARD = "pen"
PEN_LAMBDA = 0.1
PEN_WINDOW = 20

# ---------------------------------------------------------------- volatility features
ARCH_P = 5                      # ARCH(p) order
GARCH_P, GARCH_Q = 1, 1         # GARCH(1,1)
MIN_TRAIN = 500                 # days of history before the first volatility forecast
REFIT_EVERY = 21                # re-estimate model parameters every ~month (expanding window)

BASE_FEATURES = ["ret", "volchg"]
VARIANTS = {
    "A": [],                    # prices/volume only
    "B": ["arch"],              # + ARCH forecast
    "C": ["garch"],             # + GARCH forecast
    "D": ["arch", "garch"],     # + both, as separate inputs
}

# ---------------------------------------------------------------- training
SEEDS = [0, 1, 2, 3, 4]
TIMESTEPS = 100_000
EVAL_FREQ = 10_000              # validation check (for best-checkpoint choice)
