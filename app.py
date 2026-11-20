import streamlit as st
import pandas as pd
from pathlib import Path

st.set_page_config(
    page_title="Volatility-Aware Portfolio Agent",
    layout="wide"
)

BASE_DIR = Path(__file__).resolve().parent
RESULTS_DIR = BASE_DIR / "results"

st.title("Volatility-Aware Portfolio Management Agent")
st.write(
    "Portfolio management using Proximal Policy "
    "Optimization (PPO), ARCH and GARCH."
)

st.header("Portfolio Performance")

summary_path = RESULTS_DIR / "summary.csv"

if summary_path.exists():
    summary = pd.read_csv(summary_path)
    st.dataframe(summary, use_container_width=True)
else:
    st.warning("summary.csv was not found in the results folder.")

st.header("Backtesting Results")

figures = {
    "Portfolio Growth": "fig_equity.png",
    "Portfolio Drawdown": "fig_drawdown.png",
    "Sharpe Ratio Comparison": "fig_sharpe_bar.png",
    "ARCH and GARCH Comparison": "fig_arch_garch.png",
}

for title, filename in figures.items():
    image_path = RESULTS_DIR / filename

    if image_path.exists():
        st.subheader(title)
        st.image(str(image_path), use_container_width=True)
    else:
        st.warning(f"{filename} was not found.")

 
import numpy as np
from stable_baselines3 import PPO

import config as C
from common import load_processed, rf_daily
from env import PortfolioEnv, softmax
from importlib import import_module

st.header("Trained PPO Agent - Portfolio Allocation")

try:
    train_module = import_module("03_train")
    build_arrays = train_module.build_arrays

    feats, rets, _ = load_processed()
    X, R, bounds = build_arrays("D", feats, rets)

    model_path = C.MODELS_DIR / "D_s0_best.zip"
    model = PPO.load(str(model_path), device="cpu")

    test_env = PortfolioEnv(
        X, R, rf_daily(), bounds["test"],
        random_start=False
    )

    obs, _ = test_env.reset()

    action, _ = model.predict(obs, deterministic=True)

    weights = softmax(C.ACTION_TEMP * np.clip(action, -1, 1))

    allocation = pd.DataFrame({
        "Asset": list(rets.columns) + ["Cash"],
        "Allocation (%)": weights * 100
    })

    st.subheader("Agent's First Test-Period Allocation")
    st.dataframe(allocation, hide_index=True)
    st.bar_chart(allocation.set_index("Asset"))

    st.caption(
        "Portfolio weights predicted by the trained PPO agent "
        "at the beginning of the historical test period. "
        "These are simulated allocations, not live trades."
    )

except Exception as e:
    st.error(f"Unable to load trained agent: {e}")
