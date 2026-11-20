# Volatility-Aware RL Portfolio Agent (NSE / Indian market)

Research question: does giving a PPO portfolio agent **separate ARCH and GARCH volatility forecasts**
improve risk-adjusted performance compared with prices/volume alone?

| Variant | Inputs |
|---|---|
| A | returns + volume change |
| B | A + ARCH(5) forecast |
| C | A + GARCH(1,1) forecast |
| D | A + both (as separate inputs) |

## Setup (once)
```
pip install -r requirements.txt
```
Use Python 3.10-3.12 if you can (PyTorch wheels are most reliable there).
Training runs on the CPU on purpose: small MLPs are not faster on a GPU.

## Run order
```
python test_core.py          # 1. self-tests (30 seconds) - must say "All tests passed"
python 01_data.py            # 2. download prices (needs internet)
python 02_features.py        # 3. rolling ARCH/GARCH features (10-30 min; try --workers 4)
python 03_train.py --variant D --seeds 0 --timesteps 20000   # 4. quick smoke test (~1-2 min)
python 03_train.py           # 5. full run: 4 variants x 5 seeds (about 30-90 min)
python 04_baselines.py       # 6. baselines (seconds)
python 05_evaluate.py        # 7. tables, tests and figures
```
Finished runs are skipped on re-run (use `--force` to redo). Stopping and resuming is safe.
`python run_all.py` does everything in order. Add `--synthetic` to test the pipeline with fake data.

## What you get (results/)
| File | Use in the paper |
|---|---|
| `arch_garch_corr.csv`, `fig_arch_garch.png` | How similar the ARCH and GARCH signals are (justifies or challenges using both) |
| `summary.csv / summary.md` | Main results table: mean ± std over seeds |
| `stat_tests.csv` | Welch t-tests between variants (only 5 seeds, so treat p-values carefully) |
| `fig_equity.png`, `fig_drawdown.png`, `fig_sharpe_bar.png` | Main figures |
| `yearly_returns.csv` | Calendar-year breakdown (e.g. the 2022 fall, election volatility in 2024) |
| `per_run_metrics.csv` | Every single run, for the appendix |

## Design choices (write these in the Methodology section)
* **No look-ahead:** ARCH/GARCH parameters are fitted on an expanding window of past returns and
  re-fitted every 21 trading days. The forecast stored at day t uses only returns up to day t.
  `test_core.py` proves this by changing future data and checking past features do not move.
* **Split:** train 2012-2019, validation 2020-2021 (includes the COVID crash), test 2022 onward.
  Feature scaling uses training data only. The best checkpoint of each run is chosen on validation;
  the test period is used once.
* **Trading:** daily rebalancing, long-only softmax weights plus cash, 0.1% cost on traded stock value,
  cash earns 6% a year. State: last 30 days of features + current weights.
* **Reward:** return minus 0.1 x rolling 20-day standard deviation of portfolio returns
  (`--reward pen`). Differential Sharpe (`--reward dsr`) and plain log return (`--reward ret`) are also
  available for a reward-function comparison.
* **Baselines:** equal-weight (monthly), buy-and-hold equal-weight, long-only minimum-variance (monthly,
  252-day covariance), NIFTY 50 index. Same costs and cash rate as the agents.

## Honest limitations to state in the paper
* Only 9 assets and one market; results may not generalise.
* 5 seeds per variant; differences smaller than the seed-to-seed spread are not evidence of an effect.
* Yahoo Finance data is adjusted for splits/dividends but is not survivorship-free (today's NIFTY leaders).
* Hyperparameters were not tuned. Report this as a limitation.
* ARCH and GARCH are nested, so their forecasts are highly correlated. Report the correlation.
