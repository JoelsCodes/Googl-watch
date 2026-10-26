# Can GOOGL's direction be predicted? Experiment results (Oct 2026)

**Short answer: no method beat "GOOGL usually goes up" on data it had not seen.** The size of the next month's swings *is* partly predictable.

## Setup
- Data: daily GOOGL Aug 2004 to Oct 26 2026 (5,582 days, split-adjusted), plus QQQ, SPY, VIX, 10-year yield, MSFT, AAPL, AMZN, NVDA. Downloaded by `.github/workflows/fetch-data.yml` to the `data` branch.
- Question: will GOOGL be higher in 5, 21 (about a month, the app's horizon) and 63 trading days?
- Walk-forward: each prediction uses only data known that day; learned models are refit every 63 days on samples whose outcome was already known (no peeking).
- Model choice is made on **2012-2021** only. **2022-now** is a holdout, read once.
- 12 approaches: the app's 5 signals; 12-month momentum; 1-month reversal; VIX level and VIX vs its 1-year average; drawdown from the 1-year high; logistic regression and gradient boosting, each on price-only features and on price + market context (Nasdaq-100 trend, VIX, rates, mega-cap breadth, volume, earnings season); and pattern matching that finds the 50 most similar past 20-day or 60-day price shapes and averages what came next.
- Scores: AUC (0.5 = coin flip), hit rate when bullish vs bearish, and a trading test that holds GOOGL only when bullish (otherwise cash), against buy-and-hold.

## Results, 21-day horizon
| | Dev 2012-21 AUC | Holdout 2022-26 AUC | Holdout: hold when bullish, CAGR | Buy-and-hold CAGR |
|---|---|---|---|---|
| App's 5 signals | 0.43 | 0.53 | 9.6% | 20.8% |
| Drawdown from 1-yr high (picked on dev) | **0.60** | **0.50** | 14.8% | 20.8% |
| Logistic + market context | 0.46 | 0.50 | 11.6% | 20.8% |
| Gradient boosting + market context | 0.49 | 0.51 | 2.3% | 20.8% |
| Pattern match, 60-day shapes | 0.49 | 0.52 | 13.9% | 20.8% |

Full tables for 5, 21 and 63 days are in `results/`.

## What it means
1. **Direction is not predictable from these inputs.** The best development-period signal (buy after drawdowns; AUC 0.60, p = 0.02 before correcting for testing 12 ideas) fell to 0.50 on the holdout. The same happened at 5 and 63 days. Everything else hovered around 0.5 in both periods.
2. **"Patterns" did not repeat.** Matching the shape of the last 20 or 60 days to history gave coin-flip results.
3. **Machine learning did not help.** Gradient boosting with 22 features was no better than simple rules, and worse as a trading rule.
4. **Every timing rule lost to just holding.** Sitting out "bearish" periods mostly meant missing gains: GOOGL rose in 59% of 21-day windows since 2022 and 64% in 2012-21.
5. **Risk is predictable.** Today's VIX has a 0.50 rank correlation with how much GOOGL will swing over the next month (past 63-day volatility: 0.44). That is why the app's 80% range landed close to 80%.

## Caveats
- About 120 independent 21-day windows in development and 57 in the holdout, so small edges (AUC 0.52-0.55) cannot be ruled out, but none showed up consistently.
- Not tested: news and sentiment text, options-market data, fundamentals and analyst revisions, intraday data. These are where professional edges usually come from, and they are mostly not free.

## Reproduce
```
pip install numpy pandas scipy scikit-learn lightgbm
python research/experiments.py PATH_TO_DATA_BRANCH_CHECKOUT 21
```
