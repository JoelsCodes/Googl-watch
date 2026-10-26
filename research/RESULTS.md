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

## Results, next day, next hour and next 5 minutes
Same rules (walk-forward, choose on the earlier part, check once on the later part). Intraday bars come from Yahoo, which keeps only 60 days of 5-minute bars and about 3 years of hourly bars, so these samples are much shorter. "Edge" is the average gain per trade of going long when the model says up and short when it says down, before costs; about 1 basis point (0.01%) per trade is a realistic cost for GOOGL.

| Horizon | Data | Picked on development | Its development AUC | Its holdout AUC | Holdout edge per trade |
|---|---|---|---|---|---|
| Next day | daily, 2004-2026 (holdout 2022-26) | Drawdown from 1-yr high | 0.53 | 0.49 | n/a; every daily timing rule earned less than holding (3-14% a year vs 18%) |
| Next hour | 5,077 hourly bars, Nov 2023-Oct 2026 (holdout from Aug 2025) | Pattern match, last 12 bars | 0.52 | 0.51 | -1.6 bp |
| Next 5 minutes | 4,680 bars, Aug-Oct 2026 (holdout from Sep 23) | Last bar reverses | 0.53 | 0.50 | -0.2 bp |

No approach reached an edge that would survive trading costs on the holdout. The best holdout numbers (logistic regression hourly, AUC 0.52; "last bar reverses" hourly, +0.9 bp) were not the ones the development period picked, and picking after seeing the holdout would be cheating.

The size of the next bar's move is again partly predictable: the usual size at that time of day (larger after the open and near the close) has a rank correlation of 0.27 with the next 5-minute move, and recent bar sizes 0.26.

Full tables: `results/h1.txt` (next day), `results/1h.txt`, `results/5m.txt`.

## Results, non-price signals (`altdata.py`)
29 signals that are not stock-market indicators, each lined up with what was known at the close (anything published later is lagged a day), tested at 1, 5 and 21 trading days. Each signal's direction was fixed on 2012-2021, then checked once on 2022-now. Data comes from `.github/workflows/fetch-alt.yml`.

- **Calendar:** Monday, Friday, turn of the month, day before a holiday, May-October, September, the Santa rally, earnings months.
- **Moon:** illumination, days near a new moon.
- **New York weather (Central Park):** cloud cover and sunshine vs normal for the month, heavy rain, temperature vs normal.
- **Space weather:** a geomagnetic storm in the past 6 days, the daily Ap index, sunspot number.
- **Public attention (Wikipedia page views):** spikes and trends for Google + Alphabet, YouTube, Sundar Pichai, Gemini.
- **Macro mood (FRED):** newspaper-based economic policy uncertainty, the yield curve, the dollar, oil, Bitcoin over 5 and 21 days. (Junk-bond spreads and Gemini views only go back to 2023-24, too short to test.)
- **All of them together** in logistic regression and gradient boosting, walk-forward.

| | Count |
|---|---|
| Signal tests (29 signals x 3 horizons, minus short-history ones) | 81 |
| Looked significant on 2012-2021 (p < 0.05) | 13 (about 4 expected by luck) |
| Of those, also held on 2022-now in the same direction | **0** |
| Combined models, holdout AUC | 0.40-0.49 |

The 13 early "finds" included rain and cloudy New York days being followed by better returns, the day before holidays, turn of the month, earnings months, and drops in Wikipedia attention. Every one of them faded or reversed after 2021. A few signals looked good *only* on 2022-now (e.g. YouTube page-view spikes at 21 days, Bitcoin's last week at 5 days); those were not chosen in advance, and with 81 tests a few such hits are expected by chance.

Full table: `results/altdata.txt`.

## What it means
1. **Direction is not predictable from these inputs.** The best development-period signal (buy after drawdowns; AUC 0.60, p = 0.02 before correcting for testing 12 ideas) fell to 0.50 on the holdout. The same happened at 5 and 63 days. Everything else hovered around 0.5 in both periods.
2. **"Patterns" did not repeat.** Matching the shape of the last 20 or 60 days to history gave coin-flip results.
3. **Machine learning did not help.** Gradient boosting with 22 features was no better than simple rules, and worse as a trading rule.
4. **Every timing rule lost to just holding.** Sitting out "bearish" periods mostly meant missing gains: GOOGL rose in 59% of 21-day windows since 2022 and 64% in 2012-21.
5. **Risk is predictable.** Today's VIX has a 0.50 rank correlation with how much GOOGL will swing over the next month (past 63-day volatility: 0.44). That is why the app's 80% range landed close to 80%.

## Caveats
- About 120 independent 21-day windows in development and 57 in the holdout, so small edges (AUC 0.52-0.55) cannot be ruled out, but none showed up consistently.
- Not tested: news and sentiment text scored by a language model, options-market data, fundamentals and analyst revisions, credit-card or app-download data. These are where professional edges usually come from, and they are mostly not free.

## Reproduce
```
pip install numpy pandas scipy scikit-learn lightgbm
python research/experiments.py PATH_TO_DATA_BRANCH_CHECKOUT 21     # or 1, 5, 63 trading days
python research/intraday.py PATH_TO_DATA_BRANCH_CHECKOUT 5m         # or 1h
python research/altdata.py PATH_TO_DATA_BRANCH_CHECKOUT              # after running fetch-alt.yml
```
