# GOOGL Watch

A one-page phone web app for Alphabet (GOOGL): the live price, a chart of the last 365 days (plus today's intraday move), year stats, and a 30-day trend forecast.

## Use it
1. Get a free API key at https://twelvedata.com (the free plan allows 800 requests a day).
2. Open the app, tap **Data source**, paste the key and tap **Save and load**. The key stays in your browser on that device.
Shortcut: open the app once as `https://joelscodes.github.io/googl-watch/#key=YOURKEY`. It saves the key on that device and removes it from the address bar. The key is never stored in this repo.
3. On iPhone: Share → Add to Home Screen. On Android: menu → Add to Home screen.

Without a key it shows clearly labeled sample data.

## How the 30-day range works
The headline is a range: "8 times in 10, GOOGL should close between $X and $Y" in 21 trading days. Its width comes from the last 60 days of daily swings; its centre follows GOOGL's average rise over up to three years. "Chance higher" is how often a 21-day stretch actually ended higher over those three years.

The five technical signals (price vs 50-day average, 50 vs 200-day average, MACD, RSI, 60-day trend) are shown as background with their own past record. They no longer move the forecast: `research/RESULTS.md` tested them and eleven other approaches on 2004-2026 data, and none beat "it usually goes up". Adding the VIX to the range width was also tested and did not improve it on 2022-2026.

## Accuracy check
The app replays its own 30-day range on every past trading day in the last ~5 years of data (one Twelve Data request), using only prices known on that day, and scores it against where GOOGL closed 21 trading days later: how often it landed inside the 80% range, above it and below it, whether the stated chance of a rise matched how often it rose, and how far the middle estimate missed compared with assuming no change.

## News and reactions
With a free [Finnhub](https://finnhub.io/register) key (Data source panel, or `#finnhub=KEY` in the setup link), the app loads a year of Alphabet headlines, one day per request because Finnhub caps each reply to its newest stories (about 7 minutes the first time, paced under the free 60 calls a minute; days are cached in IndexedDB so later opens only fetch the last couple of days), and matches each one to the first trading session it could move: the same day if published before the 4pm ET close, otherwise the next session. For each day it shows GOOGL's move, the Nasdaq-100's move (QQQ), the difference, how unusual it was against the previous 60 days, and the next 5 days. Headlines are tagged by topic (AI, Cloud, Earnings, Legal, Analysts, YouTube, Search & ads, Waymo & bets, Devices) with an average reaction per topic, plus search and sorting. General market stories that don't mention Alphabet are left out.

The forecast does not use the news. It cannot see earnings dates either. This is not financial advice.

## Hosting
Plain static file. Settings → Pages → Deploy from branch `main`, folder `/`.
