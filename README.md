# GOOGL Watch

A one-page phone web app for Alphabet (GOOGL): the live price, a chart of the last 365 days (plus today's intraday move), year stats, and a 30-day trend forecast.

## Use it
1. Get a free API key at https://twelvedata.com (the free plan allows 800 requests a day).
2. Open the app, tap **Data source**, paste the key and tap **Save and load**. The key stays in your browser on that device.
Shortcut: open the app once as `https://joelscodes.github.io/googl-watch/#key=YOURKEY`. It saves the key on that device and removes it from the address bar. The key is never stored in this repo.
3. On iPhone: Share → Add to Home Screen. On Android: menu → Add to Home screen.

Without a key it shows clearly labeled sample data.

## How the forecast works
Five signals vote up or down: price vs the 50-day average, the 50-day vs the 200-day average, MACD momentum, RSI(14), and the slope of the last 60 closes. The projection is anchored on the average daily return of up to the last three years, with a small tilt from the vote (at most about ±2% over 21 days). An earlier version also followed the recent 60-day trend; on real GOOGL data that made it call declines that mostly did not come, so it was dropped, and its range comes from the last 60 days' volatility (80% band). The app also backtests the signals over the past year and shows how often they were right next to an "always up" baseline.

## Accuracy check
The app replays its own forecast on every past trading day in the last ~5 years of data (one Twelve Data request), using only prices known on that day, and scores it against what happened 21 trading days later: direction vs always guessing "up", target error vs assuming no change, how often the price landed in the 80% range, and the average move after Bullish, Neutral and Bearish calls.

## News and reactions
With a free [Finnhub](https://finnhub.io/register) key (Data source panel, or `#finnhub=KEY` in the setup link), the app loads a year of Alphabet headlines, one day per request because Finnhub caps each reply to its newest stories (about 7 minutes the first time, paced under the free 60 calls a minute; days are cached in IndexedDB so later opens only fetch the last couple of days), and matches each one to the first trading session it could move: the same day if published before the 4pm ET close, otherwise the next session. For each day it shows GOOGL's move, the Nasdaq-100's move (QQQ), the difference, how unusual it was against the previous 60 days, and the next 5 days. Headlines are tagged by topic (AI, Cloud, Earnings, Legal, Analysts, YouTube, Search & ads, Waymo & bets, Devices) with an average reaction per topic, plus search and sorting. General market stories that don't mention Alphabet are left out.

The forecast does not use the news. It cannot see earnings dates either. This is not financial advice.

## Hosting
Plain static file. Settings → Pages → Deploy from branch `main`, folder `/`.
