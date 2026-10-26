"""Do non-price signals predict GOOGL? Calendar effects, moon phase, New York weather, geomagnetic storms and sunspots,
Wikipedia attention, and macro mood (policy uncertainty, credit spreads, yield curve, dollar, oil, bitcoin).

Each signal is aligned to what was known at the close of day t (anything published later is lagged a day).
A signal's direction is set from the development period (2012-2021); the holdout (2022-now) is then a single check.
With ~30 signals x 3 horizons, about 1 in 20 will look "significant" in development by chance, so a signal only
counts if it also holds on the holdout with the same sign.

Usage: python research/altdata.py DATA_DIR
"""
import sys, json, math, warnings
import numpy as np, pandas as pd
from scipy import stats
import lightgbm as lgb
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.metrics import roc_auc_score

warnings.filterwarnings("ignore")
D = sys.argv[1]; A = f"{D}/alt"
DEV, HOLD = ("2012-01-01", "2022-01-01"), ("2022-01-01", "2100-01-01")

g = pd.read_csv(f"{D}/GOOGL.csv", parse_dates=["date"]).set_index("date")
idx = g.index; lc = np.log(g["Close"])
F = pd.DataFrame(index=idx)

def on_days(s, lag=0):
    """Align a calendar-day series to trading days (last value on or before each day), optionally lagged by trading days."""
    s = s.sort_index(); s = s[~s.index.duplicated()]
    out = s.reindex(s.index.union(idx)).ffill().reindex(idx)
    return out.shift(lag)

# ---- calendar
nxt = pd.Series(idx, index=idx).shift(-1); prv = pd.Series(idx, index=idx).shift(1)
F["monday"] = (idx.dayofweek == 0).astype(float)
F["friday"] = (idx.dayofweek == 4).astype(float)
month = idx.to_period("M")
last_of_month = pd.Series(month, index=idx) != pd.Series(month, index=idx).shift(-1)
F["turn_of_month"] = (last_of_month | (pd.Series(month, index=idx) != pd.Series(month, index=idx).shift(1)).rolling(3, min_periods=1).max().astype(bool)).astype(float)
gap = (nxt - pd.Series(idx, index=idx)).dt.days
F["pre_holiday"] = ((gap > 3) | ((gap > 1) & (idx.dayofweek < 4))).astype(float)
F["sell_in_may"] = idx.month.isin([5, 6, 7, 8, 9, 10]).astype(float)
F["september"] = (idx.month == 9).astype(float)
F["santa"] = (((idx.month == 12) & (idx.day >= 24)) | ((idx.month == 1) & (idx.day <= 3))).astype(float)
F["earnings_month"] = idx.month.isin([1, 4, 7, 10]).astype(float)

# ---- moon (mean synodic month from a known new moon; accurate to well under a day)
syn = 29.530588853; ref = pd.Timestamp("2000-01-06 18:14")
age = ((idx - ref) / pd.Timedelta(days=1)) % syn
F["moon_illum"] = (1 - np.cos(2 * np.pi * age / syn)) / 2
F["near_new_moon"] = ((age < 3.5) | (age > syn - 3.5)).astype(float)

# ---- weather, New York Central Park
try:
    w = json.load(open(f"{A}/weather_nyc.json"))["daily"]; w = pd.DataFrame(w).assign(time=lambda x: pd.to_datetime(x["time"])).set_index("time")
    cc = w["cloud_cover_mean"]; cc_anom = cc - cc.groupby(cc.index.month).transform("mean")
    F["nyc_cloudy"] = on_days(cc_anom)                   # morning-ish weather of the trading day itself
    F["nyc_sunshine"] = on_days(w["sunshine_duration"] - w["sunshine_duration"].groupby(w.index.month).transform("mean"))
    F["nyc_rain"] = on_days((w["precipitation_sum"] > 5).astype(float))
    F["nyc_temp_anom"] = on_days(w["temperature_2m_mean"] - w["temperature_2m_mean"].groupby(w.index.month).transform("mean"))
except Exception as e: print("weather skipped:", e)

# ---- geomagnetic activity and sunspots (GFZ): columns YYYY MM DD days days_m Bsr dB Kp1..Kp8 ap1..ap8 Ap SN F10.7obs F10.7adj D
try:
    rows = [l.split() for l in open(f"{A}/kp_sunspots.txt") if l[:1].isdigit()]
    k = pd.DataFrame([(pd.Timestamp(int(r[0]), int(r[1]), int(r[2])), float(r[23]), float(r[24])) for r in rows], columns=["d", "Ap", "SN"]).set_index("d")
    k = k.replace(-1, np.nan)
    storm = (k["Ap"] >= 30).astype(float).rolling(6, min_periods=1).max()   # a storm in the past 6 days
    F["geomag_storm_6d"] = on_days(storm, lag=1)
    F["geomag_ap"] = on_days(np.log1p(k["Ap"]), lag=1)
    F["sunspots"] = on_days(k["SN"], lag=1)
except Exception as e: print("geomagnetic skipped:", e)

# ---- Wikipedia attention (views on day t are complete only after midnight UTC, so lag a day)
def wiki(name):
    j = json.load(open(f"{A}/wiki_{name}.json"))["items"]
    return pd.Series({pd.Timestamp(x["timestamp"][:8]): x["views"] for x in j}).sort_index()
try:
    att = wiki("Google") + wiki("Alphabet_Inc.").reindex(wiki("Google").index).fillna(0)
    la = np.log(att)
    F["wiki_attention_spike"] = on_days(la - la.rolling(30).mean(), lag=1)
    F["wiki_attention_trend"] = on_days(la.rolling(7).mean() - la.rolling(90).mean(), lag=1)
    ly = np.log(wiki("YouTube")); F["wiki_youtube_spike"] = on_days(ly - ly.rolling(30).mean(), lag=1)
    lp = np.log(wiki("Sundar_Pichai")); F["wiki_ceo_spike"] = on_days(lp - lp.rolling(30).mean(), lag=1)
    lg = np.log(wiki("Google_Gemini").clip(lower=1)); F["wiki_gemini_trend"] = on_days(lg.rolling(7).mean() - lg.rolling(90).mean(), lag=1)
except Exception as e: print("wikipedia skipped:", e)

# ---- macro mood (FRED; lag a day because some series post the next morning)
def fred(s):
    x = pd.read_csv(f"{A}/fred_{s}.csv"); x.columns = ["d", "v"]
    return pd.Series(pd.to_numeric(x["v"], errors="coerce").values, index=pd.to_datetime(x["d"])).dropna()
try:
    epu = np.log(fred("USEPUINDXD").rolling(7, min_periods=3).mean())
    F["policy_uncertainty"] = on_days(epu - epu.rolling(252).mean(), lag=1)
    hy = fred("BAMLH0A0HYM2"); F["junk_spread_chg21"] = on_days(hy - hy.shift(30), lag=1)
    F["yield_curve"] = on_days(fred("T10Y2Y"), lag=1)
    dx = np.log(fred("DTWEXBGS")); F["dollar_chg21"] = on_days(dx - dx.shift(30), lag=1)
    oil = np.log(fred("DCOILWTICO").clip(lower=1)); F["oil_chg21"] = on_days(oil - oil.shift(30), lag=1)
    btc = np.log(fred("CBBTCUSD")); F["bitcoin_chg5"] = on_days(btc - btc.shift(7), lag=1); F["bitcoin_chg21"] = on_days(btc - btc.shift(30), lag=1)
except Exception as e: print("fred skipped:", e)

F = F.replace([np.inf, -np.inf], np.nan)
print(f"{F.shape[1]} signals. Coverage starts:", ", ".join(f"{c} {F[c].first_valid_index().year}" for c in F.columns if F[c].first_valid_index() is not None and F[c].first_valid_index().year > 2005))

def test(x, y, H, lo, hi, sign=1):
    m = (idx >= lo) & (idx < hi) & x.notna().values & y.notna().values
    xs, ys = x[m] * sign, y[m]
    if len(xs) < 150 or xs.nunique() < 2: return None
    nov = np.arange(0, len(xs), H)                      # non-overlapping samples for the p-value
    ic, p = stats.spearmanr(xs.iloc[nov], ys.iloc[nov])
    auc = roc_auc_score((ys > 0).astype(int), xs) if ys.gt(0).nunique() > 1 else np.nan
    return dict(ic=ic, p=p / 2 if ic > 0 else 1 - p / 2, auc=auc, n=len(nov))   # one-sided: in the chosen direction

summary = []
for H in [1, 5, 21]:
    y = lc.shift(-H) - lc
    print(f"\n=== Next {H} trading day{'s' if H > 1 else ''}: development 2012-21 → holdout 2022-now ===")
    print(f"{'signal':24s} {'dev IC':>7} {'dev p':>6} {'dev AUC':>7} | {'hold IC':>7} {'hold p':>6} {'hold AUC':>8}  verdict")
    for c in F.columns:
        d0 = test(F[c], y, H, *DEV)
        if not d0: continue
        sgn = 1 if d0["ic"] >= 0 else -1                # direction is fixed on development data
        d = test(F[c], y, H, *DEV, sign=sgn); h = test(F[c], y, H, *HOLD, sign=sgn)
        if not h: continue
        v = "HOLDS UP" if d["p"] < 0.05 and h["p"] < 0.05 else ("dev only, failed holdout" if d["p"] < 0.05 else "")
        summary.append((H, c, d["p"], h["p"], h["ic"], v))
        print(f"{('-' if sgn < 0 else '+') + c:24s} {d['ic']:7.3f} {d['p']:6.3f} {d['auc']:7.3f} | {h['ic']:7.3f} {h['p']:6.3f} {h['auc']:8.3f}  {v}")

    # all signals together, walk-forward (refit every 63 days on outcomes known at the time)
    Y = (y > 0).astype(float).where(y.notna())
    cols = [c for c in F.columns if F[c].first_valid_index() is not None and F[c].first_valid_index() < pd.Timestamp("2012-01-01")]
    for label, mk in [("Logistic, all long-history signals", lambda: make_pipeline(StandardScaler(), LogisticRegression(C=0.05, max_iter=500))),
                      ("Gradient boosting, same signals", lambda: lgb.LGBMClassifier(n_estimators=150, learning_rate=0.03, num_leaves=7, min_child_samples=100, subsample=0.8, subsample_freq=1, colsample_bytree=0.7, n_jobs=2, verbose=-1))]:
        pred = pd.Series(np.nan, index=idx); ti = np.where(idx >= DEV[0])[0]
        for s in range(0, len(ti), 63):
            blk = ti[s:s+63]; t0 = blk[0]
            Xt, Yt = F[cols].iloc[:t0-H], Y.iloc[:t0-H]; ok = Xt.notna().all(1) & Yt.notna()
            if ok.sum() < 1000: continue
            m = mk(); m.fit(Xt[ok].values, Yt[ok].values)
            Xb = F[cols].iloc[blk]; okb = Xb.notna().all(1)
            if okb.any(): pred.iloc[blk[okb.values]] = m.predict_proba(Xb[okb].values)[:, 1]
        r = [test(pred, y, H, *P) for P in (DEV, HOLD)]
        if all(r): print(f"{label:38s} dev AUC {r[0]['auc']:.3f} (p {r[0]['p']:.3f}) | holdout AUC {r[1]['auc']:.3f} (p {r[1]['p']:.3f})")

n = len(summary); dev_hits = sum(1 for s in summary if s[2] < .05); both = [s for s in summary if s[5] == "HOLDS UP"]
print(f"\n{n} signal tests. {dev_hits} looked significant in development (about {n*0.05:.0f} expected by chance alone). "
      f"{len(both)} also held on the holdout (about {dev_hits*0.05:.1f} expected by chance): " + ("; ".join(f"{c} at {h}d" for h, c, *_ in both) or "none"))
