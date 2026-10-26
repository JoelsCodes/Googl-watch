"""Can GOOGL's next 5-minute or next 1-hour move be predicted? Same rules as experiments.py: walk-forward,
models refit only on bars whose outcome was known, earlier part of the sample for choosing, later part held out.

Usage: python research/intraday.py DATA_DIR 5m|1h
"""
import sys, warnings
import numpy as np, pandas as pd
from scipy import stats
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.metrics import roc_auc_score
import lightgbm as lgb

warnings.filterwarnings("ignore")
D, IV = sys.argv[1], sys.argv[2]
COST_BP = 1.0   # assumed round-trip cost per trade in basis points (spread + slippage), generous for GOOGL

def load(sym):
    df = pd.read_csv(f"{D}/{sym}_{IV}.csv")
    df["date"] = pd.to_datetime(df["date"], utc=True).dt.tz_convert("America/New_York")
    return df.set_index("date")

g, q = load("GOOGL"), load("QQQ")
g = g[(g.index.time >= pd.Timestamp("09:30").time()) & (g.index.time < pd.Timestamp("16:00").time())]
px = pd.DataFrame({"c": g["Close"], "h": g["High"], "l": g["Low"], "o": g["Open"], "v": g["Volume"], "qc": q["Close"].reindex(g.index).ffill()}).dropna()
day = px.index.date
lc, lq = np.log(px["c"]), np.log(px["qc"])
r = lc.diff()
newday = pd.Series(day, index=px.index) != pd.Series(day, index=px.index).shift(1)
r_intra = r.where(~newday)                                  # bar returns inside the session
F = pd.DataFrame(index=px.index)
for k in [1, 3, 6, 12]:
    F[f"r{k}"] = lc.diff(k)
    F[f"q{k}"] = lq.diff(k)
F["rel3"] = F["r3"] - F["q3"]
openp = px["o"].groupby(day).transform("first")
F["since_open"] = np.log(px["c"] / openp)
prev_close = px["c"].groupby(day).transform("last").shift(1).where(newday).ffill()
F["gap"] = np.log(openp / prev_close.groupby(day).transform("first"))
F["tod"] = px.index.hour * 60 + px.index.minute
F["range1"] = (px["h"] - px["l"]) / px["c"]
lv = np.log(px["v"].replace(0, np.nan))
F["volz"] = lv - lv.groupby(F["tod"]).transform(lambda s: s.expanding().median().shift(1))
F["absr_avg12"] = r_intra.abs().rolling(12).mean()
F = F.replace([np.inf, -np.inf], np.nan)

y = r.shift(-1)                                              # next bar's return
last_bar = newday.shift(-1).fillna(True)                     # next bar is in a new session: overnight, skip
y = y.where(~last_bar)
Y = (y > 0).astype(float).where(y.notna() & (y != 0))

days = sorted(set(day)); cut = days[int(len(days) * 0.6)]
DEV = np.array([d < cut for d in day]); HOLD = ~DEV
print(f"{IV} bars: {len(px):,} from {px.index[0].date()} to {px.index[-1].date()} ({len(days)} sessions). Development before {cut}, holdout after.")

cols = list(F.columns)
logit = lambda: make_pipeline(StandardScaler(), LogisticRegression(C=0.05, max_iter=500))
gbm = lambda: lgb.LGBMClassifier(n_estimators=150, learning_rate=0.03, num_leaves=7, min_child_samples=200, subsample=0.8, subsample_freq=1, colsample_bytree=0.7, n_jobs=2, verbose=-1)

def walk(model_fn, refit_days=5, min_train=1000):
    out = pd.Series(np.nan, index=F.index); X = F[cols]
    dser = pd.Series(day, index=F.index)
    for i in range(0, len(days), refit_days):
        block = dser.isin(days[i:i + refit_days]).values
        t0 = np.argmax(block)
        Xt, Yt = X.iloc[:max(t0 - 1, 0)], Y.iloc[:max(t0 - 1, 0)]
        ok = Xt.notna().all(1) & Yt.notna()
        if ok.sum() < min_train: continue
        m = model_fn(); m.fit(Xt[ok].values, Yt[ok].values)
        Xb = X[block]; okb = Xb.notna().all(1)
        if okb.any(): out.loc[Xb.index[okb]] = m.predict_proba(Xb[okb].values)[:, 1]
    return out

def knn(L=12, k=50):
    arr = lc.values; n = len(arr); shp = np.full((n, L), np.nan)
    for t in range(L, n):
        w = arr[t-L+1:t+1] - arr[t]; s = w.std()
        if s > 0: shp[t] = w / s
    out = pd.Series(np.nan, index=F.index); yv = Y.values
    ok_hist = ~np.isnan(yv) & ~np.isnan(shp).any(1)
    for t in range(L, n):
        if np.isnan(shp[t]).any(): continue
        hv = np.where(ok_hist[:t - 1])[0]
        if len(hv) < 500: continue
        dist = ((shp[hv] - shp[t]) ** 2).sum(1)
        out.iloc[t] = yv[hv[np.argpartition(dist, k)[:k]]].mean()
    return out

S = {
    "Last bar continues (momentum)": F["r1"],
    "Last bar reverses": -F["r1"],
    "Last hour-ish trend (12 bars)": F["r12"],
    "Nasdaq-100 last bar (lead-lag)": F["q1"],
    "GOOGL lagging the market (catch-up)": -F["rel3"],
    "Move since the open reverses": -F["since_open"],
    "Logistic regression, all features": walk(logit),
    "Gradient boosting, all features": walk(gbm),
    "Pattern match, last 12 bars (kNN)": knn(12),
}

def evaluate(score, mask):
    m = mask & score.notna().values & Y.notna().values
    s, yy, rr = score[m], Y[m], y[m]
    if len(s) < 200: return None
    auc = roc_auc_score(yy, s)
    thr = score.rolling(2000, min_periods=300).median().shift(1)[m]
    call = s > thr
    acc = ((call & (yy == 1)) | (~call & (yy == 0))).mean()
    p = stats.mannwhitneyu(s[yy == 1], s[yy == 0], alternative="greater").pvalue
    # long when bullish, short when bearish, every bar; gross edge per trade in basis points
    edge = (np.where(call, rr, -rr)).mean() * 1e4
    return dict(n=len(s), auc=auc, p=p, acc=acc, base=yy.mean(), edge=edge)

for title, mask in [("DEVELOPMENT", DEV), ("HOLDOUT", HOLD)]:
    print(f"\n=== {title} ===\n{'model':40s} {'bars':>6} {'AUC':>5} {'p':>6} {'acc':>5} {'base':>5} {'edge bp/trade':>13}")
    rows = {}
    for name, sc in S.items():
        e = evaluate(sc, mask)
        if not e: continue
        rows[name] = e
        print(f"{name:40s} {e['n']:6d} {e['auc']:.3f} {e['p']:6.3f} {e['acc']:.3f} {e['base']:.3f} {e['edge']:13.2f}")
    if title == "DEVELOPMENT":
        best = max(rows, key=lambda k: rows[k]["auc"]); print(f"Picked on development AUC: {best}")
print(f"\nAn edge below about {COST_BP:.0f} bp per trade is eaten by trading costs.")

# Swing size: next bar's absolute move vs recent absolute moves and time of day
a = y.abs(); m = a.notna() & F["absr_avg12"].notna()
print(f"Rank correlation of the last 12 bars' average move with the size of the next bar's move: {stats.spearmanr(F['absr_avg12'][m], a[m]).correlation:.2f}")
tod_avg = a.groupby(F["tod"]).transform(lambda s: s.expanding().mean().shift(1))
m2 = m & tod_avg.notna()
print(f"Rank correlation of the usual move size at that time of day with the next bar's move: {stats.spearmanr(tod_avg[m2], a[m2]).correlation:.2f}")
