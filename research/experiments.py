"""Can GOOGL's 21-trading-day direction be predicted better than "it usually goes up"?

Walk-forward only: every prediction at day t uses data up to t, and models are refit on samples whose
21-day outcome was already known at t (purged). Development period 2012-2021 is used to pick a model;
2022 onward is a holdout looked at once, at the end.

Usage: python research/experiments.py DATA_DIR
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
D = sys.argv[1]
H = int(sys.argv[2]) if len(sys.argv) > 2 else 21
DEV_START, HOLD_START = "2012-01-01", "2022-01-01"

def load(sym):
    df = pd.read_csv(f"{D}/{sym}.csv", parse_dates=["date"]).set_index("date")
    return df

g = load("GOOGL"); px = pd.DataFrame({"c": g["Close"], "v": g["Volume"]})
for s in ["QQQ", "SPY", "VIX", "TNX", "MSFT", "AAPL", "AMZN", "META", "NVDA"]:
    px[s] = load(s)["Close"]
px = px.dropna(subset=["c"]).ffill()
c, lc = px["c"], np.log(px["c"])

# ---------- features (all known at the close of day t) ----------
F = pd.DataFrame(index=px.index)
for k in [1, 5, 10, 21, 63, 126, 252]:
    F[f"r{k}"] = lc.diff(k)
for k in [21, 63]:
    F[f"rel{k}"] = lc.diff(k) - np.log(px["QQQ"]).diff(k)
for k in [20, 50, 200]:
    F[f"dma{k}"] = c / c.rolling(k).mean() - 1
d = c.diff(); up, dn = d.clip(lower=0).ewm(alpha=1/14).mean(), (-d.clip(upper=0)).ewm(alpha=1/14).mean()
F["rsi"] = 100 - 100 / (1 + up / dn)
ret = lc.diff()
F["vol21"], F["vol63"] = ret.rolling(21).std(), ret.rolling(63).std()
F["volratio"] = F["vol21"] / F["vol63"]
F["volu_z"] = (np.log(px["v"]) - np.log(px["v"]).rolling(63).mean()) / np.log(px["v"]).rolling(63).std()
F["dd252"] = c / c.rolling(252).max() - 1
F["vix"] = px["VIX"]; F["vix_ch5"] = np.log(px["VIX"]).diff(5); F["vix_rel"] = px["VIX"] / px["VIX"].rolling(252).mean()
F["tnx"] = px["TNX"]; F["tnx_ch21"] = px["TNX"].diff(21)
q = np.log(px["QQQ"])
F["q21"], F["q63"], F["qdma200"] = q.diff(21), q.diff(63), px["QQQ"] / px["QQQ"].rolling(200).mean() - 1
mc = ["MSFT", "AAPL", "AMZN", "NVDA"]  # META only listed in 2012
F["breadth"] = sum((px[m] > px[m].rolling(50).mean()).astype(float) for m in mc) / len(mc)
F["month"] = px.index.month
F["earn_season"] = px.index.month.isin([1, 4, 7, 10]).astype(float)  # Alphabet reports late Jan/Apr/Jul/Oct

y = lc.shift(-H) - lc                     # forward log return, unknown at t
Y = (y > 0).astype(float).where(y.notna())
F = F.replace([np.inf, -np.inf], np.nan)

# ---------- models: each returns a score per day where higher = more bullish ----------
def walk(model_fn, cols, refit=63, min_train=1500):
    out = pd.Series(np.nan, index=F.index)
    idx = F.index[F.index >= DEV_START]
    X = F[cols]
    for start in range(0, len(idx), refit):
        block = idx[start:start + refit]
        t0 = F.index.get_loc(block[0])
        tr = slice(0, t0 - H)             # purge: only samples whose outcome was known at t0
        Xt, Yt = X.iloc[tr], Y.iloc[tr]
        ok = Xt.notna().all(1) & Yt.notna()
        if ok.sum() < min_train:
            continue
        m = model_fn(); m.fit(Xt[ok].values, Yt[ok].values)
        Xb = X.loc[block]; okb = Xb.notna().all(1)
        if okb.any():
            out.loc[Xb.index[okb]] = m.predict_proba(Xb[okb].values)[:, 1]
    return out

def knn_shapes(L=20, k=50, refit=21):
    """Pattern matching: z-normalised shape of the last L closes; average outcome of the k most similar past shapes."""
    arr = lc.values; n = len(arr)
    shp = np.full((n, L), np.nan)
    for t in range(L, n):
        w = arr[t-L+1:t+1] - arr[t]; s = w.std()
        if s > 0: shp[t] = w / s
    out = pd.Series(np.nan, index=F.index); yv = Y.values
    for t in np.where(F.index >= DEV_START)[0]:
        hist = np.arange(L, t - H)
        hv = hist[~np.isnan(yv[hist])]
        dist = ((shp[hv] - shp[t]) ** 2).sum(1)
        nn = hv[np.argpartition(dist, k)[:k]]
        out.iloc[t] = yv[nn].mean()
    return out

def app_rules():
    """The app's five signals (price vs 50-day avg, 50 vs 200-day avg, MACD, RSI, 60-day slope)."""
    s50, s200 = c.rolling(50).mean(), c.rolling(200).mean()
    macd = c.ewm(span=12, adjust=False).mean() - c.ewm(span=26, adjust=False).mean()
    hist = macd - macd.ewm(span=9, adjust=False).mean()
    slope = lc.rolling(60).apply(lambda a: np.polyfit(np.arange(60), a, 1)[0] * 252, raw=True)
    sc = np.sign(c - s50) + np.sign(s50 - s200) + np.sign(hist) + np.where(F["rsi"] < 30, 1, np.where(F["rsi"] > 70, -1, 0)) + np.where(slope > .05, 1, np.where(slope < -.05, -1, 0))
    return pd.Series(sc, index=F.index).astype(float)

base_cols = ["r5", "r21", "r63", "r252", "dma50", "dma200", "rsi", "vol21", "volratio", "dd252"]
ctx_cols = base_cols + ["rel21", "rel63", "vix", "vix_rel", "vix_ch5", "tnx_ch21", "q21", "q63", "qdma200", "breadth", "volu_z", "earn_season"]

logit = lambda: make_pipeline(StandardScaler(), LogisticRegression(C=0.05, max_iter=500))
gbm = lambda: lgb.LGBMClassifier(n_estimators=150, learning_rate=0.03, num_leaves=7, min_child_samples=100, subsample=0.8, subsample_freq=1, colsample_bytree=0.7, n_jobs=2, verbose=-1)

print(f"Horizon {H} trading days. Data {px.index[0].date()} to {px.index[-1].date()}.", flush=True)
S = {
    "App's 5 signals (score)": app_rules(),
    "12-month momentum (r252)": F["r252"],
    "1-month reversal (-r21)": -F["r21"],
    "Fear gauge: VIX level": F["vix"],
    "Fear gauge: VIX vs 1y avg": F["vix_rel"],
    "Drawdown from 1y high (deeper=buy)": -F["dd252"],
    "Logistic, price features": walk(logit, base_cols),
    "Logistic, + market context": walk(logit, ctx_cols),
    "Gradient boosting, price features": walk(gbm, base_cols),
    "Gradient boosting, + market context": walk(gbm, ctx_cols),
    "Pattern match, 20-day shapes (kNN)": knn_shapes(20),
    "Pattern match, 60-day shapes (kNN)": knn_shapes(60),
}

def evaluate(score, lo, hi):
    m = (F.index >= lo) & (F.index < hi) & score.notna().values & Y.notna().values
    s, yy, rr = score[m], Y[m], y[m]
    if len(s) < 200: return None
    auc = roc_auc_score(yy, s)
    # bullish when the score is above its own trailing median (known at the time): directional call
    thr = score.rolling(756, min_periods=252).median().shift(1)[m]
    call = (s > thr)
    acc = ((call & (yy == 1)) | (~call & (yy == 0))).mean()
    # non-overlapping samples for significance: every H-th day
    nov = np.arange(0, len(s), H)
    try: p_auc = stats.mannwhitneyu(s.iloc[nov][yy.iloc[nov] == 1], s.iloc[nov][yy.iloc[nov] == 0], alternative="greater").pvalue
    except ValueError: p_auc = np.nan
    # trading test: hold GOOGL over the next H days when bullish, else cash; non-overlapping rebalances
    strat = np.where(call.iloc[nov], rr.iloc[nov], 0.0); bh = rr.iloc[nov].values
    yrs = len(nov) * H / 252
    return dict(n=len(s), indep=len(nov), auc=auc, p=p_auc, acc=acc, base=yy.mean(), up_when_bull=yy[call].mean(), up_when_bear=yy[~call].mean(),
                ret_bull=rr[call].mean(), ret_bear=rr[~call].mean(), cagr=np.exp(strat.sum() / yrs) - 1, cagr_bh=np.exp(bh.sum() / yrs) - 1,
                inmkt=call.iloc[nov].mean())

def table(lo, hi, title, only=None):
    print(f"\n=== {title} ===")
    print(f"{'model':40s} {'AUC':>5} {'p':>6} {'acc':>5} {'base':>5} {'upIfBull':>8} {'upIfBear':>8} {'ret|bull':>8} {'ret|bear':>8} {'CAGR':>6} {'B&H':>6} {'inMkt':>5}")
    rows = {}
    for name, sc in S.items():
        if only and name not in only: continue
        r = evaluate(sc, lo, hi)
        if not r: continue
        rows[name] = r
        print(f"{name:40s} {r['auc']:.3f} {r['p']:6.3f} {r['acc']:.3f} {r['base']:.3f} {r['up_when_bull']:8.3f} {r['up_when_bear']:8.3f} {r['ret_bull']*100:7.2f}% {r['ret_bear']*100:7.2f}% {r['cagr']*100:5.1f}% {r['cagr_bh']*100:5.1f}% {r['inmkt']:.2f}")
    return rows

dev = table(DEV_START, HOLD_START, f"DEVELOPMENT {DEV_START[:4]}-2021 (model selection happens here)")
best = max(dev, key=lambda k: dev[k]["auc"])
print(f"\nPicked on development AUC: {best}")
table(HOLD_START, "2100-01-01", "HOLDOUT 2022-now (all models shown; only the picked one counts)")

# Is the size of the next month's move predictable? (the app's range)
absy = y.abs()
m = (F.index >= DEV_START) & absy.notna().values & F["vol21"].notna().values
print(f"\nVolatility check: correlation of 21-day realised vol with the size of the next {H}-day move: "
      f"{stats.spearmanr(F['vol21'][m], absy[m]).correlation:.2f} (VIX: {stats.spearmanr(F['vix'][m], absy[m]).correlation:.2f})")

# The size of future swings (volatility), unlike direction, is known to persist. Measure it properly:
fvol = ret[::-1].rolling(H).std()[::-1].shift(-1)        # realised daily vol over the next H days
m = (F.index >= DEV_START) & fvol.notna().values & F["vol63"].notna().values
for name, x in [("past 21-day vol", F["vol21"]), ("past 63-day vol", F["vol63"]), ("VIX", F["vix"])]:
    print(f"Correlation of {name} with realised vol over the next {H} days: {stats.spearmanr(x[m], fvol[m]).correlation:.2f}")
