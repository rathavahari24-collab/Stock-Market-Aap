"""
stock_analyzer.py
Technical + Fundamental + Machine-Learning stock health checker.
Data source: Yahoo Finance (via the `yfinance` library).

EDUCATIONAL USE ONLY - this is not investment advice.
"""
import io
import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib
import matplotlib.pyplot as plt
from sklearn.base import clone
from sklearn.ensemble import (AdaBoostClassifier, GradientBoostingClassifier,
                              RandomForestClassifier)
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.model_selection import TimeSeriesSplit, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier

# --------------------------------------------------------------------------
# 1. DATA DOWNLOAD
# --------------------------------------------------------------------------
def download_prices(ticker, start="2015-01-01", end=None):
    """Daily OHLCV data (split/dividend adjusted) from Yahoo Finance."""
    df = yf.download(ticker, start=start, end=end, auto_adjust=True, progress=False)
    if df is None or df.empty:
        raise ValueError(f"No price data found for '{ticker}'. "
                         "Use .NS for NSE, .BO for BSE (e.g. TCS.NS), no suffix for US stocks.")
    if isinstance(df.columns, pd.MultiIndex):          # newer yfinance versions
        df.columns = df.columns.get_level_values(0)
    df = df[["Open", "High", "Low", "Close", "Volume"]].dropna()
    df.index = pd.to_datetime(df.index).tz_localize(None)
    df.index.name = "Date"
    return df


def get_fundamentals(ticker):
    """Returns (info dict, {statement name: DataFrame or None})."""
    t = yf.Ticker(ticker)
    try:
        info = t.info or {}
    except Exception:
        info = {}
    statements = {}
    for name, attr in [("Income_Statement", "financials"),
                       ("Balance_Sheet", "balance_sheet"),
                       ("Cash_Flow", "cashflow")]:
        try:
            s = getattr(t, attr)
            statements[name] = s if (s is not None and not s.empty) else None
        except Exception:
            statements[name] = None
    return info, statements


# --------------------------------------------------------------------------
# 2. TECHNICAL INDICATORS (pure pandas - no extra library needed)
# --------------------------------------------------------------------------
def add_indicators(df):
    d = df.copy()
    c, h, l, v = d["Close"], d["High"], d["Low"], d["Volume"]

    # Trend: moving averages
    for n in (20, 50, 200):
        d[f"SMA_{n}"] = c.rolling(n).mean()
    d["EMA_12"] = c.ewm(span=12, adjust=False).mean()
    d["EMA_26"] = c.ewm(span=26, adjust=False).mean()

    # Momentum: MACD
    d["MACD"] = d["EMA_12"] - d["EMA_26"]
    d["MACD_signal"] = d["MACD"].ewm(span=9, adjust=False).mean()
    d["MACD_hist"] = d["MACD"] - d["MACD_signal"]

    # Momentum: RSI (Wilder)
    delta = c.diff()
    gain = delta.clip(lower=0).ewm(alpha=1 / 14, adjust=False).mean()
    loss = (-delta.clip(upper=0)).ewm(alpha=1 / 14, adjust=False).mean()
    d["RSI_14"] = 100 - 100 / (1 + gain / loss)

    # Momentum: Stochastic oscillator
    low14, high14 = l.rolling(14).min(), h.rolling(14).max()
    d["Stoch_K"] = (c - low14) / (high14 - low14) * 100
    d["Stoch_D"] = d["Stoch_K"].rolling(3).mean()

    # Volatility: Bollinger Bands and ATR
    mid, sd = c.rolling(20).mean(), c.rolling(20).std()
    d["BB_upper"], d["BB_lower"] = mid + 2 * sd, mid - 2 * sd
    d["BB_pct"] = (c - d["BB_lower"]) / (d["BB_upper"] - d["BB_lower"])
    tr = pd.concat([h - l, (h - c.shift()).abs(), (l - c.shift()).abs()], axis=1).max(axis=1)
    d["ATR_14"] = tr.ewm(alpha=1 / 14, adjust=False).mean()

    # Volume: On-Balance Volume
    d["OBV"] = (np.sign(c.diff()).fillna(0) * v).cumsum()

    # Returns / risk
    d["ret_1"], d["ret_5"], d["ret_20"] = c.pct_change(), c.pct_change(5), c.pct_change(20)
    d["volatility_20"] = d["ret_1"].rolling(20).std()
    d["vol_ratio"] = v / v.rolling(20).mean()
    for n in (20, 50, 200):
        d[f"dist_sma{n}"] = c / d[f"SMA_{n}"] - 1
    d["atr_pct"] = d["ATR_14"] / c
    d["macd_hist_pct"] = d["MACD_hist"] / c
    return d


def technical_score(d):
    """Rule-based score for the latest day. Returns (score 0-100, signals table)."""
    r = d.iloc[-1]
    obv_avg = d["OBV"].rolling(20).mean().iloc[-1]
    rules = []

    def add(name, value, point, why):
        rules.append({"Indicator": name, "Value": value, "Signal": {1: "Bullish", -1: "Bearish", 0: "Neutral"}[point],
                      "Meaning": why, "_p": point})

    add("Price vs SMA200", round(r["Close"] / r["SMA_200"] - 1, 4) if pd.notna(r["SMA_200"]) else None,
        0 if pd.isna(r["SMA_200"]) else (1 if r["Close"] > r["SMA_200"] else -1), "Long-term trend")
    add("Price vs SMA50", round(r["Close"] / r["SMA_50"] - 1, 4) if pd.notna(r["SMA_50"]) else None,
        0 if pd.isna(r["SMA_50"]) else (1 if r["Close"] > r["SMA_50"] else -1), "Medium-term trend")
    add("SMA50 vs SMA200", round(r["SMA_50"] - r["SMA_200"], 2) if pd.notna(r["SMA_200"]) else None,
        0 if pd.isna(r["SMA_200"]) else (1 if r["SMA_50"] > r["SMA_200"] else -1), "Golden / death cross state")
    add("MACD vs Signal", round(r["MACD_hist"], 3), 1 if r["MACD_hist"] > 0 else -1, "Momentum direction")
    rsi = r["RSI_14"]
    add("RSI (14)", round(rsi, 1), 1 if rsi < 30 else (-1 if rsi > 70 else 0), "<30 oversold, >70 overbought")
    bb = r["BB_pct"]
    add("Bollinger %B", round(bb, 2), 1 if bb < 0.05 else (-1 if bb > 0.95 else 0), "Near lower / upper band")
    add("Volume flow (OBV)", round(r["OBV"], 0), 0 if pd.isna(obv_avg) else (1 if r["OBV"] > obv_avg else -1),
        "Money flowing in or out")
    table = pd.DataFrame(rules)
    score = (table["_p"].sum() + len(table)) / (2 * len(table)) * 100
    return round(score, 1), table.drop(columns="_p")


# --------------------------------------------------------------------------
# 3. FUNDAMENTAL ANALYSIS
# --------------------------------------------------------------------------
# (label, yfinance key, "good" test, "bad" test). Thresholds are general rules of thumb.
FUND_RULES = [
    ("P/E ratio (trailing)", "trailingPE", lambda x: 0 < x < 25, lambda x: x <= 0 or x > 40),
    ("Price / Book", "priceToBook", lambda x: 0 < x < 3, lambda x: x > 6),
    ("Return on Equity", "returnOnEquity", lambda x: x > 0.15, lambda x: x < 0.08),
    ("Debt / Equity (%)", "debtToEquity", lambda x: x < 100, lambda x: x > 200),
    ("Profit margin", "profitMargins", lambda x: x > 0.10, lambda x: x < 0.03),
    ("Revenue growth", "revenueGrowth", lambda x: x > 0.10, lambda x: x < 0),
    ("Earnings growth", "earningsGrowth", lambda x: x > 0.10, lambda x: x < 0),
    ("Current ratio", "currentRatio", lambda x: x > 1.5, lambda x: x < 1),
]


def fundamental_score(info):
    """Returns (score 0-100 or None if no data, table)."""
    rows, pts = [], []
    for label, key, good, bad in FUND_RULES:
        x = info.get(key)
        if x is None or (isinstance(x, float) and np.isnan(x)):
            continue
        p = 1 if good(x) else (-1 if bad(x) else 0)
        pts.append(p)
        rows.append({"Metric": label, "Value": round(float(x), 4),
                     "Verdict": {1: "Good", -1: "Weak", 0: "Average"}[p]})
    if not pts:
        return None, pd.DataFrame(columns=["Metric", "Value", "Verdict"])
    return round((sum(pts) + len(pts)) / (2 * len(pts)) * 100, 1), pd.DataFrame(rows)


def company_snapshot(info):
    keys = ["longName", "sector", "industry", "marketCap", "currentPrice", "fiftyTwoWeekHigh",
            "fiftyTwoWeekLow", "dividendYield", "beta", "trailingEps", "bookValue"]
    return pd.DataFrame({"Item": keys, "Value": [info.get(k, "NA") for k in keys]})


# --------------------------------------------------------------------------
# 4. MACHINE LEARNING (will the price be higher in `horizon` trading days?)
# --------------------------------------------------------------------------
ML_FEATURES = ["RSI_14", "Stoch_K", "Stoch_D", "BB_pct", "macd_hist_pct", "atr_pct",
               "ret_1", "ret_5", "ret_20", "volatility_20", "vol_ratio",
               "dist_sma20", "dist_sma50", "dist_sma200"]


def build_ml_data(d, horizon=5):
    x = d[ML_FEATURES].replace([np.inf, -np.inf], np.nan)
    future = d["Close"].shift(-horizon)
    ok = x.notna().all(axis=1)
    labelled = ok & future.notna()
    X = x[labelled]
    y = (future[labelled] > d["Close"][labelled]).astype(int)
    X_latest = x[ok].iloc[[-1]]
    return X, y, X_latest


def get_models():
    def pipe(m):
        return Pipeline([("scale", StandardScaler()), ("model", m)])
    return {
        "Logistic Regression": pipe(LogisticRegression(max_iter=1000)),
        "Decision Tree": pipe(DecisionTreeClassifier(max_depth=5, random_state=42)),
        "Random Forest": pipe(RandomForestClassifier(n_estimators=200, max_depth=6, min_samples_leaf=20,
                                                     random_state=42, n_jobs=-1)),
        "AdaBoost": pipe(AdaBoostClassifier(random_state=42)),
        "Gradient Boosting": pipe(GradientBoostingClassifier(max_depth=2, n_estimators=100, random_state=42)),
    }


def train_and_evaluate(d, horizon=5):
    """Time-ordered split (no shuffling, so no peeking into the future)."""
    X, y, X_latest = build_ml_data(d, horizon)
    if len(X) < 400:
        raise ValueError("Not enough history for the ML model (need roughly 2+ years of data).")
    split = int(len(X) * 0.8)
    X_tr, y_tr = X.iloc[:split - horizon], y.iloc[:split - horizon]   # gap avoids label overlap
    X_te, y_te = X.iloc[split:], y.iloc[split:]
    tscv = TimeSeriesSplit(n_splits=5, gap=horizon)

    rows, models = [], get_models()
    for name, m in models.items():
        cv = cross_val_score(m, X_tr, y_tr, cv=tscv, scoring="accuracy").mean()
        m.fit(X_tr, y_tr)
        p = m.predict(X_te)
        rows.append({"Model": name, "CV Accuracy": cv,
                     "Test Accuracy": accuracy_score(y_te, p),
                     "Test Precision": precision_score(y_te, p, zero_division=0),
                     "Test Recall": recall_score(y_te, p, zero_division=0),
                     "Test F1": f1_score(y_te, p, zero_division=0)})
    results = pd.DataFrame(rows).sort_values("CV Accuracy", ascending=False).reset_index(drop=True)
    best = results.loc[0, "Model"]                                    # chosen on CV, not on test
    final = clone(models[best]).fit(X, y)
    prob_up = float(final.predict_proba(X_latest)[0, 1])
    baseline = float(max(y_te.mean(), 1 - y_te.mean()))               # "always guess the majority" accuracy
    return {"results": results, "best_name": best, "model": final, "prob_up": prob_up,
            "baseline": baseline, "features": ML_FEATURES}


# --------------------------------------------------------------------------
# 5. FINAL VERDICT
# --------------------------------------------------------------------------
def final_verdict(tech, fund, prob_up):
    parts = {"Technical": (tech, 0.4), "Fundamental": (fund, 0.4), "ML outlook": (prob_up * 100, 0.2)}
    parts = {k: v for k, v in parts.items() if v[0] is not None}
    total_w = sum(w for _, w in parts.values())
    overall = sum(s * w for s, w in parts.values()) / total_w
    if overall >= 65:
        label = "STRONG - performing well"
    elif overall >= 50:
        label = "MODERATE - mildly positive"
    elif overall >= 35:
        label = "WEAK - mildly negative"
    else:
        label = "POOR - performing badly"
    return round(overall, 1), label


def analyze(ticker, start="2015-01-01", horizon=5):
    prices = download_prices(ticker, start)
    d = add_indicators(prices)
    info, statements = get_fundamentals(ticker)
    t_score, t_table = technical_score(d)
    f_score, f_table = fundamental_score(info)
    ml = train_and_evaluate(d, horizon)
    overall, label = final_verdict(t_score, f_score, ml["prob_up"])
    return {"ticker": ticker, "prices": prices, "data": d, "info": info, "statements": statements,
            "tech_score": t_score, "tech_table": t_table, "fund_score": f_score, "fund_table": f_table,
            "ml": ml, "overall": overall, "label": label, "horizon": horizon}


# --------------------------------------------------------------------------
# 6. CHARTS AND EXCEL EXPORT
# --------------------------------------------------------------------------
def plot_dashboard(d, ticker, last_n=500):
    x = d.tail(last_n)
    fig, ax = plt.subplots(3, 1, figsize=(12, 9), sharex=True, gridspec_kw={"height_ratios": [3, 1, 1]})
    ax[0].plot(x.index, x["Close"], label="Close", color="black", lw=1.2)
    for n, col in [(20, "tab:blue"), (50, "tab:orange"), (200, "tab:red")]:
        ax[0].plot(x.index, x[f"SMA_{n}"], label=f"SMA {n}", color=col, lw=1)
    ax[0].fill_between(x.index, x["BB_lower"], x["BB_upper"], alpha=0.12, color="grey", label="Bollinger")
    ax[0].set_title(f"{ticker} - Price, Moving Averages, Bollinger Bands"); ax[0].legend(loc="upper left", ncol=5)
    ax[1].plot(x.index, x["RSI_14"], color="purple"); ax[1].axhline(70, ls="--", c="r"); ax[1].axhline(30, ls="--", c="g")
    ax[1].set_ylabel("RSI")
    ax[2].bar(x.index, x["MACD_hist"], color=np.where(x["MACD_hist"] >= 0, "green", "red"), width=1)
    ax[2].plot(x.index, x["MACD"], c="blue", lw=1); ax[2].plot(x.index, x["MACD_signal"], c="orange", lw=1)
    ax[2].set_ylabel("MACD")
    fig.tight_layout()
    return fig


def build_excel_bytes(prices, indicators, info, statements, tech_table, fund_table, ml_results):
    """Creates the Excel workbook in memory and returns bytes."""
    buf = io.BytesIO()
    with pd.ExcelWriter(buf, engine="openpyxl") as xw:
        prices.to_excel(xw, sheet_name="Price_Data")
        indicators.to_excel(xw, sheet_name="Technical_Indicators")
        company_snapshot(info).astype(str).to_excel(xw, sheet_name="Company_Snapshot", index=False)
        fund_table.to_excel(xw, sheet_name="Fundamental_Ratios", index=False)
        tech_table.to_excel(xw, sheet_name="Technical_Signals", index=False)
        ml_results.to_excel(xw, sheet_name="ML_Model_Results", index=False)
        for name, df in statements.items():
            if df is not None:
                df.to_excel(xw, sheet_name=name)
    return buf.getvalue()
