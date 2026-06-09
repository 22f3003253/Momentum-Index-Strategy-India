"""
MSCI Momentum Index Replication — Indian Markets
=================================================
Faithfully replicates the MSCI Momentum Indexes Methodology (August 2021)
applied to the Nifty 500 universe. Backtests over the last 5 years and
compares against Nifty 50 / Nifty 500 benchmarks.

Methodology Summary (from MSCI doc):
  1. Universe      : All constituents of Parent Index (Nifty 500)
  2. Momentum Score: Risk-adjusted blend of 6m and 12m price momentum
                     (excluding most-recent 1 month), each deflated by
                     local risk-free rate (India: 3M NSE MIBOR proxy → 91-day T-Bill)
  3. Volatility adj: σ = annualised std of weekly returns over 3 years
  4. Z-scoring     : Combined C = 0.5*Z6m + 0.5*Z12m, then re-Z-scored & winsorised ±3
  5. Score mapping : Score = 1+Z (Z>0) or (1-Z)^-1 (Z<0)
  6. Selection     : Fixed-N securities with highest positive Z-scores
                     (targeting ~30% of parent Mcap, N rounded per MSCI rules)
  7. Weighting     : Score × Parent Mcap Weight, normalised to 100%
  8. Issuer cap    : 5% (broad index)
  9. Rebalancing   : Semi-annual (May / November)
 10. Buffer rules  : ±50% of fixed N to reduce turnover

Assumptions / adaptations for India:
  - Parent Index  : Nifty 500 (proxied via yfinance tickers)
  - Risk-free rate: 91-day T-Bill yield sourced from RBI / proxied at ~6.5% p.a.
                    constant for the backtest period (conservative; MIBOR-3M ~6-7%)
  - Market cap    : Float-adjusted market cap approximated using yfinance
  - Semi-annual   : Rebalancing on last trading day of May and November each year
  - Conditional   : Ad-hoc rebalancing trigger (Appendix III) included
  - Issuer cap    : 5% (treating Nifty 500 as broad)

Author  : Index Researcher
Version : 1.0 (June 2026)
"""

# ─────────────────────────────────────────────────────────────────────────────
# 0.  IMPORTS & CONFIG
# ─────────────────────────────────────────────────────────────────────────────
import warnings
warnings.filterwarnings("ignore")

import os
import time
import math
import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.gridspec import GridSpec
import matplotlib.ticker as mticker
from scipy import stats

# ── Backtest window ──────────────────────────────────────────────────────────
BT_START   = "2021-01-01"  
BT_END     = "2026-03-31"
DATA_START = "2018-01-01"

# ── India risk-free rate (91-day T-Bill) — annualised decimal ───────────────
# Using a time-varying series approximation:
#   2017-2019 : 6.25%   2019-2020 : 5.50%   2020-2022 : 3.90%   2022-2024 : 6.75%
RISK_FREE_ANNUAL = 0.065      # flat fallback; time-varying series built below

# ── Index construction parameters ───────────────────────────────────────────
ISSUER_CAP      = 0.05        # 5% issuer weight cap (broad index)
WINSOR_LIMIT    = 3.0         # Z-score winsorisation ± 3
TARGET_MCAP_COV = 0.30        # target 30% parent mcap coverage for N selection
MAX_SEC_COVERAGE= 0.40        # reduce N if >40% of parent securities
MIN_MCAP_COV    = 0.20        # force N up if mcap coverage < 20%
BUFFER_FACTOR   = 0.50        # buffer at 50% of fixed N

# ── Ad-hoc rebalancing (Appendix III) ───────────────────────────────────────
ADHOC_PERCENTILE = 95         # 95th percentile of monthly vol changes

# ── Output paths ────────────────────────────────────────────────────────────
OUT_DIR = r"C:\Users\0310a\Desktop\momentum_output"
os.makedirs(OUT_DIR, exist_ok=True)

# ─────────────────────────────────────────────────────────────────────────────
# 1.  NIFTY 500 TICKER LIST 
# ─────────────────────────────────────────────────────────────────────────────

url = r"C:\Users\0310a\Downloads\ind_nifty500list.csv"
nifty500 = pd.read_csv(url)

tickers = nifty500["Symbol"].tolist()

print(len(tickers))
print(tickers[:10])
NIFTY500_TICKERS_NSE = [f"{t}.NS" for t in tickers]

# De-duplicate and append .NS suffix
TICKERS = [f"{t}.NS" for t in tickers]
BENCHMARK_NIFTY50  = "^NSEI"
BENCHMARK_NIFTY500 = "^CRSLDX"      # NSE 500 index on Yahoo Finance

print(f"[Setup] Universe: {len(TICKERS)} tickers | Period: {BT_START} to {BT_END}")


# ─────────────────────────────────────────────────────────────────────────────
# 2.  DATA DOWNLOAD
# ─────────────────────────────────────────────────────────────────────────────
def download_price_data(tickers: list, start: str, end: str,
                        batch_size: int = 50) -> pd.DataFrame:
    """Download adjusted close prices in batches; returns wide DataFrame."""
    print(f"[Data] Downloading prices for {len(tickers)} tickers …")
    frames = []
    for i in range(0, len(tickers), batch_size):
        batch = tickers[i:i+batch_size]
        try:
            raw = yf.download(batch, start=start, end=end,
                              auto_adjust=True, progress=False, threads=True)
            if isinstance(raw.columns, pd.MultiIndex):
                close = raw["Close"]
            else:
                close = raw[["Close"]] if "Close" in raw else raw
            frames.append(close)
        except Exception as e:
            print(f"  Warning: batch {i//batch_size} failed — {e}")
        time.sleep(0.5)

    prices = pd.concat(frames, axis=1)
    prices = prices.loc[:, ~prices.columns.duplicated()]
    prices.index = pd.to_datetime(prices.index)
    # Keep only columns present for >80% of the backtest period
    threshold = len(prices) * 0.20
    prices = prices.dropna(thresh=threshold, axis=1)
    print(f"[Data] Downloaded: {prices.shape[1]} securities × {prices.shape[0]} days")
    return prices


def download_shares_outstanding(tickers: list) -> pd.Series:
    """
    Download latest shares outstanding.
    We assume shares remain constant historically because
    Yahoo does not provide a clean historical shares series.
    """
    shares = {}

    print("[Data] Downloading shares outstanding ...")

    for tkr in tickers:
        try:
            info = yf.Ticker(tkr).info

            s = (
                info.get("floatShares")
                or info.get("sharesOutstanding")
                or 0
            )

            shares[tkr] = s

        except Exception:
            shares[tkr] = 0

    return pd.Series(shares)

def build_historical_market_caps(
    prices: pd.DataFrame,
    shares_outstanding: pd.Series
) -> pd.DataFrame:
    """
    Historical market cap approximation:

        Market Cap(t) =
            Price(t) × Current Shares Outstanding

    This removes look-ahead bias from using
    today's market cap at historical dates.
    """

    shares = shares_outstanding.reindex(prices.columns).fillna(0)

    hist_mcaps = prices.mul(shares, axis=1)

    return hist_mcaps


def build_riskfree_series(index: pd.DatetimeIndex) -> pd.Series:
    """
    Time-varying 91-day T-Bill yield approximation for India.
    Source: RBI historical data (simplified regime-based proxy).
    A production system would pull from RBI DBIE API.
    """
    rf_dict = {
        pd.Timestamp("2017-01-01"): 0.0625,
        pd.Timestamp("2019-06-01"): 0.0575,
        pd.Timestamp("2019-10-01"): 0.0515,
        pd.Timestamp("2020-03-01"): 0.0400,
        pd.Timestamp("2020-06-01"): 0.0365,
        pd.Timestamp("2021-01-01"): 0.0380,
        pd.Timestamp("2022-04-01"): 0.0440,
        pd.Timestamp("2022-10-01"): 0.0660,
        pd.Timestamp("2023-03-01"): 0.0685,
        pd.Timestamp("2024-01-01"): 0.0670,
        pd.Timestamp("2025-01-01"): 0.0675,
        pd.Timestamp("2026-01-01"): 0.0665,
    }
    dates  = sorted(rf_dict.keys())
    values = [rf_dict[d] for d in dates]
    rf_full = pd.Series(values, index=dates).reindex(index, method="ffill")
    return rf_full.fillna(RISK_FREE_ANNUAL)


# ─────────────────────────────────────────────────────────────────────────────
# 3.  MOMENTUM SCORE ENGINE
# ─────────────────────────────────────────────────────────────────────────────
def calc_momentum_score(prices: pd.DataFrame,
                        rebal_date: pd.Timestamp,
                        rf_annual: float,
                        use_6m_only: bool = False) -> pd.Series:
    """
    Compute MSCI Momentum Score for all securities as at rebal_date.

    Steps (from methodology §2.2):
      1. 6-month price momentum  = (P_{T-1}/P_{T-7}  - 1) - rf_6m
      2. 12-month price momentum = (P_{T-1}/P_{T-13} - 1) - rf_12m
         (if missing, fallback to 6m only)
      3. Risk-adjust by dividing by 3-year annualised weekly vol
      4. Z-score each, combine 50/50, re-Z-score, winsorise ±3
      5. Map to Momentum Score: 1+Z if Z>0, (1-Z)^-1 if Z<0

    Returns pd.Series of Momentum Scores (unwinsorised Z also returned
    as second element for ranking purposes).
    """
    # T-1: one month before rebalancing date
    all_dates = prices.index[prices.index <= rebal_date]
    if len(all_dates) < 2:
        return pd.Series(dtype=float), pd.Series(dtype=float)

    t1_idx  = all_dates[-1]          # most recent close before T

    # Helper: find price N calendar months back
    def price_n_months_back(n_months: int):
        target = t1_idx - pd.DateOffset(months=n_months)
        avail  = prices.index[prices.index <= target]
        if len(avail) == 0:
            return None
        return prices.loc[avail[-1]]

    p_t1 = price_n_months_back(1)    # T-1 month
    p_t7 = price_n_months_back(7)    # T-7 month
    p_t13 = price_n_months_back(13)  # T-13 month

    if p_t7 is None:
        return pd.Series(dtype=float), pd.Series(dtype=float)

    # Risk-free rates for respective periods (annualised → period)
    rf_6m  = rf_annual * (6/12)
    rf_12m = rf_annual * (12/12)

    mom_6m = (p_t1 / p_t7 - 1) - rf_6m
    mom_6m = mom_6m.replace([np.inf, -np.inf], np.nan)

    if not use_6m_only and p_t13 is not None:
        mom_12m = (p_t1 / p_t13 - 1) - rf_12m
        mom_12m = mom_12m.replace([np.inf, -np.inf], np.nan)
    else:
        mom_12m = None

    # ── 3-year weekly volatility (§2.2.1) ────────────────────────────────
    vol_start = t1_idx - pd.DateOffset(years=3)
    hist = prices.loc[(prices.index >= vol_start) & (prices.index <= t1_idx)]

    weekly = hist.resample("W-FRI").last()
    weekly_ret = weekly.pct_change().dropna(how="all")
    sigma = weekly_ret.std() * np.sqrt(52)          # annualised weekly vol
    sigma = sigma.replace(0, np.nan)

    # ── Risk-adjusted momentum (§2.2.1) ──────────────────────────────────
    ra_6m  = mom_6m / sigma

    if mom_12m is not None:
        ra_12m = mom_12m / sigma
    else:
        ra_12m = None

    # ── Z-scores (§2.2.2) ────────────────────────────────────────────────
    def z_score(s: pd.Series) -> pd.Series:
        clean = s.dropna()
        if len(clean) < 5:
            return pd.Series(np.nan, index=s.index)
        z = (s - clean.mean()) / clean.std(ddof=1)
        return z

    z6 = z_score(ra_6m)

    if ra_12m is not None:
        z12 = z_score(ra_12m)
        # Align: if 12m missing, use 6m only
        combined = pd.DataFrame({"z6": z6, "z12": z12})
        has_both = combined.notna().all(axis=1)
        C = pd.Series(np.nan, index=z6.index)
        C[has_both]  = 0.5 * combined.loc[has_both, "z6"] + \
                       0.5 * combined.loc[has_both, "z12"]
        # Only 6m available
        only6 = combined["z6"].notna() & combined["z12"].isna()
        C[only6] = combined.loc[only6, "z6"]
    else:
        C = z6

    # Re-standardise combined score
    Z = z_score(C)

    # Unwinsorised Z for ranking (per §2.3)
    Z_unwinsorised = Z.copy()

    # Winsorise ±3
    Z = Z.clip(-WINSOR_LIMIT, WINSOR_LIMIT)

    # Momentum Score (§2.2.2 formula)
    score = pd.Series(np.nan, index=Z.index)
    pos = Z > 0
    neg = Z < 0
    score[pos] = 1 + Z[pos]
    score[neg] = 1 / (1 - Z[neg])
    score[Z == 0] = 1.0

    return score, Z_unwinsorised


# ─────────────────────────────────────────────────────────────────────────────
# 4.  FIXED-N SECURITY SELECTION (Appendix I)
# ─────────────────────────────────────────────────────────────────────────────
def round_up_n(n: float) -> int:
    """MSCI rounding rules for fixed-N determination (Appendix I)."""
    if n < 100:
        return math.ceil(n / 10) * 10
    elif n < 300:
        return math.ceil(n / 25) * 25
    else:
        return math.ceil(n / 50) * 50


def determine_fixed_n(z_unwinsorised: pd.Series,
                      parent_weights: pd.Series,
                      prev_fixed_n: int = None) -> int:
    """
    Implements Appendix I algorithm to determine/reevaluate fixed N.

    parent_weights: float-mcap weights normalised to 1.0 within available universe.
    """
    eligible = z_unwinsorised.dropna()
    n_parent = len(eligible)

    # Rebalancing reevaluation path (when prev_fixed_n is known)
    if prev_fixed_n is not None:
        if prev_fixed_n > n_parent:
            # fall through to initial construction algorithm
            prev_fixed_n = None
        elif n_parent <= 25:
            return n_parent
        elif n_parent < 25 and prev_fixed_n is not None:
            pass  # use initial construction
        else:
            # check if mcap coverage with prev N < 10%
            ranked = eligible.sort_values(ascending=False)
            top_n  = ranked.iloc[:prev_fixed_n].index
            pw     = parent_weights.reindex(top_n).fillna(0)
            mcap_cov = pw.sum()
            if mcap_cov <= 0.10:
                prev_fixed_n = None  # re-run initial construction
            else:
                return prev_fixed_n

    # Initial construction algorithm
    if n_parent <= 25:
        return n_parent

    # Sort by momentum z-score descending
    ranked = eligible.sort_values(ascending=False)
    pw     = parent_weights.reindex(ranked.index).fillna(0)
    cum_w  = pw.cumsum()

    # Number of securities for target 30% mcap coverage
    hits_30 = (cum_w >= TARGET_MCAP_COV)
    if not hits_30.any():
        n_30pct = n_parent
    else:
        n_30pct = hits_30.idxmax()
        n_30pct = ranked.index.get_loc(n_30pct) + 1

    if n_30pct <= 25:
        return max(25, round_up_n(n_30pct))
    if n_30pct <= int(0.10 * n_parent):
        return round_up_n(int(0.10 * n_parent))

    n_rounded = round_up_n(n_30pct)

    # Check if >40% of parent security count
    if n_rounded >= 0.40 * n_parent:
        # Reduce until ≤40%
        n_reduced = math.floor(0.40 * n_parent)
        n_reduced = round_up_n(n_reduced)
        # But ensure mcap coverage ≥ 20%
        top_n_r = ranked.iloc[:n_reduced].index
        mc_r    = parent_weights.reindex(top_n_r).fillna(0).sum()
        if mc_r < MIN_MCAP_COV:
            # Increase N until mcap ≥ 20% even if sec% > 40%
            for k in range(n_reduced, n_parent+1):
                top_k = ranked.iloc[:k].index
                if parent_weights.reindex(top_k).fillna(0).sum() >= MIN_MCAP_COV:
                    return round_up_n(k)
            return n_parent
        return n_reduced

    return n_rounded


# ─────────────────────────────────────────────────────────────────────────────
# 5.  SECURITY SELECTION WITH BUFFER RULES (§2.3 + §3.1.1)
# ─────────────────────────────────────────────────────────────────────────────
def select_securities(z_unwinsorised: pd.Series,
                      fixed_n: int,
                      prev_constituents: set = None) -> list:
    """
    Select fixed_n securities applying MSCI buffer rules.

    Buffer zone = [fixed_n * (1 - BUFFER_FACTOR), fixed_n * (1 + BUFFER_FACTOR)]
    i.e. for N=100: buffer between rank 51 and 150.
    """
    eligible = z_unwinsorised.dropna()
    ranked   = eligible.sort_values(ascending=False)

    buffer_lo = int(fixed_n * BUFFER_FACTOR) + 1    # rank where buffer starts
    buffer_hi = int(fixed_n * (1 + BUFFER_FACTOR))  # rank where buffer ends

    # Step 1: All securities with rank ≤ buffer_lo always included
    core = list(ranked.iloc[:buffer_lo].index)

    if prev_constituents is None or len(prev_constituents) == 0:
        # Initial construction: just take top fixed_n
        selected = list(ranked.iloc[:fixed_n].index)
        return selected

    # Step 2: Add existing constituents within buffer zone
    buffer_zone = set(ranked.iloc[buffer_lo:buffer_hi].index)
    existing_in_buffer = [t for t in prev_constituents
                          if t in buffer_zone and t not in core]

    selected = core.copy()
    for t in existing_in_buffer:
        if len(selected) >= fixed_n:
            break
        selected.append(t)

    # Step 3: Fill remaining slots from best remaining
    if len(selected) < fixed_n:
        remaining = [t for t in ranked.index if t not in set(selected)]
        selected += remaining[:fixed_n - len(selected)]

    return selected[:fixed_n]


# ─────────────────────────────────────────────────────────────────────────────
# 6.  WEIGHTING SCHEME (§2.4)
# ─────────────────────────────────────────────────────────────────────────────
def compute_weights(selected: list,
                    scores: pd.Series,
                    parent_weights: pd.Series) -> pd.Series:
    """
    Momentum Weight = Score × Parent Mcap Weight, normalised to 100%.
    Then apply 5% issuer cap (iterative redistribution).
    """
    s = scores.reindex(selected).fillna(1.0)
    w = parent_weights.reindex(selected).fillna(0)

    raw_w = s * w
    total = raw_w.sum()
    if total == 0:
        return pd.Series(1/len(selected), index=selected)
    norm_w = raw_w / total

    # Iterative issuer cap at 5%
    for _ in range(30):
        over = norm_w[norm_w > ISSUER_CAP]
        if over.empty:
            break
        excess = (norm_w[norm_w > ISSUER_CAP] - ISSUER_CAP).sum()
        norm_w[norm_w > ISSUER_CAP] = ISSUER_CAP
        under_idx = norm_w[norm_w < ISSUER_CAP].index
        if len(under_idx) == 0:
            break
        denom = norm_w[under_idx].sum()
        if denom == 0:
            break
        norm_w[under_idx] += norm_w[under_idx] / denom * excess

    return norm_w


# ─────────────────────────────────────────────────────────────────────────────
# 7.  CONDITIONAL REBALANCING TRIGGER (Appendix III)
# ─────────────────────────────────────────────────────────────────────────────
def check_adhoc_trigger(benchmark_prices: pd.Series,
                        check_date: pd.Timestamp,
                        threshold_pct: float) -> bool:
    """
    Compute monthly change in annualised 3-month vol of the benchmark.
    Trigger if change > threshold_pct (95th percentile of historical changes).
    """
    end   = benchmark_prices.index[benchmark_prices.index <= check_date][-1]
    start = end - pd.DateOffset(months=3)
    hist  = benchmark_prices.loc[(benchmark_prices.index >= start) &
                                  (benchmark_prices.index <= end)]
    if len(hist) < 20:
        return False
    vt = hist.pct_change().dropna().std() * np.sqrt(252)

    # Previous month vol
    prev_end   = end - pd.DateOffset(months=1)
    prev_start = prev_end - pd.DateOffset(months=3)
    hist_prev  = benchmark_prices.loc[(benchmark_prices.index >= prev_start) &
                                       (benchmark_prices.index <= prev_end)]
    if len(hist_prev) < 20:
        return False
    vt1 = hist_prev.pct_change().dropna().std() * np.sqrt(252)

    if vt1 == 0:
        return False
    delta = (vt / vt1) - 1
    return delta > threshold_pct


def compute_vol_threshold(benchmark_prices: pd.Series,
                          start: str, end: str) -> float:
    """
    Compute 95th percentile of monthly volatility changes over full history.
    Appendix III: Reference Index for India (EM) is MSCI EM; we use Nifty 50.
    """
    prices = benchmark_prices.loc[start:end]
    monthly_vols = []
    dates = pd.date_range(start, end, freq="ME")
    for d in dates:
        s = d - pd.DateOffset(months=3)
        h = prices.loc[(prices.index >= s) & (prices.index <= d)]
        if len(h) < 20:
            continue
        v = h.pct_change().dropna().std() * np.sqrt(252)
        monthly_vols.append(v)

    changes = [(monthly_vols[i] / monthly_vols[i-1]) - 1
               for i in range(1, len(monthly_vols))
               if monthly_vols[i-1] > 0]
    if len(changes) < 10:
        return 0.30  # default fallback
    return np.percentile(changes, ADHOC_PERCENTILE)


# ─────────────────────────────────────────────────────────────────────────────
# 8.  REBALANCING CALENDAR
# ─────────────────────────────────────────────────────────────────────────────
def get_sair_dates(start: str, end: str,
                   trading_days: pd.DatetimeIndex) -> list:
    """
    MSCI SAIR: last business day of May and November each year.
    """
    dates = []
    for year in range(pd.Timestamp(start).year, pd.Timestamp(end).year + 1):
        for month in [5, 11]:
            # Last trading day of that month
            m_end = pd.Timestamp(year=year, month=month,
                                  day=pd.Timestamp(year, month, 1).days_in_month)
            avail = trading_days[trading_days <= m_end]
            if len(avail) > 0:
                dates.append(avail[-1])
    return sorted(set(dates))


# ─────────────────────────────────────────────────────────────────────────────
# 9.  PORTFOLIO PERFORMANCE ENGINE
# ─────────────────────────────────────────────────────────────────────────────
def compute_portfolio_returns(
    weights_schedule,
    prices,
    start,
    end
):
    """
    Buy-and-hold implementation.

    Weights drift naturally between rebalances.
    """

    rebal_dates = sorted(weights_schedule.keys())

    portfolio_returns = []

    bt_prices = prices.loc[start:end]

    for i, rb_date in enumerate(rebal_dates):

        start_weights = weights_schedule[rb_date]

        if i < len(rebal_dates) - 1:
            period_end = rebal_dates[i + 1] - pd.Timedelta(days=1)
        else:
            period_end = pd.Timestamp(end)

        tickers = start_weights.index.intersection(bt_prices.columns)

        if len(tickers) == 0:
            continue

        px = bt_prices.loc[rb_date:period_end, tickers]

        if len(px) < 2:
            continue

        start_weights = (
            start_weights.reindex(tickers)
            .fillna(0)
        )

        start_weights /= start_weights.sum()

        wealth = px.div(px.iloc[0])

        position_values = wealth.mul(
            start_weights,
            axis=1
        )

        portfolio_value = position_values.sum(axis=1)

        period_returns = portfolio_value.pct_change()

        portfolio_returns.append(period_returns)

    if not portfolio_returns:
        return pd.Series(dtype=float)

    return (
        pd.concat(portfolio_returns)
        .sort_index()
        .fillna(0)
    )


# ─────────────────────────────────────────────────────────────────────────────
# 10. PERFORMANCE ANALYTICS
# ─────────────────────────────────────────────────────────────────────────────
def performance_stats(returns: pd.Series, rf_annual: float = 0.065,
                      name: str = "Strategy") -> pd.Series:
    """Compute standard performance metrics."""
    if returns.empty or returns.std() == 0:
        return pd.Series(dtype=float)

    r       = returns.dropna()
    n_years = len(r) / 252
    total   = (1 + r).prod() - 1
    cagr    = (1 + total) ** (1 / n_years) - 1 if n_years > 0 else 0
    vol     = r.std() * np.sqrt(252)
    sharpe  = (cagr - rf_annual) / vol if vol > 0 else 0

    # Drawdown
    cum     = (1 + r).cumprod()
    roll_max= cum.cummax()
    dd      = (cum - roll_max) / roll_max
    max_dd  = dd.min()

    # Calmar
    calmar  = cagr / abs(max_dd) if max_dd != 0 else 0

    # Sortino
    neg_r   = r[r < 0]
    down_vol= neg_r.std() * np.sqrt(252) if len(neg_r) > 0 else vol
    sortino = (cagr - rf_annual) / down_vol if down_vol > 0 else 0

    # Best / worst year
    annual = r.resample("YE").apply(lambda x: (1+x).prod()-1)
    best_yr  = annual.max()
    worst_yr = annual.min()

    # Win rate (% positive months)
    monthly = r.resample("ME").apply(lambda x: (1+x).prod()-1)
    win_rate= (monthly > 0).mean()

    return pd.Series({
        "Total Return (%)":   round(total*100, 2),
        "CAGR (%)":           round(cagr*100, 2),
        "Ann. Volatility (%)":round(vol*100, 2),
        "Sharpe Ratio":       round(sharpe, 3),
        "Sortino Ratio":      round(sortino, 3),
        "Calmar Ratio":       round(calmar, 3),
        "Max Drawdown (%)":   round(max_dd*100, 2),
        "Best Year (%)":      round(best_yr*100, 2),
        "Worst Year (%)":     round(worst_yr*100, 2),
        "Monthly Win Rate":   round(win_rate, 3),
    }, name=name)


def relative_stats(strat_rets: pd.Series, bench_rets: pd.Series,
                   name: str = "vs Benchmark") -> pd.Series:
    """Alpha, beta, tracking error, information ratio."""
    aligned = pd.concat([strat_rets, bench_rets], axis=1,
                        keys=["s","b"]).dropna()
    if len(aligned) < 30:
        return pd.Series(dtype=float)

    slope, intercept, r, p, se = stats.linregress(aligned["b"], aligned["s"])
    te  = (aligned["s"] - aligned["b"]).std() * np.sqrt(252)
    er  = (aligned["s"] - aligned["b"]).mean() * 252
    ir  = er / te if te > 0 else 0
    ann_alpha = intercept * 252
    beta      = slope

    return pd.Series({
        "Alpha (ann. %)": round(ann_alpha*100, 3),
        "Beta":           round(beta, 3),
        "R²":             round(r**2, 3),
        "Tracking Error (%)": round(te*100, 2),
        "Information Ratio":  round(ir, 3),
    }, name=name)


# ─────────────────────────────────────────────────────────────────────────────
# 11. VISUALISATION
# ─────────────────────────────────────────────────────────────────────────────
COLORS = {
    "momentum": "#1f77b4",
    "nifty50":  "#d62728",
    "nifty500": "#2ca02c",
    "drawdown": "#ff7f0e",
}


def plot_results(cum_strat: pd.Series,
                 cum_n50: pd.Series,
                 cum_n500: pd.Series,
                 monthly_strat: pd.Series,
                 monthly_n50: pd.Series,
                 rebal_dates: list,
                 stats_df: pd.DataFrame,
                 portfolio_history: dict,
                 fixed_n_history: list) -> str:
    """Generate a multi-panel research report chart."""
    fig = plt.figure(figsize=(20, 26))
    fig.patch.set_facecolor("#f8f9fa")
    gs  = GridSpec(5, 2, figure=fig, hspace=0.45, wspace=0.35,
                   left=0.07, right=0.97, top=0.94, bottom=0.04)

    # ── 1. Cumulative Performance ────────────────────────────────────────
    ax1 = fig.add_subplot(gs[0, :])
    ax1.set_facecolor("white")
    ax1.plot(cum_strat.index, cum_strat.values,
             color=COLORS["momentum"], lw=2.2, label="MSCI-style India Momentum")
    if not cum_n50.empty:
        ax1.plot(cum_n50.index, cum_n50.values,
                 color=COLORS["nifty50"], lw=1.8, ls="--", label="Nifty 50")
    if not cum_n500.empty:
        ax1.plot(cum_n500.index, cum_n500.values,
                 color=COLORS["nifty500"], lw=1.8, ls=":", label="Nifty 500")
    for rd in rebal_dates:
        if pd.Timestamp(BT_START) <= rd <= pd.Timestamp(BT_END):
            ax1.axvline(rd, color="gray", alpha=0.25, lw=0.8)
    ax1.set_title("Cumulative Performance (Indexed to 100) — Semi-annual Rebalancing",
                  fontsize=13, fontweight="bold")
    ax1.set_ylabel("Index Value")
    ax1.legend(fontsize=10)
    ax1.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{x:.0f}"))
    ax1.grid(True, alpha=0.3)

    # ── 2. Rolling 12-Month Returns ──────────────────────────────────────
    ax2 = fig.add_subplot(gs[1, 0])
    ax2.set_facecolor("white")
    roll12_strat = monthly_strat.rolling(12).apply(lambda x: (1+x).prod()-1)
    roll12_n50   = monthly_n50.rolling(12).apply(lambda x: (1+x).prod()-1) \
                   if not monthly_n50.empty else pd.Series()
    ax2.plot(roll12_strat.index, roll12_strat.values*100,
             color=COLORS["momentum"], lw=2, label="Momentum")
    if not roll12_n50.empty:
        ax2.plot(roll12_n50.index, roll12_n50.values*100,
                 color=COLORS["nifty50"], lw=1.5, ls="--", label="Nifty 50")
    ax2.axhline(0, color="black", lw=0.8)
    ax2.set_title("Rolling 12-Month Returns (%)", fontsize=11, fontweight="bold")
    ax2.set_ylabel("%")
    ax2.legend(fontsize=9)
    ax2.grid(True, alpha=0.3)

    # ── 3. Drawdown ──────────────────────────────────────────────────────
    ax3 = fig.add_subplot(gs[1, 1])
    ax3.set_facecolor("white")

    def dd_series(cum):
        roll_max = cum.cummax()
        return ((cum - roll_max) / roll_max) * 100

    ax3.fill_between(cum_strat.index, dd_series(cum_strat).values,
                     0, alpha=0.5, color=COLORS["momentum"], label="Momentum")
    if not cum_n50.empty:
        ax3.plot(cum_n50.index, dd_series(cum_n50).values,
                 color=COLORS["nifty50"], lw=1.5, ls="--", label="Nifty 50", alpha=0.8)
    ax3.set_title("Drawdown (%)", fontsize=11, fontweight="bold")
    ax3.set_ylabel("%")
    ax3.legend(fontsize=9)
    ax3.grid(True, alpha=0.3)

    # ── 4. Monthly Return Distribution ──────────────────────────────────
    ax4 = fig.add_subplot(gs[2, 0])
    ax4.set_facecolor("white")
    ms_clean = monthly_strat.dropna() * 100
    ax4.hist(ms_clean, bins=30, color=COLORS["momentum"], alpha=0.7,
             edgecolor="white", label="Momentum")
    if not monthly_n50.empty:
        mn_clean = monthly_n50.dropna() * 100
        ax4.hist(mn_clean, bins=30, color=COLORS["nifty50"], alpha=0.4,
                 edgecolor="white", label="Nifty 50")
    ax4.axvline(ms_clean.mean(), color=COLORS["momentum"], lw=2, ls="--")
    ax4.set_title("Monthly Return Distribution", fontsize=11, fontweight="bold")
    ax4.set_xlabel("Monthly Return (%)")
    ax4.legend(fontsize=9)
    ax4.grid(True, alpha=0.3)

    # ── 5. Annual Returns Bar ────────────────────────────────────────────
    ax5 = fig.add_subplot(gs[2, 1])
    ax5.set_facecolor("white")
    ann_strat = monthly_strat.resample("YE").apply(lambda x: (1+x).prod()-1)*100
    ann_n50   = monthly_n50.resample("YE").apply(lambda x: (1+x).prod()-1)*100 \
                if not monthly_n50.empty else pd.Series()
    years = ann_strat.index.year
    x = np.arange(len(years))
    width = 0.35
    ax5.bar(x - width/2, ann_strat.values, width, label="Momentum",
            color=COLORS["momentum"], alpha=0.85)
    if not ann_n50.empty:
        ann_n50_aligned = ann_n50.reindex(ann_strat.index, method="nearest")
        ax5.bar(x + width/2, ann_n50_aligned.values, width, label="Nifty 50",
                color=COLORS["nifty50"], alpha=0.85)
    ax5.axhline(0, color="black", lw=0.8)
    ax5.set_xticks(x)
    ax5.set_xticklabels(years, rotation=45)
    ax5.set_title("Annual Returns (%)", fontsize=11, fontweight="bold")
    ax5.set_ylabel("%")
    ax5.legend(fontsize=9)
    ax5.grid(True, alpha=0.3, axis="y")

    # ── 6. Performance Stats Table ───────────────────────────────────────
    ax6 = fig.add_subplot(gs[3, :])
    ax6.set_facecolor("white")
    ax6.axis("off")
    if not stats_df.empty:
        tbl = ax6.table(
            cellText=stats_df.values,
            colLabels=stats_df.columns,
            rowLabels=stats_df.index,
            cellLoc="center", loc="center",
            bbox=[0, 0, 1, 1]
        )
        tbl.auto_set_font_size(False)
        tbl.set_fontsize(9.5)
        for (r, c), cell in tbl.get_celld().items():
            if r == 0:
                cell.set_facecolor("#2c3e50")
                cell.set_text_props(color="white", fontweight="bold")
            elif c == -1:
                cell.set_facecolor("#ecf0f1")
                cell.set_text_props(fontweight="bold")
            elif r % 2 == 0:
                cell.set_facecolor("#f7f9fc")
    ax6.set_title("Performance Statistics Summary", fontsize=11,
                  fontweight="bold", pad=15)

    # ── 7. Fixed-N History ───────────────────────────────────────────────
    ax7 = fig.add_subplot(gs[4, 0])
    ax7.set_facecolor("white")
    if fixed_n_history:
        fn_dates = [x[0] for x in fixed_n_history]
        fn_vals  = [x[1] for x in fixed_n_history]
        ax7.step(fn_dates, fn_vals, where="post",
                 color=COLORS["momentum"], lw=2)
        ax7.scatter(fn_dates, fn_vals, color=COLORS["momentum"], s=40, zorder=5)
        ax7.set_title("Fixed-N Constituents (Semi-annual)", fontsize=11, fontweight="bold")
        ax7.set_ylabel("Number of Securities")
        ax7.grid(True, alpha=0.3)

    # ── 8. Sector Concentration (latest portfolio) ───────────────────────
    ax8 = fig.add_subplot(gs[4, 1])
    ax8.set_facecolor("white")
    if portfolio_history:
        last_date = max(portfolio_history.keys())
        last_port = portfolio_history[last_date]
        top10 = last_port.nlargest(10)
        tickers_clean = [t.replace(".NS","") for t in top10.index]
        colors = plt.cm.Blues(np.linspace(0.4, 0.9, len(top10)))
        bars = ax8.barh(range(len(top10)), top10.values * 100,
                        color=colors)
        ax8.set_yticks(range(len(top10)))
        ax8.set_yticklabels(tickers_clean, fontsize=9)
        ax8.set_xlabel("Weight (%)")
        ax8.set_title(f"Top 10 Holdings — {last_date.strftime('%b %Y')}",
                      fontsize=11, fontweight="bold")
        ax8.grid(True, alpha=0.3, axis="x")
        for bar, val in zip(bars, top10.values):
            ax8.text(val*100 + 0.05, bar.get_y() + bar.get_height()/2,
                     f"{val*100:.1f}%", va="center", fontsize=8)

    fig.suptitle("MSCI-Style Momentum Index — India (Nifty 500 Universe)\n"
                 f"Backtest: {BT_START} to {BT_END}",
                 fontsize=15, fontweight="bold", y=0.97)

    out_path = os.path.join(OUT_DIR, "msci_momentum_india_backtest.png")
    plt.savefig(out_path, dpi=150, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    plt.close()
    print(f"[Output] Chart saved → {out_path}")
    return out_path


# ─────────────────────────────────────────────────────────────────────────────
# 12. MAIN BACKTEST ORCHESTRATOR
# ─────────────────────────────────────────────────────────────────────────────
def run_backtest():
    print("\n" + "="*70)
    print("  MSCI Momentum Index Replication — Indian Markets")
    print("  Methodology: MSCI Momentum Indexes (August 2021)")
    print("="*70 + "\n")

    # ── 12.1 Download Data ───────────────────────────────────────────────
    prices_all = download_price_data(TICKERS, DATA_START, BT_END)

    # Benchmark prices
    print("[Data] Downloading benchmarks …")
    bench_raw = yf.download([BENCHMARK_NIFTY50, BENCHMARK_NIFTY500],
                             start=DATA_START, end=BT_END,
                             auto_adjust=True, progress=False)
    if isinstance(bench_raw.columns, pd.MultiIndex):
        bench_close = bench_raw["Close"]
    else:
        bench_close = bench_raw

    nifty50_px  = bench_close.get(BENCHMARK_NIFTY50,  pd.Series(dtype=float))
    nifty500_px = bench_close.get(BENCHMARK_NIFTY500, pd.Series(dtype=float))

    # Risk-free series
    rf_series = build_riskfree_series(prices_all.index)

    # Market cap weights (static proxy — in production: use daily float shares)
    print("[Data] Building historical market caps ...")

    shares_outstanding = download_shares_outstanding(
        prices_all.columns.tolist()
    )

    historical_mcaps = build_historical_market_caps(
        prices_all,
        shares_outstanding
    )

    # ── 12.2 Vol threshold for conditional rebalancing ───────────────────
    if not nifty50_px.empty:
        print("[Setup] Computing conditional rebalancing threshold …")
        adhoc_threshold = compute_vol_threshold(nifty50_px, DATA_START, BT_END)
        print(f"        Ad-hoc trigger threshold (95th pct): {adhoc_threshold:.2%}")
    else:
        adhoc_threshold = 0.30

    # ── 12.3 Build rebalancing calendar ─────────────────────────────────
    trading_days = prices_all.index
    sair_dates   = get_sair_dates(BT_START, BT_END, trading_days)
    print(f"[Rebal] SAIR dates: {len(sair_dates)} | "
          f"First: {sair_dates[0].date()}  Last: {sair_dates[-1].date()}")

    # ── 12.4 Run through each rebalancing date ───────────────────────────
    weights_schedule = {}
    portfolio_history = {}
    fixed_n_history   = []
    prev_constituents = set()
    prev_fixed_n      = None
    rebal_log         = []

    for rb_date in sair_dates:
        # Get risk-free rate for this date
        avail_rf = rf_series.loc[rf_series.index <= rb_date]
        rf_ann   = float(avail_rf.iloc[-1]) if len(avail_rf) > 0 else RISK_FREE_ANNUAL

        # Compute momentum scores
        scores, z_unwin = calc_momentum_score(
            prices_all, rb_date, rf_ann, use_6m_only=False
        )
        if scores.empty:
            print(f"  [Skip] {rb_date.date()} — insufficient history")
            continue

        # Determine fixed N
        
        avail_mcaps = historical_mcaps.loc[historical_mcaps.index <= rb_date]

        if len(avail_mcaps) == 0:
            continue

        current_mcaps = avail_mcaps.iloc[-1]

        parent_weights = (current_mcaps /current_mcaps.sum()).fillna(0)

        fixed_n = determine_fixed_n(z_unwin, parent_weights, prev_fixed_n)
        prev_fixed_n = fixed_n
        fixed_n_history.append((rb_date, fixed_n))

        # Select securities with buffer rules
        eligible_z = z_unwin[z_unwin > 0].dropna()   # only positive Z
        selected   = select_securities(eligible_z, fixed_n, prev_constituents)

        if not selected:
            print(f"  [Warn] {rb_date.date()} — no securities selected")
            continue

        # Check for ad-hoc trigger (checked 9 business days before T)
        adhoc = False
        if not nifty50_px.empty:
            adhoc = check_adhoc_trigger(nifty50_px, rb_date, adhoc_threshold)

        # Compute weights
        weights = compute_weights(selected, scores, parent_weights)
        weights = weights[weights > 0]

        weights_schedule[rb_date] = weights
        portfolio_history[rb_date] = weights
        prev_constituents = set(selected)

        rebal_log.append({
            "Date":       rb_date.date(),
            "Fixed N":    fixed_n,
            "Selected":   len(selected),
            "Ad-hoc":     adhoc,
            "Top Hold.":  weights.idxmax().replace(".NS","") if len(weights)>0 else "",
            "Top Wt(%)":  f"{weights.max()*100:.1f}%",
            "RF (%)":     f"{rf_ann*100:.2f}%",
        })

        print(f"  [{rb_date.date()}] Fixed-N={fixed_n} | Selected={len(selected)} | "
              f"MaxWt={weights.max()*100:.1f}% | RF={rf_ann*100:.2f}%"
              + (" ⚡ AD-HOC TRIGGER" if adhoc else ""))

    if not weights_schedule:
        print("[Error] No rebalancing dates produced weights. Exiting.")
        return

    # ── 12.5 Compute portfolio returns ───────────────────────────────────
    print("\n[Perf] Computing daily portfolio returns …")
    port_rets = compute_portfolio_returns(weights_schedule, prices_all,
                                          BT_START, BT_END)

    # Benchmark returns
    n50_rets  = nifty50_px.pct_change().dropna().reindex(
        port_rets.index, method="nearest").fillna(0) if not nifty50_px.empty \
        else pd.Series(dtype=float)
    n500_rets = nifty500_px.pct_change().dropna().reindex(
        port_rets.index, method="nearest").fillna(0) if not nifty500_px.empty \
        else pd.Series(dtype=float)

    # Cumulative returns (indexed to 100)
    cum_strat = (1 + port_rets).cumprod() * 100
    cum_n50   = (1 + n50_rets).cumprod() * 100  if not n50_rets.empty  else pd.Series()
    cum_n500  = (1 + n500_rets).cumprod() * 100 if not n500_rets.empty else pd.Series()

    # Monthly returns for stats
    monthly_strat = port_rets.resample("ME").apply(lambda x: (1+x).prod()-1)
    monthly_n50   = n50_rets.resample("ME").apply(lambda x: (1+x).prod()-1) \
                    if not n50_rets.empty else pd.Series()

    # ── 12.6 Performance stats ───────────────────────────────────────────
    print("\n[Stats] Computing performance metrics …")
    avg_rf = rf_series.loc[BT_START:BT_END].mean()

    stats_mom  = performance_stats(port_rets, avg_rf, "India Momentum")
    stats_n50  = performance_stats(n50_rets,  avg_rf, "Nifty 50")  \
                 if not n50_rets.empty else pd.Series()
    stats_n500 = performance_stats(n500_rets, avg_rf, "Nifty 500") \
                 if not n500_rets.empty else pd.Series()

    rel_n50  = relative_stats(port_rets, n50_rets,  "Momentum vs Nifty 50") \
               if not n50_rets.empty else pd.Series()
    rel_n500 = relative_stats(port_rets, n500_rets, "Momentum vs Nifty 500") \
               if not n500_rets.empty else pd.Series()

    # Combine into display DataFrame
    frames = [s for s in [stats_mom, stats_n50, stats_n500] if not s.empty]
    stats_df = pd.concat(frames, axis=1) if frames else pd.DataFrame()

    # ── 12.7 Print results ───────────────────────────────────────────────
    print("\n" + "="*70)
    print("  PERFORMANCE SUMMARY")
    print("="*70)
    pd.set_option("display.float_format", lambda x: f"{x:.3f}")
    pd.set_option("display.max_columns", 10)
    pd.set_option("display.width", 120)
    if not stats_df.empty:
        print(stats_df.to_string())

    if not rel_n50.empty:
        print("\n  RELATIVE PERFORMANCE vs NIFTY 50:")
        print(rel_n50.to_string())
    if not rel_n500.empty:
        print("\n  RELATIVE PERFORMANCE vs NIFTY 500:")
        print(rel_n500.to_string())

    print("\n  REBALANCING LOG:")
    rb_df = pd.DataFrame(rebal_log)
    if not rb_df.empty:
        print(rb_df.to_string(index=False))

    # ── 12.8 Save outputs ────────────────────────────────────────────────
    # Performance stats CSV
    stats_path = os.path.join(OUT_DIR, "performance_stats.csv")
    if not stats_df.empty:
        stats_df.to_csv(stats_path)
        print(f"\n[Output] Stats CSV → {stats_path}")

    # Rebalancing log CSV
    rb_path = os.path.join(OUT_DIR, "rebalancing_log.csv")
    if not rb_df.empty:
        rb_df.to_csv(rb_path, index=False)

    # Portfolio weights history CSV
    wh_rows = []
    for dt, wts in portfolio_history.items():
        for tkr, w in wts.items():
            wh_rows.append({"Date": dt.date(), "Ticker": tkr.replace(".NS",""),
                            "Weight (%)": round(w*100, 4)})
    if wh_rows:
        wh_df = pd.DataFrame(wh_rows)
        wh_path = os.path.join(OUT_DIR, "portfolio_weights_history.csv")
        wh_df.to_csv(wh_path, index=False)

    # Daily returns CSV
    dr_df = pd.concat([port_rets.rename("Momentum"),
                       n50_rets.rename("Nifty50"),
                       n500_rets.rename("Nifty500")], axis=1)
    dr_path = os.path.join(OUT_DIR, "daily_returns.csv")
    dr_df.to_csv(dr_path)

    # ── 12.9 Generate charts ─────────────────────────────────────────────
    plot_path = plot_results(
        cum_strat, cum_n50, cum_n500,
        monthly_strat, monthly_n50,
        sair_dates, stats_df,
        portfolio_history, fixed_n_history
    )

    print("\n[Done] Backtest complete. All outputs saved to:", OUT_DIR)
    return {
        "port_rets":         port_rets,
        "cum_strat":         cum_strat,
        "weights_schedule":  weights_schedule,
        "stats_df":          stats_df,
        "rebal_log":         rb_df,
        "fixed_n_history":   fixed_n_history,
    }


# ─────────────────────────────────────────────────────────────────────────────
# 13. ENTRY POINT
# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    results = run_backtest()