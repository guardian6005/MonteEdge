"""
Module 4: Black-Scholes Synthetic Option Pricer
─────────────────────────────────────────────────
For each signal from Module 3:
  1. Select strike based on conviction (ATM / ITM-1 / OTM-1)
  2. Estimate IV from recent realized volatility (HV proxy)
  3. Price option at ENTRY using Black-Scholes
  4. Compute Greeks (Delta, Gamma, Theta, Vega, Rho)
  5. Forward-simulate Nifty via GBM to estimate exit premiums
  6. Output: entry premium, SL premium, target premium, expected P&L

Why synthetic pricing?
─────────────────────────────────────────────────
• Historical NSE option chain data is hard to fetch reliably
• BS + HV gives reproducible, defensible pricing
• Industry standard for backtesting option strategies
• Lets us isolate STRATEGY edge from PRICING noise
"""

import numpy as np
import pandas as pd
from scipy.stats import norm
from datetime import datetime, timedelta
import os
import sys
import importlib.util

# ──────────────────────────────────────────────
# IMPORT SIBLING MODULES
# ──────────────────────────────────────────────
THIS_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, THIS_DIR)

def _load(name, filename):
    spec = importlib.util.spec_from_file_location(name, os.path.join(THIS_DIR, filename))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod

levels_mod = _load("levels_mod", "02_levels.py")


# ──────────────────────────────────────────────
# CONFIGURATION
# ──────────────────────────────────────────────
RISK_FREE_RATE         = 0.0665    # India 10-yr G-Sec ~6.65% (2026)
NIFTY_STRIKE_STEP      = 50        # Nifty strikes spaced at ₹50
TRADING_DAYS_PER_YEAR  = 252
MINUTES_PER_DAY        = 375       # NSE trading minutes (9:15-15:30)

# Volatility estimation
HV_LOOKBACK_DAYS       = 30        # Days of daily returns for HV
HV_FLOOR               = 0.08      # Minimum IV assumed (8%)
HV_CEIL                = 0.50      # Maximum IV assumed (50%)
IV_PREMIUM_FACTOR      = 1.10      # IV typically 5-15% above HV → use 1.10x

# Strike selection (offset from spot in number of strike steps)
STRIKE_OFFSET = {
    "VERY HIGH": -1,   # ITM-1 (deeper delta, ~0.6 for PE)
    "HIGH":       0,   # ATM
    "MEDIUM":     0,   # ATM
    "LOW":        1,   # OTM-1 (cheap lottery — but we'll likely skip)
}

# Weekly expiry assumption: signal day to next Tuesday (Nifty weekly)
# We'll compute actual days-to-expiry dynamically.

# Forward simulation for exit pricing
EXIT_SIM_PATHS         = 1000      # GBM paths from entry to forecast exits
EXIT_HORIZON_MINUTES   = 90        # Forecast 90 min ahead (intraday hold)

DATA_DIR = "/home/cecadmin/nifty_false_breakout_mc/data"
OUT_DIR  = "/home/cecadmin/nifty_false_breakout_mc/outputs"


# ══════════════════════════════════════════════
# BLACK-SCHOLES FORMULAS
# ══════════════════════════════════════════════
def _d1_d2(S, K, T, r, sigma):
    """Compute d1, d2 for Black-Scholes."""
    if T <= 0 or sigma <= 0:
        return np.nan, np.nan
    d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    return d1, d2


def bs_price(S, K, T, r, sigma, option_type='CE'):
    """
    Black-Scholes price for European option.

    Args:
        S : Spot price (Nifty)
        K : Strike price
        T : Time to expiry (in years)
        r : Risk-free rate (decimal)
        sigma : Implied volatility (decimal)
        option_type : 'CE' or 'PE'

    Returns: Theoretical option premium
    """
    if T <= 0:
        # At/after expiry — intrinsic value only
        if option_type == 'CE':
            return max(S - K, 0)
        else:
            return max(K - S, 0)

    d1, d2 = _d1_d2(S, K, T, r, sigma)
    if np.isnan(d1):
        return 0.0

    if option_type == 'CE':
        price = S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
    else:  # PE
        price = K * np.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)

    return max(price, 0.05)  # floor at 5 paise (real-world minimum tick)


def bs_greeks(S, K, T, r, sigma, option_type='CE'):
    """Compute Delta, Gamma, Theta, Vega, Rho."""
    if T <= 0 or sigma <= 0:
        return {'delta': 0, 'gamma': 0, 'theta': 0, 'vega': 0, 'rho': 0}

    d1, d2 = _d1_d2(S, K, T, r, sigma)
    pdf_d1 = norm.pdf(d1)

    if option_type == 'CE':
        delta = norm.cdf(d1)
        theta = (- (S * pdf_d1 * sigma) / (2 * np.sqrt(T))
                 - r * K * np.exp(-r * T) * norm.cdf(d2))
        rho   = K * T * np.exp(-r * T) * norm.cdf(d2) / 100
    else:  # PE
        delta = norm.cdf(d1) - 1
        theta = (- (S * pdf_d1 * sigma) / (2 * np.sqrt(T))
                 + r * K * np.exp(-r * T) * norm.cdf(-d2))
        rho   = -K * T * np.exp(-r * T) * norm.cdf(-d2) / 100

    gamma = pdf_d1 / (S * sigma * np.sqrt(T))
    vega  = S * pdf_d1 * np.sqrt(T) / 100   # per 1% IV change
    theta_per_day = theta / 365             # per calendar day

    return {
        'delta': round(delta, 4),
        'gamma': round(gamma, 6),
        'theta': round(theta_per_day, 4),
        'vega':  round(vega, 4),
        'rho':   round(rho, 4),
    }


# ══════════════════════════════════════════════
# VOLATILITY ESTIMATION (Historical Volatility → IV proxy)
# ══════════════════════════════════════════════
def estimate_hv(daily_df, as_of_date, lookback_days=HV_LOOKBACK_DAYS):
    """
    Annualized historical volatility from daily log returns
    over the past `lookback_days` BEFORE `as_of_date`.
    """
    as_of = pd.Timestamp(as_of_date).normalize()
    prior = daily_df[daily_df.index < as_of].tail(lookback_days)
    if len(prior) < 5:
        return 0.15  # fallback

    log_ret = np.log(prior['close'] / prior['close'].shift(1)).dropna()
    daily_std = log_ret.std()
    annualized = daily_std * np.sqrt(TRADING_DAYS_PER_YEAR)
    return float(np.clip(annualized, HV_FLOOR, HV_CEIL))


def estimate_iv(daily_df, as_of_date):
    """IV proxy = HV × IV_PREMIUM_FACTOR."""
    hv = estimate_hv(daily_df, as_of_date)
    return round(hv * IV_PREMIUM_FACTOR, 4)


# ══════════════════════════════════════════════
# EXPIRY CALCULATION (Nifty Weekly = Tuesday since 2024)
# ══════════════════════════════════════════════
def days_to_next_weekly_expiry(signal_dt, expiry_weekday=1):
    """
    NSE Nifty weekly expiry = Tuesday (weekday=1).
    Returns time-to-expiry in YEARS (for BS T input).

    For intraday trades, we use fractional days based on
    minutes remaining until next expiry's 15:30 IST close.
    """
    sig_dt = pd.Timestamp(signal_dt)
    days_ahead = (expiry_weekday - sig_dt.weekday()) % 7
    if days_ahead == 0:
        # Same day as expiry — check if before 15:30
        if sig_dt.time() < pd.Timestamp("15:30").time():
            days_ahead = 0  # expires today
        else:
            days_ahead = 7

    expiry_dt = (sig_dt.normalize() + pd.Timedelta(days=days_ahead)).replace(
        hour=15, minute=30)
    delta = expiry_dt - sig_dt
    days_fraction = delta.total_seconds() / (365.25 * 24 * 3600)
    return max(days_fraction, 1e-4), expiry_dt


# ══════════════════════════════════════════════
# STRIKE SELECTION
# ══════════════════════════════════════════════
def select_strike(spot, conviction, direction, step=NIFTY_STRIKE_STEP):
    """
    Pick strike based on conviction.

    For BUYING options (PE for short, CE for long):
      • VERY HIGH → ITM-1 (one strike in-the-money)
      • HIGH/MED  → ATM
      • LOW       → OTM-1 (cheap)
    """
    atm = round(spot / step) * step
    offset_steps = STRIKE_OFFSET.get(conviction, 0)

    if direction == 'long':   # buying CE → ITM = strike BELOW spot
        strike = atm + (offset_steps * step)
    else:                     # buying PE → ITM = strike ABOVE spot
        strike = atm - (offset_steps * step)

    return int(strike)


# ══════════════════════════════════════════════
# FORWARD GBM SIMULATION (for exit premium forecast)
# ══════════════════════════════════════════════
def simulate_nifty_paths(S0, mu, sigma, horizon_minutes,
                          n_paths=EXIT_SIM_PATHS, seed=None):
    """
    Simulate GBM paths for Nifty over `horizon_minutes`.

    Returns: array of shape (n_paths, horizon_steps+1)
    """
    if seed is not None:
        np.random.seed(seed)

    # Convert minutes → years for time step
    dt_years = (1.0 / MINUTES_PER_DAY) / TRADING_DAYS_PER_YEAR
    n_steps = int(horizon_minutes)

    # Standard GBM increments
    Z = np.random.randn(n_paths, n_steps)
    drift = (mu - 0.5 * sigma ** 2) * dt_years
    diffusion = sigma * np.sqrt(dt_years) * Z

    log_returns = drift + diffusion
    log_prices = np.log(S0) + np.cumsum(log_returns, axis=1)
    prices = np.exp(log_prices)

    # Prepend S0 to each path
    S0_col = np.full((n_paths, 1), S0)
    paths = np.hstack([S0_col, prices])
    return paths


def forecast_exit_premium_distribution(entry_spot, strike, T_entry, r, sigma,
                                        option_type, horizon_minutes=EXIT_HORIZON_MINUTES,
                                        n_paths=EXIT_SIM_PATHS):
    """
    Forward-simulate Nifty paths and re-price option at each path's end.
    Returns distribution of forecast exit premiums.
    """
    paths = simulate_nifty_paths(entry_spot, mu=r, sigma=sigma,
                                  horizon_minutes=horizon_minutes,
                                  n_paths=n_paths, seed=42)

    # Time at horizon (T decays by horizon_minutes worth of years)
    T_horizon = max(T_entry - (horizon_minutes / MINUTES_PER_DAY) / TRADING_DAYS_PER_YEAR,
                    1e-4)

    final_spots = paths[:, -1]
    exit_premiums = np.array([
        bs_price(s, strike, T_horizon, r, sigma, option_type)
        for s in final_spots
    ])

    return {
        'mean':   round(float(exit_premiums.mean()), 2),
        'median': round(float(np.median(exit_premiums)), 2),
        'p10':    round(float(np.percentile(exit_premiums, 10)), 2),
        'p25':    round(float(np.percentile(exit_premiums, 25)), 2),
        'p75':    round(float(np.percentile(exit_premiums, 75)), 2),
        'p90':    round(float(np.percentile(exit_premiums, 90)), 2),
        'std':    round(float(exit_premiums.std()), 2),
    }


# ══════════════════════════════════════════════
# CORE: PRICE A SINGLE SIGNAL
# ══════════════════════════════════════════════
def price_signal(signal, daily_df, run_exit_sim=True):
    """
    Given one signal dict from Module 3, attach BS pricing data.
    """
    spot         = signal['nifty_spot']
    direction    = signal['direction']
    option_type  = signal['option_type']
    conviction   = signal['conviction']
    sig_time     = pd.Timestamp(signal['signal_time'])

    # 1. Strike
    strike = select_strike(spot, conviction, direction)

    # 2. Time to expiry
    T_years, expiry_dt = days_to_next_weekly_expiry(sig_time)

    # 3. Volatility
    iv = estimate_iv(daily_df, sig_time)

    # 4. Entry premium + Greeks
    entry_premium = bs_price(spot, strike, T_years, RISK_FREE_RATE, iv, option_type)
    greeks        = bs_greeks(spot, strike, T_years, RISK_FREE_RATE, iv, option_type)

    # ──────────────────────────────────────────────
    # V2 CONFIG — Target/SL multipliers
    # ──────────────────────────────────────────────
    # Set TARGET_MULT to 2.0 for V1, 1.75 for V2
    TARGET_MULT = 1.75   # ← V2 (was 2.0 in V1)
    SL_MULT     = 0.60   # unchanged

    # 5. Target & SL premiums
    #    Target = entry × TARGET_MULT  → reward
    #    SL     = entry × SL_MULT      → risk
    target_premium = round(entry_premium * TARGET_MULT, 2)
    sl_premium     = round(entry_premium * SL_MULT, 2)

    result = {
        # Strike & expiry
        'strike':         strike,
        'expiry_dt':      expiry_dt,
        'days_to_expiry': round(T_years * 365.25, 2),
        'T_years':        round(T_years, 5),

        # Volatility
        'iv_used':        iv,

        # Premiums
        'entry_premium':  round(entry_premium, 2),
        'target_premium': target_premium,
        'sl_premium':     sl_premium,

        # Greeks
        **{f"greek_{k}": v for k, v in greeks.items()},

        # Risk-reward on PREMIUM
        'rrr_premium':    round((target_premium - entry_premium) /
                                (entry_premium - sl_premium), 2),
    }

    # 6. Forward simulation for expected exit price (optional)
    if run_exit_sim:
        exit_dist = forecast_exit_premium_distribution(
            spot, strike, T_years, RISK_FREE_RATE, iv, option_type
        )
        result['fwd_sim_mean']   = exit_dist['mean']
        result['fwd_sim_median'] = exit_dist['median']
        result['fwd_sim_p10']    = exit_dist['p10']
        result['fwd_sim_p90']    = exit_dist['p90']

        # Expected edge: positive fwd_sim_mean - entry_premium = edge!
        result['fwd_edge_pts']   = round(exit_dist['mean'] - entry_premium, 2)
        result['fwd_edge_pct']   = round(
            (exit_dist['mean'] - entry_premium) / entry_premium * 100, 2)

    return result


# ══════════════════════════════════════════════
# BATCH: PRICE ALL SIGNALS
# ══════════════════════════════════════════════
def price_all_signals(signals_csv_path, daily_df, output_csv_path=None,
                       run_exit_sim=True):
    """
    Load signals from CSV, attach BS pricing, save augmented CSV.
    """
    df_sigs = pd.read_csv(signals_csv_path, parse_dates=['signal_time', 'breakout_time'])
    print(f"📥 Loaded {len(df_sigs)} signals from {signals_csv_path}")

    pricing_rows = []
    for idx, row in df_sigs.iterrows():
        sig = row.to_dict()
        try:
            pricing = price_signal(sig, daily_df, run_exit_sim=run_exit_sim)
            pricing_rows.append(pricing)
        except Exception as e:
            print(f"  ⚠️ Row {idx}: {e}")
            pricing_rows.append({})

    df_pricing = pd.DataFrame(pricing_rows)
    df_final = pd.concat([df_sigs.reset_index(drop=True), df_pricing], axis=1)

    if output_csv_path:
        df_final.to_csv(output_csv_path, index=False)
        print(f"💾 Saved priced signals: {output_csv_path}")

    return df_final


# ══════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════
if __name__ == "__main__":
    print("=" * 60)
    print("MODULE 4: BLACK-SCHOLES SYNTHETIC PRICER")
    print("=" * 60)

    # Load daily for volatility estimation
    daily = pd.read_csv(f"{DATA_DIR}/nifty_daily.csv",
                        index_col=0, parse_dates=True)
    daily.columns = [c.lower() for c in daily.columns]
    print(f"\n✅ Loaded {len(daily)} daily candles for HV estimation")

    # ── SELF-TEST: Price a sample ATM PE for today ──
    print("\n" + "─" * 60)
    print("🧪 SELF-TEST: ATM Nifty PE")
    print("─" * 60)
    test_spot = 23900
    test_strike = 23900
    test_iv = estimate_iv(daily, daily.index[-1])
    test_T = 3 / 365  # 3 days to expiry
    test_premium = bs_price(test_spot, test_strike, test_T,
                             RISK_FREE_RATE, test_iv, 'PE')
    test_greeks = bs_greeks(test_spot, test_strike, test_T,
                             RISK_FREE_RATE, test_iv, 'PE')
    print(f"  Spot: ₹{test_spot}  Strike: ₹{test_strike}")
    print(f"  IV (annualized): {test_iv*100:.2f}%  (HV proxy)")
    print(f"  T: {test_T*365:.1f} days to expiry")
    print(f"  → Premium: ₹{test_premium:.2f}")
    print(f"  → Delta:   {test_greeks['delta']:.3f}")
    print(f"  → Gamma:   {test_greeks['gamma']:.6f}")
    print(f"  → Theta:   ₹{test_greeks['theta']:.2f}/day")
    print(f"  → Vega:    ₹{test_greeks['vega']:.2f}/1%IV")

    # ── PRICE ALL SIGNALS ──
    signals_csv = f"{OUT_DIR}/signals_log.csv"
    if not os.path.exists(signals_csv):
        print(f"\n⚠️ {signals_csv} not found. Run Module 3 first.")
        sys.exit(1)

    print("\n" + "─" * 60)
    print("📊 PRICING ALL SIGNALS FROM MODULE 3")
    print("─" * 60)
    out_csv = f"{OUT_DIR}/signals_priced.csv"
    df_priced = price_all_signals(signals_csv, daily, output_csv_path=out_csv,
                                    run_exit_sim=True)

    # Summary stats
    print("\n📊 PREMIUM SUMMARY:")
    print(df_priced[['entry_premium', 'target_premium', 'sl_premium',
                     'iv_used', 'rrr_premium']].describe().round(2).to_string())

    print("\n📊 AVG GREEKS BY CONVICTION (absolute values, properly handled):")
    g_cols = ['greek_delta', 'greek_gamma', 'greek_theta', 'greek_vega']

    # Fix: use abs(delta) since CE>0 and PE<0 would cancel out
    df_abs = df_priced.copy()
    df_abs['greek_delta'] = df_abs['greek_delta'].abs()

    print(df_abs.groupby('conviction')[g_cols].mean().round(4).to_string())

    # Also show split by CE/PE for full clarity
    print("\n📊 AVG GREEKS BY CONVICTION × OPTION TYPE (true signed values):")
    print(df_priced.groupby(['conviction', 'option_type'])[g_cols]
          .mean().round(4).to_string())

    if 'fwd_edge_pct' in df_priced.columns:
        print("\n📊 FORWARD-SIMULATED EDGE BY CONVICTION:")
        edge_cols = ['entry_premium', 'fwd_sim_mean', 'fwd_edge_pts', 'fwd_edge_pct']
        print(df_priced.groupby('conviction')[edge_cols].mean().round(2).to_string())

    print("\n📋 SAMPLE OF 10 PRICED SIGNALS:")
    show_cols = ['date', 'signal_time', 'direction', 'option_type',
                 'nifty_spot', 'strike', 'entry_premium',
                 'target_premium', 'sl_premium', 'iv_used',
                 'greek_delta', 'greek_theta', 'conviction']
    print(df_priced[show_cols].head(10).to_string(index=False))

    print(f"\n✅ Module 4 complete. Full output: {out_csv}\n")
