"""
Module 5: Realized Backtest Engine
─────────────────────────────────────────────────
For each signal from Module 4, replay against ACTUAL
subsequent Nifty 5m candles to compute realized P&L.

Logic per signal:
  1. Walk forward through next N candles (up to 90 min)
  2. At each candle, re-price option using:
       - Current real Nifty spot
       - Decaying T (time-to-expiry shrinks)
       - Constant IV (entry IV held — simplification)
  3. Exit on FIRST of:
       a) Target premium hit  → WIN  (+1.0R)
       b) SL premium hit      → LOSS (-1.0R)
       c) Time-stop (90 min)  → exit at current premium

Outputs:
  • signals_backtested.csv  — every trade with realized outcome
  • Calibrated win rate + RRR (feeds Module 6 Monte Carlo)
  • Per-conviction stats   — which tier actually works?
  • Visual: equity curve, MFE/MAE plot
"""

import pandas as pd
import numpy as np
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

bs_mod = _load("bs_mod", "04_bs_pricer.py")


# ──────────────────────────────────────────────
# CONFIGURATION
# ──────────────────────────────────────────────
TIME_STOP_MINUTES   = 90        # Max hold time (intraday)
CANDLES_PER_MIN     = 1 / 5     # 5m candles → 18 candles in 90 min
SESSION_END_TIME    = "15:15"   # Force-exit before close

# Capital + risk sizing (for equity curve)
START_CAPITAL       = 100_000   # ₹1L starting
RISK_PCT_PER_TRADE  = 2.0       # Risk 2% per trade
NIFTY_LOT_SIZE      = 75        # Current Nifty lot size (post 2024 revision)

# Re-pricing assumptions
HOLD_IV_CONSTANT    = True      # Simpler & defensible; can toggle to vol-of-vol model later
INTRABAR_RESOLUTION = 'close'   # 'close' = check exits on candle close only
                                # 'hl'    = also check if H/L touched levels (more aggressive)

# ──────────────────────────────────────────────
# V2 FILTER MODE
# ──────────────────────────────────────────────
# Options:
#   'V1' = use ALL signals (original)
#   'V2' = filter Type B only (drop failed_breakout)
STRATEGY_VERSION = 'V1'   # ← Toggle between 'V1' and 'V2'

# Type B trigger types (swing-based)
TYPE_B_TRIGGERS = ['swing_high_rejection', 'swing_low_reclaim']

DATA_DIR = "/home/cecadmin/nifty_false_breakout_mc/data"
OUT_DIR  = "/home/cecadmin/nifty_false_breakout_mc/outputs"


# ══════════════════════════════════════════════
# CORE: BACKTEST A SINGLE SIGNAL
# ══════════════════════════════════════════════
def backtest_signal(signal_row, intraday_df, verbose=False):
    """
    Replay one signal forward and compute realized P&L.
    """
    # Unpack signal
    entry_time     = pd.Timestamp(signal_row['signal_time'])
    direction      = signal_row['direction']
    option_type    = signal_row['option_type']
    spot_entry     = float(signal_row['nifty_spot'])
    strike         = int(signal_row['strike'])
    entry_premium  = float(signal_row['entry_premium'])
    target_premium = float(signal_row['target_premium'])
    sl_premium     = float(signal_row['sl_premium'])
    iv             = float(signal_row['iv_used'])
    T_entry        = float(signal_row['T_years'])
    r              = bs_mod.RISK_FREE_RATE

    # Forward window
    end_time = min(
        entry_time + pd.Timedelta(minutes=TIME_STOP_MINUTES),
        pd.Timestamp(f"{entry_time.date()} {SESSION_END_TIME}")
    )

    # Get all candles in window (strictly AFTER entry_time)
    fwd = intraday_df[(intraday_df.index > entry_time) &
                       (intraday_df.index <= end_time)].copy()

    if fwd.empty:
        return {
            'outcome':       'no_data',
            'exit_time':     None,
            'exit_spot':     spot_entry,
            'exit_premium':  entry_premium,
            'hold_minutes':  0,
            'r_multiple':    0.0,
            'mfe_premium':   entry_premium,
            'mae_premium':   entry_premium,
        }

    # Track MFE / MAE (max favorable / adverse excursion)
    mfe_premium = entry_premium
    mae_premium = entry_premium
    outcome     = 'time_stop'
    exit_time   = fwd.index[-1]
    exit_spot   = fwd['close'].iloc[-1]
    exit_premium = entry_premium

    # Walk forward candle by candle
    for ts, row in fwd.iterrows():
        # Time decay: T shrinks as we move forward
        minutes_elapsed = (ts - entry_time).total_seconds() / 60
        T_now = max(T_entry -
                    (minutes_elapsed / bs_mod.MINUTES_PER_DAY) /
                    bs_mod.TRADING_DAYS_PER_YEAR, 1e-5)

        # Re-price option at this candle
        if INTRABAR_RESOLUTION == 'hl':
            # Check both extremes for the candle (more realistic for SL/Target)
            spot_high = row['high']
            spot_low  = row['low']

            prem_high = bs_mod.bs_price(spot_high, strike, T_now, r, iv, option_type)
            prem_low  = bs_mod.bs_price(spot_low,  strike, T_now, r, iv, option_type)

            # For BUYING options: PE gains when spot falls, CE gains when spot rises
            if option_type == 'PE':
                best_prem  = prem_low   # PE max when spot low
                worst_prem = prem_high  # PE min when spot high
            else:  # CE
                best_prem  = prem_high  # CE max when spot high
                worst_prem = prem_low

            mfe_premium = max(mfe_premium, best_prem)
            mae_premium = min(mae_premium, worst_prem)

            # Check exits (favorable hit first = optimistic but standard)
            if best_prem >= target_premium:
                outcome       = 'target_hit'
                exit_time     = ts
                exit_spot     = spot_low if option_type == 'PE' else spot_high
                exit_premium  = target_premium
                break
            if worst_prem <= sl_premium:
                outcome       = 'sl_hit'
                exit_time     = ts
                exit_spot     = spot_high if option_type == 'PE' else spot_low
                exit_premium  = sl_premium
                break

        else:  # 'close' resolution
            spot_now = row['close']
            prem_now = bs_mod.bs_price(spot_now, strike, T_now, r, iv, option_type)
            mfe_premium = max(mfe_premium, prem_now)
            mae_premium = min(mae_premium, prem_now)

            if prem_now >= target_premium:
                outcome       = 'target_hit'
                exit_time     = ts
                exit_spot     = spot_now
                exit_premium  = target_premium
                break
            if prem_now <= sl_premium:
                outcome       = 'sl_hit'
                exit_time     = ts
                exit_spot     = spot_now
                exit_premium  = sl_premium
                break

            exit_time     = ts
            exit_spot     = spot_now
            exit_premium  = prem_now

    # Compute R-multiple
    risk_per_unit   = entry_premium - sl_premium     # ₹ at risk per option
    reward_per_unit = exit_premium  - entry_premium  # ₹ gained (or lost)
    r_multiple      = reward_per_unit / risk_per_unit if risk_per_unit > 0 else 0

    hold_minutes = (exit_time - entry_time).total_seconds() / 60

    if verbose:
        print(f"  {entry_time.time()} {direction:5s} {option_type} K={strike} "
              f"prem {entry_premium:.1f}→{exit_premium:.1f} | "
              f"{outcome} in {hold_minutes:.0f}min | R={r_multiple:+.2f}")

    return {
        'outcome':       outcome,
        'exit_time':     exit_time,
        'exit_spot':     round(float(exit_spot), 2),
        'exit_premium':  round(float(exit_premium), 2),
        'hold_minutes':  round(hold_minutes, 1),
        'r_multiple':    round(float(r_multiple), 3),
        'mfe_premium':   round(float(mfe_premium), 2),
        'mae_premium':   round(float(mae_premium), 2),
    }


# ══════════════════════════════════════════════
# BATCH BACKTEST
# ══════════════════════════════════════════════
def backtest_all_signals(priced_csv, intraday_df, output_csv=None, verbose=False):
    """Load priced signals → backtest all → save augmented CSV."""
    df = pd.read_csv(priced_csv, parse_dates=['signal_time', 'breakout_time'])
    print(f"📥 Loaded {len(df)} priced signals")

    results = []
    for idx, row in df.iterrows():
        try:
            res = backtest_signal(row, intraday_df, verbose=verbose)
        except Exception as e:
            print(f"  ⚠️ Row {idx}: {e}")
            res = {'outcome': 'error', 'r_multiple': 0,
                   'exit_time': None, 'exit_spot': None,
                   'exit_premium': None, 'hold_minutes': 0,
                   'mfe_premium': None, 'mae_premium': None}
        results.append(res)

    df_res = pd.DataFrame(results)
    df_final = pd.concat([df.reset_index(drop=True), df_res], axis=1)

    if output_csv:
        df_final.to_csv(output_csv, index=False)
        print(f"💾 Saved backtest results: {output_csv}")

    return df_final


# ══════════════════════════════════════════════
# ANALYTICS
# ══════════════════════════════════════════════
def summarize_backtest(df):
    """Print key statistics + return calibrated metrics dict."""
    # Filter valid trades
    valid = df[df['outcome'].isin(['target_hit', 'sl_hit', 'time_stop'])].copy()

    if valid.empty:
        print("⚠️ No valid trades!")
        return None

    n_total = len(valid)
    n_wins  = (valid['r_multiple'] > 0).sum()
    n_loss  = (valid['r_multiple'] < 0).sum()
    n_flat  = (valid['r_multiple'] == 0).sum()
    win_rate = n_wins / n_total

    wins  = valid[valid['r_multiple'] > 0]['r_multiple']
    losses = valid[valid['r_multiple'] < 0]['r_multiple']

    avg_win  = wins.mean()  if len(wins)  > 0 else 0
    avg_loss = losses.mean() if len(losses) > 0 else 0
    rrr_realized = abs(avg_win / avg_loss) if avg_loss != 0 else np.inf

    # Expectancy per trade (in R)
    expectancy_R = (win_rate * avg_win) + ((1 - win_rate) * avg_loss)

    # Breakeven win rate
    breakeven_wr = 1 / (1 + rrr_realized) if rrr_realized > 0 else 1.0

    # Outcome breakdown
    print("\n" + "═" * 60)
    print("📊 BACKTEST SUMMARY — CALIBRATED EDGE")
    print("═" * 60)
    print(f"\n  Total trades       : {n_total}")
    print(f"  Wins (target)      : {n_wins}")
    print(f"  Losses (SL)        : {n_loss}")
    print(f"  Time-stop / flat   : {n_flat}")
    print(f"\n  ► WIN RATE         : {win_rate*100:>6.2f}%")
    print(f"  ► AVG WIN (R)      : {avg_win:>+6.3f}R")
    print(f"  ► AVG LOSS (R)     : {avg_loss:>+6.3f}R")
    print(f"  ► REALIZED RRR     : 1:{rrr_realized:.2f}")
    print(f"  ► EXPECTANCY/trade : {expectancy_R:>+6.3f}R")
    print(f"  ► BREAKEVEN WR     : {breakeven_wr*100:>6.2f}%")
    print(f"  ► EDGE             : {(win_rate - breakeven_wr)*100:>+6.2f}%")

    verdict = "✅ POSITIVE EDGE" if expectancy_R > 0 else "❌ NEGATIVE EDGE"
    print(f"\n  VERDICT: {verdict}")

    # Outcome distribution
    print("\n📊 OUTCOME DISTRIBUTION:")
    print(valid['outcome'].value_counts().to_string())

    # By conviction
    print("\n📊 PERFORMANCE BY CONVICTION:")
    by_conv = valid.groupby('conviction').agg(
        trades=('r_multiple', 'count'),
        win_rate=('r_multiple', lambda x: (x > 0).mean()),
        avg_R=('r_multiple', 'mean'),
        total_R=('r_multiple', 'sum'),
    ).round(3)
    by_conv['win_rate'] = (by_conv['win_rate'] * 100).round(1).astype(str) + '%'
    print(by_conv.to_string())

    # By direction
    print("\n📊 PERFORMANCE BY DIRECTION:")
    by_dir = valid.groupby('direction').agg(
        trades=('r_multiple', 'count'),
        win_rate=('r_multiple', lambda x: (x > 0).mean()),
        avg_R=('r_multiple', 'mean'),
    ).round(3)
    by_dir['win_rate'] = (by_dir['win_rate'] * 100).round(1).astype(str) + '%'
    print(by_dir.to_string())

    # By trigger type
    if 'trigger_type' in valid.columns:
        print("\n📊 PERFORMANCE BY TRIGGER TYPE:")
        by_trig = valid.groupby('trigger_type').agg(
            trades=('r_multiple', 'count'),
            win_rate=('r_multiple', lambda x: (x > 0).mean()),
            avg_R=('r_multiple', 'mean'),
        ).round(3)
        by_trig['win_rate'] = (by_trig['win_rate'] * 100).round(1).astype(str) + '%'
        print(by_trig.to_string())

    # Avg hold time
    print(f"\n📊 AVG HOLD TIME: {valid['hold_minutes'].mean():.1f} min")
    print(f"📊 AVG MFE: {(valid['mfe_premium'] - valid['entry_premium']).mean():.2f} ₹")
    print(f"📊 AVG MAE: {(valid['mae_premium'] - valid['entry_premium']).mean():.2f} ₹")

    return {
        'n_trades':     n_total,
        'win_rate':     round(win_rate, 4),
        'avg_win_R':    round(avg_win, 3),
        'avg_loss_R':   round(avg_loss, 3),
        'rrr_realized': round(rrr_realized, 2),
        'expectancy_R': round(expectancy_R, 4),
        'breakeven_wr': round(breakeven_wr, 4),
        'edge_pct':     round((win_rate - breakeven_wr) * 100, 2),
        'verdict':      verdict,
    }


# ══════════════════════════════════════════════
# EQUITY CURVE PLOT
# ══════════════════════════════════════════════
def plot_equity_curve(df, save_path=None):
    """Cumulative R curve + drawdown."""
    import matplotlib.pyplot as plt

    valid = df[df['outcome'].isin(['target_hit', 'sl_hit', 'time_stop'])].copy()
    valid = valid.sort_values('signal_time').reset_index(drop=True)

    valid['cum_R']  = valid['r_multiple'].cumsum()
    valid['peak']   = valid['cum_R'].cummax()
    valid['dd']     = valid['cum_R'] - valid['peak']

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 8),
                                     sharex=True,
                                     gridspec_kw={'height_ratios': [3, 1]})

    # Equity
    ax1.plot(valid.index, valid['cum_R'], color='#1f77b4',
             linewidth=1.8, label='Cumulative R')
    ax1.fill_between(valid.index, 0, valid['cum_R'],
                     where=(valid['cum_R'] >= 0), alpha=0.15, color='green')
    ax1.fill_between(valid.index, 0, valid['cum_R'],
                     where=(valid['cum_R'] < 0), alpha=0.15, color='red')
    ax1.axhline(0, color='gray', linewidth=0.8)
    ax1.set_ylabel('Cumulative R')
    ax1.set_title('Realized Backtest — Equity Curve (in R-multiples)',
                  fontweight='bold')
    ax1.grid(alpha=0.3)
    ax1.legend()

    # Drawdown
    ax2.fill_between(valid.index, valid['dd'], 0, color='red', alpha=0.4)
    ax2.set_ylabel('Drawdown (R)')
    ax2.set_xlabel('Trade #')
    ax2.grid(alpha=0.3)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=130, bbox_inches='tight')
        print(f"💾 Equity curve saved: {save_path}")
    plt.show()


# ══════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════
if __name__ == "__main__":
    print("=" * 60)
    print("MODULE 5: REALIZED BACKTEST ENGINE")
    print("=" * 60)

    # Load intraday data
    intra = pd.read_csv(f"{DATA_DIR}/nifty_5m.csv",
                         index_col=0, parse_dates=True)
    intra.columns = [c.lower() for c in intra.columns]
    if intra.index.tz is not None:
        intra.index = intra.index.tz_convert("Asia/Kolkata").tz_localize(None)
    intra = intra.between_time("09:15", "15:30")
    print(f"\n✅ Loaded {len(intra)} intraday candles")

    # Backtest
    priced_csv = f"{OUT_DIR}/signals_priced.csv"
    if not os.path.exists(priced_csv):
        print(f"⚠️ {priced_csv} not found — run Module 4 first")
        sys.exit(1)

    # Apply V2 filter BEFORE backtesting (faster, cleaner)
    df_priced = pd.read_csv(priced_csv,
                              parse_dates=['signal_time', 'breakout_time'])

    print(f"\n📊 STRATEGY VERSION: {STRATEGY_VERSION}")
    print(f"   Total signals available: {len(df_priced)}")

    if STRATEGY_VERSION == 'V2':
        before = len(df_priced)
        df_priced = df_priced[df_priced['trigger_type'].isin(TYPE_B_TRIGGERS)]
        after = len(df_priced)
        print(f"   V2 filter applied: dropped {before - after} Type A signals")
        print(f"   Remaining (Type B): {after}")

        # Save V2-filtered priced signals
        v2_priced_csv = f"{OUT_DIR}/signals_priced_V2.csv"
        df_priced.to_csv(v2_priced_csv, index=False)
        priced_csv = v2_priced_csv  # use filtered version for backtest

    # Run backtest
    out_csv = f"{OUT_DIR}/signals_backtested_{STRATEGY_VERSION}.csv"
    df_bt = backtest_all_signals(priced_csv, intra,
                                  output_csv=out_csv, verbose=False)

    # Summary
    metrics = summarize_backtest(df_bt)

    # Save calibrated metrics for Module 6
    if metrics:
        meta_csv = f"{OUT_DIR}/calibrated_metrics_{STRATEGY_VERSION}.csv"
        pd.DataFrame([metrics]).to_csv(meta_csv, index=False)
        print(f"\n💾 Calibrated metrics saved: {meta_csv}")
        print("   → These feed Module 6 Monte Carlo")

    # Plot
    chart_dir = f"{OUT_DIR}/charts"
    os.makedirs(chart_dir, exist_ok=True)
    plot_equity_curve(df_bt,
        save_path=f"{chart_dir}/equity_curve_{STRATEGY_VERSION}.png")

    print("\n✅ Module 5 complete.\n")
