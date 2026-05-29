"""
Module 2: Multi-Timeframe Levels Engine
─────────────────────────────────────────────────
For any given trading day D, computes:

  • TDH/TDL : Running session High/Low (updates each candle)
  • PDH/PDL : Previous trading day's H/L
  • PWH/PWL : Previous COMPLETED week's H/L
  • PMH/PML : Previous COMPLETED month's H/L
  • ATH     : All-time high up to day D (look-ahead-safe)

Design principle: NO LOOK-AHEAD BIAS
─────────────────────────────────────────────────
Every level for day D uses ONLY data strictly before D
(except TDH/TDL which uses today's data UP TO current candle).
This ensures backtests reflect what you'd actually know live.
"""

import pandas as pd
import numpy as np
import sys
import os

# Allow importing sibling module 00_plot_utils
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import importlib.util
spec = importlib.util.spec_from_file_location(
    "plot_utils",
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "00_plot_utils.py")
)
plot_utils = importlib.util.module_from_spec(spec)
spec.loader.exec_module(plot_utils)

# ──────────────────────────────────────────────
# CONFIGURATION
# ──────────────────────────────────────────────
DATA_DIR = "/home/cecadmin/nifty_false_breakout_mc/data"


# ──────────────────────────────────────────────
# DATA LOADERS
# ──────────────────────────────────────────────
def load_all_data():
    """Load all 4 CSVs produced by Module 1."""
    daily = pd.read_csv(f"{DATA_DIR}/nifty_daily.csv",
                        index_col=0, parse_dates=True)
    weekly = pd.read_csv(f"{DATA_DIR}/nifty_weekly.csv",
                         index_col=0, parse_dates=True)
    monthly = pd.read_csv(f"{DATA_DIR}/nifty_monthly.csv",
                          index_col=0, parse_dates=True)
    intraday = pd.read_csv(f"{DATA_DIR}/nifty_5m.csv",
                           index_col=0, parse_dates=True)

    # Normalize column names
    for df in [daily, weekly, monthly, intraday]:
        df.columns = [c.lower() for c in df.columns]

    return daily, weekly, monthly, intraday


# ──────────────────────────────────────────────
# HTF LEVEL EXTRACTORS (look-ahead-safe)
# ──────────────────────────────────────────────
def get_prev_day(daily, target_date):
    """Previous trading day's H/L/C (strictly before target_date)."""
    prior = daily[daily.index.normalize() < pd.Timestamp(target_date).normalize()]
    if prior.empty:
        return None
    row = prior.iloc[-1]
    return {
        'date':  prior.index[-1].date(),
        'high':  row['high'],
        'low':   row['low'],
        'close': row['close'],
    }


def get_prev_week(weekly, target_date):
    """Previous COMPLETED week's H/L/C."""
    target_ts = pd.Timestamp(target_date).normalize()
    # 'date' index = week-ending Friday
    prior = weekly[weekly.index < target_ts]
    if prior.empty:
        return None
    row = prior.iloc[-1]
    return {
        'week_end': prior.index[-1].date(),
        'high':  row['week_high'],
        'low':   row['week_low'],
        'close': row['week_close'],
    }


def get_prev_month(monthly, target_date):
    """Previous COMPLETED month's H/L/C."""
    target_ts = pd.Timestamp(target_date).normalize()
    prior = monthly[monthly.index < target_ts]
    if prior.empty:
        return None
    row = prior.iloc[-1]
    return {
        'month_end': prior.index[-1].date(),
        'high':  row['month_high'],
        'low':   row['month_low'],
        'close': row['month_close'],
    }


def get_ath(daily, target_date):
    """All-time high using only data strictly BEFORE target_date."""
    prior = daily[daily.index.normalize() < pd.Timestamp(target_date).normalize()]
    if prior.empty:
        return None
    ath_idx = prior['high'].idxmax()
    return {
        'date':  ath_idx.date(),
        'high':  prior.loc[ath_idx, 'high'],
        'close': prior.loc[ath_idx, 'close'],
    }


# ──────────────────────────────────────────────
# INTRADAY RUNNING TDH / TDL (session-wide)
# ──────────────────────────────────────────────
def compute_running_tdh_tdl(intraday_day):
    """
    For a single day's intraday DataFrame, computes
    cumulative running high/low at EVERY candle.

    Returns the same DF with two new columns: tdh, tdl
    """
    df = intraday_day.copy().sort_index()
    df['tdh'] = df['high'].cummax()
    df['tdl'] = df['low'].cummin()
    return df


# ──────────────────────────────────────────────
# MASTER FUNCTION: get all levels for a day
# ──────────────────────────────────────────────
def get_levels_for_day(target_date, daily, weekly, monthly, intraday):
    """
    For a given trading day, returns a structured dict of all levels.

    target_date : str 'YYYY-MM-DD' or pd.Timestamp
    """
    target_date = pd.Timestamp(target_date).normalize()

    # Intraday for that specific day
    intra_day = intraday[intraday.index.normalize() == target_date]
    if intra_day.empty:
        raise ValueError(f"No intraday data for {target_date.date()}")
    intra_day = compute_running_tdh_tdl(intra_day)

    levels = {
        'date': target_date.date(),
        'intraday': intra_day,
        'PDH_PDL': get_prev_day(daily, target_date),
        'PWH_PWL': get_prev_week(weekly, target_date),
        'PMH_PML': get_prev_month(monthly, target_date),
        'ATH':     get_ath(daily, target_date),
    }

    # End-of-day TDH/TDL summary (for final logging)
    levels['TDH_final'] = intra_day['tdh'].iloc[-1]
    levels['TDL_final'] = intra_day['tdl'].iloc[-1]

    return levels


# ──────────────────────────────────────────────
# CONFLUENCE DETECTOR
# ──────────────────────────────────────────────
def find_confluence_zones(levels, tolerance_pct=0.25):
    """
    Find clusters of levels within ±tolerance_pct of each other.

    Returns list of confluence zones, each with:
      - center_price
      - members      : list of (label, price, weight)
      - total_weight : sum of priority weights
    """
    # Build flat list: (label, price, weight)
    weights = {'TDH': 5, 'TDL': 5, 'PDH': 4, 'PDL': 4,
               'PWH': 3, 'PWL': 3, 'PMH': 2, 'PML': 2, 'ATH': 5}

    points = []
    if levels['PDH_PDL']:
        points.append(('PDH', levels['PDH_PDL']['high'], weights['PDH']))
        points.append(('PDL', levels['PDH_PDL']['low'],  weights['PDL']))
    if levels['PWH_PWL']:
        points.append(('PWH', levels['PWH_PWL']['high'], weights['PWH']))
        points.append(('PWL', levels['PWH_PWL']['low'],  weights['PWL']))
    if levels['PMH_PML']:
        points.append(('PMH', levels['PMH_PML']['high'], weights['PMH']))
        points.append(('PML', levels['PMH_PML']['low'],  weights['PML']))
    if levels['ATH']:
        points.append(('ATH', levels['ATH']['high'], weights['ATH']))
    # Also include final TDH/TDL of the day for end-of-day analysis
    points.append(('TDH', levels['TDH_final'], weights['TDH']))
    points.append(('TDL', levels['TDL_final'], weights['TDL']))

    # Sort by price ascending
    points.sort(key=lambda x: x[1])

    # Cluster: greedy single-pass
    zones = []
    used = [False] * len(points)
    for i, (lbl, px, w) in enumerate(points):
        if used[i]:
            continue
        cluster = [(lbl, px, w)]
        used[i] = True
        for j in range(i + 1, len(points)):
            if used[j]:
                continue
            if abs(points[j][1] - px) / px * 100 <= tolerance_pct:
                cluster.append(points[j])
                used[j] = True
        if len(cluster) >= 1:
            center = np.mean([c[1] for c in cluster])
            total_w = sum(c[2] for c in cluster)   # ← FIXED
            zones.append({
                'center_price': round(center, 2),
                'members': cluster,
                'total_weight': total_w,
            })

    # Sort zones by total_weight descending (strongest first)
    zones.sort(key=lambda z: z['total_weight'], reverse=True)
    return zones


# ──────────────────────────────────────────────
# VISUALIZATION (sample day) — with smart y-axis clipping
# ──────────────────────────────────────────────
def plot_day_with_levels(levels, save_path=None, y_scale_mode='auto'):
    """
    Plot intraday 5m candles + all reference levels.

    y_scale_mode options:
      'auto'    : Clip to PML..PMH range (recommended) — far ATH shown as annotation
      'full'    : Include ATH in y-axis (old behavior, compressed)
      'tight'   : Clip to ±1.5% around day's TDH/TDL (zoomed-in view)
    """
    import matplotlib.pyplot as plt
    import matplotlib.dates as mdates

    intra = levels['intraday']
    fig, ax = plt.subplots(figsize=(14, 7))

    # Plot price line
    ax.plot(intra.index, intra['close'], color='#1f77b4',
            linewidth=1.4, label='Nifty 5m Close', zorder=3)

    # Running TDH/TDL band
    ax.fill_between(intra.index, intra['tdl'], intra['tdh'],
                    color='gray', alpha=0.08,
                    label='Running TDH/TDL band', zorder=1)

    # ── COLLECT ALL HTF LEVELS ──
    htf_lines = []
    if levels['PDH_PDL']:
        pdh = plot_utils.htf_marker_high(levels['PDH_PDL']['high'],
                                         levels['PDH_PDL']['close'])
        pdl = plot_utils.htf_marker_low(levels['PDH_PDL']['low'],
                                        levels['PDH_PDL']['close'])
        htf_lines += [('PDH', pdh), ('PDL', pdl)]
    if levels['PWH_PWL']:
        pwh = plot_utils.htf_marker_high(levels['PWH_PWL']['high'],
                                         levels['PWH_PWL']['close'])
        pwl = plot_utils.htf_marker_low(levels['PWH_PWL']['low'],
                                        levels['PWH_PWL']['close'])
        htf_lines += [('PWH', pwh), ('PWL', pwl)]
    if levels['PMH_PML']:
        pmh = plot_utils.htf_marker_high(levels['PMH_PML']['high'],
                                         levels['PMH_PML']['close'])
        pml = plot_utils.htf_marker_low(levels['PMH_PML']['low'],
                                        levels['PMH_PML']['close'])
        htf_lines += [('PMH', pmh), ('PML', pml)]

    ath_y = None
    if levels['ATH']:
        ath_y = plot_utils.htf_marker_high(levels['ATH']['high'],
                                           levels['ATH']['close'])

    # ── DETERMINE Y-AXIS RANGE ──
    if y_scale_mode == 'auto':
        # Clip to PML..PMH with 1% padding (monthly range = full context)
        if levels['PMH_PML']:
            y_min = levels['PMH_PML']['low']  * 0.99
            y_max = levels['PMH_PML']['high'] * 1.01
        else:
            y_min = min(intra['low'].min(),  levels['TDL_final']) * 0.995
            y_max = max(intra['high'].max(), levels['TDH_final']) * 1.005
    elif y_scale_mode == 'tight':
        # Zoom to today's range ±1.5%
        y_min = levels['TDL_final'] * 0.985
        y_max = levels['TDH_final'] * 1.015
    else:  # 'full' — include ATH (old compressed behavior)
        y_min = min(intra['low'].min(), levels['TDL_final']) * 0.99
        y_max = max(ath_y if ath_y else 0, levels['TDH_final']) * 1.01

    ax.set_ylim(y_min, y_max)

    # ── DRAW HTF HORIZONTALS (only those visible in y range) ──
    for label, y in htf_lines:
        if y_min <= y <= y_max:
            color, ls, _, _, _ = plot_utils.LEVEL_STYLE[label]
            ax.axhline(y, color=color, linestyle=ls, linewidth=1.3,
                       alpha=0.85, label=f"{label}: {y:.1f}", zorder=2)
        else:
            # Off-chart annotation (arrow at edge)
            color = plot_utils.LEVEL_STYLE[label][0]
            arrow_y = y_max * 0.998 if y > y_max else y_min * 1.002
            arrow_dir = '↑' if y > y_max else '↓'
            ax.annotate(
                f"{arrow_dir} {label}: {y:.1f} (off-chart)",
                xy=(intra.index[-1], arrow_y),
                xytext=(intra.index[-1], arrow_y),
                color=color, fontsize=8, fontweight='bold',
                ha='right', va='top' if y > y_max else 'bottom',
                bbox=dict(boxstyle='round,pad=0.3',
                          facecolor='white', edgecolor=color, alpha=0.85)
            )

    # ── ATH HANDLING (always check separately due to distance) ──
    if ath_y is not None:
        if y_min <= ath_y <= y_max:
            ax.axhline(ath_y, color='#FFD700', linestyle='-',
                       linewidth=1.5, alpha=0.9,
                       label=f"ATH: {ath_y:.1f}")
        else:
            # Big golden annotation at top — most important off-chart level
            ax.annotate(
                f"👑 ATH: {levels['ATH']['high']:.1f}  "
                f"({levels['ATH']['date']}) — OFF CHART ↑",
                xy=(0.5, 0.97), xycoords='axes fraction',
                fontsize=10, fontweight='bold', color='#B8860B',
                ha='center', va='top',
                bbox=dict(boxstyle='round,pad=0.5',
                          facecolor='#FFF8DC',
                          edgecolor='#FFD700', linewidth=1.5)
            )

    # ── TRADE TF: final TDH/TDL ──
    ax.axhline(levels['TDH_final'], color='#FF4444', linestyle='--',
               linewidth=1.2, alpha=0.7,
               label=f"TDH (final): {levels['TDH_final']:.1f}", zorder=2)
    ax.axhline(levels['TDL_final'], color='#44AA44', linestyle='--',
               linewidth=1.2, alpha=0.7,
               label=f"TDL (final): {levels['TDL_final']:.1f}", zorder=2)

    # Cosmetics
    distance_to_ath = ((levels['ATH']['high'] - levels['TDH_final'])
                       / levels['TDH_final'] * 100) if levels['ATH'] else 0
    ax.set_title(
        f"Nifty 50 — Multi-Timeframe Levels  |  {levels['date']}  "
        f"(y-scale: {y_scale_mode})\n"
        f"Distance to ATH: {distance_to_ath:.2f}%",
        fontsize=12, fontweight='bold'
    )
    ax.set_xlabel("Time (IST)")
    ax.set_ylabel("Nifty Price")
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
    ax.grid(alpha=0.25)
    ax.legend(loc='center left', bbox_to_anchor=(1.01, 0.5),
              fontsize=8, frameon=True)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=130, bbox_inches='tight')
        print(f"💾 Chart saved: {save_path}")
    plt.show()


# ──────────────────────────────────────────────
# MAIN — demo for a sample day
# ──────────────────────────────────────────────
if __name__ == "__main__":
    print("=" * 60)
    print("MODULE 2: MULTI-TIMEFRAME LEVELS ENGINE")
    print("=" * 60)

    daily, weekly, monthly, intraday = load_all_data()
    print(f"\n✅ Loaded: {len(daily)} daily | {len(weekly)} weekly | "
          f"{len(monthly)} monthly | {len(intraday)} intraday candles")

    # Pick the most recent FULL trading day in intraday data
    unique_days = sorted(intraday.index.normalize().unique())
    target_date = unique_days[-2]   # last fully-formed day
    print(f"\n🎯 Target day: {target_date.date()}")

    # Get all levels
    levels = get_levels_for_day(target_date, daily, weekly, monthly, intraday)

    # Print neatly
    print("\n" + "─" * 60)
    print(f"📊 LEVELS FOR {levels['date']}")
    print("─" * 60)
    print(f"  TDH (final)  : {levels['TDH_final']:>10.2f}")
    print(f"  TDL (final)  : {levels['TDL_final']:>10.2f}")
    if levels['PDH_PDL']:
        print(f"  PDH          : {levels['PDH_PDL']['high']:>10.2f}  "
              f"(date: {levels['PDH_PDL']['date']})")
        print(f"  PDL          : {levels['PDH_PDL']['low']:>10.2f}")
    if levels['PWH_PWL']:
        print(f"  PWH          : {levels['PWH_PWL']['high']:>10.2f}  "
              f"(week end: {levels['PWH_PWL']['week_end']})")
        print(f"  PWL          : {levels['PWH_PWL']['low']:>10.2f}")
    if levels['PMH_PML']:
        print(f"  PMH          : {levels['PMH_PML']['high']:>10.2f}  "
              f"(month end: {levels['PMH_PML']['month_end']})")
        print(f"  PML          : {levels['PMH_PML']['low']:>10.2f}")
    if levels['ATH']:
        print(f"  ATH          : {levels['ATH']['high']:>10.2f}  "
              f"(date: {levels['ATH']['date']})")

    # Confluence zones
    print("\n" + "─" * 60)
    print("🎯 CONFLUENCE ZONES (clustered within ±0.25%)")
    print("─" * 60)
    zones = find_confluence_zones(levels, tolerance_pct=0.25)
    for i, z in enumerate(zones[:6], 1):
        members = ", ".join(f"{m[0]}@{m[1]:.1f}" for m in z['members'])
        strength = "🔥 VERY STRONG" if z['total_weight'] >= 7 else \
                   "⚡ MEDIUM"      if z['total_weight'] >= 4 else "💭 WEAK"
        print(f"  {i}. Center: {z['center_price']:.2f}  "
              f"Weight: {z['total_weight']}  {strength}")
        print(f"     Members: {members}")

    # Plot with tight view (locked-in default)
    out_dir = "/home/cecadmin/nifty_false_breakout_mc/outputs/charts"
    os.makedirs(out_dir, exist_ok=True)
    save_path = f"{out_dir}/levels_{levels['date']}.png"
    plot_day_with_levels(levels, save_path=save_path, y_scale_mode='tight')

    print("\n✅ Module 2 complete.\n")
