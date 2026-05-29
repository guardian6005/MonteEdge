"""
Module 3: False Breakout Signal Engine
─────────────────────────────────────────────────
Detects false breakout reversal setups on 5m Nifty data
using multi-timeframe confluence + speed/volume filters.

Pipeline:
  1. Detect breakouts above TDH / below TDL
  2. Wait up to 40 candles for reversal back through level
  3. Score confluence with PDH/PWH/PMH/ATH/PDL/PWL/PML
  4. Apply speed + volume filters
  5. Emit trade signal with conviction tier
"""

import pandas as pd
import numpy as np
import os
import sys
import importlib.util

# Import sibling modules
THIS_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, THIS_DIR)

def _load(name, filename):
    spec = importlib.util.spec_from_file_location(name, os.path.join(THIS_DIR, filename))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod

plot_utils = _load("plot_utils", "00_plot_utils.py")
levels_mod = _load("levels_mod", "02_levels.py")


# ──────────────────────────────────────────────
# CONFIGURATION
# ──────────────────────────────────────────────
OBSERVATION_END     = "10:15"   # No trades before this
SESSION_END         = "15:15"   # No new entries after this (15 min before close)
REVERSAL_WINDOW     = 24        # Max candles to wait for reversal (24 × 5m = 2 hrs)
SPEED_FAST          = 3         # ≤3 candles = HIGH conviction
SPEED_MEDIUM        = 10        # 4-10 candles = MEDIUM
VOLUME_AVG_LOOKBACK = 20        # Bars to compute avg volume
VOLUME_MULTIPLIER   = 1.5       # Reversal candle volume threshold
CONFLUENCE_TOLERANCE = 0.25     # ±0.25% to count as near a HTF level
MIN_BREAKOUT_PCT     = 0.05     # Breakout must exceed TDH/TDL by ≥0.05% (avoid noise)

DATA_DIR = "/home/cecadmin/nifty_false_breakout_mc/data"
OUT_DIR  = "/home/cecadmin/nifty_false_breakout_mc/outputs"


# ──────────────────────────────────────────────
# HELPER: build list of HTF levels for confluence check
# ──────────────────────────────────────────────
def _build_htf_level_list(day_levels):
    """Returns list of (label, price, weight) for nearby-level scoring."""
    weights = {'PDH': 4, 'PDL': 4, 'PWH': 3, 'PWL': 3,
               'PMH': 2, 'PML': 2, 'ATH': 5}
    lst = []
    if day_levels['PDH_PDL']:
        lst += [('PDH', day_levels['PDH_PDL']['high'], weights['PDH']),
                ('PDL', day_levels['PDH_PDL']['low'],  weights['PDL'])]
    if day_levels['PWH_PWL']:
        lst += [('PWH', day_levels['PWH_PWL']['high'], weights['PWH']),
                ('PWL', day_levels['PWH_PWL']['low'],  weights['PWL'])]
    if day_levels['PMH_PML']:
        lst += [('PMH', day_levels['PMH_PML']['high'], weights['PMH']),
                ('PML', day_levels['PMH_PML']['low'],  weights['PML'])]
    if day_levels['ATH']:
        lst += [('ATH', day_levels['ATH']['high'], weights['ATH'])]
    return lst


def _confluence_score(price, htf_levels, tolerance_pct=CONFLUENCE_TOLERANCE):
    """Sum weights of HTF levels within tolerance% of given price."""
    score = 0
    near = []
    for label, lvl_price, weight in htf_levels:
        if abs(lvl_price - price) / price * 100 <= tolerance_pct:
            score += weight
            near.append((label, lvl_price, weight))
    # TDH/TDL itself contributes weight 5 (the breakout level)
    score += 5
    return score, near


def _conviction_tier(score, speed_candles):
    """Combine confluence + speed → final conviction."""
    if score >= 10 and speed_candles <= SPEED_FAST:
        return "VERY HIGH"
    if score >= 7 and speed_candles <= SPEED_MEDIUM:
        return "HIGH"
    if score >= 5:
        return "MEDIUM"
    return "LOW"


def _strike_recommendation(conviction):
    """Map conviction → suggested option strike type."""
    return {
        "VERY HIGH": "ITM-1 (deep delta ~0.6)",
        "HIGH":      "ATM (delta ~0.5)",
        "MEDIUM":    "ATM",
        "LOW":       "Skip OR small-size ATM",
    }[conviction]


# ──────────────────────────────────────────────
# TRENDING MARKET FILTER (Gate #1: ADX-based)
# ──────────────────────────────────────────────
def detect_intraday_trend(df_slice, adx_period=14, lookback_bars=10):
    """
    Detect trend direction + strength using ADX + slope.

    Returns:
        direction : 'bullish' | 'bearish' | 'neutral'
        strength  : 'strong' | 'moderate' | 'weak'
        adx_value : float
    """
    if len(df_slice) < adx_period + 2:
        return 'neutral', 'weak', 0.0

    high  = df_slice['high']
    low   = df_slice['low']
    close = df_slice['close']

    # True Range
    tr = pd.concat([
        high - low,
        (high - close.shift(1)).abs(),
        (low  - close.shift(1)).abs()
    ], axis=1).max(axis=1)
    atr = tr.rolling(adx_period).mean()

    # Directional Movement
    plus_dm  = high.diff().clip(lower=0)
    minus_dm = (-low.diff()).clip(lower=0)
    plus_dm  = plus_dm.where(plus_dm > minus_dm, 0)
    minus_dm = minus_dm.where(minus_dm > plus_dm, 0)

    plus_di  = 100 * plus_dm.rolling(adx_period).mean()  / atr
    minus_di = 100 * minus_dm.rolling(adx_period).mean() / atr

    dx  = (100 * (plus_di - minus_di).abs() /
                  (plus_di + minus_di).replace(0, np.nan)).fillna(0)
    adx = dx.rolling(adx_period).mean()

    latest_adx      = adx.iloc[-1] if not adx.empty else 0
    latest_plus_di  = plus_di.iloc[-1]  if not plus_di.empty  else 0
    latest_minus_di = minus_di.iloc[-1] if not minus_di.empty else 0

    # Slope of last N closes
    recent = close.iloc[-lookback_bars:]
    if len(recent) >= 3:
        slope = np.polyfit(range(len(recent)), recent.values, 1)[0]
    else:
        slope = 0

    # Classify
    if latest_adx >= 25:
        strength = 'strong'
    elif latest_adx >= 18:
        strength = 'moderate'
    else:
        strength = 'weak'

    if latest_plus_di > latest_minus_di and slope > 0:
        direction = 'bullish'
    elif latest_minus_di > latest_plus_di and slope < 0:
        direction = 'bearish'
    else:
        direction = 'neutral'

    return direction, strength, round(float(latest_adx), 2)


def is_counter_trend_blocked(signal_direction, df_slice,
                              adx_threshold=20):
    """
    Block signal if it goes against a moderate/strong trend.

    Returns:
        blocked (bool), reason (str), trend_meta (dict)
    """
    direction, strength, adx_val = detect_intraday_trend(df_slice)

    meta = {'trend_dir': direction, 'trend_strength': strength,
            'adx': adx_val}

    if strength in ('strong', 'moderate') and adx_val >= adx_threshold:
        if direction == 'bearish' and signal_direction == 'long':
            return True, (f"counter-trend LONG in BEARISH trend "
                          f"(ADX={adx_val}, {strength})"), meta
        if direction == 'bullish' and signal_direction == 'short':
            return True, (f"counter-trend SHORT in BULLISH trend "
                          f"(ADX={adx_val}, {strength})"), meta

    return False, f"trend={direction}, ADX={adx_val}, {strength}", meta


# ──────────────────────────────────────────────
# TRENDING MARKET FILTER (Gate #2: Freefall detection)
# ──────────────────────────────────────────────
FREEFALL_BARS         = 6        # last N candles
FREEFALL_POINTS_DROP  = 40       # min total drop (in Nifty points)
FREEFALL_FALLING_PCT  = 0.75     # ≥75% of bars must be falling

def is_in_freefall(df_slice, signal_direction):
    """
    Quick momentum check: is price in continuous fall/rise?
    Blocks LONG if freefall down, SHORT if freefall up.
    """
    if len(df_slice) < FREEFALL_BARS + 1:
        return False, "insufficient bars"

    recent = df_slice['close'].iloc[-FREEFALL_BARS:]
    total_change = recent.iloc[-1] - recent.iloc[0]
    diffs = recent.diff().dropna()

    falling_count = (diffs < 0).sum()
    rising_count  = (diffs > 0).sum()
    pct_falling = falling_count / len(diffs)
    pct_rising  = rising_count  / len(diffs)

    # Freefall DOWN — block LONGs
    if (signal_direction == 'long'
            and total_change <= -FREEFALL_POINTS_DROP
            and pct_falling >= FREEFALL_FALLING_PCT):
        return True, (f"freefall DOWN: {total_change:.1f} pts in "
                      f"last {FREEFALL_BARS} bars ({pct_falling*100:.0f}% red)")

    # Freefall UP — block SHORTs
    if (signal_direction == 'short'
            and total_change >= FREEFALL_POINTS_DROP
            and pct_rising >= FREEFALL_FALLING_PCT):
        return True, (f"freefall UP: +{total_change:.1f} pts in "
                      f"last {FREEFALL_BARS} bars ({pct_rising*100:.0f}% green)")

    return False, "no freefall"


# ──────────────────────────────────────────────
# ADX DIRECTION CHECK (is ADX rising or falling?)
# ──────────────────────────────────────────────
def is_adx_rising(df_slice, adx_period=14, lookback=3):
    """
    Check if ADX has been RISING over last `lookback` computed values.
    Rising ADX = trend strengthening (good for directional trades).
    Flat/falling ADX = chop (bad for Type B swing signals).
    """
    if len(df_slice) < adx_period * 2 + lookback:
        return False

    high  = df_slice['high']
    low   = df_slice['low']
    close = df_slice['close']

    tr = pd.concat([
        high - low,
        (high - close.shift(1)).abs(),
        (low  - close.shift(1)).abs()
    ], axis=1).max(axis=1)
    atr = tr.rolling(adx_period).mean()

    plus_dm  = high.diff().clip(lower=0)
    minus_dm = (-low.diff()).clip(lower=0)
    plus_dm  = plus_dm.where(plus_dm > minus_dm, 0)
    minus_dm = minus_dm.where(minus_dm > plus_dm, 0)

    plus_di  = 100 * plus_dm.rolling(adx_period).mean()  / atr
    minus_di = 100 * minus_dm.rolling(adx_period).mean() / atr

    dx  = (100 * (plus_di - minus_di).abs() /
                  (plus_di + minus_di).replace(0, np.nan)).fillna(0)
    adx = dx.rolling(adx_period).mean()

    recent_adx = adx.dropna().iloc[-lookback:]
    if len(recent_adx) < lookback:
        return False

    # ADX is "rising" if last value > first value in lookback window
    return float(recent_adx.iloc[-1]) > float(recent_adx.iloc[0])


# ──────────────────────────────────────────────
# GATE PARAMETERS (RECALIBRATED — balanced strictness)
# ──────────────────────────────────────────────

# Gate 3: Reversal quality
REVERSAL_QUALITY_MIN_LOWER_HIGHS  = 0.40   # was 0.50 → loosened
REVERSAL_QUALITY_MAX_HIGHER_LOWS  = 0.55   # was 0.40 → loosened
REVERSAL_QUALITY_MIN_DROP_PCT     = 0.08   # was 0.10
REVERSAL_QUALITY_MIN_DROP_POINTS  = 12     # was 15

# Gate 4: Day structure (now uses RECENT window only)
SWING_WINDOW                       = 3       # reverted to 3 for better swing detection
SWING_LOOKBACK_BARS                = 20      # was 40 → ONLY last 100min, not whole day
MAX_LOWER_LOWS_FOR_CE              = 3       # was 2 → allow normal reversal context
MAX_HIGHER_HIGHS_FOR_PE            = 3       # was 2
MIN_BOUNCE_RETRACE_PCT             = 0.15    # was 0.20 → loosened

# ──────────────────────────────────────────────
# REVERSAL QUALITY FILTER (Gate #3: Sluggishness detection)
# ──────────────────────────────────────────────

def assess_reversal_quality(df_slice, breakout_idx, current_idx,
                             broken_level, direction):
    """
    Assesses HOW the reversal happened (not just that it happened).

    For a SHORT (failed up-breakout) we want:
      • Mostly lower highs in the reversal window
      • Very few higher lows (sluggishness flag)
      • Meaningful drop from intraday peak
      • Candles with real range (not tiny dojis)

    Returns: (is_quality, label, metrics_dict)
    """
    # Window from breakout to current candle
    window = df_slice.iloc[breakout_idx: current_idx + 1]

    if len(window) < 3:
        return False, "too_few_candles", {}

    highs  = window['high'].values
    lows   = window['low'].values
    closes = window['close'].values

    # Find the actual extreme (peak for short, trough for long)
    if direction == 'short':
        peak       = highs.max()
        peak_idx   = highs.argmax()
        post_peak  = window.iloc[peak_idx:]
        if len(post_peak) < 2:
            return False, "no_post_peak_action", {}

        ph = post_peak['high'].values
        pl = post_peak['low'].values
        pc = post_peak['close'].values

        # Count lower highs / higher lows AFTER the peak
        lower_highs  = sum(1 for j in range(1, len(ph)) if ph[j] < ph[j-1])
        higher_lows  = sum(1 for j in range(1, len(pl)) if pl[j] > pl[j-1])
        n_steps = len(ph) - 1

        lh_ratio = lower_highs / n_steps if n_steps else 0
        hl_ratio = higher_lows / n_steps if n_steps else 0

        # Drop from peak
        drop_pts = peak - pc[-1]
        drop_pct = drop_pts / peak * 100

        # Avg candle range (vs the move size)
        avg_range = np.mean(ph - pl)

        metrics = {
            'lh_ratio':    round(lh_ratio, 2),
            'hl_ratio':    round(hl_ratio, 2),
            'drop_points': round(drop_pts, 2),
            'drop_pct':    round(drop_pct, 3),
            'avg_range':   round(avg_range, 2),
            'peak':        round(peak, 2),
        }

        # Decision
        if hl_ratio > REVERSAL_QUALITY_MAX_HIGHER_LOWS:
            return False, f"sluggish_higher_lows(hl={hl_ratio:.0%})", metrics
        if lh_ratio < REVERSAL_QUALITY_MIN_LOWER_HIGHS:
            return False, f"weak_no_lower_highs(lh={lh_ratio:.0%})", metrics
        if drop_pts < REVERSAL_QUALITY_MIN_DROP_POINTS:
            return False, f"shallow_drop({drop_pts:.1f}pts)", metrics
        if drop_pct < REVERSAL_QUALITY_MIN_DROP_PCT:
            return False, f"shallow_drop_pct({drop_pct:.2f}%)", metrics

        return True, "strong_reversal", metrics

    else:  # direction == 'long' (mirror logic)
        trough     = lows.min()
        trough_idx = lows.argmin()
        post = window.iloc[trough_idx:]
        if len(post) < 2:
            return False, "no_post_trough_action", {}

        ph = post['high'].values
        pl = post['low'].values
        pc = post['close'].values

        higher_highs = sum(1 for j in range(1, len(ph)) if ph[j] > ph[j-1])
        lower_highs  = sum(1 for j in range(1, len(ph)) if ph[j] < ph[j-1])
        n_steps = len(ph) - 1

        hh_ratio = higher_highs / n_steps if n_steps else 0
        lh_ratio = lower_highs  / n_steps if n_steps else 0

        rise_pts = pc[-1] - trough
        rise_pct = rise_pts / trough * 100
        avg_range = np.mean(ph - pl)

        metrics = {
            'hh_ratio':    round(hh_ratio, 2),
            'lh_ratio':    round(lh_ratio, 2),  # for longs, "sluggish" = lower highs persist
            'rise_points': round(rise_pts, 2),
            'rise_pct':    round(rise_pct, 3),
            'avg_range':   round(avg_range, 2),
            'trough':      round(trough, 2),
        }

        if lh_ratio > REVERSAL_QUALITY_MAX_HIGHER_LOWS:
            return False, f"sluggish_lower_highs(lh={lh_ratio:.0%})", metrics
        if hh_ratio < REVERSAL_QUALITY_MIN_LOWER_HIGHS:
            return False, f"weak_no_higher_highs(hh={hh_ratio:.0%})", metrics
        if rise_pts < REVERSAL_QUALITY_MIN_DROP_POINTS:
            return False, f"shallow_rise({rise_pts:.1f}pts)", metrics
        if rise_pct < REVERSAL_QUALITY_MIN_DROP_PCT:
            return False, f"shallow_rise_pct({rise_pct:.2f}%)", metrics

        return True, "strong_reversal", metrics


# ──────────────────────────────────────────────
# STRUCTURAL TREND FILTER (Gate #4: Lower-Lows / Higher-Highs detection)
# ──────────────────────────────────────────────
SWING_WINDOW          = 3       # candles on each side to confirm a swing point
SWING_LOOKBACK_BARS   = 40      # look at last N candles (200 min on 5m)
MAX_LOWER_LOWS_FOR_CE = 2       # >2 consecutive lower swing lows = block CE
MAX_HIGHER_HIGHS_FOR_PE = 2     # >2 consecutive higher swing highs = block PE
MIN_BOUNCE_RETRACE_PCT = 0.20   # CE bounce must retrace ≥20% of recent fall

def find_swing_points(df_slice, window=SWING_WINDOW):
    """
    Identify swing highs and lows.
    A swing low = bar whose low is lower than `window` bars on each side.
    Returns: (swing_highs, swing_lows) as lists of (timestamp, price)
    """
    swing_highs = []
    swing_lows  = []
    highs = df_slice['high'].values
    lows  = df_slice['low'].values
    idxs  = df_slice.index

    for i in range(window, len(df_slice) - window):
        is_swing_high = all(highs[i] >= highs[i-j] for j in range(1, window+1)) and \
                        all(highs[i] >= highs[i+j] for j in range(1, window+1))
        is_swing_low  = all(lows[i]  <= lows[i-j]  for j in range(1, window+1)) and \
                        all(lows[i]  <= lows[i+j]  for j in range(1, window+1))
        if is_swing_high:
            swing_highs.append((idxs[i], highs[i]))
        if is_swing_low:
            swing_lows.append((idxs[i], lows[i]))
    return swing_highs, swing_lows


def assess_day_structure(df_slice, signal_direction):
    """
    Recent structural bias detector with REVERSAL CONFIRMATION.

    Key change from v1:
    • Only looks at the RECENT window (last ~20 candles = 100min on 5m)
      → A genuine reversal NEEDS prior weakness; we should not block
        a CE trade just because the morning was bearish.
    • Requires either:
        (a) Structure has already shifted (HL formed after LLs), OR
        (b) Strong bounce magnitude proves momentum shift
    """
    recent = df_slice.iloc[-min(SWING_LOOKBACK_BARS, len(df_slice)):]
    if len(recent) < SWING_WINDOW * 2 + 3:
        return True, "insufficient_data", {}

    swing_highs, swing_lows = find_swing_points(recent, window=SWING_WINDOW)

    # ── For LONG (CE) trades ──
    if signal_direction == 'long':
        # Count consecutive lower lows at END of swing series
        ll_run = 0
        for k in range(len(swing_lows)-1, 0, -1):
            if swing_lows[k][1] < swing_lows[k-1][1]:
                ll_run += 1
            else:
                break

        # Check if there's already been a Higher Low (reversal structure)
        has_higher_low = False
        if len(swing_lows) >= 2:
            # Most recent swing low compared to one before
            has_higher_low = swing_lows[-1][1] > swing_lows[-2][1]

        # Bounce magnitude check
        recent_low   = recent['low'].min()
        current      = recent['close'].iloc[-1]
        recent_high  = recent['high'].max()
        fall_size    = recent_high - recent_low
        bounce_size  = current - recent_low
        retrace_pct  = bounce_size / fall_size if fall_size > 0 else 0

        metrics = {
            'll_run': int(ll_run),
            'has_higher_low': bool(has_higher_low),
            'retrace_pct': round(retrace_pct, 3),
            'recent_low': round(recent_low, 2),
            'bounce_size': round(bounce_size, 2),
            'n_swings_low': len(swing_lows),
        }

        # ── DECISION ──
        # Pass if structure HAS reversed OR bounce is strong
        if has_higher_low:
            return True, "structure_reversed_HL_confirmed", metrics
        if retrace_pct >= 0.35:   # strong bounce ≥ 35% retrace = momentum shift
            return True, f"strong_bounce({retrace_pct:.0%})", metrics

        # Block: continuing weakness without reversal confirmation
        if ll_run > MAX_LOWER_LOWS_FOR_CE:
            return False, (f"continuous_lower_lows({ll_run}) "
                           f"no_HL_no_strong_bounce(retrace={retrace_pct:.0%})"), metrics
        if retrace_pct < MIN_BOUNCE_RETRACE_PCT and ll_run >= 2:
            return False, (f"dead_cat_bounce(retrace={retrace_pct:.0%} "
                           f"of {fall_size:.0f}pt fall)"), metrics
        return True, "structure_ok", metrics

    # ── For SHORT (PE) trades (mirror) ──
    else:
        hh_run = 0
        for k in range(len(swing_highs)-1, 0, -1):
            if swing_highs[k][1] > swing_highs[k-1][1]:
                hh_run += 1
            else:
                break

        has_lower_high = False
        if len(swing_highs) >= 2:
            has_lower_high = swing_highs[-1][1] < swing_highs[-2][1]

        recent_low   = recent['low'].min()
        recent_high  = recent['high'].max()
        current      = recent['close'].iloc[-1]
        rise_size    = recent_high - recent_low
        pullback     = recent_high - current
        retrace_pct  = pullback / rise_size if rise_size > 0 else 0

        metrics = {
            'hh_run': int(hh_run),
            'has_lower_high': bool(has_lower_high),
            'retrace_pct': round(retrace_pct, 3),
            'recent_high': round(recent_high, 2),
            'pullback': round(pullback, 2),
            'n_swings_high': len(swing_highs),
        }

        if has_lower_high:
            return True, "structure_reversed_LH_confirmed", metrics
        if retrace_pct >= 0.35:
            return True, f"strong_pullback({retrace_pct:.0%})", metrics

        if hh_run > MAX_HIGHER_HIGHS_FOR_PE:
            return False, (f"continuous_higher_highs({hh_run}) "
                           f"no_LH_no_strong_pullback(retrace={retrace_pct:.0%})"), metrics
        if retrace_pct < MIN_BOUNCE_RETRACE_PCT and hh_run >= 2:
            return False, (f"weak_pullback(retrace={retrace_pct:.0%} "
                           f"of {rise_size:.0f}pt rise)"), metrics
        return True, "structure_ok", metrics


# ──────────────────────────────────────────────
# TRIGGER TYPE B PARAMETERS (TIGHTENED v2)
# ──────────────────────────────────────────────
SWING_TEST_TOLERANCE_PCT  = 0.15    # reverted from 0.20
SWING_RECLAIM_LOOKBACK     = 15     # reverted from 20
SWING_MIN_HL_DIFF_POINTS   = 8      # reverted from 6
SWING_MIN_ADX_FOR_TYPE_B   = 15     # NEW: gate ADX floor
SWING_REQUIRE_HTF_LEVEL    = False  # was True → now allow TDL/TDH but with stricter rules
SWING_TDL_ONLY_MIN_HL_DIFF = 15     # reverted from 12
SWING_COOLDOWN_BARS        = 6      # NEW: prevent same-direction signals within N bars

# ──────────────────────────────────────────────
# TRIGGER TYPE B: Swing-Low Reclaim / Swing-High Rejection
# ──────────────────────────────────────────────

def detect_swing_reversal(df_slice, current_idx, htf_levels,
                            day_levels, direction_hint='auto'):
    """
    TIGHTENED v2 — Detects swing-low reclaim / swing-high rejection
    near a SIGNIFICANT HTF level (PDH/PWH/PMH/PDL/PWL/PML/ATH).

    Key v2 changes:
    • Min swing diff raised 3 → 8 points (eliminates noise)
    • Test tolerance tightened 0.20% → 0.10%
    • Requires confluence with a real HTF level (TDH/TDL alone no longer enough
      unless SWING_REQUIRE_HTF_LEVEL=False)
    """
    if current_idx < SWING_RECLAIM_LOOKBACK + 3:
        return None

    window = df_slice.iloc[current_idx - SWING_RECLAIM_LOOKBACK : current_idx + 1]
    if len(window) < 5:
        return None

    current = df_slice.iloc[current_idx]
    current_close = current['close']

    swing_highs, swing_lows = find_swing_points(window, window=SWING_WINDOW)

    # ── BULLISH: SWING-LOW RECLAIM ──
    if direction_hint in ('auto', 'long') and len(swing_lows) >= 2:
        prev_low = swing_lows[-2][1]
        last_low = swing_lows[-1][1]

        if last_low > prev_low + SWING_MIN_HL_DIFF_POINTS:
            test_price = (prev_low + last_low) / 2

            # Find nearest HTF SUPPORT level (must be PDL/PWL/PML)
            nearest_support = None
            nearest_label   = None
            best_dist_pct   = float('inf')
            for label, lvl_price, weight in htf_levels:
                if label in ('PDL', 'PWL', 'PML'):
                    dist_pct = abs(lvl_price - test_price) / test_price * 100
                    if dist_pct <= SWING_TEST_TOLERANCE_PCT and dist_pct < best_dist_pct:
                        nearest_support = lvl_price
                        nearest_label   = label
                        best_dist_pct   = dist_pct

            # TDL fallback (only if not requiring HTF level)
            tested_tdl = False
            tdl_value = day_levels['intraday'].iloc[:current_idx+1]['tdl'].iloc[-1]
            if abs(tdl_value - test_price) / test_price * 100 <= SWING_TEST_TOLERANCE_PCT:
                tested_tdl = True

            # ── Level validation ──
            using_tdl_only = False
            if nearest_support is None and tested_tdl:
                using_tdl_only = True
            elif nearest_support is None and not tested_tdl:
                return None

            # If TDL-only, require larger swing diff
            min_diff = SWING_TDL_ONLY_MIN_HL_DIFF if using_tdl_only else SWING_MIN_HL_DIFF_POINTS
            if (last_low - prev_low) < min_diff:
                return None

            if current_close > last_low + min_diff:
                if current_close >= current.get('open', current_close):
                    return {
                        'trigger_type':   'swing_low_reclaim',
                        'direction':      'long',
                        'tested_level':   nearest_support if nearest_support else tdl_value,
                        'tested_label':   nearest_label if nearest_label else 'TDL',
                        'prev_swing_low': prev_low,
                        'last_swing_low': last_low,
                        'hl_diff':        last_low - prev_low,
                        'tdl_only':       using_tdl_only,
                    }

    # ── BEARISH: SWING-HIGH REJECTION ──
    if direction_hint in ('auto', 'short') and len(swing_highs) >= 2:
        prev_high = swing_highs[-2][1]
        last_high = swing_highs[-1][1]

        if last_high < prev_high - SWING_MIN_HL_DIFF_POINTS:
            test_price = (prev_high + last_high) / 2

            nearest_resistance = None
            nearest_label      = None
            best_dist_pct      = float('inf')
            for label, lvl_price, weight in htf_levels:
                if label in ('PDH', 'PWH', 'PMH', 'ATH'):
                    dist_pct = abs(lvl_price - test_price) / test_price * 100
                    if dist_pct <= SWING_TEST_TOLERANCE_PCT and dist_pct < best_dist_pct:
                        nearest_resistance = lvl_price
                        nearest_label      = label
                        best_dist_pct      = dist_pct

            tested_tdh = False
            tdh_value = day_levels['intraday'].iloc[:current_idx+1]['tdh'].iloc[-1]
            if abs(tdh_value - test_price) / test_price * 100 <= SWING_TEST_TOLERANCE_PCT:
                tested_tdh = True

            using_tdh_only = False
            if nearest_resistance is None and tested_tdh:
                using_tdh_only = True
            elif nearest_resistance is None and not tested_tdh:
                return None

            min_diff = SWING_TDL_ONLY_MIN_HL_DIFF if using_tdh_only else SWING_MIN_HL_DIFF_POINTS
            if (prev_high - last_high) < min_diff:
                return None

            if current_close < last_high - min_diff:
                if current_close <= current.get('open', current_close):
                    return {
                        'trigger_type':    'swing_high_rejection',
                        'direction':       'short',
                        'tested_level':    nearest_resistance if nearest_resistance else tdh_value,
                        'tested_label':    nearest_label if nearest_label else 'TDH',
                        'prev_swing_high': prev_high,
                        'last_swing_high': last_high,
                        'lh_diff':         prev_high - last_high,
                        'tdh_only':        using_tdh_only,
                    }

    return None


# ──────────────────────────────────────────────
# CORE: detect signals for ONE trading day
# ──────────────────────────────────────────────
def detect_signals_for_day(target_date, daily, weekly, monthly, intraday,
                            verbose=False):
    """
    Returns list of signal dicts for the given day.
    """
    target_date = pd.Timestamp(target_date).normalize()
    day_levels = levels_mod.get_levels_for_day(
        target_date, daily, weekly, monthly, intraday)

    df = day_levels['intraday'].copy()
    htf_levels = _build_htf_level_list(day_levels)

    # Rolling avg volume for filter
    df['avg_vol'] = df['volume'].rolling(VOLUME_AVG_LOOKBACK, min_periods=1).mean()

    signals = []
    obs_cutoff = pd.Timestamp(f"{target_date.date()} {OBSERVATION_END}")
    session_end = pd.Timestamp(f"{target_date.date()} {SESSION_END}")

    # Track which breakouts have been resolved (avoid duplicates)
    pending_breakouts = []  # list of dicts

    for i, (ts, row) in enumerate(df.iterrows()):
        if ts < obs_cutoff or ts > session_end:
            continue

        # ── STEP 1: Detect new breakout (price piercing TDH or TDL) ──
        # TDH BREAKOUT (failed up move → potential SHORT)
        if row['high'] > df['tdh'].iloc[max(0, i-1)] * (1 + MIN_BREAKOUT_PCT/100):
            broken_level = df['tdh'].iloc[max(0, i-1)]
            pending_breakouts.append({
                'direction':       'short',  # if reverses, we go SHORT
                'breakout_time':   ts,
                'breakout_idx':    i,
                'broken_level':    broken_level,
                'breakout_price':  row['high'],
            })

        # TDL BREAKDOWN (failed down move → potential LONG)
        if row['low'] < df['tdl'].iloc[max(0, i-1)] * (1 - MIN_BREAKOUT_PCT/100):
            broken_level = df['tdl'].iloc[max(0, i-1)]
            pending_breakouts.append({
                'direction':       'long',
                'breakout_time':   ts,
                'breakout_idx':    i,
                'broken_level':    broken_level,
                'breakout_price':  row['low'],
            })

        # ── STEP 2: Check pending breakouts for reversal ──
        resolved = []
        for pb in pending_breakouts:
            age = i - pb['breakout_idx']
            if age == 0:
                continue
            if age > REVERSAL_WINDOW:
                resolved.append(pb)  # expired
                continue

            reversed_back = False
            if pb['direction'] == 'short':
                # Reversal = close back BELOW broken_level (held as resistance)
                if row['close'] < pb['broken_level']:
                    reversed_back = True
            else:  # long
                if row['close'] > pb['broken_level']:
                    reversed_back = True

            if reversed_back:
                # ── PRE-SIGNAL GATES ──
                df_slice = df.iloc[:i+1]

                # GATE 1: Counter-trend block
                blocked_a, reason_a, trend_meta = is_counter_trend_blocked(
                    pb['direction'], df_slice, adx_threshold=25)
                # GATE 2: Freefall block
                blocked_b, reason_b = is_in_freefall(
                    df_slice, pb['direction'])
                # GATE 3: Reversal quality block
                quality_ok, quality_label, quality_metrics = assess_reversal_quality(
                    df_slice, pb['breakout_idx'], i,
                    pb['broken_level'], pb['direction'])
                # GATE 4: Day-wide structure block
                struct_ok, struct_label, struct_metrics = assess_day_structure(
                    df_slice, pb['direction'])

                if blocked_a or blocked_b or not quality_ok or not struct_ok:
                    if blocked_a:
                        block_reason = reason_a
                    elif blocked_b:
                        block_reason = reason_b
                    elif not quality_ok:
                        block_reason = f"reversal_quality: {quality_label}"
                    else:
                        block_reason = f"day_structure: {struct_label}"
                    if verbose:
                        print(f"  ⛔ {ts.time()} | {pb['direction'].upper():5s} "
                              f"BLOCKED — {block_reason}")
                    resolved.append(pb)
                    continue

                # ── STEP 3-5: Score & filter ──
                score, near = _confluence_score(pb['broken_level'], htf_levels)
                vol_ok = row['volume'] >= row['avg_vol'] * VOLUME_MULTIPLIER
                conviction = _conviction_tier(score, age)

                signal = {
                    'date':           target_date.date(),
                    'signal_time':    ts,
                    'direction':      pb['direction'],
                    'option_type':    'PE' if pb['direction'] == 'short' else 'CE',
                    'broken_level':   round(pb['broken_level'], 2),
                    'breakout_time':  pb['breakout_time'],
                    'breakout_price': round(pb['breakout_price'], 2),
                    'reversal_price': round(row['close'], 2),
                    'speed_candles':  age,
                    'confluence_score': score,
                    'nearby_levels':  near,
                    'volume_ok':      bool(vol_ok),
                    'conviction':     conviction,
                    'strike_hint':    _strike_recommendation(conviction),
                    'nifty_spot':     round(row['close'], 2),
                    'trend_dir':      trend_meta['trend_dir'],
                    'trend_strength': trend_meta['trend_strength'],
                    'adx':            trend_meta['adx'],
                    # Quality metadata
                    'rev_quality':    quality_label,
                    'rev_metrics':    quality_metrics,
                    # NEW structure metadata
                    'day_structure':  struct_label,
                    'struct_metrics': struct_metrics,
                    # NEW trigger type
                    'trigger_type':   'failed_breakout',
                }
                signals.append(signal)
                resolved.append(pb)

                if verbose:
                    print(f"  🎯 {ts.time()} | {pb['direction'].upper():5s} | "
                          f"Lvl {pb['broken_level']:.1f} → Rev {row['close']:.1f} | "
                          f"Speed {age} | Score {score} | {conviction} | "
                          f"trend={trend_meta['trend_dir']}({trend_meta['adx']})")

        # ── TRIGGER TYPE B: Swing reversal at HTF level (TIGHTENED v2) ──
        if ts >= obs_cutoff and ts <= session_end:
            swing_sig = detect_swing_reversal(
                df, i, htf_levels, day_levels, direction_hint='auto')
            if swing_sig:
                df_slice = df.iloc[:i+1]

                # Apply ALL gates including new ADX floor
                blocked_a, reason_a, trend_meta = is_counter_trend_blocked(
                    swing_sig['direction'], df_slice, adx_threshold=25)
                blocked_b, reason_b = is_in_freefall(
                    df_slice, swing_sig['direction'])
                struct_ok, struct_label, struct_metrics = assess_day_structure(
                    df_slice, swing_sig['direction'])

                # NEW: Type B mandatory ADX floor (block chop/flat conditions)
                adx_too_low = trend_meta['adx'] < SWING_MIN_ADX_FOR_TYPE_B

                # NEW: Cooldown — prevent clustering of same-direction signals
                in_cooldown = False
                cooldown_threshold = ts - pd.Timedelta(minutes=5 * SWING_COOLDOWN_BARS)
                for s in signals[-10:]:  # check last 10 signals
                    if (s['direction'] == swing_sig['direction']
                            and s['signal_time'] >= cooldown_threshold):
                        in_cooldown = True
                        break

                # NEW: For TDL/TDH-only signals, require higher confluence
                tdl_only_block = False
                if swing_sig.get('tdl_only') or swing_sig.get('tdh_only'):
                    # Pre-check: will confluence score be sufficient?
                    pre_score, _ = _confluence_score(swing_sig['tested_level'], htf_levels)
                    if pre_score < 5:  # lowered from 7 to 5 for TDL-only
                        tdl_only_block = True

                if (blocked_a or blocked_b or not struct_ok
                        or adx_too_low or in_cooldown or tdl_only_block):
                    if verbose:
                        reason = (reason_a if blocked_a else
                                  reason_b if blocked_b else
                                  f"struct:{struct_label}" if not struct_ok else
                                  f"adx_too_low({trend_meta['adx']})" if adx_too_low else
                                  "cooldown" if in_cooldown else
                                  "tdl_only_low_confluence")
                        print(f"  ⛔[B] {ts.time()} | {swing_sig['direction'].upper():5s} "
                              f"BLOCKED — {reason}")
                    pass  # do nothing
                else:
                    score, near = _confluence_score(
                        swing_sig['tested_level'], htf_levels)
                    vol_ok = row['volume'] >= row['avg_vol'] * VOLUME_MULTIPLIER
                    conviction = _conviction_tier(score, 5)

                    already_signaled = any(
                        s['signal_time'] == ts and s['direction'] == swing_sig['direction']
                        for s in signals)

                    if not already_signaled:
                        signal = {
                            'date':           target_date.date(),
                            'signal_time':    ts,
                            'direction':      swing_sig['direction'],
                            'option_type':    'PE' if swing_sig['direction'] == 'short' else 'CE',
                            'broken_level':   round(swing_sig['tested_level'], 2),
                            'breakout_time':  ts,
                            'breakout_price': round(swing_sig['tested_level'], 2),
                            'reversal_price': round(row['close'], 2),
                            'speed_candles':  5,
                            'confluence_score': score,
                            'nearby_levels':  near,
                            'volume_ok':      bool(vol_ok),
                            'conviction':     conviction,
                            'strike_hint':    _strike_recommendation(conviction),
                            'nifty_spot':     round(row['close'], 2),
                            'trend_dir':      trend_meta['trend_dir'],
                            'trend_strength': trend_meta['trend_strength'],
                            'adx':            trend_meta['adx'],
                            'rev_quality':    'type_B_swing',
                            'rev_metrics':    {'tested': swing_sig['tested_label'],
                                                'hl_diff': swing_sig.get('hl_diff') or swing_sig.get('lh_diff')},
                            'day_structure':  struct_label,
                            'struct_metrics': struct_metrics,
                            'trigger_type':   swing_sig['trigger_type'],
                        }
                        signals.append(signal)
                        if verbose:
                            print(f"  🎯[B] {ts.time()} | {swing_sig['direction'].upper():5s} | "
                                  f"{swing_sig['trigger_type']} at "
                                  f"{swing_sig['tested_label']}={swing_sig['tested_level']:.1f} | "
                                  f"Score {score} | {conviction}")

        # Remove resolved/expired
        for r in resolved:
            pending_breakouts.remove(r)

    return signals, day_levels


# ──────────────────────────────────────────────
# RUN ACROSS ALL AVAILABLE DAYS
# ──────────────────────────────────────────────
def run_signal_detection_all_days():
    daily, weekly, monthly, intraday = levels_mod.load_all_data()
    unique_days = sorted(intraday.index.normalize().unique())

    print(f"📅 Scanning {len(unique_days)} trading days for signals...\n")

    all_signals = []
    for d in unique_days:
        try:
            sigs, _ = detect_signals_for_day(
                d, daily, weekly, monthly, intraday, verbose=False)
            if sigs:
                all_signals.extend(sigs)
        except Exception as e:
            print(f"  ⚠️ {d.date()}: {e}")

    return all_signals


# ──────────────────────────────────────────────
# MAIN
# ──────────────────────────────────────────────
if __name__ == "__main__":
    print("=" * 60)
    print("MODULE 3: FALSE BREAKOUT SIGNAL ENGINE")
    print("=" * 60)

    all_sigs = run_signal_detection_all_days()
    print(f"\n✅ Total signals detected: {len(all_sigs)}")

    if not all_sigs:
        print("⚠️ No signals — consider relaxing filters")
        sys.exit(0)

    # Build DataFrame
    df_sigs = pd.DataFrame(all_sigs)

    # Conviction breakdown
    print("\n📊 CONVICTION BREAKDOWN:")
    print(df_sigs['conviction'].value_counts().to_string())

    print("\n📊 DIRECTION BREAKDOWN:")
    print(df_sigs['direction'].value_counts().to_string())

    print("\n📊 OPTION TYPE BREAKDOWN:")
    print(df_sigs['option_type'].value_counts().to_string())

    # Avg confluence by conviction
    print("\n📊 AVG CONFLUENCE SCORE BY CONVICTION:")
    print(df_sigs.groupby('conviction')['confluence_score'].mean().round(2).to_string())

    # Trigger type breakdown (Type A vs Type B)
    if 'trigger_type' in df_sigs.columns:
        print("\n📊 TRIGGER TYPE BREAKDOWN:")
        print(df_sigs['trigger_type'].value_counts().to_string())

    # Save
    os.makedirs(OUT_DIR, exist_ok=True)

    # Flatten nearby_levels for CSV
    df_save = df_sigs.copy()
    df_save['nearby_levels'] = df_save['nearby_levels'].apply(
        lambda x: ", ".join(f"{m[0]}@{m[1]:.1f}" for m in x) if x else ""
    )
    csv_path = f"{OUT_DIR}/signals_log.csv"
    df_save.to_csv(csv_path, index=False)
    print(f"\n💾 Signal log saved: {csv_path}")

    # Sample preview — RECENT signals (newest first)
    print(f"\n📋 LAST 15 SIGNALS (most recent first — for sanity check):")
    cols = ['date', 'signal_time', 'direction', 'broken_level',
            'speed_candles', 'confluence_score', 'conviction',
            'trend_dir', 'adx']
    recent = df_sigs.sort_values('signal_time', ascending=False).head(15)
    print(recent[cols].to_string(index=False))

    # Specifically inspect 26-May-2026 — our problem case
    print(f"\n🔍 SIGNALS ON 2026-05-26 (the bearish-day test):")
    may26 = df_sigs[df_sigs['date'].astype(str) == '2026-05-26']
    if len(may26) == 0:
        print("   ✅ No signals fired — trend filter successfully blocked counter-trend trades!")
    else:
        print(may26[cols].to_string(index=False))

    # Verify May 18 — the sluggish reversal day
    print(f"\n🔍 SIGNALS ON 2026-05-18 (sluggish reversal test):")
    cols = ['date', 'signal_time', 'direction', 'broken_level',
            'speed_candles', 'confluence_score', 'conviction',
            'trend_dir', 'adx', 'rev_quality']
    may18 = df_sigs[df_sigs['date'].astype(str) == '2026-05-18']
    if len(may18) == 0:
        print("   ✅ No signals — sluggish reversal filter blocked the bad 13:35 trade!")
    else:
        print(may18[cols].to_string(index=False))

    # Verify May 12 — dead cat bounce CE trade
    print(f"\n🔍 SIGNALS ON 2026-05-12 (dead-cat-bounce test):")
    cols = ['date', 'signal_time', 'direction', 'broken_level',
            'speed_candles', 'confluence_score', 'conviction',
            'trend_dir', 'adx', 'rev_quality', 'day_structure']
    may12 = df_sigs[df_sigs['date'].astype(str) == '2026-05-12']
    if len(may12) == 0:
        print("   ✅ No signals — structure filter blocked dead cat bounces!")
    else:
        print(may12[cols].to_string(index=False))

    # Verify May 14 — should ideally capture the 11:10 reversal
    print(f"\n🔍 SIGNALS ON 2026-05-14 (genuine reversal capture):")
    may14 = df_sigs[df_sigs['date'].astype(str) == '2026-05-14']
    if len(may14) == 0:
        print("   ⚠️ No signals captured — may need to RELAX filters for genuine reversals")
    else:
        print(may14[cols].to_string(index=False))

    print("\n✅ Module 3 complete.\n")
