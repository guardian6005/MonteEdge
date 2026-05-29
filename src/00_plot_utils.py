"""
Module 0: Plotting Utilities
─────────────────────────────────────────────────
Shared visualization helpers used across all modules.

TWO-TIER MARKER SYSTEM:
─────────────────────────────────────────────────
1. TRADE TIMEFRAME (3m/5m candles):
   → Marker at EXACT close price
   → Precision required for entry/exit signals

2. HIGHER TIMEFRAMES (Daily/Weekly/Monthly H/L):
   → Marker blended between close and wick tip
   → Cleaner visuals when multiple levels cluster
   → Wick information still preserved (just not at tip)

Why this split?
• Trade TF: We act on the close → marker must match action
• Context TF: We just need a visible reference zone, 
  and wicks often overlap, so blending declutters charts
"""

# ──────────────────────────────────────────────
# CONFIGURATION
# ──────────────────────────────────────────────
# Blend factor for HIGHER timeframe markers only (Daily/Weekly/Monthly/ATH)
# 0.0 = at close | 0.5 = midway | 1.0 = at wick tip
HTF_WICK_BLEND = 0.35   # 35% from close toward wick ← recommended


# ──────────────────────────────────────────────
# TIER 1: TRADE TIMEFRAME MARKERS (3m / 5m)
# ──────────────────────────────────────────────
def trade_marker_price(close):
    """
    For 3m/5m trade candles → marker sits at EXACT close.
    Used for: entry signals, exit signals, intraday TDH/TDL updates.
    """
    return close


# ──────────────────────────────────────────────
# TIER 2: HIGHER TIMEFRAME MARKERS (D / W / M / ATH)
# ──────────────────────────────────────────────
def htf_marker_high(high, close, blend=HTF_WICK_BLEND):
    """
    For Daily/Weekly/Monthly HIGH → marker blended toward wick tip.

    Example: high=24100, close=24050, blend=0.35
    → marker at 24050 + (24100-24050)*0.35 = 24067.5
    """
    return close + (high - close) * blend


def htf_marker_low(low, close, blend=HTF_WICK_BLEND):
    """
    For Daily/Weekly/Monthly LOW → marker blended toward wick tip.

    Example: low=23900, close=23950, blend=0.35
    → marker at 23950 - (23950-23900)*0.35 = 23932.5
    """
    return close - (close - low) * blend


# ──────────────────────────────────────────────
# UNIFIED HELPER (auto-routes based on level type)
# ──────────────────────────────────────────────
def get_marker_y(level_type, high=None, low=None, close=None, side='high'):
    """
    Auto-route to correct marker formula based on level type.

    level_type: 'TRADE', 'PDH', 'PDL', 'PWH', 'PWL', 'PMH', 'PML', 'ATH'
    side:       'high' or 'low' (ignored for TRADE)

    Returns: Y-coordinate for marker placement
    """
    TRADE_TF_LEVELS = {'TRADE', 'TDH', 'TDL'}      # exact close
    HTF_LEVELS      = {'PDH', 'PDL', 'PWH', 'PWL', 'PMH', 'PML', 'ATH'}

    if level_type in TRADE_TF_LEVELS:
        return trade_marker_price(close)
    elif level_type in HTF_LEVELS:
        if side == 'high':
            return htf_marker_high(high, close)
        else:
            return htf_marker_low(low, close)
    else:
        raise ValueError(f"Unknown level_type: {level_type}")


# ──────────────────────────────────────────────
# LEVEL STYLE MAP (color / linestyle / priority / marker symbol)
# ──────────────────────────────────────────────
LEVEL_STYLE = {
    # label : (color,    linestyle, priority_weight, marker, tier)
    'TDH':  ('#FF4444', '--',  5, '▼', 'TRADE'),  # Today High (running)
    'TDL':  ('#44AA44', '--',  5, '▲', 'TRADE'),  # Today Low (running)
    'PDH':  ('#FF8866', ':',   4, '▽', 'HTF'),    # Prev Day High
    'PDL':  ('#66AA88', ':',   4, '△', 'HTF'),    # Prev Day Low
    'PWH':  ('#CC4488', '-.',  3, '◆', 'HTF'),    # Prev Week High
    'PWL':  ('#44AACC', '-.',  3, '◆', 'HTF'),    # Prev Week Low
    'PMH':  ('#884488', '-',   2, '★', 'HTF'),    # Prev Month High
    'PML':  ('#448844', '-',   2, '★', 'HTF'),    # Prev Month Low
    'ATH':  ('#FFD700', '-',   5, '👑', 'HTF'),   # All-Time High
}


# ──────────────────────────────────────────────
# QUICK TEST
# ──────────────────────────────────────────────
if __name__ == "__main__":
    print("=" * 60)
    print("TWO-TIER MARKER SYSTEM TEST")
    print("=" * 60)

    # ── Trade timeframe (3m/5m candle) ──
    trade_close = 24050.75
    print(f"\n🎯 TRADE TIMEFRAME (5m candle)")
    print(f"   Candle close: {trade_close}")
    print(f"   Marker Y:     {trade_marker_price(trade_close):.2f}  ← exact close")

    # ── Higher timeframe (Daily / Weekly / Monthly) ──
    h, l, c = 24100, 23900, 24050
    print(f"\n📊 HIGHER TIMEFRAME (Daily/Weekly/Monthly candle)")
    print(f"   Candle: high={h}, low={l}, close={c}")
    print(f"   HIGH marker Y: {htf_marker_high(h, c):.2f}  (wick tip was {h})")
    print(f"   LOW  marker Y: {htf_marker_low(l, c):.2f}  (wick tip was {l})")
    print(f"   Blend factor: {HTF_WICK_BLEND*100:.0f}% from close toward wick")

    # ── Unified router test ──
    print(f"\n🔀 UNIFIED ROUTER TEST")
    print(f"   TDH (running): {get_marker_y('TDH', close=trade_close):.2f}")
    print(f"   PDH:           {get_marker_y('PDH', high=h, close=c, side='high'):.2f}")
    print(f"   PWL:           {get_marker_y('PWL', low=l,  close=c, side='low'):.2f}")
    print(f"   ATH:           {get_marker_y('ATH', high=26373.2, close=26200, side='high'):.2f}")
