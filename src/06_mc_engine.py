"""
Module 6: Bootstrap Monte Carlo Engine
─────────────────────────────────────────────────
Simulates 10,000 "careers" × 200 trades each, sampling
from the ACTUAL distribution of your backtested R-multiples.

Bootstrap (not parametric) → preserves the full shape of
your real trade outcomes (some +2R wins, some +0.3R wins,
some -0.4R losses, some -1.0R losses, etc.)

Outputs (per strategy version):
  • mc_results_<V>.csv     : final R, MDD, streak, etc. for each career
  • mc_paths_<V>.csv       : 100 sample equity curves (for plotting)
  • mc_summary_<V>.json    : aggregated statistics

Final comparison: V1 vs V2 side-by-side analysis + chart.
"""

import pandas as pd
import numpy as np
import os
import sys
import json
import importlib.util

# ──────────────────────────────────────────────
# CONFIGURATION
# ──────────────────────────────────────────────
N_SIMULATIONS    = 10_000     # number of careers
TRADES_PER_CAREER = 200       # ~1 year of active trading
START_CAPITAL    = 100_000    # ₹1L (notional, used for % drawdown)
RISK_PCT         = 2.0        # 2% risk per trade (₹2,000 = 1R)
RANDOM_SEED      = 42         # for reproducibility

# Save 100 sample paths for spaghetti plot
N_SAMPLE_PATHS   = 100

OUT_DIR  = "/home/cecadmin/nifty_false_breakout_mc/outputs"
CHART_DIR = f"{OUT_DIR}/charts"
os.makedirs(CHART_DIR, exist_ok=True)


# ══════════════════════════════════════════════
# BOOTSTRAP SAMPLING
# ══════════════════════════════════════════════
def load_trade_pools(backtested_csv):
    """
    Load backtested signals and split R-multiples into:
      - wins_pool   : array of R-multiples for winning trades
      - losses_pool : array of R-multiples for losing trades (negative)
      - win_rate    : calibrated win rate
    """
    df = pd.read_csv(backtested_csv)

    # Only include trades with valid outcomes
    valid = df[df['outcome'].isin(['target_hit', 'sl_hit', 'time_stop'])].copy()

    wins_pool   = valid[valid['r_multiple'] > 0]['r_multiple'].values
    losses_pool = valid[valid['r_multiple'] < 0]['r_multiple'].values
    flat_pool   = valid[valid['r_multiple'] == 0]['r_multiple'].values

    win_rate = len(wins_pool) / len(valid)

    return {
        'wins_pool':   wins_pool,
        'losses_pool': losses_pool,
        'flat_pool':   flat_pool,
        'win_rate':    win_rate,
        'n_trades':    len(valid),
        'avg_win':     float(wins_pool.mean()) if len(wins_pool) else 0,
        'avg_loss':    float(losses_pool.mean()) if len(losses_pool) else 0,
    }


def simulate_one_career(wins_pool, losses_pool, win_rate,
                         n_trades=TRADES_PER_CAREER, rng=None):
    """
    Bootstrap-sample one career of N trades.
    Returns the equity curve in R-multiples (cumulative).
    """
    if rng is None:
        rng = np.random.default_rng()

    # Pre-sample all trade outcomes (vectorized — fast)
    uniform_draws = rng.random(n_trades)
    is_win = uniform_draws < win_rate

    # For wins: bootstrap sample from wins_pool
    # For losses: bootstrap sample from losses_pool
    n_wins   = int(is_win.sum())
    n_losses = n_trades - n_wins

    win_samples   = rng.choice(wins_pool,   size=n_wins,   replace=True) if n_wins   > 0 else np.array([])
    loss_samples  = rng.choice(losses_pool, size=n_losses, replace=True) if n_losses > 0 else np.array([])

    # Reconstruct in original order
    outcomes = np.empty(n_trades)
    outcomes[is_win]  = win_samples
    outcomes[~is_win] = loss_samples

    # Cumulative equity curve (in R-units)
    equity_curve = np.cumsum(outcomes)
    return outcomes, equity_curve


# ══════════════════════════════════════════════
# CAREER STATISTICS
# ══════════════════════════════════════════════
def analyze_career(outcomes, equity_curve, start_capital=START_CAPITAL,
                    risk_pct=RISK_PCT):
    """
    Compute key statistics for one simulated career.
    """
    # Convert R-multiples to ₹ using risk amount
    rupee_per_R = start_capital * (risk_pct / 100)  # ₹2,000 for 2% of ₹1L
    rupee_curve = start_capital + equity_curve * rupee_per_R

    # Final P&L
    final_R = equity_curve[-1]
    final_rupees = rupee_curve[-1]
    pct_return = (final_rupees / start_capital - 1) * 100

    # Max drawdown (in R and %)
    running_max = np.maximum.accumulate(rupee_curve)
    drawdowns   = rupee_curve - running_max
    max_dd_rupees = drawdowns.min()
    max_dd_pct  = (drawdowns / running_max).min() * 100

    # Longest losing streak
    is_loss = outcomes < 0
    max_streak = 0
    cur_streak = 0
    for x in is_loss:
        if x:
            cur_streak += 1
            max_streak = max(max_streak, cur_streak)
        else:
            cur_streak = 0

    # Sharpe-equivalent (mean / std of trade R)
    sharpe = outcomes.mean() / outcomes.std() if outcomes.std() > 0 else 0
    # Annualized assuming 200 trades = 1 year
    sharpe_annual = sharpe * np.sqrt(TRADES_PER_CAREER)

    # Win rate this career
    realized_wr = (outcomes > 0).sum() / len(outcomes)

    return {
        'final_R':         round(float(final_R), 2),
        'final_rupees':    round(float(final_rupees), 2),
        'pct_return':      round(float(pct_return), 2),
        'max_dd_rupees':   round(float(max_dd_rupees), 2),
        'max_dd_pct':      round(float(max_dd_pct), 2),
        'max_loss_streak': int(max_streak),
        'sharpe_trade':    round(float(sharpe), 4),
        'sharpe_annual':   round(float(sharpe_annual), 4),
        'realized_wr':     round(float(realized_wr), 4),
    }


# ══════════════════════════════════════════════
# MAIN MONTE CARLO LOOP
# ══════════════════════════════════════════════
def run_monte_carlo(pool_data, version_label):
    """
    Run N_SIMULATIONS careers and collect statistics + sample paths.
    """
    rng = np.random.default_rng(seed=RANDOM_SEED)
    print(f"\n🎲 Running {N_SIMULATIONS:,} simulations × {TRADES_PER_CAREER} trades "
          f"for {version_label}...")

    all_stats   = []
    sample_paths = []   # store 100 random paths for plotting

    for sim_i in range(N_SIMULATIONS):
        outcomes, eq_curve = simulate_one_career(
            pool_data['wins_pool'],
            pool_data['losses_pool'],
            pool_data['win_rate'],
            rng=rng,
        )
        stats = analyze_career(outcomes, eq_curve)
        stats['sim_id'] = sim_i
        all_stats.append(stats)

        # Save first N_SAMPLE_PATHS paths
        if sim_i < N_SAMPLE_PATHS:
            sample_paths.append(eq_curve)

        # Progress
        if (sim_i + 1) % 2500 == 0:
            print(f"  ... {sim_i + 1:,}/{N_SIMULATIONS:,} done")

    df_stats = pd.DataFrame(all_stats)
    paths_array = np.array(sample_paths)

    print(f"✅ Completed {N_SIMULATIONS:,} simulations for {version_label}")
    return df_stats, paths_array


# ══════════════════════════════════════════════
# SUMMARY STATISTICS
# ══════════════════════════════════════════════
def summarize_mc(df_stats, version_label, pool_data):
    """Aggregate statistics across all simulations."""

    summary = {
        'version':              version_label,
        'n_simulations':        N_SIMULATIONS,
        'trades_per_career':    TRADES_PER_CAREER,
        'start_capital':        START_CAPITAL,
        'risk_pct_per_trade':   RISK_PCT,

        # Input calibration
        'calibrated_win_rate':  round(pool_data['win_rate'], 4),
        'calibrated_avg_win':   round(pool_data['avg_win'], 3),
        'calibrated_avg_loss':  round(pool_data['avg_loss'], 3),
        'n_real_trades':        pool_data['n_trades'],

        # Final P&L distribution
        'median_final_R':       float(df_stats['final_R'].median().round(2)),
        'mean_final_R':         float(df_stats['final_R'].mean().round(2)),
        'p10_final_R':          float(df_stats['final_R'].quantile(0.10).round(2)),
        'p25_final_R':          float(df_stats['final_R'].quantile(0.25).round(2)),
        'p75_final_R':          float(df_stats['final_R'].quantile(0.75).round(2)),
        'p90_final_R':          float(df_stats['final_R'].quantile(0.90).round(2)),

        # Return %
        'median_pct_return':    float(df_stats['pct_return'].median().round(2)),
        'p10_pct_return':       float(df_stats['pct_return'].quantile(0.10).round(2)),
        'p90_pct_return':       float(df_stats['pct_return'].quantile(0.90).round(2)),

        # Probability metrics
        'probability_profit':       float(((df_stats['final_R'] > 0).mean() * 100).round(2)),
        'probability_2x_capital':   float(((df_stats['pct_return'] >= 100).mean() * 100).round(2)),
        'probability_loss':         float(((df_stats['final_R'] < 0).mean() * 100).round(2)),

        # Drawdown
        'median_max_dd_pct':    float(df_stats['max_dd_pct'].median().round(2)),
        'p90_worst_max_dd_pct': float(df_stats['max_dd_pct'].quantile(0.10).round(2)),  # worst = lowest
        'risk_of_20pct_dd':     float(((df_stats['max_dd_pct'] <= -20).mean() * 100).round(2)),
        'risk_of_30pct_dd':     float(((df_stats['max_dd_pct'] <= -30).mean() * 100).round(2)),
        'risk_of_50pct_dd_ruin': float(((df_stats['max_dd_pct'] <= -50).mean() * 100).round(2)),

        # Losing streaks
        'median_max_loss_streak': float(df_stats['max_loss_streak'].median()),
        'p90_max_loss_streak':    float(df_stats['max_loss_streak'].quantile(0.90)),
        'max_observed_streak':    int(df_stats['max_loss_streak'].max()),

        # Sharpe
        'median_sharpe_annual': float(df_stats['sharpe_annual'].median().round(3)),
        'mean_sharpe_annual':   float(df_stats['sharpe_annual'].mean().round(3)),
    }
    return summary


def print_summary(summary):
    """Pretty-print summary."""
    v = summary['version']
    print("\n" + "═" * 65)
    print(f"📊 MONTE CARLO RESULTS — {v}")
    print("═" * 65)

    print(f"\n  CALIBRATION INPUTS:")
    print(f"    Win Rate         : {summary['calibrated_win_rate']*100:>6.2f}%")
    print(f"    Avg Win          : {summary['calibrated_avg_win']:>+6.3f}R")
    print(f"    Avg Loss         : {summary['calibrated_avg_loss']:>+6.3f}R")
    print(f"    Real trades used : {summary['n_real_trades']}")

    print(f"\n  FINAL P&L DISTRIBUTION (after 200 trades):")
    print(f"    Median           : {summary['median_final_R']:>+8.2f}R")
    print(f"    10th percentile  : {summary['p10_final_R']:>+8.2f}R  (worst case 90% conf)")
    print(f"    25th percentile  : {summary['p25_final_R']:>+8.2f}R")
    print(f"    75th percentile  : {summary['p75_final_R']:>+8.2f}R")
    print(f"    90th percentile  : {summary['p90_final_R']:>+8.2f}R  (best case 90% conf)")

    print(f"\n  RETURN % (₹1L starting capital, 2% risk/trade):")
    print(f"    Median return    : {summary['median_pct_return']:>+8.2f}%")
    print(f"    10th pct return  : {summary['p10_pct_return']:>+8.2f}%")
    print(f"    90th pct return  : {summary['p90_pct_return']:>+8.2f}%")

    print(f"\n  PROBABILITY METRICS:")
    print(f"    ► Profitable     : {summary['probability_profit']:>6.2f}%")
    print(f"    ► Lose money     : {summary['probability_loss']:>6.2f}%")
    print(f"    ► 2x capital     : {summary['probability_2x_capital']:>6.2f}%")

    print(f"\n  DRAWDOWN ANALYSIS:")
    print(f"    Median max DD    : {summary['median_max_dd_pct']:>6.2f}%")
    print(f"    Worst 10% DD     : {summary['p90_worst_max_dd_pct']:>6.2f}%")
    print(f"    Risk of >20% DD  : {summary['risk_of_20pct_dd']:>6.2f}%")
    print(f"    Risk of >30% DD  : {summary['risk_of_30pct_dd']:>6.2f}%")
    print(f"    Risk of >50% RUIN: {summary['risk_of_50pct_dd_ruin']:>6.2f}%")

    print(f"\n  LOSING STREAKS:")
    print(f"    Median longest   : {summary['median_max_loss_streak']:>6.0f}")
    print(f"    90th percentile  : {summary['p90_max_loss_streak']:>6.0f}")
    print(f"    Max observed     : {summary['max_observed_streak']:>6}")

    print(f"\n  RISK-ADJUSTED RETURN:")
    print(f"    Median Sharpe    : {summary['median_sharpe_annual']:>6.3f}")
    print(f"    Mean Sharpe      : {summary['mean_sharpe_annual']:>6.3f}")


# ══════════════════════════════════════════════
# VISUALIZATION — THE LINKEDIN KILLER CHART
# ══════════════════════════════════════════════
def plot_v1_vs_v2(paths_v1, paths_v2, summary_v1, summary_v2, save_path):
    """
    Side-by-side spaghetti plot comparing V1 and V2.
    """
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 2, figsize=(16, 10))

    # ── TOP LEFT: V1 paths ──
    ax = axes[0, 0]
    for path in paths_v1:
        ax.plot(path, color='#1f77b4', alpha=0.08, linewidth=0.7)
    median_path_v1 = np.median(paths_v1, axis=0)
    p10_path_v1    = np.percentile(paths_v1, 10, axis=0)
    p90_path_v1    = np.percentile(paths_v1, 90, axis=0)
    ax.plot(median_path_v1, color='#1f77b4', linewidth=2.5, label='Median')
    ax.plot(p10_path_v1,    color='red',     linewidth=1.2, linestyle='--',
            label='10th percentile')
    ax.plot(p90_path_v1,    color='green',   linewidth=1.2, linestyle='--',
            label='90th percentile')
    ax.axhline(0, color='black', linewidth=0.8)
    ax.set_title(f"V1 (All Signals): {N_SAMPLE_PATHS} sample equity curves\n"
                 f"WR={summary_v1['calibrated_win_rate']*100:.1f}%, "
                 f"Expectancy=+{summary_v1['median_final_R']/TRADES_PER_CAREER:.3f}R/trade",
                 fontsize=11, fontweight='bold')
    ax.set_xlabel("Trade #")
    ax.set_ylabel("Cumulative R")
    ax.legend(loc='upper left', fontsize=9)
    ax.grid(alpha=0.3)

    # ── TOP RIGHT: V2 paths ──
    ax = axes[0, 1]
    for path in paths_v2:
        ax.plot(path, color='#2ca02c', alpha=0.08, linewidth=0.7)
    median_path_v2 = np.median(paths_v2, axis=0)
    p10_path_v2    = np.percentile(paths_v2, 10, axis=0)
    p90_path_v2    = np.percentile(paths_v2, 90, axis=0)
    ax.plot(median_path_v2, color='#2ca02c', linewidth=2.5, label='Median')
    ax.plot(p10_path_v2,    color='red',     linewidth=1.2, linestyle='--',
            label='10th percentile')
    ax.plot(p90_path_v2,    color='green',   linewidth=1.2, linestyle='--',
            label='90th percentile')
    ax.axhline(0, color='black', linewidth=0.8)
    ax.set_title(f"V2 (Type B Only): {N_SAMPLE_PATHS} sample equity curves\n"
                 f"WR={summary_v2['calibrated_win_rate']*100:.1f}%, "
                 f"Expectancy=+{summary_v2['median_final_R']/TRADES_PER_CAREER:.3f}R/trade",
                 fontsize=11, fontweight='bold')
    ax.set_xlabel("Trade #")
    ax.set_ylabel("Cumulative R")
    ax.legend(loc='upper left', fontsize=9)
    ax.grid(alpha=0.3)

    # ── BOTTOM LEFT: Final P&L distribution histogram ──
    ax = axes[1, 0]
    ax.hist([s['final_R'] for s in [
        {'final_R': summary_v1['p10_final_R']}]], bins=1)  # placeholder
    # Use full dataframes instead — we'll pass them in
    ax.clear()
    # We'll plot from stored stats — need to pass these. For now use percentile bars:
    # Re-plot using actual all-simulation data
    ax.set_visible(False)  # we'll replace below

    # Re-do bottom left correctly with the data we have
    axes[1, 0].set_visible(True)
    bins = np.linspace(
        min(summary_v1['p10_final_R'], summary_v2['p10_final_R']) - 5,
        max(summary_v1['p90_final_R'], summary_v2['p90_final_R']) + 5,
        60
    )
    # We don't have full distributions here — show key percentiles as bars
    metrics = ['p10_final_R', 'median_final_R', 'p90_final_R']
    x_pos = np.arange(len(metrics))
    width = 0.35
    axes[1, 0].bar(x_pos - width/2,
                    [summary_v1[m] for m in metrics],
                    width, color='#1f77b4', label='V1', alpha=0.8)
    axes[1, 0].bar(x_pos + width/2,
                    [summary_v2[m] for m in metrics],
                    width, color='#2ca02c', label='V2', alpha=0.8)
    axes[1, 0].set_xticks(x_pos)
    axes[1, 0].set_xticklabels(['10th pct\n(worst)', 'Median', '90th pct\n(best)'])
    axes[1, 0].set_ylabel('Final R (after 200 trades)')
    axes[1, 0].set_title('Final P&L Distribution: V1 vs V2',
                          fontsize=11, fontweight='bold')
    axes[1, 0].axhline(0, color='black', linewidth=0.8)
    axes[1, 0].legend()
    axes[1, 0].grid(alpha=0.3, axis='y')

    # ── BOTTOM RIGHT: Risk metrics comparison ──
    ax = axes[1, 1]
    risk_metrics = ['probability_profit', 'probability_loss',
                     'risk_of_20pct_dd', 'risk_of_30pct_dd', 'risk_of_50pct_dd_ruin']
    risk_labels = ['Profit %', 'Loss %', '>20% DD %', '>30% DD %', '>50% Ruin %']
    x_pos = np.arange(len(risk_metrics))
    axes[1, 1].bar(x_pos - width/2,
                    [summary_v1[m] for m in risk_metrics],
                    width, color='#1f77b4', label='V1', alpha=0.8)
    axes[1, 1].bar(x_pos + width/2,
                    [summary_v2[m] for m in risk_metrics],
                    width, color='#2ca02c', label='V2', alpha=0.8)
    axes[1, 1].set_xticks(x_pos)
    axes[1, 1].set_xticklabels(risk_labels, rotation=15, ha='right')
    axes[1, 1].set_ylabel('Probability (%)')
    axes[1, 1].set_title('Risk Metrics: V1 vs V2', fontsize=11, fontweight='bold')
    axes[1, 1].legend()
    axes[1, 1].grid(alpha=0.3, axis='y')

    plt.suptitle(
        "Monte Carlo Comparison: V1 (All Signals) vs V2 (Type B Only)\n"
        f"{N_SIMULATIONS:,} simulations × {TRADES_PER_CAREER} trades each",
        fontsize=13, fontweight='bold', y=1.00
    )
    plt.tight_layout()
    plt.savefig(save_path, dpi=130, bbox_inches='tight')
    print(f"\n💾 Comparison chart saved: {save_path}")
    plt.show()


# ══════════════════════════════════════════════
# COMPARISON TABLE
# ══════════════════════════════════════════════
def print_comparison_table(s1, s2):
    """Side-by-side comparison."""
    print("\n" + "═" * 80)
    print("🏆 V1 vs V2 — MONTE CARLO COMPARISON")
    print("═" * 80)
    rows = [
        ("Calibrated win rate",     f"{s1['calibrated_win_rate']*100:.2f}%",    f"{s2['calibrated_win_rate']*100:.2f}%"),
        ("Calibrated avg win",      f"{s1['calibrated_avg_win']:+.3f}R",         f"{s2['calibrated_avg_win']:+.3f}R"),
        ("Calibrated avg loss",     f"{s1['calibrated_avg_loss']:+.3f}R",        f"{s2['calibrated_avg_loss']:+.3f}R"),
        ("Real trades sampled",     f"{s1['n_real_trades']}",                    f"{s2['n_real_trades']}"),
        ("",                        "",                                          ""),
        ("Median final R",          f"{s1['median_final_R']:+.2f}R",             f"{s2['median_final_R']:+.2f}R"),
        ("Worst 10% (p10)",         f"{s1['p10_final_R']:+.2f}R",                f"{s2['p10_final_R']:+.2f}R"),
        ("Best 10% (p90)",          f"{s1['p90_final_R']:+.2f}R",                f"{s2['p90_final_R']:+.2f}R"),
        ("",                        "",                                          ""),
        ("Median return %",         f"{s1['median_pct_return']:+.1f}%",          f"{s2['median_pct_return']:+.1f}%"),
        ("",                        "",                                          ""),
        ("Probability of profit",   f"{s1['probability_profit']:.1f}%",          f"{s2['probability_profit']:.1f}%"),
        ("Probability of 2x",       f"{s1['probability_2x_capital']:.1f}%",      f"{s2['probability_2x_capital']:.1f}%"),
        ("Probability of loss",     f"{s1['probability_loss']:.1f}%",            f"{s2['probability_loss']:.1f}%"),
        ("",                        "",                                          ""),
        ("Median max DD",           f"{s1['median_max_dd_pct']:.2f}%",           f"{s2['median_max_dd_pct']:.2f}%"),
        ("Risk of >20% DD",         f"{s1['risk_of_20pct_dd']:.1f}%",            f"{s2['risk_of_20pct_dd']:.1f}%"),
        ("Risk of >30% DD",         f"{s1['risk_of_30pct_dd']:.1f}%",            f"{s2['risk_of_30pct_dd']:.1f}%"),
        ("Risk of >50% ruin",       f"{s1['risk_of_50pct_dd_ruin']:.1f}%",       f"{s2['risk_of_50pct_dd_ruin']:.1f}%"),
        ("",                        "",                                          ""),
        ("Median Sharpe (annual)",  f"{s1['median_sharpe_annual']:.3f}",         f"{s2['median_sharpe_annual']:.3f}"),
        ("Median max loss streak",  f"{int(s1['median_max_loss_streak'])}",      f"{int(s2['median_max_loss_streak'])}"),
    ]
    print(f"\n  {'METRIC':<28} {'V1 (All Signals)':<22} {'V2 (Type B Only)':<22}")
    print("  " + "─" * 76)
    for metric, v1, v2 in rows:
        print(f"  {metric:<28} {v1:<22} {v2:<22}")
    print()


# ══════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════
if __name__ == "__main__":
    print("=" * 65)
    print("MODULE 6: BOOTSTRAP MONTE CARLO ENGINE")
    print("=" * 65)
    print(f"\nConfiguration:")
    print(f"  Simulations per version : {N_SIMULATIONS:,}")
    print(f"  Trades per career       : {TRADES_PER_CAREER}")
    print(f"  Starting capital        : ₹{START_CAPITAL:,}")
    print(f"  Risk per trade          : {RISK_PCT}% (₹{START_CAPITAL*RISK_PCT/100:,.0f}/R)")

    # Load V1 and V2 backtest data
    v1_csv = f"{OUT_DIR}/signals_backtested_V1.csv"
    v2_csv = f"{OUT_DIR}/signals_backtested_V2.csv"

    if not os.path.exists(v1_csv) or not os.path.exists(v2_csv):
        print(f"\n⚠️ Missing backtested CSVs. Run Module 5 for both V1 and V2 first.")
        sys.exit(1)

    pool_v1 = load_trade_pools(v1_csv)
    pool_v2 = load_trade_pools(v2_csv)

    print(f"\n✅ Loaded V1: {pool_v1['n_trades']} trades, WR={pool_v1['win_rate']*100:.2f}%")
    print(f"✅ Loaded V2: {pool_v2['n_trades']} trades, WR={pool_v2['win_rate']*100:.2f}%")

    # ── RUN V1 ──
    df_v1, paths_v1 = run_monte_carlo(pool_v1, "V1")
    summary_v1 = summarize_mc(df_v1, "V1", pool_v1)
    print_summary(summary_v1)

    # ── RUN V2 ──
    df_v2, paths_v2 = run_monte_carlo(pool_v2, "V2")
    summary_v2 = summarize_mc(df_v2, "V2", pool_v2)
    print_summary(summary_v2)

    # ── SAVE OUTPUTS ──
    df_v1.to_csv(f"{OUT_DIR}/mc_results_V1.csv", index=False)
    df_v2.to_csv(f"{OUT_DIR}/mc_results_V2.csv", index=False)
    np.savetxt(f"{OUT_DIR}/mc_paths_V1.csv", paths_v1, delimiter=",")
    np.savetxt(f"{OUT_DIR}/mc_paths_V2.csv", paths_v2, delimiter=",")
    with open(f"{OUT_DIR}/mc_summary_V1.json", 'w') as f:
        json.dump(summary_v1, f, indent=2)
    with open(f"{OUT_DIR}/mc_summary_V2.json", 'w') as f:
        json.dump(summary_v2, f, indent=2)
    print(f"\n💾 Saved: mc_results_V1.csv, mc_results_V2.csv")
    print(f"💾 Saved: mc_paths_V1.csv,  mc_paths_V2.csv")
    print(f"💾 Saved: mc_summary_V1.json, mc_summary_V2.json")

    # ── COMPARISON ──
    print_comparison_table(summary_v1, summary_v2)

    # ── KILLER CHART ──
    chart_path = f"{CHART_DIR}/monte_carlo_V1_vs_V2.png"
    plot_v1_vs_v2(paths_v1, paths_v2, summary_v1, summary_v2, chart_path)

    print("\n" + "═" * 65)
    print("✅ Module 6 complete. Ready for Module 7 (Analytics + Story).")
    print("═" * 65)
