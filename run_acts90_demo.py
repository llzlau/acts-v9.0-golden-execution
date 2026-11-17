"""
# Module: run_acts90_demo.py
# Business: ACTS 9.0 Golden Execution Engine Public Demo Launcher
# Dependencies: acts90_core.pyd (Binary C-Extension), numpy, pandas
# Output: Standard Backtest Performance Metrics, NAV Curve, and Trade Execution Audit
"""

import os
import sys
import argparse
from pathlib import Path
import pandas as pd
import numpy as np

# Ensure binary C-extension can be loaded from current script directory
current_dir = Path(__file__).resolve().parent
sys.path.insert(0, str(current_dir))

try:
    import acts90_core
except ImportError as e:
    print(f"[FATAL] Failed to import compiled binary core 'acts90_core': {e}")
    print("Please ensure 'acts90_core.cp311-win_amd64.pyd' is in the current directory and running on Python 3.11 64-bit Windows.")
    sys.exit(1)


def parse_args():
    parser = argparse.ArgumentParser(description="ACTS 9.0 Golden Execution Engine - Backtest Verification Runner")
    parser.add_argument("--capital", "-c", type=float, default=5000000.0,
                        help="Initial capital in RMB (e.g. 1000000 for Small-Cap / 5000000 for Institutional Standard, default: 5000000)")
    parser.add_argument("--data-dir", "-d", type=str, default="../antigravity-trading/data/back_adjusted",
                        help="Path to processed daily continuous futures directory")
    parser.add_argument("--basis-file", "-b", type=str, default="../antigravity-trading/data/spot_basis/ALL_SPOT_BASIS_DAILY.csv",
                        help="Path to basis_daily.csv file")
    parser.add_argument("--contracts-dir", type=str, default="../antigravity-trading/data/contracts",
                        help="Path to raw contract lifecycle directory (optional for EC/SC)")
    parser.add_argument("--start", "-s", type=str, default="2020-01-01",
                        help="Backtest start date (YYYY-MM-DD)")
    parser.add_argument("--end", "-e", type=str, default="2026-11-13",
                        help="Backtest end date (YYYY-MM-DD)")
    parser.add_argument("--slip", type=float, default=2.0,
                        help="Slippage in ticks (default: 2.0)")
    return parser.parse_args()


def calculate_metrics(df_nav: pd.DataFrame, initial_nav: float):
    if df_nav.empty:
        return {}
    
    end_nav = df_nav['equity'].iloc[-1]
    tot_return = (end_nav - initial_nav) / initial_nav
    
    days = (df_nav.index[-1] - df_nav.index[0]).days
    years = max(days / 365.25, 0.1)
    cagr = (end_nav / initial_nav) ** (1.0 / years) - 1.0
    
    rolling_peak = df_nav['equity'].cummax()
    drawdowns = (df_nav['equity'] - rolling_peak) / rolling_peak
    max_dd = drawdowns.min()
    
    daily_rets = df_nav['equity'].pct_change().dropna()
    mean_ret = daily_rets.mean() * 252
    std_ret = daily_rets.std() * np.sqrt(252)
    sharpe = (mean_ret - 0.02) / std_ret if std_ret > 0 else 0.0
    calmar = (cagr / abs(max_dd)) if abs(max_dd) > 0 else 0.0
    
    return {
        'initial_nav': initial_nav,
        'final_nav': end_nav,
        'tot_return': tot_return,
        'cagr': cagr,
        'max_dd': max_dd,
        'sharpe': sharpe,
        'calmar': calmar,
        'mean_margin_ratio': df_nav['margin_ratio'].mean(),
        'max_margin_ratio': df_nav['margin_ratio'].max(),
        'years': years
    }


def main():
    args = parse_args()
    
    data_path = Path(args.data_dir).resolve()
    basis_path = Path(args.basis_file).resolve()
    contracts_path = Path(args.contracts_dir).resolve() if args.contracts_dir else None
    
    print("=" * 80)
    print(f" ACTS 9.0 Golden Execution Engine | Binary Core Verification")
    print("=" * 80)
    print(f" Core Module   : {acts90_core.__file__}")
    print(f" Initial Nav   : ¥{args.capital:,.2f}")
    print(f" Evaluation    : {args.start} -> {args.end}")
    print(f" Slippage      : {args.slip} ticks per trade")
    print(f" Data Directory: {data_path}")
    print(f" Basis Lookup  : {basis_path}")
    print("-" * 80)
    
    if not data_path.exists():
        print(f"[ERROR] Data directory does not exist: {data_path}")
        print("Please specify a valid path with --data-dir <path>")
        sys.exit(1)
        
    if not basis_path.exists():
        print(f"[ERROR] Basis file does not exist: {basis_path}")
        print("Please specify a valid path with --basis-file <path>")
        sys.exit(1)
        
    print("[1/3] Loading daily commodity universe & calculating KER / Donchian / ATR...")
    dfs = acts90_core.load_data(str(data_path))
    print(f"      Loaded {len(dfs)} active commodity assets.")
    
    print("[2/3] Parsing basis structure & multi-contract lifecycle matrices...")
    basis_records = acts90_core.load_basis_lookup(str(basis_path))
    lifecycle_matrix = acts90_core.build_lifecycle_matrix(str(basis_path), str(contracts_path) if contracts_path and contracts_path.exists() else None)
    print(f"      Constructed dynamic rollover mapping across multi-year timeline.")
    
    print("[3/3] Simulating ACTS 9.0 Golden Execution...")
    df_nav, unified_trades, positions = acts90_core.simulate_acts90(
        dfs,
        basis_records,
        lifecycle_matrix,
        initial_nav=args.capital,
        slip_ticks=args.slip,
        start_date=args.start,
        end_date=args.end
    )
    
    metrics = calculate_metrics(df_nav, args.capital)
    
    print("\n" + "=" * 80)
    print(" ACTS 9.0 OFFICIAL BACKTEST AUDIT REPORT")
    print("=" * 80)
    print(f" Initial NAV     : ¥{metrics['initial_nav']:,.2f}")
    print(f" Final NAV       : ¥{metrics['final_nav']:,.2f}")
    print(f" Net Profit      : ¥{(metrics['final_nav'] - metrics['initial_nav']):,.2f}")
    print(f" Total Return    : {metrics['tot_return'] * 100:+.2f}%")
    print(f" CAGR (Annual)   : {metrics['cagr'] * 100:+.2f}%")
    print(f" Max Drawdown    : {metrics['max_dd'] * 100:.2f}%")
    print(f" Sharpe Ratio    : {metrics['sharpe']:.2f}")
    print(f" Calmar Ratio    : {metrics['calmar']:.2f}")
    print(f" Mean Margin Occ : {metrics['mean_margin_ratio'] * 100:.2f}%")
    print(f" Max Margin Occ  : {metrics['max_margin_ratio'] * 100:.2f}%")
    print(f" Closed Trades   : {len(unified_trades)}")
    
    if unified_trades:
        win_trades = [t for t in unified_trades if (t.get('pnl_a', 0.0) + t.get('pnl_b', 0.0)) > 0]
        win_rate = len(win_trades) / len(unified_trades) * 100
        print(f" Win Rate        : {win_rate:.2f}% ({len(win_trades)}/{len(unified_trades)})")
    
    print("=" * 80)
    
    # Save output summary
    out_csv = current_dir / f"nav_curve_{int(args.capital)}.csv"
    df_nav.to_csv(out_csv)
    print(f"[INFO] Daily NAV curve exported to: {out_csv.name}")


if __name__ == "__main__":
    main()
