#!/usr/bin/env python3
"""
Comprehensive Market Microstructure Analysis for Academic Paper
ETH/USDC Binance Futures Order Book Analysis

This script analyzes real market data from Binance ETH/USDC futures to extract
sophisticated market microstructure insights for academic publication.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

import os
import glob
from typing import Dict, List, Tuple, Optional
from datetime import datetime, timedelta
import json

class MarketMicrostructureAnalyzer:
    """
    Comprehensive market microstructure analyzer for ETH/USDC order book data.
    Extracts sophisticated insights suitable for academic publication.
    """
    
    def __init__(self, data_path: str = "/home/gaen/Documents/RL"):
        self.data_path = data_path
        self.results = {}
        self.statistics = {}
        
    def load_processed_data(self) -> pd.DataFrame:
        """Load processed order book data from trimmed CSV files."""
        print("Loading processed order book data...")
        
        # Load the largest dataset for comprehensive analysis
        df = pd.read_csv(f"{self.data_path}/orderbook_trimmed_large.csv")
        
        # Convert datetime columns
        df['datetime'] = pd.to_datetime(df['datetime'], unit='s')
        df['expanded_datetime'] = pd.to_datetime(df['expanded_datetime'])
        
        # Calculate mid-price for each observation
        df['mid_price'] = (df['ask1'] + df['bid1']) / 2
        
        # Calculate spread
        df['spread'] = df['ask1'] - df['bid1']
        df['spread_bps'] = (df['spread'] / df['mid_price']) * 10000
        
        print(f"Loaded {len(df):,} order book snapshots")
        print(f"Time range: {df['datetime'].min()} to {df['datetime'].max()}")
        
        return df
    
    def analyze_bid_ask_spreads(self, df: pd.DataFrame) -> Dict:
        """Comprehensive bid-ask spread analysis."""
        print("\n=== BID-ASK SPREAD ANALYSIS ===")
        
        spread_stats = {
            'mean_spread': df['spread'].mean(),
            'median_spread': df['spread'].median(),
            'std_spread': df['spread'].std(),
            'min_spread': df['spread'].min(),
            'max_spread': df['spread'].max(),
            'q25_spread': df['spread'].quantile(0.25),
            'q75_spread': df['spread'].quantile(0.75),
            'mean_spread_bps': df['spread_bps'].mean(),
            'median_spread_bps': df['spread_bps'].median(),
            'std_spread_bps': df['spread_bps'].std(),
        }
        
        # Calculate spread volatility using rolling standard deviation
        df['spread_volatility'] = df['spread'].rolling(window=100, min_periods=50).std()
        spread_stats['avg_spread_volatility'] = df['spread_volatility'].mean()
        
        # Analyze spread distribution
        spread_stats['spread_distribution'] = {
            'skewness': stats.skew(df['spread'].dropna()),
            'kurtosis': stats.kurtosis(df['spread'].dropna()),
            'jarque_bera_stat': stats.jarque_bera(df['spread'].dropna())[0],
            'jarque_bera_pvalue': stats.jarque_bera(df['spread'].dropna())[1]
        }
        
        # Time-based spread analysis
        df['hour'] = df['datetime'].dt.hour
        hourly_spreads = df.groupby('hour')['spread_bps'].agg(['mean', 'std', 'count'])
        spread_stats['hourly_patterns'] = hourly_spreads.to_dict()
        
        # Correlation with price volatility
        df['price_volatility'] = df['mid_price'].rolling(window=100, min_periods=50).std()
        spread_vol_corr = df['spread'].corr(df['price_volatility'])
        spread_stats['spread_volatility_correlation'] = spread_vol_corr
        
        print(f"Average spread: {spread_stats['mean_spread']:.4f} USDC ({spread_stats['mean_spread_bps']:.2f} bps)")
        print(f"Median spread: {spread_stats['median_spread']:.4f} USDC ({spread_stats['median_spread_bps']:.2f} bps)")
        print(f"Spread volatility: {spread_stats['std_spread']:.4f} USDC")
        print(f"Spread range: {spread_stats['min_spread']:.4f} - {spread_stats['max_spread']:.4f} USDC")
        
        return spread_stats
    
    def analyze_order_book_depth(self, df: pd.DataFrame) -> Dict:
        """Analyze order book depth and liquidity characteristics."""
        print("\n=== ORDER BOOK DEPTH & LIQUIDITY ANALYSIS ===")
        
        depth_stats = {}
        
        # Calculate depth at each level
        for level in range(1, 11):
            # Bid side depth
            bid_qty_col = f'bidqty{level}'
            ask_qty_col = f'askqty{level}'
            
            if bid_qty_col in df.columns and ask_qty_col in df.columns:
                depth_stats[f'avg_bid_depth_level_{level}'] = df[bid_qty_col].mean()
                depth_stats[f'avg_ask_depth_level_{level}'] = df[ask_qty_col].mean()
        
        # Calculate cumulative depth
        bid_cols = [f'bidqty{i}' for i in range(1, 11) if f'bidqty{i}' in df.columns]
        ask_cols = [f'askqty{i}' for i in range(1, 11) if f'askqty{i}' in df.columns]
        
        df['total_bid_depth'] = df[bid_cols].sum(axis=1)
        df['total_ask_depth'] = df[ask_cols].sum(axis=1)
        df['total_depth'] = df['total_bid_depth'] + df['total_ask_depth']
        
        depth_stats['avg_total_bid_depth'] = df['total_bid_depth'].mean()
        depth_stats['avg_total_ask_depth'] = df['total_ask_depth'].mean()
        depth_stats['avg_total_depth'] = df['total_depth'].mean()
        
        # Order book imbalance
        df['order_imbalance'] = (df['total_bid_depth'] - df['total_ask_depth']) / df['total_depth']
        depth_stats['avg_order_imbalance'] = df['order_imbalance'].mean()
        depth_stats['std_order_imbalance'] = df['order_imbalance'].std()
        
        # Volume-weighted average price impact
        # Calculate how much price would move for different order sizes
        order_sizes = [1, 5, 10, 50, 100]  # ETH
        
        for size in order_sizes:
            # Calculate market impact for buying
            buy_impact = self.calculate_market_impact(df, size, 'buy')
            sell_impact = self.calculate_market_impact(df, size, 'sell')
            
            depth_stats[f'avg_buy_impact_{size}eth'] = buy_impact.mean()
            depth_stats[f'avg_sell_impact_{size}eth'] = sell_impact.mean()
        
        # Liquidity measures
        depth_stats['liquidity_ratio'] = df['total_depth'].mean() / df['spread'].mean()
        
        print(f"Average total order book depth: {depth_stats['avg_total_depth']:.2f} ETH")
        print(f"Average bid depth: {depth_stats['avg_total_bid_depth']:.2f} ETH")
        print(f"Average ask depth: {depth_stats['avg_total_ask_depth']:.2f} ETH")
        print(f"Average order imbalance: {depth_stats['avg_order_imbalance']:.4f}")
        
        return depth_stats
    
    def calculate_market_impact(self, df: pd.DataFrame, order_size: float, side: str) -> pd.Series:
        """Calculate market impact for given order size."""
        impacts = []
        
        for idx, row in df.iterrows():
            if side == 'buy':
                # Calculate cost of buying order_size ETH
                remaining_size = order_size
                total_cost = 0
                
                for level in range(1, 11):
                    ask_price_col = f'ask{level}'
                    ask_qty_col = f'askqty{level}'
                    
                    if ask_price_col in row and ask_qty_col in row:
                        available_qty = row[ask_qty_col]
                        if available_qty > 0 and remaining_size > 0:
                            trade_qty = min(remaining_size, available_qty)
                            total_cost += trade_qty * row[ask_price_col]
                            remaining_size -= trade_qty
                
                if remaining_size <= 0:
                    avg_price = total_cost / order_size
                    impact = (avg_price - row['ask1']) / row['ask1'] * 10000  # bps
                    impacts.append(impact)
                else:
                    impacts.append(np.nan)  # Insufficient liquidity
            
            else:  # sell
                # Calculate proceeds from selling order_size ETH
                remaining_size = order_size
                total_proceeds = 0
                
                for level in range(1, 11):
                    bid_price_col = f'bid{level}'
                    bid_qty_col = f'bidqty{level}'
                    
                    if bid_price_col in row and bid_qty_col in row:
                        available_qty = row[bid_qty_col]
                        if available_qty > 0 and remaining_size > 0:
                            trade_qty = min(remaining_size, available_qty)
                            total_proceeds += trade_qty * row[bid_price_col]
                            remaining_size -= trade_qty
                
                if remaining_size <= 0:
                    avg_price = total_proceeds / order_size
                    impact = (row['bid1'] - avg_price) / row['bid1'] * 10000  # bps
                    impacts.append(impact)
                else:
                    impacts.append(np.nan)  # Insufficient liquidity
        
        return pd.Series(impacts, index=df.index)
    
    def analyze_price_volume_dynamics(self, df: pd.DataFrame) -> Dict:
        """Analyze price and volume dynamics."""
        print("\n=== PRICE & VOLUME DYNAMICS ANALYSIS ===")
        
        dynamics_stats = {}
        
        # Price characteristics
        dynamics_stats['tick_size'] = self.calculate_tick_size(df)
        dynamics_stats['avg_price_level'] = df['mid_price'].mean()
        dynamics_stats['price_volatility'] = df['mid_price'].std()
        dynamics_stats['price_range'] = df['mid_price'].max() - df['mid_price'].min()
        
        # Volume characteristics
        dynamics_stats['avg_order_size'] = df['bidqty1'].mean()  # Typical order size at best bid
        dynamics_stats['volume_distribution'] = {
            'q25': df['bidqty1'].quantile(0.25),
            'q50': df['bidqty1'].quantile(0.5),
            'q75': df['bidqty1'].quantile(0.75),
            'q90': df['bidqty1'].quantile(0.9),
            'q95': df['bidqty1'].quantile(0.95),
            'q99': df['bidqty1'].quantile(0.99)
        }
        
        # Price-volume correlation
        dynamics_stats['price_volume_correlation'] = df['mid_price'].corr(df['total_depth'])
        
        # Volatility analysis
        df['returns'] = df['mid_price'].pct_change()
        df['realized_volatility'] = df['returns'].rolling(window=100, min_periods=50).std() * np.sqrt(8640)  # Annualized (assuming 100ms intervals)
        
        dynamics_stats['avg_realized_volatility'] = df['realized_volatility'].mean()
        dynamics_stats['volatility_of_volatility'] = df['realized_volatility'].std()
        
        # Autocorrelation analysis
        returns_autocorr = []
        for lag in range(1, 21):
            autocorr = df['returns'].autocorr(lag)
            returns_autocorr.append(autocorr)
        
        dynamics_stats['returns_autocorrelation'] = returns_autocorr
        dynamics_stats['mean_reversion_coefficient'] = -returns_autocorr[0]  # Negative of lag-1 autocorr
        
        print(f"Estimated tick size: {dynamics_stats['tick_size']:.4f} USDC")
        print(f"Average price level: {dynamics_stats['avg_price_level']:.2f} USDC")
        print(f"Price volatility: {dynamics_stats['price_volatility']:.4f} USDC")
        print(f"Average order size: {dynamics_stats['avg_order_size']:.4f} ETH")
        print(f"Mean reversion coefficient: {dynamics_stats['mean_reversion_coefficient']:.6f}")
        
        return dynamics_stats
    
    def calculate_tick_size(self, df: pd.DataFrame) -> float:
        """Calculate the effective tick size from price data."""
        # Calculate price differences
        bid_diffs = []
        ask_diffs = []
        
        for level in range(1, 10):
            bid_col = f'bid{level}'
            next_bid_col = f'bid{level+1}'
            ask_col = f'ask{level}'
            next_ask_col = f'ask{level+1}'
            
            if bid_col in df.columns and next_bid_col in df.columns:
                diffs = (df[bid_col] - df[next_bid_col]).abs()
                bid_diffs.extend(diffs[diffs > 0].tolist())
            
            if ask_col in df.columns and next_ask_col in df.columns:
                diffs = (df[next_ask_col] - df[ask_col]).abs()
                ask_diffs.extend(diffs[diffs > 0].tolist())
        
        all_diffs = bid_diffs + ask_diffs
        if all_diffs:
            # Find the most common small difference as tick size
            return np.quantile(all_diffs, 0.1)  # 10th percentile as conservative estimate
        return 0.01  # Default fallback
    
    def analyze_market_making_opportunities(self, df: pd.DataFrame) -> Dict:
        """Analyze market making opportunities and profitability."""
        print("\n=== MARKET MAKING OPPORTUNITIES ANALYSIS ===")
        
        mm_stats = {}
        
        # Calculate potential profit per trade
        df['potential_profit'] = df['spread'] / 2  # Half spread capture
        df['potential_profit_bps'] = df['potential_profit'] / df['mid_price'] * 10000
        
        mm_stats['avg_profit_per_trade'] = df['potential_profit'].mean()
        mm_stats['avg_profit_per_trade_bps'] = df['potential_profit_bps'].mean()
        
        # Risk analysis
        df['inventory_risk'] = df['price_volatility'] * df['avg_order_size']  # Simplified risk measure
        mm_stats['avg_inventory_risk'] = df['inventory_risk'].mean()
        
        # Optimal spread analysis
        optimal_spreads = []
        for spread_multiple in [0.5, 1.0, 1.5, 2.0, 2.5, 3.0]:
            # Calculate probability of being filled (simplified)
            target_spread = df['spread'].median() * spread_multiple
            fill_probability = (df['spread'] >= target_spread).mean()
            expected_profit = fill_probability * (target_spread / 2)
            optimal_spreads.append({
                'spread_multiple': spread_multiple,
                'target_spread': target_spread,
                'fill_probability': fill_probability,
                'expected_profit': expected_profit
            })
        
        mm_stats['optimal_spread_analysis'] = optimal_spreads
        
        # Inventory turnover analysis
        # Simulate inventory turnover based on order sizes and frequencies
        avg_order_size = df['bidqty1'].mean()
        order_frequency = len(df) / ((df['datetime'].max() - df['datetime'].min()).total_seconds() / 60)  # per minute
        
        mm_stats['avg_order_size'] = avg_order_size
        mm_stats['order_frequency_per_minute'] = order_frequency
        mm_stats['estimated_daily_turnover'] = avg_order_size * order_frequency * 1440  # ETH per day
        
        # Market regime analysis
        df['volatility_regime'] = pd.cut(df['realized_volatility'], 
                                       bins=3, labels=['Low', 'Medium', 'High'])
        
        regime_stats = df.groupby('volatility_regime').agg({
            'spread_bps': ['mean', 'std'],
            'total_depth': ['mean', 'std'],
            'order_imbalance': ['mean', 'std']
        }).round(4)
        
        mm_stats['regime_analysis'] = regime_stats.to_dict()
        
        print(f"Average profit per trade: {mm_stats['avg_profit_per_trade']:.4f} USDC ({mm_stats['avg_profit_per_trade_bps']:.2f} bps)")
        print(f"Order frequency: {mm_stats['order_frequency_per_minute']:.2f} per minute")
        print(f"Estimated daily turnover: {mm_stats['estimated_daily_turnover']:.2f} ETH")
        
        return mm_stats
    
    def analyze_statistical_characteristics(self, df: pd.DataFrame) -> Dict:
        """Analyze advanced statistical characteristics."""
        print("\n=== STATISTICAL CHARACTERISTICS ANALYSIS ===")
        
        stats_data = {}
        
        # Mean reversion analysis
        df['price_deviation'] = df['mid_price'] - df['mid_price'].rolling(window=1000, min_periods=500).mean()
        
        # Ornstein-Uhlenbeck process parameters
        def estimate_ou_parameters(prices):
            """Estimate Ornstein-Uhlenbeck parameters for mean reversion."""
            returns = np.diff(np.log(prices))
            prices_lag = prices[:-1]
            
            # Simple regression approach
            X = np.column_stack([np.ones(len(returns)), np.log(prices_lag)])
            y = returns
            
            try:
                coeffs = np.linalg.lstsq(X, y, rcond=None)[0]
                alpha = -coeffs[1]  # Speed of mean reversion
                mu = coeffs[0] / alpha if alpha != 0 else 0  # Long-term mean
                sigma = np.std(returns)  # Volatility
                
                return {'alpha': alpha, 'mu': mu, 'sigma': sigma}
            except:
                return {'alpha': 0, 'mu': 0, 'sigma': 0}
        
        ou_params = estimate_ou_parameters(df['mid_price'].dropna().values)
        stats_data['ou_parameters'] = ou_params
        
        # Half-life of mean reversion
        if ou_params['alpha'] > 0:
            stats_data['mean_reversion_half_life'] = np.log(2) / ou_params['alpha']
        else:
            stats_data['mean_reversion_half_life'] = np.inf
        
        # Order flow toxicity (simplified Kyle's lambda)
        # Calculate price impact per unit volume
        df['volume_imbalance'] = df['total_bid_depth'] - df['total_ask_depth']
        
        # Rolling correlation between price changes and volume imbalance
        price_changes = df['mid_price'].diff()
        volume_imbalance = df['volume_imbalance']
        
        toxic_correlation = price_changes.corr(volume_imbalance)
        stats_data['order_flow_toxicity'] = abs(toxic_correlation)
        
        # Adverse selection cost
        # Measure how much prices move after large orders
        large_order_threshold = df['bidqty1'].quantile(0.9)
        large_order_mask = (df['bidqty1'] > large_order_threshold) | (df['askqty1'] > large_order_threshold)
        
        if large_order_mask.sum() > 100:
            future_returns = df['mid_price'].shift(-10) / df['mid_price'] - 1
            adverse_selection = future_returns[large_order_mask].mean()
            stats_data['adverse_selection_cost'] = abs(adverse_selection) * 10000  # bps
        else:
            stats_data['adverse_selection_cost'] = 0
        
        # Hurst exponent for long-term dependency
        def hurst_exponent(ts):
            """Calculate Hurst exponent."""
            ts = np.array(ts)
            N = len(ts)
            if N < 100:
                return 0.5
            
            # Calculate R/S statistic
            lags = range(2, min(N//4, 100))
            rs = []
            
            for lag in lags:
                # Divide series into periods of length lag
                periods = N // lag
                if periods < 2:
                    continue
                    
                rs_values = []
                for i in range(periods):
                    start_idx = i * lag
                    end_idx = start_idx + lag
                    period_data = ts[start_idx:end_idx]
                    
                    if len(period_data) == lag:
                        mean_val = np.mean(period_data)
                        deviations = period_data - mean_val
                        cumulative_deviations = np.cumsum(deviations)
                        R = np.max(cumulative_deviations) - np.min(cumulative_deviations)
                        S = np.std(period_data)
                        
                        if S > 0:
                            rs_values.append(R / S)
                
                if rs_values:
                    rs.append(np.mean(rs_values))
            
            if len(rs) > 5:
                # Linear regression of log(R/S) vs log(lag)
                log_lags = np.log(lags[:len(rs)])
                log_rs = np.log(rs)
                
                try:
                    coeffs = np.polyfit(log_lags, log_rs, 1)
                    hurst = coeffs[0]
                    return max(0, min(1, hurst))  # Bound between 0 and 1
                except:
                    return 0.5
            
            return 0.5
        
        stats_data['hurst_exponent'] = hurst_exponent(df['mid_price'].dropna().values[-10000:])
        
        # Market efficiency metrics
        # Variance ratio test
        returns = df['returns'].dropna()
        if len(returns) > 1000:
            # Calculate variance ratios for different horizons
            variance_ratios = []
            for k in [2, 4, 8, 16]:
                if len(returns) > k * 100:
                    k_period_returns = returns.rolling(window=k, min_periods=k).sum().dropna()
                    if len(k_period_returns) > 10:
                        vr = k_period_returns.var() / (k * returns.var())
                        variance_ratios.append(vr)
            
            stats_data['variance_ratios'] = variance_ratios
            stats_data['market_efficiency_score'] = np.mean([abs(vr - 1) for vr in variance_ratios])
        
        # Latency requirements (estimated from data frequency)
        time_diffs = df['datetime'].diff().dt.total_seconds()
        stats_data['avg_update_frequency'] = 1 / time_diffs.mean()  # Hz
        stats_data['min_update_interval'] = time_diffs.min()
        stats_data['max_update_interval'] = time_diffs.max()
        
        print(f"Mean reversion half-life: {stats_data['mean_reversion_half_life']:.2f} periods")
        print(f"Order flow toxicity: {stats_data['order_flow_toxicity']:.6f}")
        print(f"Adverse selection cost: {stats_data['adverse_selection_cost']:.2f} bps")
        print(f"Hurst exponent: {stats_data['hurst_exponent']:.4f}")
        print(f"Average update frequency: {stats_data['avg_update_frequency']:.2f} Hz")
        
        return stats_data
    
    def generate_comprehensive_report(self, df: pd.DataFrame) -> Dict:
        """Generate comprehensive market microstructure report."""
        print("\n" + "="*60)
        print("COMPREHENSIVE MARKET MICROSTRUCTURE ANALYSIS")
        print("ETH/USDC Binance Futures Order Book Data")
        print("="*60)
        
        # Run all analyses
        spread_analysis = self.analyze_bid_ask_spreads(df)
        depth_analysis = self.analyze_order_book_depth(df)
        dynamics_analysis = self.analyze_price_volume_dynamics(df)
        mm_analysis = self.analyze_market_making_opportunities(df)
        statistical_analysis = self.analyze_statistical_characteristics(df)
        
        # Compile comprehensive report
        report = {
            'data_overview': {
                'total_observations': len(df),
                'time_period': {
                    'start': df['datetime'].min().isoformat(),
                    'end': df['datetime'].max().isoformat(),
                    'duration_hours': (df['datetime'].max() - df['datetime'].min()).total_seconds() / 3600
                },
                'price_range': {
                    'min': df['mid_price'].min(),
                    'max': df['mid_price'].max(),
                    'mean': df['mid_price'].mean(),
                    'std': df['mid_price'].std()
                }
            },
            'spread_analysis': spread_analysis,
            'depth_analysis': depth_analysis,
            'dynamics_analysis': dynamics_analysis,
            'market_making_analysis': mm_analysis,
            'statistical_analysis': statistical_analysis
        }
        
        # Calculate confidence intervals for key metrics
        n_bootstrap = 1000
        key_metrics = ['spread', 'total_depth', 'returns']
        
        for metric in key_metrics:
            if metric in df.columns:
                data = df[metric].dropna()
                if len(data) > 100:
                    bootstrap_means = []
                    for _ in range(n_bootstrap):
                        sample = np.random.choice(data, size=len(data)//10, replace=True)
                        bootstrap_means.append(np.mean(sample))
                    
                    report[f'{metric}_confidence_interval'] = {
                        'mean': np.mean(bootstrap_means),
                        'lower_95': np.percentile(bootstrap_means, 2.5),
                        'upper_95': np.percentile(bootstrap_means, 97.5)
                    }
        
        return report
    
    def save_results(self, report: Dict, filename: str = "market_microstructure_report.json"):
        """Save analysis results to JSON file."""
        output_path = f"{self.data_path}/paper_analysis/academic_paper/{filename}"
        
        # Convert numpy types to Python types for JSON serialization
        def convert_numpy_types(obj):
            if isinstance(obj, np.integer):
                return int(obj)
            elif isinstance(obj, np.floating):
                return float(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, dict):
                return {k: convert_numpy_types(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [convert_numpy_types(item) for item in obj]
            return obj
        
        clean_report = convert_numpy_types(report)
        
        with open(output_path, 'w') as f:
            json.dump(clean_report, f, indent=2)
        
        print(f"\nResults saved to: {output_path}")
        return output_path

def main():
    """Main analysis function."""
    analyzer = MarketMicrostructureAnalyzer()
    
    # Load data
    df = analyzer.load_processed_data()
    
    # Generate comprehensive report
    report = analyzer.generate_comprehensive_report(df)
    
    # Save results
    analyzer.save_results(report)
    
    print("\n" + "="*60)
    print("ANALYSIS COMPLETE")
    print("="*60)
    print("\nKey Insights for Academic Paper:")
    print(f"• Dataset: {len(df):,} high-frequency order book snapshots")
    print(f"• Average spread: {report['spread_analysis']['mean_spread_bps']:.2f} bps")
    print(f"• Average depth: {report['depth_analysis']['avg_total_depth']:.2f} ETH")
    print(f"• Mean reversion half-life: {report['statistical_analysis']['mean_reversion_half_life']:.2f} periods")
    print(f"• Market efficiency score: {report['statistical_analysis'].get('market_efficiency_score', 'N/A')}")
    print(f"• Order flow toxicity: {report['statistical_analysis']['order_flow_toxicity']:.6f}")
    
    return report

if __name__ == "__main__":
    report = main()