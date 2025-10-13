#!/usr/bin/env python3
"""
Enhanced Market Data Analyzer for ETH/USDC Order Book Data
Extracts detailed statistical insights for academic paper enhancement.
"""

import pandas as pd
import numpy as np
import os
import glob
import json
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

class EnhancedMarketAnalyzer:
    def __init__(self, data_path="/home/gaen/Documents/billions_db/orderbooks/"):
        self.data_path = data_path
        self.results = {}
        
    def load_sample_data(self, max_rows=50000):
        """Load a sample of ETH/USDC orderbook data for analysis."""
        print("Loading ETH/USDC orderbook sample...")
        
        # Find ETH/USDC files
        ethusdc_pattern = os.path.join(self.data_path, "**/*ethusdc*.csv")
        files = glob.glob(ethusdc_pattern, recursive=True)
        
        if not files:
            print("No ETH/USDC files found")
            return False
            
        # Load first file with sample
        file_path = files[0]
        print(f"Loading sample from: {os.path.basename(file_path)}")
        
        try:
            self.data = pd.read_csv(file_path, nrows=max_rows)
            print(f"Loaded {len(self.data)} rows with {len(self.data.columns)} columns")
            print("Columns:", list(self.data.columns))
            return True
        except Exception as e:
            print(f"Error loading data: {e}")
            return False
    
    def analyze_spread_statistics(self):
        """Analyze bid-ask spread statistics."""
        print("\nAnalyzing spread statistics...")
        
        if not hasattr(self, 'data'):
            return None
            
        # Calculate spreads using bid1/ask1 (best bid/ask)
        valid_data = self.data[(self.data['bid1'] > 0) & (self.data['ask1'] > 0)]
        
        if len(valid_data) == 0:
            print("No valid bid/ask data found")
            return None
            
        spreads = valid_data['ask1'] - valid_data['bid1']
        midprices = (valid_data['bid1'] + valid_data['ask1']) / 2
        spread_bps = (spreads / midprices) * 10000
        
        # Calculate tick size (minimum price increment)
        price_diffs = valid_data['bid1'].diff().dropna()
        tick_size = price_diffs[price_diffs > 0].min()
        
        spread_stats = {
            'mean_spread_usd': float(spreads.mean()),
            'median_spread_usd': float(spreads.median()),
            'std_spread_usd': float(spreads.std()),
            'mean_spread_bps': float(spread_bps.mean()),
            'median_spread_bps': float(spread_bps.median()),
            'std_spread_bps': float(spread_bps.std()),
            'min_spread_bps': float(spread_bps.min()),
            'max_spread_bps': float(spread_bps.max()),
            'mean_midprice': float(midprices.mean()),
            'price_volatility': float(midprices.std() / midprices.mean()),
            'tick_size': float(tick_size) if not pd.isna(tick_size) else None,
            'valid_observations': len(valid_data)
        }
        
        print(f"Average spread: ${spread_stats['mean_spread_usd']:.4f} ({spread_stats['mean_spread_bps']:.2f} bps)")
        print(f"Median spread: ${spread_stats['median_spread_usd']:.4f} ({spread_stats['median_spread_bps']:.2f} bps)")
        print(f"Average midprice: ${spread_stats['mean_midprice']:.2f}")
        print(f"Tick size: ${spread_stats['tick_size']:.2f}" if spread_stats['tick_size'] else "Tick size: N/A")
        
        self.results['spread_statistics'] = spread_stats
        return spread_stats
    
    def analyze_depth_statistics(self):
        """Analyze order book depth statistics."""
        print("\nAnalyzing depth statistics...")
        
        if not hasattr(self, 'data'):
            return None
            
        # Calculate total depth for bid and ask sides
        bid_qty_cols = [f'bidqty{i}' for i in range(1, 11)]
        ask_qty_cols = [f'askqty{i}' for i in range(1, 11)]
        
        # Ensure columns exist
        bid_qty_cols = [col for col in bid_qty_cols if col in self.data.columns]
        ask_qty_cols = [col for col in ask_qty_cols if col in self.data.columns]
        
        if not bid_qty_cols or not ask_qty_cols:
            print("No quantity columns found")
            return None
            
        bid_depths = self.data[bid_qty_cols].sum(axis=1)
        ask_depths = self.data[ask_qty_cols].sum(axis=1)
        
        # Remove zero values
        valid_mask = (bid_depths > 0) & (ask_depths > 0)
        bid_depths = bid_depths[valid_mask]
        ask_depths = ask_depths[valid_mask]
        
        if len(bid_depths) == 0:
            print("No valid depth data")
            return None
            
        # Calculate imbalance
        imbalance = (bid_depths - ask_depths) / (bid_depths + ask_depths)
        
        depth_stats = {
            'mean_bid_depth': float(bid_depths.mean()),
            'mean_ask_depth': float(ask_depths.mean()),
            'median_bid_depth': float(bid_depths.median()),
            'median_ask_depth': float(ask_depths.median()),
            'total_depth_mean': float((bid_depths + ask_depths).mean()),
            'depth_imbalance_mean': float(imbalance.mean()),
            'depth_imbalance_std': float(imbalance.std()),
            'bid_depth_volatility': float(bid_depths.std()),
            'ask_depth_volatility': float(ask_depths.std()),
            'num_levels': len(bid_qty_cols),
            'valid_observations': len(bid_depths)
        }
        
        print(f"Average bid depth: {depth_stats['mean_bid_depth']:.2f} ETH")
        print(f"Average ask depth: {depth_stats['mean_ask_depth']:.2f} ETH")
        print(f"Average total depth: {depth_stats['total_depth_mean']:.2f} ETH")
        print(f"Average imbalance: {depth_stats['depth_imbalance_mean']:.4f}")
        print(f"Order book levels: {depth_stats['num_levels']}")
        
        self.results['depth_statistics'] = depth_stats
        return depth_stats
    
    def analyze_price_levels(self):
        """Analyze price level distributions."""
        print("\nAnalyzing price level distributions...")
        
        if not hasattr(self, 'data'):
            return None
            
        # Analyze price increments between levels
        bid_price_cols = [f'bid{i}' for i in range(1, 11)]
        ask_price_cols = [f'ask{i}' for i in range(1, 11)]
        
        bid_price_cols = [col for col in bid_price_cols if col in self.data.columns]
        ask_price_cols = [col for col in ask_price_cols if col in self.data.columns]
        
        level_stats = {}
        
        if bid_price_cols and ask_price_cols:
            # Calculate price increments
            bid_increments = []
            ask_increments = []
            
            for i in range(len(bid_price_cols) - 1):
                bid_diff = self.data[bid_price_cols[i]] - self.data[bid_price_cols[i + 1]]
                bid_increments.append(bid_diff[bid_diff > 0])
                
            for i in range(len(ask_price_cols) - 1):
                ask_diff = self.data[ask_price_cols[i + 1]] - self.data[ask_price_cols[i]]
                ask_increments.append(ask_diff[ask_diff > 0])
            
            if bid_increments and ask_increments:
                all_bid_increments = pd.concat(bid_increments)
                all_ask_increments = pd.concat(ask_increments)
                
                level_stats = {
                    'mean_bid_increment': float(all_bid_increments.mean()),
                    'mean_ask_increment': float(all_ask_increments.mean()),
                    'median_bid_increment': float(all_bid_increments.median()),
                    'median_ask_increment': float(all_ask_increments.median()),
                    'price_precision': 2,  # Based on ETH/USDC typical precision
                    'levels_analyzed': len(bid_price_cols)
                }
                
                print(f"Mean bid increment: ${level_stats['mean_bid_increment']:.4f}")
                print(f"Mean ask increment: ${level_stats['mean_ask_increment']:.4f}")
        
        self.results['level_statistics'] = level_stats
        return level_stats
    
    def analyze_market_quality_metrics(self):
        """Calculate market quality metrics."""
        print("\nAnalyzing market quality metrics...")
        
        if not hasattr(self, 'data'):
            return None
            
        valid_data = self.data[(self.data['bid1'] > 0) & (self.data['ask1'] > 0)]
        
        if len(valid_data) == 0:
            return None
            
        # Calculate metrics
        spreads = valid_data['ask1'] - valid_data['bid1']
        midprices = (valid_data['bid1'] + valid_data['ask1']) / 2
        
        # Price efficiency (lower volatility = more efficient)
        price_returns = midprices.pct_change().dropna()
        
        # Liquidity proxy (higher depth = more liquid)
        bid_qty_cols = [f'bidqty{i}' for i in range(1, 11) if f'bidqty{i}' in self.data.columns]
        ask_qty_cols = [f'askqty{i}' for i in range(1, 11) if f'askqty{i}' in self.data.columns]
        
        total_bid_liquidity = valid_data[bid_qty_cols].sum(axis=1)
        total_ask_liquidity = valid_data[ask_qty_cols].sum(axis=1)
        total_liquidity = total_bid_liquidity + total_ask_liquidity
        
        quality_metrics = {
            'liquidity_score': float(total_liquidity.mean()),
            'spread_efficiency': float(1 / spreads.mean()),  # Lower spread = higher efficiency
            'price_volatility': float(price_returns.std() * np.sqrt(252 * 24 * 60)),  # Annualized volatility
            'market_depth_usd': float((total_liquidity * midprices).mean()),
            'bid_ask_balance': float((total_bid_liquidity / total_liquidity).mean()),
            'liquidity_volatility': float(total_liquidity.std() / total_liquidity.mean())
        }
        
        print(f"Liquidity score: {quality_metrics['liquidity_score']:.2f} ETH")
        print(f"Market depth: ${quality_metrics['market_depth_usd']:,.0f}")
        print(f"Annualized volatility: {quality_metrics['price_volatility']:.2%}")
        
        self.results['quality_metrics'] = quality_metrics
        return quality_metrics
    
    def generate_trading_insights(self):
        """Generate insights relevant to RL trading strategies."""
        print("\nGenerating trading insights...")
        
        insights = {}
        
        if 'spread_statistics' in self.results:
            spread_stats = self.results['spread_statistics']
            
            # Calculate potential profit per trade
            avg_spread_usd = spread_stats['mean_spread_usd']
            transaction_cost_bps = 10  # Assume 10 bps transaction cost
            net_spread_bps = spread_stats['mean_spread_bps'] - (2 * transaction_cost_bps)
            
            insights['profitability'] = {
                'gross_spread_bps': spread_stats['mean_spread_bps'],
                'transaction_cost_bps': transaction_cost_bps * 2,  # Round trip
                'net_spread_bps': net_spread_bps,
                'profitable': net_spread_bps > 0,
                'profit_per_eth': avg_spread_usd - (spread_stats['mean_midprice'] * transaction_cost_bps * 2 / 10000)
            }
        
        if 'depth_statistics' in self.results:
            depth_stats = self.results['depth_statistics']
            
            insights['liquidity'] = {
                'adequate_depth': depth_stats['mean_bid_depth'] > 10,  # >10 ETH considered adequate
                'balanced_book': abs(depth_stats['depth_imbalance_mean']) < 0.2,  # <20% imbalance
                'stable_liquidity': depth_stats['bid_depth_volatility'] / depth_stats['mean_bid_depth'] < 1
            }
        
        if 'quality_metrics' in self.results:
            quality = self.results['quality_metrics']
            
            insights['market_quality'] = {
                'high_liquidity': quality['liquidity_score'] > 50,
                'low_volatility': quality['price_volatility'] < 0.5,
                'efficient_pricing': quality['spread_efficiency'] > 100
            }
        
        self.results['trading_insights'] = insights
        
        print("Trading Strategy Insights:")
        if 'profitability' in insights:
            prof = insights['profitability']
            print(f"  Net spread: {prof['net_spread_bps']:.2f} bps ({'Profitable' if prof['profitable'] else 'Unprofitable'})")
            print(f"  Profit per ETH: ${prof['profit_per_eth']:.4f}")
        
        if 'liquidity' in insights:
            liq = insights['liquidity']
            print(f"  Adequate depth: {liq['adequate_depth']}")
            print(f"  Balanced book: {liq['balanced_book']}")
        
        return insights
    
    def save_enhanced_results(self):
        """Save comprehensive analysis results."""
        output_dir = "/home/gaen/Documents/RL/paper_analysis/results/"
        os.makedirs(output_dir, exist_ok=True)
        
        # Save detailed results
        results_file = os.path.join(output_dir, "enhanced_market_analysis.json")
        with open(results_file, 'w') as f:
            json.dump({
                'analysis_timestamp': datetime.now().isoformat(),
                'data_source': self.data_path,
                'analysis_type': 'Enhanced ETH/USDC Market Microstructure',
                'results': self.results
            }, f, indent=2, default=str)
        
        # Generate summary for paper integration
        summary_file = os.path.join(output_dir, "market_insights_for_paper.txt")
        with open(summary_file, 'w') as f:
            f.write("MARKET INSIGHTS FOR ACADEMIC PAPER\n")
            f.write("=" * 50 + "\n\n")
            
            if 'spread_statistics' in self.results:
                spread = self.results['spread_statistics']
                f.write(f"Average Bid-Ask Spread: ${spread['mean_spread_usd']:.4f} ({spread['mean_spread_bps']:.2f} bps)\n")
                f.write(f"Spread Range: {spread['min_spread_bps']:.2f} - {spread['max_spread_bps']:.2f} bps\n")
                f.write(f"Average ETH Price: ${spread['mean_midprice']:.2f}\n\n")
            
            if 'depth_statistics' in self.results:
                depth = self.results['depth_statistics']
                f.write(f"Average Order Book Depth: {depth['total_depth_mean']:.2f} ETH\n")
                f.write(f"Bid/Ask Imbalance: {depth['depth_imbalance_mean']:.4f}\n")
                f.write(f"Order Book Levels: {depth['num_levels']}\n\n")
            
            if 'trading_insights' in self.results:
                insights = self.results['trading_insights']
                if 'profitability' in insights:
                    prof = insights['profitability']
                    f.write(f"Market Making Viability: {'Profitable' if prof['profitable'] else 'Unprofitable'}\n")
                    f.write(f"Net Spread After Costs: {prof['net_spread_bps']:.2f} bps\n")
        
        print(f"Enhanced results saved to: {results_file}")
        print(f"Paper insights saved to: {summary_file}")
        return results_file

def main():
    """Main enhanced analysis function."""
    print("Starting Enhanced ETH/USDC Market Analysis...")
    
    analyzer = EnhancedMarketAnalyzer()
    
    # Load sample data
    if not analyzer.load_sample_data():
        print("Failed to load data. Exiting.")
        return
    
    # Run comprehensive analysis
    analyzer.analyze_spread_statistics()
    analyzer.analyze_depth_statistics()
    analyzer.analyze_price_levels()
    analyzer.analyze_market_quality_metrics()
    analyzer.generate_trading_insights()
    
    # Save results
    analyzer.save_enhanced_results()
    
    print("\nEnhanced analysis complete!")
    return analyzer.results

if __name__ == "__main__":
    main()