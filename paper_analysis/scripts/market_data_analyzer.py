#!/usr/bin/env python3
"""
Market Data Analyzer for ETH/USDC Order Book Data
Extracts statistical insights for academic paper enhancement.
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

class MarketDataAnalyzer:
    def __init__(self, data_path="/home/gaen/Documents/billions_db/orderbooks/"):
        self.data_path = data_path
        self.results = {}
        self.orderbook_data = []
        
    def load_orderbook_data(self):
        """Load all available ETH/USDC orderbook data."""
        print("Loading ETH/USDC orderbook data...")
        
        # Find all ETH/USDC CSV files
        ethusdc_pattern = os.path.join(self.data_path, "**/*ethusdc*.csv")
        files = glob.glob(ethusdc_pattern, recursive=True)
        
        print(f"Found {len(files)} ETH/USDC orderbook files")
        
        for file_path in files[:3]:  # Limit to first 3 files for analysis
            try:
                print(f"Processing: {os.path.basename(file_path)}")
                df = pd.read_csv(file_path)
                
                # Add file source for tracking
                df['source_file'] = os.path.basename(file_path)
                self.orderbook_data.append(df)
                
            except Exception as e:
                print(f"Error loading {file_path}: {e}")
        
        if self.orderbook_data:
            self.combined_data = pd.concat(self.orderbook_data, ignore_index=True)
            print(f"Combined dataset shape: {self.combined_data.shape}")
            return True
        return False
    
    def analyze_spread_dynamics(self):
        """Analyze bid-ask spread patterns."""
        print("\nAnalyzing spread dynamics...")
        
        if not hasattr(self, 'combined_data'):
            return None
            
        # Calculate spreads for each level
        spread_analysis = {}
        
        # Find bid/ask columns
        bid_cols = [col for col in self.combined_data.columns if 'bid_price' in col.lower()]
        ask_cols = [col for col in self.combined_data.columns if 'ask_price' in col.lower()]
        
        if not bid_cols or not ask_cols:
            print("No bid/ask price columns found")
            return None
            
        # Analyze best bid/ask (level 1)
        if bid_cols[0] in self.combined_data.columns and ask_cols[0] in self.combined_data.columns:
            best_bid = self.combined_data[bid_cols[0]]
            best_ask = self.combined_data[ask_cols[0]]
            
            # Remove invalid data
            valid_mask = (best_bid > 0) & (best_ask > 0) & (best_ask > best_bid)
            best_bid = best_bid[valid_mask]
            best_ask = best_ask[valid_mask]
            
            if len(best_bid) > 0:
                spreads = best_ask - best_bid
                midprices = (best_bid + best_ask) / 2
                spread_bps = (spreads / midprices) * 10000
                
                spread_analysis = {
                    'mean_spread_usd': spreads.mean(),
                    'median_spread_usd': spreads.median(),
                    'std_spread_usd': spreads.std(),
                    'mean_spread_bps': spread_bps.mean(),
                    'median_spread_bps': spread_bps.median(),
                    'std_spread_bps': spread_bps.std(),
                    'min_spread_bps': spread_bps.min(),
                    'max_spread_bps': spread_bps.max(),
                    'mean_midprice': midprices.mean(),
                    'price_volatility': midprices.std() / midprices.mean(),
                    'valid_observations': len(spreads)
                }
                
                print(f"Average spread: ${spread_analysis['mean_spread_usd']:.4f}")
                print(f"Average spread: {spread_analysis['mean_spread_bps']:.2f} bps")
                print(f"Midprice volatility: {spread_analysis['price_volatility']:.4f}")
        
        self.results['spread_analysis'] = spread_analysis
        return spread_analysis
    
    def analyze_orderbook_depth(self):
        """Analyze order book depth and liquidity."""
        print("\nAnalyzing orderbook depth...")
        
        if not hasattr(self, 'combined_data'):
            return None
            
        depth_analysis = {}
        
        # Find quantity columns
        bid_qty_cols = [col for col in self.combined_data.columns if 'bid_qty' in col.lower() or 'bid_volume' in col.lower()]
        ask_qty_cols = [col for col in self.combined_data.columns if 'ask_qty' in col.lower() or 'ask_volume' in col.lower()]
        
        if bid_qty_cols and ask_qty_cols:
            # Analyze total depth (sum across all levels)
            total_bid_qty = self.combined_data[bid_qty_cols].sum(axis=1)
            total_ask_qty = self.combined_data[ask_qty_cols].sum(axis=1)
            
            # Remove invalid data
            valid_mask = (total_bid_qty > 0) & (total_ask_qty > 0)
            total_bid_qty = total_bid_qty[valid_mask]
            total_ask_qty = total_ask_qty[valid_mask]
            
            if len(total_bid_qty) > 0:
                depth_analysis = {
                    'mean_bid_depth': total_bid_qty.mean(),
                    'mean_ask_depth': total_ask_qty.mean(),
                    'median_bid_depth': total_bid_qty.median(),
                    'median_ask_depth': total_ask_qty.median(),
                    'depth_imbalance': (total_bid_qty - total_ask_qty).mean(),
                    'depth_volatility_bid': total_bid_qty.std(),
                    'depth_volatility_ask': total_ask_qty.std(),
                    'num_levels': len(bid_qty_cols),
                    'valid_observations': len(total_bid_qty)
                }
                
                print(f"Average bid depth: {depth_analysis['mean_bid_depth']:.2f} ETH")
                print(f"Average ask depth: {depth_analysis['mean_ask_depth']:.2f} ETH")
                print(f"Number of levels: {depth_analysis['num_levels']}")
        
        self.results['depth_analysis'] = depth_analysis
        return depth_analysis
    
    def analyze_market_patterns(self):
        """Analyze temporal and volume patterns."""
        print("\nAnalyzing market patterns...")
        
        if not hasattr(self, 'combined_data'):
            return None
            
        pattern_analysis = {}
        
        # Try to parse timestamps if available
        timestamp_cols = [col for col in self.combined_data.columns if 'time' in col.lower() or 'timestamp' in col.lower()]
        
        if timestamp_cols:
            try:
                # Convert timestamp to datetime
                ts_col = timestamp_cols[0]
                self.combined_data['datetime'] = pd.to_datetime(self.combined_data[ts_col], unit='ms', errors='coerce')
                
                if not self.combined_data['datetime'].isna().all():
                    # Extract hour patterns
                    self.combined_data['hour'] = self.combined_data['datetime'].dt.hour
                    
                    # Analyze hourly patterns
                    hourly_stats = self.combined_data.groupby('hour').agg({
                        'datetime': 'count'
                    }).rename(columns={'datetime': 'observations'})
                    
                    pattern_analysis = {
                        'temporal_coverage_hours': len(hourly_stats),
                        'peak_activity_hour': hourly_stats['observations'].idxmax(),
                        'min_activity_hour': hourly_stats['observations'].idxmin(),
                        'total_observations': len(self.combined_data),
                        'time_span_hours': (self.combined_data['datetime'].max() - 
                                          self.combined_data['datetime'].min()).total_seconds() / 3600
                    }
                    
                    print(f"Time span: {pattern_analysis['time_span_hours']:.1f} hours")
                    print(f"Peak activity: {pattern_analysis['peak_activity_hour']}:00")
                    
            except Exception as e:
                print(f"Error parsing timestamps: {e}")
        
        self.results['pattern_analysis'] = pattern_analysis
        return pattern_analysis
    
    def calculate_market_statistics(self):
        """Calculate comprehensive market statistics."""
        print("\nCalculating comprehensive market statistics...")
        
        if not hasattr(self, 'combined_data'):
            return None
            
        # Basic dataset information
        basic_stats = {
            'total_rows': len(self.combined_data),
            'total_columns': len(self.combined_data.columns),
            'data_files_processed': len(self.orderbook_data),
            'memory_usage_mb': self.combined_data.memory_usage(deep=True).sum() / 1024 / 1024
        }
        
        # Column analysis
        column_types = {
            'price_columns': len([col for col in self.combined_data.columns if 'price' in col.lower()]),
            'volume_columns': len([col for col in self.combined_data.columns if any(x in col.lower() for x in ['qty', 'volume', 'size'])]),
            'bid_columns': len([col for col in self.combined_data.columns if 'bid' in col.lower()]),
            'ask_columns': len([col for col in self.combined_data.columns if 'ask' in col.lower()])
        }
        
        self.results['basic_stats'] = basic_stats
        self.results['column_types'] = column_types
        
        print(f"Dataset: {basic_stats['total_rows']:,} rows, {basic_stats['total_columns']} columns")
        print(f"Price columns: {column_types['price_columns']}, Volume columns: {column_types['volume_columns']}")
        
        return basic_stats
    
    def generate_summary_report(self):
        """Generate comprehensive summary report."""
        print("\n" + "="*60)
        print("ETH/USDC MARKET DATA ANALYSIS SUMMARY")
        print("="*60)
        
        summary = {
            'analysis_timestamp': datetime.now().isoformat(),
            'data_source': self.data_path,
            'results': self.results
        }
        
        # Print key insights
        if 'spread_analysis' in self.results and self.results['spread_analysis']:
            spread = self.results['spread_analysis']
            print(f"\nSPREAD ANALYSIS:")
            print(f"  Average Spread: ${spread.get('mean_spread_usd', 'N/A'):.4f} ({spread.get('mean_spread_bps', 'N/A'):.2f} bps)")
            print(f"  Spread Range: {spread.get('min_spread_bps', 'N/A'):.2f} - {spread.get('max_spread_bps', 'N/A'):.2f} bps")
            print(f"  Average Midprice: ${spread.get('mean_midprice', 'N/A'):.2f}")
        
        if 'depth_analysis' in self.results and self.results['depth_analysis']:
            depth = self.results['depth_analysis']
            print(f"\nDEPTH ANALYSIS:")
            print(f"  Average Bid Depth: {depth.get('mean_bid_depth', 'N/A'):.2f} ETH")
            print(f"  Average Ask Depth: {depth.get('mean_ask_depth', 'N/A'):.2f} ETH")
            print(f"  Order Book Levels: {depth.get('num_levels', 'N/A')}")
        
        if 'basic_stats' in self.results:
            stats = self.results['basic_stats']
            print(f"\nDATASET SUMMARY:")
            print(f"  Total Observations: {stats.get('total_rows', 'N/A'):,}")
            print(f"  Files Processed: {stats.get('data_files_processed', 'N/A')}")
            print(f"  Memory Usage: {stats.get('memory_usage_mb', 'N/A'):.1f} MB")
        
        # Save results to JSON
        output_path = "/home/gaen/Documents/RL/paper_analysis/results/market_analysis_results.json"
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        with open(output_path, 'w') as f:
            json.dump(summary, f, indent=2, default=str)
        
        print(f"\nResults saved to: {output_path}")
        return summary

def main():
    """Main analysis function."""
    print("Starting ETH/USDC Market Data Analysis...")
    
    analyzer = MarketDataAnalyzer()
    
    # Load data
    if not analyzer.load_orderbook_data():
        print("No data loaded. Exiting.")
        return
    
    # Run analyses
    analyzer.calculate_market_statistics()
    analyzer.analyze_spread_dynamics()
    analyzer.analyze_orderbook_depth()
    analyzer.analyze_market_patterns()
    
    # Generate summary
    summary = analyzer.generate_summary_report()
    
    print("\nAnalysis complete!")
    return summary

if __name__ == "__main__":
    main()