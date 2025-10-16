#!/usr/bin/env python3
"""
Real-time monitoring script for extended rebated HFT training
Provides comprehensive status updates and performance tracking
"""

import os
import time
import json
import psutil
import subprocess
from datetime import datetime, timedelta
from pathlib import Path
import pandas as pd
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
LOG_DIR = PROJECT_ROOT / "runs" / "logs"

def get_training_status():
    """Check if training process is running."""
    try:
        pid_path = LOG_DIR / 'training.pid'
        with pid_path.open('r') as f:
            pid = int(f.read().strip())
        
        if psutil.pid_exists(pid):
            process = psutil.Process(pid)
            return {
                'running': True,
                'pid': pid,
                'cpu_percent': process.cpu_percent(),
                'memory_mb': process.memory_info().rss / 1024 / 1024,
                'status': process.status()
            }
        else:
            return {'running': False, 'reason': 'Process not found'}
    except FileNotFoundError:
        return {'running': False, 'reason': 'PID file not found'}
    except Exception as e:
        return {'running': False, 'reason': f'Error: {e}'}

def get_gpu_status():
    """Get GPU utilization if available."""
    try:
        result = subprocess.run(['nvidia-smi', '--query-gpu=utilization.gpu,memory.used,memory.total,temperature.gpu', '--format=csv,noheader,nounits'], 
                              capture_output=True, text=True)
        if result.returncode == 0:
            line = result.stdout.strip()
            gpu_util, mem_used, mem_total, temp = line.split(', ')
            return {
                'available': True,
                'utilization': int(gpu_util),
                'memory_used_mb': int(mem_used),
                'memory_total_mb': int(mem_total),
                'temperature': int(temp),
                'memory_percent': int(mem_used) / int(mem_total) * 100
            }
        else:
            return {'available': False}
    except:
        return {'available': False}

def parse_training_progress():
    """Parse training progress from log file."""
    try:
        log_path = LOG_DIR / 'extended_training.log'
        if not log_path.exists():
            return {'found': False, 'reason': 'Log file not found'}
        
        progress_info = {
            'found': True,
            'validations': [],
            'latest_timestep': 0,
            'latest_reward': None,
            'latest_pnl': None,
            'errors': [],
            'warnings': []
        }
        
        with log_path.open('r') as f:
            lines = f.readlines()
        
        # Parse recent lines for progress
        for line in lines[-1000:]:  # Check last 1000 lines
            line = line.strip()
            
            # Look for validation results
            if 'Extended Validation' in line and 'steps:' in line:
                try:
                    parts = line.split()
                    validation_num = int(parts[parts.index('Validation') + 1].replace('#', ''))
                    timestep_idx = parts.index('steps:') - 1
                    timestep = int(parts[timestep_idx].replace(',', ''))
                    progress_info['latest_timestep'] = max(progress_info['latest_timestep'], timestep)
                except:
                    pass
            
            # Look for reward information
            if 'Mean Reward:' in line:
                try:
                    reward = float(line.split('Mean Reward:')[1].strip())
                    progress_info['latest_reward'] = reward
                except:
                    pass
            
            # Look for PnL information
            if 'Mean PnL:' in line:
                try:
                    pnl = float(line.split('Mean PnL:')[1].strip())
                    progress_info['latest_pnl'] = pnl
                except:
                    pass
            
            # Track errors and warnings
            if 'ERROR' in line:
                progress_info['errors'].append(line)
            elif 'WARNING' in line:
                progress_info['warnings'].append(line)
        
        return progress_info
        
    except Exception as e:
        return {'found': False, 'reason': f'Error parsing log: {e}'}

def get_tensorboard_status():
    """Check TensorBoard status."""
    try:
        pid_path = LOG_DIR / 'tensorboard.pid'
        with pid_path.open('r') as f:
            pid = int(f.read().strip())
        
        if psutil.pid_exists(pid):
            return {'running': True, 'pid': pid, 'url': 'http://localhost:6006'}
        else:
            return {'running': False, 'reason': 'Process not found'}
    except FileNotFoundError:
        return {'running': False, 'reason': 'PID file not found'}
    except Exception as e:
        return {'running': False, 'reason': f'Error: {e}'}

def estimate_completion_time(current_timestep, target_timestep, start_time):
    """Estimate completion time based on current progress."""
    if current_timestep <= 0:
        return "Unknown"
    
    elapsed_hours = (datetime.now() - start_time).total_seconds() / 3600
    progress_ratio = current_timestep / target_timestep
    
    if progress_ratio <= 0:
        return "Unknown"
    
    total_estimated_hours = elapsed_hours / progress_ratio
    remaining_hours = total_estimated_hours - elapsed_hours
    
    completion_time = datetime.now() + timedelta(hours=remaining_hours)
    
    return {
        'remaining_hours': remaining_hours,
        'estimated_completion': completion_time.strftime('%Y-%m-%d %H:%M:%S'),
        'progress_percent': progress_ratio * 100
    }

def print_status_report():
    """Print comprehensive status report."""
    print("\n" + "="*80)
    print(f"🖥️  EXTENDED REBATED HFT TRAINING - STATUS REPORT")
    print(f"📅 {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("="*80)
    
    # Training process status
    training_status = get_training_status()
    print(f"\n🚀 TRAINING PROCESS:")
    if training_status['running']:
        print(f"   Status: ✅ RUNNING (PID: {training_status['pid']})")
        print(f"   CPU Usage: {training_status['cpu_percent']:.1f}%")
        print(f"   Memory: {training_status['memory_mb']:.0f} MB")
    else:
        print(f"   Status: ❌ NOT RUNNING ({training_status['reason']})")
    
    # GPU status
    gpu_status = get_gpu_status()
    print(f"\n🎮 GPU STATUS:")
    if gpu_status['available']:
        print(f"   Utilization: {gpu_status['utilization']}%")
        print(f"   Memory: {gpu_status['memory_used_mb']}/{gpu_status['memory_total_mb']} MB ({gpu_status['memory_percent']:.1f}%)")
        print(f"   Temperature: {gpu_status['temperature']}°C")
    else:
        print(f"   Status: ❌ NO GPU DETECTED")
    
    # Training progress
    progress = parse_training_progress()
    print(f"\n📊 TRAINING PROGRESS:")
    if progress['found']:
        target_timesteps = 50000000  # 50M target
        
        if progress['latest_timestep'] > 0:
            print(f"   Current Timestep: {progress['latest_timestep']:,}")
            print(f"   Target Timesteps: {target_timesteps:,}")
            
            # Calculate progress
            progress_pct = (progress['latest_timestep'] / target_timesteps) * 100
            print(f"   Progress: {progress_pct:.2f}%")
            
            # Progress bar
            bar_length = 50
            filled_length = int(bar_length * progress['latest_timestep'] / target_timesteps)
            bar = '█' * filled_length + '░' * (bar_length - filled_length)
            print(f"   [{bar}] {progress_pct:.1f}%")
            
            # Estimate completion time (simplified)
            if progress_pct > 0.1:  # At least 0.1% progress
                try:
                    # Read start time from log
                    start_time = datetime.now() - timedelta(hours=1)  # Rough estimate
                    completion_est = estimate_completion_time(progress['latest_timestep'], target_timesteps, start_time)
                    if completion_est != "Unknown":
                        print(f"   ETA: {completion_est['estimated_completion']}")
                        print(f"   Remaining: {completion_est['remaining_hours']:.1f} hours")
                except:
                    pass
        
        if progress['latest_reward'] is not None:
            print(f"   Latest Reward: {progress['latest_reward']:.4f}")
        if progress['latest_pnl'] is not None:
            print(f"   Latest PnL: ${progress['latest_pnl']:.2f}")
        
        # Show recent errors/warnings
        if progress['errors']:
            print(f"   Recent Errors: {len(progress['errors'])}")
        if progress['warnings']:
            print(f"   Recent Warnings: {len(progress['warnings'])}")
    else:
        print(f"   Status: ❌ NO PROGRESS DATA ({progress['reason']})")
    
    # TensorBoard status
    tb_status = get_tensorboard_status()
    print(f"\n📈 TENSORBOARD:")
    if tb_status['running']:
        print(f"   Status: ✅ RUNNING (PID: {tb_status['pid']})")
        print(f"   URL: {tb_status['url']}")
    else:
        print(f"   Status: ❌ NOT RUNNING ({tb_status['reason']})")
    
    # System resources
    print(f"\n💻 SYSTEM RESOURCES:")
    cpu_percent = psutil.cpu_percent(interval=1)
    memory = psutil.virtual_memory()
    disk = psutil.disk_usage('.')
    
    print(f"   CPU Usage: {cpu_percent:.1f}%")
    print(f"   Memory: {memory.percent:.1f}% ({memory.used/1024/1024/1024:.1f}/{memory.total/1024/1024/1024:.1f} GB)")
    print(f"   Disk Free: {disk.free/1024/1024/1024:.1f} GB")
    
    print("\n" + "="*80)

def continuous_monitoring(interval_seconds=300):
    """Run continuous monitoring with specified interval."""
    print("🔍 STARTING CONTINUOUS MONITORING")
    print(f"Refresh interval: {interval_seconds} seconds")
    print("Press Ctrl+C to stop monitoring")
    
    try:
        while True:
            print_status_report()
            
            # Check if training is still running
            status = get_training_status()
            if not status['running']:
                print("\n⚠️  TRAINING PROCESS NOT RUNNING - MONITORING STOPPED")
                break
            
            print(f"\n⏰ Next update in {interval_seconds} seconds...")
            time.sleep(interval_seconds)
            
    except KeyboardInterrupt:
        print("\n👋 Monitoring stopped by user")

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == "continuous":
        interval = int(sys.argv[2]) if len(sys.argv) > 2 else 300
        continuous_monitoring(interval)
    else:
        print_status_report()
        print("\n💡 USAGE:")
        print("  Single report:      python monitor_extended_training.py")
        print("  Continuous (5min):  python monitor_extended_training.py continuous")
        print("  Continuous (custom): python monitor_extended_training.py continuous 600")
