#!/usr/bin/env python3
"""
Training monitoring script with disk space protection
"""
import os
import time
import subprocess
import signal
import psutil
from pathlib import Path
import json

def get_process_status(pid):
    """Check if process is running"""
    try:
        process = psutil.Process(pid)
        return process.is_running(), process.status()
    except psutil.NoSuchProcess:
        return False, "not_found"

def get_disk_usage():
    """Get disk usage stats"""
    usage = psutil.disk_usage('/')
    return {
        'total': usage.total / (1024**3),  # GB
        'used': usage.used / (1024**3),    # GB  
        'free': usage.free / (1024**3),    # GB
        'percent': (usage.used / usage.total) * 100
    }

def get_log_size(log_path: Path):
    """Get log file size in MB"""
    if log_path.exists():
        return log_path.stat().st_size / (1024**2)
    return 0

def get_latest_training_step(log_path):
    """Extract latest training step from log"""
    if not log_path.exists():
        return None

    try:
        # Get last few lines efficiently
        result = subprocess.run(['tail', '-20', str(log_path)], 
                              capture_output=True, text=True)
        lines = result.stdout.strip().split('\n')
        
        for line in reversed(lines):
            if 'Step ' in line and ' - Inventory:' in line:
                parts = line.split('Step ')[1].split(' - Inventory:')
                step = int(parts[0])
                inventory_part = parts[1].split(', Cash: ')
                inventory = float(inventory_part[0])
                cash = float(inventory_part[1])
                return step, inventory, cash
    except:
        pass
    return None, None, None

def emergency_stop_training(pid, reason):
    """Emergency stop training process"""
    print(f"🚨 EMERGENCY STOP: {reason}")
    try:
        os.kill(pid, signal.SIGTERM)
        print(f"Sent SIGTERM to process {pid}")
        time.sleep(5)
        if get_process_status(pid)[0]:
            os.kill(pid, signal.SIGKILL)
            print(f"Sent SIGKILL to process {pid}")
    except ProcessLookupError:
        print(f"Process {pid} already terminated")

def monitor_training():
    """Main monitoring loop"""
    project_root = Path(__file__).resolve().parents[2]
    log_dir = project_root / 'runs' / 'logs'
    pid_file = log_dir / 'training.pid'
    log_file = log_dir / 'extended_training.log'

    if not pid_file.exists():
        print("❌ No training PID file found")
        return

    with pid_file.open('r') as f:
        pid = int(f.read().strip())
    
    print(f"📊 Monitoring training process {pid}")
    
    # Thresholds
    MAX_LOG_SIZE_MB = 100    # 100MB log limit
    MIN_FREE_DISK_GB = 50    # 50GB minimum free space
    MAX_DISK_PERCENT = 90    # 90% disk usage limit
    
    while True:
        # Check process status
        is_running, status = get_process_status(pid)
        if not is_running:
            print(f"❌ Training process {pid} stopped (status: {status})")
            break
            
        # Check disk usage
        disk = get_disk_usage()
        log_size = get_log_size(log_file)
        step, inventory, cash = get_latest_training_step(log_file)
        
        if step is not None:
            # Calculate MTM (assuming current price ~2389)
            mtm = cash + (inventory * 2389.0) if cash is not None and inventory is not None else 0
            print(f"⚡ Step: {step:,} | Inventory: {inventory:.2f} | Cash: ${cash:,.2f} | MTM: ${mtm:,.2f}")
        else:
            print("⚡ Waiting for training data...")
        print(f"💾 Log: {log_size:.1f}MB | Disk: {disk['percent']:.1f}% used ({disk['free']:.1f}GB free)")
        
        # Emergency checks
        if disk['free'] < MIN_FREE_DISK_GB:
            emergency_stop_training(pid, f"Low disk space: {disk['free']:.1f}GB remaining")
            break
            
        if disk['percent'] > MAX_DISK_PERCENT:
            emergency_stop_training(pid, f"High disk usage: {disk['percent']:.1f}%")
            break
            
        if log_size > MAX_LOG_SIZE_MB:
            print(f"⚠️  WARNING: Log file is {log_size:.1f}MB (>{MAX_LOG_SIZE_MB}MB)")
            # Consider log rotation here
            
        time.sleep(90)  # 90 second intervals
        
if __name__ == "__main__":
    monitor_training()
