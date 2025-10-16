import os
from datetime import datetime

def setup_logging(base_name="SAC_HFT"):
    """
    Setup logging directory for training
    
    Args:
        base_name: Base name for the log directory
        
    Returns:
        str: Path to the log directory
    """
    # Create base run directory if it doesn't exist
    base_log_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'runs')
    os.makedirs(base_log_dir, exist_ok=True)
    
    # Create specific log directory with timestamp
    current_time = datetime.now().strftime('%Y%m%d-%H%M%S')
    log_dir = os.path.join(base_log_dir, f'{base_name}_{current_time}')
    os.makedirs(log_dir, exist_ok=True)
    
    return log_dir
