#!/bin/bash
# Extended Rebated HFT Training Experiment
# Runs with nohup for uninterrupted 48-72 hour training

echo "🚀 LAUNCHING EXTENDED REBATED HFT TRAINING"
echo "=========================================="
echo "Expected runtime: 48-72 hours"
echo "Target timesteps: 50,000,000"
echo "Process will run in background with nohup"
echo "=========================================="

# Set up environment
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${REPO_ROOT}"

# Check GPU availability
echo "🔍 GPU Status:"
nvidia-smi --query-gpu=name,memory.total,memory.free --format=csv,noheader,nounits 2>/dev/null || echo "No GPU detected - will use CPU"

# Check disk space
echo "💾 Disk Space:"
df -h . | tail -1

# Check memory
echo "🧠 Memory Status:"
free -h

echo ""
echo "🚀 Starting extended training..."
LOG_DIR="${REPO_ROOT}/runs/logs"
mkdir -p "$LOG_DIR"
echo "Logs will be written to: ${LOG_DIR}/extended_training.log"
echo "Process ID will be saved to: ${LOG_DIR}/training.pid"

# Kill any existing training processes
pkill -f "scripts/training/run_rebated_training.py" 2>/dev/null || true

# Start training with nohup
nohup python scripts/training/run_rebated_training.py \
  > "${LOG_DIR}/extended_training.log" 2>&1 &

# Save process ID
echo $! > "${LOG_DIR}/training.pid"

echo "✅ Extended training started successfully!"
echo "Process ID: $(cat training.pid)"
echo ""
echo "📊 MONITORING COMMANDS:"
echo "  View logs:      tail -f extended_training.log"
echo "  Check progress: grep 'Validation' extended_training.log"
echo "  Monitor GPU:    watch -n 5 nvidia-smi"
echo "  Stop training:  kill \$(cat training.pid)"
echo ""
echo "🖥️ TENSORBOARD COMMANDS:"
echo "  Start TensorBoard: ./start_tensorboard.sh"
echo "  View dashboard: http://localhost:6006"
echo ""
echo "⏱️ Training will run for approximately 48-72 hours"
echo "   Check back periodically to monitor progress"

# Wait a moment to check if process started successfully
sleep 5

if ps -p $(cat training.pid) > /dev/null; then
    echo "✅ Training process confirmed running (PID: $(cat training.pid))"
else
    echo "❌ Training process may have failed to start"
    echo "Check extended_training.log for errors"
fi
