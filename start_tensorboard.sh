#!/bin/bash
# Start TensorBoard for extended training monitoring

echo "🖥️ STARTING TENSORBOARD FOR EXTENDED TRAINING"
echo "============================================="

# Find the latest log directory
LATEST_LOG_DIR=$(find logs -name "extended_rebated_*" -type d | sort | tail -1)

if [ -z "$LATEST_LOG_DIR" ]; then
    echo "❌ No extended training logs found"
    echo "Make sure training has started and created log directories"
    exit 1
fi

echo "📊 Monitoring logs in: $LATEST_LOG_DIR"
echo "🌐 TensorBoard will be available at: http://localhost:6006"
echo "📈 Dashboard will show:"
echo "  - Training progress and reward curves"
echo "  - Validation metrics every 50k steps"
echo "  - Economic performance (PnL, rebates)"
echo "  - System health metrics"
echo ""

# Kill any existing TensorBoard processes
pkill -f "tensorboard" 2>/dev/null || true

# Start TensorBoard
echo "🚀 Starting TensorBoard..."
nohup tensorboard --logdir=$LATEST_LOG_DIR --host=0.0.0.0 --port=6006 > tensorboard.log 2>&1 &

# Save TensorBoard PID
echo $! > tensorboard.pid

echo "✅ TensorBoard started successfully!"
echo "Process ID: $(cat tensorboard.pid)"
echo ""
echo "📊 TENSORBOARD ACCESS:"
echo "  Local:  http://localhost:6006"
echo "  Remote: http://$(hostname -I | awk '{print $1}'):6006"
echo ""
echo "🔧 TENSORBOARD COMMANDS:"
echo "  Stop TensorBoard: kill \$(cat tensorboard.pid)"
echo "  View TB logs:     tail -f tensorboard.log"
echo ""

# Wait to confirm TensorBoard started
sleep 3

if ps -p $(cat tensorboard.pid) > /dev/null; then
    echo "✅ TensorBoard confirmed running (PID: $(cat tensorboard.pid))"
    echo "🌐 Dashboard should be accessible at http://localhost:6006"
else
    echo "❌ TensorBoard may have failed to start"
    echo "Check tensorboard.log for errors"
fi