import os
import numpy as np
from flask import Flask, render_template_string, request, redirect, url_for, flash
from tym import HFTEnv, config  # your custom environment

###############################################################################
#   1. Global Environment Setup
###############################################################################
# Create a single global instance of your custom environment.
env = HFTEnv(config)

# Reset it to get an initial observation and info.
obs, info = env.reset()

# Keep a history of steps, midprice, best bid/ask, and mark-to-market (MTM).
history = {
    "step": [],
    "midprice": [],
    "best_bid": [],
    "best_ask": [],
    "mtm": [],
    "reward": [],
    "total_reward": [],
    "trades": {
        "step": [],
        "price": [],
        "type": []  # "buy" or "sell"
    }
}
current_step = 0
total_reward = 0

###############################################################################
#   2. Flask App Configuration
###############################################################################
app = Flask(__name__)

# For production, store your secret key in an environment variable or a vault.
app.secret_key = os.environ.get("FLASK_SECRET_KEY", "CHANGE_THIS_IN_PRODUCTION")


###############################################################################
#   3. HTML Template with Bootstrap & Chart.js
###############################################################################
# Using render_template_string for demonstration.
# In a real project, place this template in a separate .html file.
template = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <title>HFT Environment Simulator</title>
    <!-- Bootstrap 5 (CSS) -->
    <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.3.0/dist/css/bootstrap.min.css" rel="stylesheet">
    <!-- Chart.js -->
    <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
    <style>
        body {
            background: #f8f9fa;
        }
        .container {
            margin-top: 30px;
            margin-bottom: 50px;
        }
        .header {
            margin-bottom: 20px;
        }
        .card {
            margin-bottom: 20px;
        }
        .chart-container {
            position: relative;
            height: 400px;
        }
    </style>
</head>
<body>
    <!-- Navbar -->
    <nav class="navbar navbar-expand-lg navbar-dark bg-primary">
      <div class="container-fluid">
        <a class="navbar-brand" href="#">HFT Simulator</a>
      </div>
    </nav>
    
    <div class="container">
        <div class="header text-center">
            <h1>High-Frequency Trading Environment</h1>
            <p class="lead">Step through the simulation, place orders, and watch the market evolve in real time.</p>
        </div>

        <!-- Flash Messages -->
        {% with messages = get_flashed_messages() %}
          {% if messages %}
            <div class="alert alert-info">
              {% for message in messages %}
                <div>{{ message }}</div>
              {% endfor %}
            </div>
          {% endif %}
        {% endwith %}

        <!-- Simulation Controls -->
        <div class="card">
            <div class="card-header">
                Simulation Controls
            </div>
            <div class="card-body">
                <form method="post" action="{{ url_for('step') }}">
                    <div class="row mb-3">
                        <div class="col">
                            <label for="signed_volume" class="form-label">Signed Volume</label>
                            <input type="number" step="0.01" class="form-control" id="signed_volume" name="signed_volume" required>
                            <small class="form-text text-muted">Positive for buy, negative for sell.</small>
                        </div>
                        <div class="col">
                            <label for="price_offset" class="form-label">Price Offset</label>
                            <input type="number" step="0.01" class="form-control" id="price_offset" name="price_offset" required>
                        </div>
                        <div class="col">
                            <label for="cancel_fraction" class="form-label">Cancel Fraction</label>
                            <input type="number" step="0.01" min="0" max="1" class="form-control" id="cancel_fraction" name="cancel_fraction" required>
                        </div>
                    </div>
                    <div class="d-flex justify-content-between">
                        <div>
                            <button type="submit" class="btn btn-success">Step Environment</button>
                            <button type="button" class="btn btn-primary" id="randomActionBtn">Random Action</button>
                        </div>
                        <a href="{{ url_for('reset') }}" class="btn btn-danger">Reset Environment</a>
                    </div>
                </form>
                
                <script>
                    document.getElementById('randomActionBtn').addEventListener('click', function() {
                        fetch('{{ url_for("random_action") }}')
                            .then(response => response.json())
                            .then(data => {
                                document.getElementById('signed_volume').value = data.signed_volume;
                                document.getElementById('price_offset').value = data.price_offset;
                                document.getElementById('cancel_fraction').value = data.cancel_fraction;
                                // Automatically submit the form after setting values
                                document.querySelector('form').submit();
                            });
                    });
                </script>
            </div>
        </div>

        <!-- Current Step -->
        <div class="card">
            <div class="card-header">
                Current Step
            </div>
            <div class="card-body">
                <p class="fs-5">{{ current_step }}</p>
            </div>
        </div>

        <!-- Combined Environment Info and Observation -->
        <div class="card">
            <div class="card-header">
                Environment State
            </div>
            <div class="card-body">
                <div class="row">
                    <!-- Normalized Features -->
                    <div class="col-md-6">
                        <h6 class="mb-3">Normalized Features</h6>
                        <div style="max-height: 300px; overflow-y: auto;">
                            <table class="table table-sm table-striped">
                                <thead>
                                    <tr>
                                        <th>Feature</th>
                                        <th>Value</th>
                                    </tr>
                                </thead>
                                <tbody>
                                    {% for i, val in enumerate(obs) %}
                                    <tr>
                                        <td>Feature {{ i + 1 }}</td>
                                        <td>{{ '{:.4f}'.format(val) }}</td>
                                    </tr>
                                    {% endfor %}
                                </tbody>
                            </table>
                        </div>
                    </div>
                    
                    <!-- Environment Info -->
                    <div class="col-md-6">
                        <h6 class="mb-3">Environment Info</h6>
                        <div style="max-height: 300px; overflow-y: auto;">
                            <table class="table table-sm table-bordered">
                                <tbody>
                                    {% for key, value in info.items() %}
                                    <tr>
                                        <th>{{ key.replace('_', ' ') | title }}</th>
                                        <td>{{ value }}</td>
                                    </tr>
                                    {% endfor %}
                                </tbody>
                            </table>
                        </div>
                    </div>
                </div>
            </div>
        </div>

        <!-- History Chart -->
        <div class="row mt-4">
            <!-- Price Chart -->
            <div class="col-12 mb-4">
                <div class="card">
                    <div class="card-header">
                        Market Prices & Trades
                    </div>
                    <div class="card-body">
                        <div class="chart-container">
                            <canvas id="priceChart"></canvas>
                        </div>
                    </div>
                </div>
            </div>
            
            <!-- MTM Chart -->
            <div class="col-12 mb-4">
                <div class="card">
                    <div class="card-header">
                        Mark-to-Market (MTM) & Rewards
                    </div>
                    <div class="card-body">
                        <div class="chart-container">
                            <canvas id="mtmChart"></canvas>
                        </div>
                    </div>
                </div>
            </div>
        </div>
    </div>

    <!-- Chart.js script for the history -->
    <script>
        {% if history.step|length > 0 %}
        // Price Chart
        const priceCtx = document.getElementById('priceChart').getContext('2d');
        const priceChart = new Chart(priceCtx, {
            type: 'line',
            data: {
                labels: {{ history.step|tojson }},
                datasets: [
                    {
                        label: 'Midprice',
                        data: {{ history.midprice|tojson }},
                        borderColor: 'rgba(75, 192, 192, 1)',
                        fill: false,
                        tension: 0.1
                    },
                    {
                        label: 'Best Bid',
                        data: {{ history.best_bid|tojson }},
                        borderColor: 'rgba(255, 99, 132, 1)',
                        fill: false,
                        tension: 0.1
                    },
                    {
                        label: 'Best Ask',
                        data: {{ history.best_ask|tojson }},
                        borderColor: 'rgba(54, 162, 235, 1)',
                        fill: false,
                        tension: 0.1
                    }
                ]
            },
            options: {
                responsive: true,
                interaction: {
                    mode: 'index',
                    intersect: false,
                },
                scales: {
                    x: {
                        title: {
                            display: true,
                            text: 'Step'
                        }
                    }
                }
            }
        });

        // Add trade points
        const buyPoints = {
            label: 'Buys',
            data: {{ history.trades.step|tojson }}.map((step, i) => ({
                x: step,
                y: {{ history.trades.price|tojson }}[i]
            })).filter((_, i) => {{ history.trades.type|tojson }}[i] === 'buy'),
            backgroundColor: 'green',
            pointStyle: 'triangle',
            pointRadius: 8,
            showLine: false
        };

        const sellPoints = {
            label: 'Sells',
            data: {{ history.trades.step|tojson }}.map((step, i) => ({
                x: step,
                y: {{ history.trades.price|tojson }}[i]
            })).filter((_, i) => {{ history.trades.type|tojson }}[i] === 'sell'),
            backgroundColor: 'red',
            pointStyle: 'triangle',
            pointRadius: 8,
            showLine: false
        };

        priceChart.data.datasets.push(buyPoints);
        priceChart.data.datasets.push(sellPoints);
        priceChart.update();

        // MTM and Rewards Chart
        const mtmCtx = document.getElementById('mtmChart').getContext('2d');
        const mtmChart = new Chart(mtmCtx, {
            type: 'line',
            data: {
                labels: {{ history.step|tojson }},
                datasets: [
                    {
                        label: 'MTM',
                        data: {{ history.mtm|tojson }},
                        borderColor: 'rgba(153, 102, 255, 1)',
                        fill: false,
                        tension: 0.1,
                        yAxisID: 'y'
                    },
                    {
                        label: 'Total Reward',
                        data: {{ history.total_reward|tojson }},
                        borderColor: 'rgba(255, 159, 64, 1)',
                        fill: false,
                        tension: 0.1,
                        yAxisID: 'y1'
                    }
                ]
            },
            options: {
                responsive: true,
                interaction: {
                    mode: 'index',
                    intersect: false,
                },
                scales: {
                    x: {
                        title: {
                            display: true,
                            text: 'Step'
                        }
                    },
                    y: {
                        type: 'linear',
                        display: true,
                        position: 'left',
                        title: {
                            display: true,
                            text: 'MTM'
                        }
                    },
                    y1: {
                        type: 'linear',
                        display: true,
                        position: 'right',
                        title: {
                            display: true,
                            text: 'Total Reward'
                        },
                        grid: {
                            drawOnChartArea: false
                        }
                    }
                }
            }
        });
        {% endif %}
    </script>

    <!-- Bootstrap 5 (JS) -->
    <script src="https://cdn.jsdelivr.net/npm/bootstrap@5.3.0/dist/js/bootstrap.bundle.min.js"></script>
</body>
</html>
"""

###############################################################################
#   4. Flask Routes
###############################################################################

@app.route("/", methods=["GET"])
def index():
    """
    Main page: Renders the simulation dashboard, including:
    - Current observation
    - Environment info
    - History chart
    - Step controls
    """
    # Pass the current observation, info, history, and current step to the template
    current_obs = env._get_observation()
    current_info = env._get_info()
    return render_template_string(
        template,
        obs=current_obs,
        info=current_info,
        history=history,
        current_step=current_step,
        enumerate=enumerate
    )
@app.route("/step", methods=["POST"])
def step():
    """
    Process a single step in the environment using the form inputs for
    signed volume, price offset, and cancel fraction.
    """
    global current_step, obs, info, total_reward

    try:
        signed_volume = float(request.form["signed_volume"])
        price_offset = float(request.form["price_offset"])
        cancel_fraction = float(request.form["cancel_fraction"])
    except ValueError:
        flash("Invalid input. Please enter numeric values.")
        return redirect(url_for("index"))

    # Validate the input ranges a bit more strictly if desired:
    if not (-1.0 <= signed_volume <= 1.0):
        flash("Signed volume must be between -1.0 and 1.0.")
        return redirect(url_for("index"))
    if not (-1.0 <= price_offset <= 1.0):
        flash("Price offset must be between -1.0 and 1.0.")
        return redirect(url_for("index"))
    if not (0.0 <= cancel_fraction <= 1.0):
        flash("Cancel fraction must be between 0.0 and 1.0.")
        return redirect(url_for("index"))

    # Create the action for the environment
    action = np.array([signed_volume, price_offset, cancel_fraction], dtype=np.float32)

    # Step the environment
    new_obs, reward, terminated, truncated, new_info = env.step(action)
    current_step += 1
    total_reward += reward

    # Update the global observation/info references
    obs = new_obs
    info = new_info

    # Record history for the chart
    history["step"].append(current_step)
    history["midprice"].append(float(env.midprice))
    history["best_bid"].append(float(env.best_bid))
    history["best_ask"].append(float(env.best_ask))
    history["mtm"].append(float(new_info["mtm"]))
    history["reward"].append(float(reward))
    history["total_reward"].append(float(total_reward))
    
    # Track trades (assuming signed_volume indicates trade direction)
    if abs(signed_volume) > 0:
        history["trades"]["step"].append(current_step)
        history["trades"]["price"].append(float(env.midprice))
        history["trades"]["type"].append("buy" if signed_volume > 0 else "sell")

    flash(f"Step {current_step} executed. Reward: {reward:.2f}, Total Reward: {total_reward:.2f}")

    if terminated or truncated:
        flash("Episode terminated or truncated. Please reset the environment.")

    return redirect(url_for("index"))


@app.route("/reset", methods=["GET"])
def reset():
    """
    Reset the environment to start a new episode.
    """
    global env, obs, info, current_step, history, total_reward
    env = HFTEnv(config)
    obs, info = env.reset()
    current_step = 0
    total_reward = 0

    history = {
        "step": [],
        "midprice": [],
        "best_bid": [],
        "best_ask": [],
        "mtm": [],
        "reward": [],
        "total_reward": [],
        "trades": {
            "step": [],
            "price": [],
            "type": []
        }
    }
    flash("Environment reset successfully.")
    return redirect(url_for("index"))

@app.route("/random_action")
def random_action():
    """Generate random actions within valid ranges with appropriate rounding."""
    # Generate raw random values
    signed_volume = np.random.uniform(-1, 1)
    price_offset = np.random.uniform(-1, 1)
    cancel_fraction = np.random.uniform(0.0, 0.2)
    
    # Round to 2 decimal places for precise control
    return {
        "signed_volume": float(round(signed_volume, 2)),
        "price_offset": float(round(price_offset, 2)),
        "cancel_fraction": float(round(cancel_fraction, 2))
    }

###############################################################################
#   5. Run the App
###############################################################################
if __name__ == "__main__":
    # For production, typically run via a WSGI server (e.g., gunicorn).
    # If you want a specific port, set it here. E.g., 5000:
    port = int(os.environ.get("PORT", 5000))
    # debug=False is safer in production
    app.run(host="0.0.0.0", port=port, debug=False)
