# train.py
import logging
import os
import time
from copy import deepcopy
import json
from datetime import datetime
from pathlib import Path
import torch
import numpy as np
import pandas as pd
import psutil
try:
    import GPUtil  # type: ignore
except ImportError:  # pragma: no cover - optional dependency
    GPUtil = None  # Fallback handled in get_optimal_batch_size

try:
    import gymnasium as gym
except ImportError as exc:  # pragma: no cover - surface a clear error
    raise ImportError(
        "gymnasium is required to run training scripts. Install it with `pip install gymnasium`."
    ) from exc
import importlib
from stable_baselines3 import SAC
from stable_baselines3.common.logger import configure as configure_logger # For resuming logging
from stable_baselines3.common.env_util import make_vec_env # Potentially needed if agent uses VecEnv

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data" / "raw"
RUNS_DIR = PROJECT_ROOT / "runs"
LOG_DIR = RUNS_DIR / "logs"


# --- Default Configuration (will be overridden if resuming, except total_timesteps) ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')

config = {
    # --- Data and Environment ---
    "csv_path": str((DATA_DIR / "orderbook_trimmed_small.csv").resolve()),
    "env_path": "rltrader.envs.two_sided",
    "env_class": "TwoSidedMarketEnv",

    # --- Core Simulation Parameters ---
    "initial_capital": 20000.0,
    "max_steps": 500000, # Max steps to load from CSV (can be overridden by dataset size)
    "episode_length": 400, # Max steps per training episode

    # --- Market Microstructure ---
    "order_book_levels": 10,
    "tick_size": 0.01,
    "lot_size": 0.005,

    # --- Agent Actions & Constraints ---
    "price_offset_ticks": 50,       # Range for placing orders relative to BBO (scaled from action)
    "max_order_volume": 2.5,        # Max volume per order (scaled from action)
    "max_active_orders": 10,
    "max_inventory": 5.0,           # Max absolute inventory allowed
    "allowed_aggressiveness_ticks": 5, # Max ticks aggressive placement allowed (e.g., crossing spread)

    # --- Latency & Costs ---
    "latency_steps_long": 0,        # Steps until BUY order is active
    "latency_steps_short": 0,       # Steps until SELL order is active
    "transaction_cost_long": 0.0000, # Proportional cost for BUY execution (e.g., 0.0004 = 0.04%)
    "transaction_cost_short": 0.0000, # Proportional cost for SELL execution
    "taker_penalty": 0.000,         # Additional penalty per unit volume for aggressive (taker) orders

    # --- Reward Components ---
    "inventory_penalty": 0.0000,     # Quadratic penalty factor for holding inventory
    "invalid_order_penalty": 0.0,    # Penalty for placing invalid orders (e.g., too far)
    "activity_bonus": 0.00,         # Bonus per unit volume executed (maker incentive)
    "quoting_reward_enabled": True,
    "quoting_reward_amount": 0.00001, # Reward per side for tight quotes
    "quoting_reward_max_ticks": 5,   # Max ticks from BBO for quoting reward

    # --- Action Space Features ---
    "explicit_cancel_enabled": True,
    "explicit_cancel_threshold": 0.7,  # Action[4] > threshold triggers cancel
    "explicit_cancel_penalty": 0.00001,# Penalty for explicit cancel action
    "explicit_cancel_clears_pending": True, # Cancel pending orders too?
    "do_nothing_threshold": 0,      # Action[5] > threshold skips market actions (range -1 to 1)

    # --- Observation Space Features ---
    "obs_qty_norm_scale": 1.0, # Multiply normalized quantities by this factor
    "obs_price_norm_scale": 100.0, # Divide normalized prices (in ticks) by this factor

    # --- Agent Creation Parameters (Used if *not* resuming) ---
    "hidden_dim": 256,
    "features_dim": 128,
    "lstm_layers": 1,
    "num_heads": 4,
    "net_arch_pi": [128, 128],
    "net_arch_qf": [128, 128],

    # --- Training Hyperparameters ---
    "batch_size": None, # Will be calculated or loaded
    # <<< --- THIS VALUE IS USED TO OVERRIDE THE LOADED CONFIG WHEN RESUMING --- >>>
    "total_timesteps": 30000000, # Example: Increased target total timesteps
    "log_interval": 10, # Log SB3 metrics every N episodes

    # --- Deprecated/Unused (kept for reference/compatibility if needed) ---
    # "latency_steps": 0,
    # "transaction_cost": 0.000,
    # "details": {"reward_type": "mtm if "},
}

csv_path = Path(config["csv_path"]).resolve()
if not csv_path.exists():
    raise FileNotFoundError(
        f"Dataset not found at {csv_path}. Please place a limit order book CSV there before running training."
    )

# --- Helper Functions ---

def get_env_class(module_path, class_name):
    """Dynamically import the environment class."""
    try:
        module = importlib.import_module(module_path)
        return getattr(module, class_name)
    except ImportError as e:
        logging.error(f"Could not import environment module: {module_path}. Error: {e}")
        raise
    except AttributeError:
        logging.error(f"Could not find class '{class_name}' in module '{module_path}'.")
        raise

def get_optimal_batch_size(memory_per_sample_gb=0.001, max_default_bs=512, reserve_factor=0.8):
    """Calculate optimal batch size based on available GPU VRAM."""
    if not torch.cuda.is_available() or GPUtil is None:
        logging.warning("GPU not available, using default batch size.")
        return max_default_bs // 2 # Smaller default for CPU
    try:
        gpu = GPUtil.getGPUs()[0]
        free_memory_gb = (gpu.memoryFree / 1024) * reserve_factor # Free memory in GB, with reserve
        optimal_size = int(free_memory_gb / memory_per_sample_gb)
        optimal_size = max(32, min(optimal_size, max_default_bs)) # Clamp
        logging.info(f"GPU Free Memory: {gpu.memoryFree/1024:.2f}GB. Calculated Optimal Batch Size: {optimal_size}")
        return optimal_size
    except Exception as e:
        logging.warning(f"GPU info unavailable or error during calculation, using default batch size {max_default_bs}. Error: {e}")
        return max_default_bs

def estimate_training_time(data_size_gb, batch_size, steps_per_gb=1_000_000):
    """Estimate training time based on data size and batch size."""
    samples_per_second = 2000 if torch.cuda.is_available() else 100
    total_samples = data_size_gb * steps_per_gb
    if samples_per_second == 0 or batch_size == 0:
         logging.warning("Cannot estimate training time with zero samples/sec or batch size.")
         return "N/A"
    estimated_seconds = total_samples / samples_per_second
    return f"{estimated_seconds / 3600:.1f} hours"

def setup_training_dir(base_log_dir: Path | None = None, config_to_save=None):
    """Create training directory and save config."""
    base_dir = base_log_dir if base_log_dir is not None else LOG_DIR
    base_dir.mkdir(parents=True, exist_ok=True)
    run_name = datetime.now().strftime("%Y%m%d-%H%M%S")
    log_dir = (base_dir / run_name).resolve()
    log_dir.mkdir(parents=True, exist_ok=True)
    logging.info(f"Log directory created: {log_dir}")

    if config_to_save:
        config_path = log_dir / "config.json"
        try:
            # Use a copy to avoid modifying the original global config if passed directly
            config_copy = deepcopy(config_to_save)
            with config_path.open('w') as f:
                json.dump(config_copy, f, indent=4)
            logging.info(f"Configuration saved to {config_path}")
        except TypeError as e:
            logging.error(f"Failed to serialize config to JSON: {e}. Check config for non-serializable types.")
            try:
                with (log_dir / "config_error.txt").open('w') as f:
                    f.write(str(config_to_save))
            except: pass
        except Exception as e:
            logging.error(f"Failed to save configuration: {e}")
    else:
        logging.warning("No configuration provided to save.")

    return str(log_dir)

def find_existing_runs(base_log_dir: Path | None = None):
    """Finds existing training run directories."""
    base_dir = base_log_dir if base_log_dir is not None else LOG_DIR
    if not base_dir.exists():
        return []
    potential_runs = []
    for item in base_dir.iterdir():
        if item.is_dir():
            try:
                datetime.strptime(item.name, "%Y%m%d-%H%M%S")
                potential_runs.append(str(item))
            except ValueError:
                continue
    return sorted(potential_runs, reverse=True) # Newest first

def load_config_from_run(run_dir):
    """Loads config.json from a specified run directory."""
    config_path = os.path.join(run_dir, "config.json")
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"config.json not found in {run_dir}")
    try:
        with open(config_path, 'r') as f:
            loaded_config = json.load(f)
        logging.info(f"Successfully loaded configuration from {config_path}")
        return loaded_config
    except json.JSONDecodeError as e:
        logging.error(f"Error decoding JSON from {config_path}: {e}")
        raise
    except Exception as e:
        logging.error(f"Failed to load config from {config_path}: {e}")
        raise

def find_model_to_load(run_dir):
    """Finds the latest saved model (.zip) in a run directory."""
    interrupted_model = os.path.join(run_dir, "interrupted_model.zip")
    final_model = os.path.join(run_dir, "final_model.zip")

    if os.path.exists(interrupted_model):
        logging.info(f"Found interrupted model: {interrupted_model}")
        return interrupted_model
    elif os.path.exists(final_model):
        logging.info(f"Found final model: {final_model}")
        return final_model
    else:
        logging.warning(f"No 'interrupted_model.zip' or 'final_model.zip' found in {run_dir}")
        return None

# --- Main Training Function ---

def train(run_config, log_dir, resume_from_path=None):
    """Trains or resumes training the SAC agent."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logging.info(f"Using device: {device}")

    # --- Environment Creation (using run_config) ---
    if 'csv_path' not in run_config or not os.path.exists(run_config['csv_path']):
        raise FileNotFoundError(f"Configuration must include a valid 'csv_path'. Path not found: {run_config.get('csv_path')}")

    HFTEnv = get_env_class(run_config["env_path"], run_config["env_class"])
    try:
        # Create ONE instance - used for loading or passed to create_sac_agent
        base_env = HFTEnv(run_config)
        logging.info("Base environment created.")
        # Print obs space size after creation
        obs_space = base_env.observation_space
        if isinstance(obs_space, gym.spaces.Box):
             logging.info(f"Observation space size: {obs_space.shape[0]}")
        else:
             logging.info(f"Observation space: {obs_space}")

    except Exception as e:
        logging.error(f"Failed to create environment using config: {e}", exc_info=True)
        raise

    # Import agent creator AFTER setting up logging/config
    try:
        from rltrader.agents import create_sac_agent
    except ImportError as e:
        logging.error(f"Could not import agent creation function: {e}", exc_info=True)
        raise

    model = None
    if resume_from_path:
        # --- Resume Training ---
        logging.info(f"Attempting to resume training from: {resume_from_path}")
        try:
            # Load the model. Pass the env; SB3 uses it to check spaces
            # and potentially for loading wrappers like VecNormalize if they were saved separately.
            # Important: SB3 handles internal wrapping (like Monitor, DummyVecEnv) during load if env is passed
            model = SAC.load(
                resume_from_path,
                env=base_env, # Provide the base env instance, SB3 will wrap it as needed
                device=device
                # If you have custom policies/features, you might need custom_objects=...
            )
            logging.info(f"Model loaded successfully. Current timesteps: {model.num_timesteps}")

            # --- CORRECTED LOGGER CONFIGURATION ---
            # Configure the logger to output to the specified directory
            new_logger = configure_logger(folder=log_dir, format_strings=["stdout", "tensorboard", "csv"]) # Added csv
            model.set_logger(new_logger)
            # --- END CORRECTION ---

            logging.info(f"Logger reset to continue logging in: {log_dir}")

        except Exception as e:
            logging.error(f"Failed to load model from {resume_from_path}: {e}", exc_info=True)
            logging.error("Cannot resume. Consider starting a new training run.")
            raise # Re-raise the exception to stop the script
    else:
        # --- Start New Training ---
        logging.info("Starting a new training run.")
        # Get batch size dynamically for new runs
        batch_size = run_config.get("batch_size") or get_optimal_batch_size()
        logging.info(f"Using batch size: {batch_size}")
        run_config["batch_size"] = batch_size # Store it in config for reference

        try:
            # Estimate training time for new run
            data_size_gb = os.path.getsize(run_config['csv_path']) / 1e9
            logging.info(f"Dataset size: {data_size_gb:.2f}GB")
            est_time = estimate_training_time(data_size_gb, batch_size)
            logging.info(f"Roughly Estimated training time: {est_time}")
        except Exception as e:
            logging.warning(f"Could not estimate training time: {e}")

        try:
            # create_sac_agent should handle env wrapping if necessary (e.g., using make_vec_env)
            # Pass the base_env we created
             model, _ = create_sac_agent( # We already have log_dir
                env=base_env, # Pass the base env instance
                hidden_dim=run_config.get('hidden_dim', 256),
                features_dim=run_config.get('features_dim', 128),
                lstm_layers=run_config.get('lstm_layers', 1),
                num_heads=run_config.get('num_heads', 4),
                batch_size=batch_size,
                log_dir=log_dir,
                episode_length=run_config["episode_length"],
                net_arch_pi=run_config.get('net_arch_pi', [128, 128]),
                net_arch_qf=run_config.get('net_arch_qf', [128, 128]),
                # Add any other necessary args from run_config needed by create_sac_agent
             )
             logging.info("New SAC model created and configured.")
        except Exception as e:
             logging.error(f"Error during new agent creation: {e}", exc_info=True)
             raise # Critical error


    # --- Training Loop ---
    if not model:
        logging.error("Model could not be loaded or created. Aborting training.")
        return None, log_dir # Return None model

    start_time = time.time()
    # Ensure total_timesteps is treated as an integer from the FINAL run_config
    total_timesteps = int(run_config.get("total_timesteps", 1000000)) # Default if missing
    log_interval = run_config.get("log_interval", 10)

    # Calculate remaining timesteps using the potentially updated total_timesteps
    remaining_timesteps = total_timesteps - model.num_timesteps
    if remaining_timesteps <= 0:
        logging.info(f"Model already trained for {model.num_timesteps} timesteps (target: {total_timesteps}). Skipping training.")
    else:
        logging.info(f"Starting training for {remaining_timesteps} additional timesteps (current: {model.num_timesteps}, target: {total_timesteps})...")
        try:
            # Train the model - SB3 handles continuing from model.num_timesteps
            model.learn(
                total_timesteps=total_timesteps, # Target total steps for the whole run
                log_interval=log_interval,
                progress_bar=True,
                reset_num_timesteps=False # IMPORTANT: Do not reset counter when resuming
            )
            logging.info("Training finished successfully.")
        except Exception as e:
            logging.error(f"Error during model.learn: {e}", exc_info=True)
            # Save model progress even if interrupted
            interrupted_model_path = os.path.join(log_dir, "interrupted_model")
            try:
                model.save(interrupted_model_path)
                logging.info(f"Interrupted model saved due to error at: {interrupted_model_path}.zip")
            except Exception as save_e:
                logging.error(f"Could not save interrupted model: {save_e}")
            # Re-raise the training error after attempting to save
            raise e

    training_time_hours = (time.time() - start_time) / 3600
    logging.info(f"Actual training time (this session): {training_time_hours:.2f} hours")

    # --- Save Final Model ---
    final_model_path = os.path.join(log_dir, "final_model")
    try:
        model.save(final_model_path)
        logging.info(f"Final model saved at: {final_model_path}.zip")
        # Optional: remove interrupted model if final save succeeds
        interrupted_model_check = os.path.join(log_dir, "interrupted_model.zip")
        if os.path.exists(interrupted_model_check):
            try:
                os.remove(interrupted_model_check)
                logging.info("Removed previous interrupted model file.")
            except OSError as remove_e:
                 logging.warning(f"Could not remove interrupted model file {interrupted_model_check}: {remove_e}")
    except Exception as e:
        logging.error(f"Failed to save final model: {e}")

    # --- Close Environment ---
    try:
        # SB3 models store the env (potentially wrapped) in model.env
        if hasattr(model, 'env') and model.env is not None:
            model.env.close()
            logging.info("Training environment closed.")
        elif hasattr(base_env, 'close'):
             base_env.close() # Close the base env if model.env wasn't available
             logging.info("Base environment closed.")
    except Exception as e:
        logging.warning(f"Exception while closing environment: {e}")

    return model, log_dir

# --- Evaluation Function (Corrected - No is_recurrent check) ---

def evaluate(model, env, n_episodes=10):
    """
    Evaluate the trained model and log both total and adjusted (per-step) rewards.
    Uses the environment's render method for step-by-step logging if implemented.
    Handles potential VecEnv structure.
    """
    if not hasattr(env, 'render'):
         logging.warning("Evaluation environment does not have a `render` method. Step details won't be logged.")

    # Determine if the environment is vectorized
    is_vec_env = hasattr(env, 'num_envs') and env.num_envs > 1
    # if is_vec_env: # No need to log this repeatedly if called multiple times
        # logging.warning("Evaluation running on a vectorized environment...")

    device = model.device # Get device from the model
    all_episode_rewards = []
    all_adjusted_rewards = []
    all_episode_pnls = []
    all_episode_steps = []

    logging.info(f"--- Starting Evaluation ({n_episodes} episodes) ---")

    episodes_ran = 0
    while episodes_ran < n_episodes:
        current_episode_reward = 0.0
        current_episode_steps = 0
        current_episode_pnl = None

        # Reset environment
        obs, info = env.reset() # Returns obs and info dict (or list of dicts for VecEnv)

        # Initialize states for model.predict. It will be None for non-recurrent policies.
        states = None

        terminated = np.array([False] * env.num_envs) if is_vec_env else np.array([False])
        truncated = np.array([False] * env.num_envs) if is_vec_env else np.array([False])
        done = False

        logging.info(f"--- Evaluation Episode {episodes_ran + 1}/{n_episodes} ---")
        while not done:
            current_episode_steps += 1
            # Predict action(s) - Pass 'states', predict handles it internally
            action, states = model.predict(obs, state=states, deterministic=True)

            # Step environment
            try:
                obs, reward, terminated, truncated, info = env.step(action)

                # Handle VecEnv returns
                if is_vec_env:
                    step_reward = reward[0]
                    done = terminated[0] or truncated[0]
                    step_info = info[0]
                else:
                    step_reward = reward
                    done = terminated or truncated
                    step_info = info

                current_episode_reward += step_reward

                # Render/Log step details if method exists
                if hasattr(env, 'render'):
                    env.render()

                # Extract PnL from step info if available
                pnl_step = step_info.get('episode_pnl')
                if pnl_step is not None:
                    current_episode_pnl = pnl_step

                # Check for terminal observation info in VecEnv case
                if is_vec_env and done:
                    # Use get() with default to avoid KeyError if 'final_info' is missing
                    terminal_info = step_info.get('final_info', None)
                    if terminal_info:
                         # Also use get() for 'episode_pnl' within terminal_info
                        pnl_final = terminal_info.get('episode_pnl')
                        if pnl_final is not None:
                            current_episode_pnl = pnl_final

            except Exception as e:
                 logging.error(f"Error during env.step in evaluation episode {episodes_ran + 1}, step {current_episode_steps}: {e}", exc_info=True)
                 done = True
                 break # Stop this episode loop on error

        # --- Episode End Logging ---
        episodes_ran += 1

        # Get final PnL if not captured during steps (Improved logic)
        if current_episode_pnl is None: # Only try to find if not already set
            final_info_to_check = None
            if is_vec_env:
                if isinstance(step_info, dict) and isinstance(step_info.get('final_info'), dict):
                     final_info_to_check = step_info['final_info']
            elif isinstance(step_info, dict):
                 if isinstance(step_info.get('episode'), dict):
                     final_info_to_check = step_info['episode']
                 elif 'episode_pnl' in step_info:
                     final_info_to_check = step_info

            if final_info_to_check:
                current_episode_pnl = final_info_to_check.get('episode_pnl', 'N/A')
            else:
                 current_episode_pnl = 'N/A'

        all_episode_rewards.append(current_episode_reward)
        all_episode_steps.append(current_episode_steps)
        if current_episode_steps > 0:
            avg_reward = current_episode_reward / current_episode_steps
            all_adjusted_rewards.append(avg_reward)
        else:
             all_adjusted_rewards.append(0)

        if isinstance(current_episode_pnl, (float, int)):
            all_episode_pnls.append(current_episode_pnl)

        logging.info(f"Episode {episodes_ran} Finished: Steps={current_episode_steps}, Total Reward={current_episode_reward:.4f}, Final PnL={current_episode_pnl}")

        # Break outer loop if enough episodes ran
        if episodes_ran >= n_episodes:
            break

    # --- Close Evaluation Environment ---
    try:
        if hasattr(env, 'close'):
            env.close()
            logging.info("Evaluation environment closed.")
    except Exception as e:
        logging.warning(f"Exception while closing evaluation environment: {e}")

    # --- Overall Evaluation Statistics ---
    mean_total_reward = np.mean(all_episode_rewards) if all_episode_rewards else 0.0
    std_total_reward = np.std(all_episode_rewards) if all_episode_rewards else 0.0
    mean_adjusted_reward = np.mean(all_adjusted_rewards) if all_adjusted_rewards else 0.0
    std_adjusted_reward = np.std(all_adjusted_rewards) if all_adjusted_rewards else 0.0
    mean_pnl = np.mean(all_episode_pnls) if all_episode_pnls else 0.0
    std_pnl = np.std(all_episode_pnls) if all_episode_pnls else 0.0
    mean_steps = np.mean(all_episode_steps) if all_episode_steps else 0.0

    logging.info("--- Evaluation Summary ---")
    logging.info(f"Episodes Run: {len(all_episode_rewards)}")
    logging.info(f"Mean Total Reward: {mean_total_reward:.4f} +/- {std_total_reward:.4f}")
    logging.info(f"Mean Adjusted Reward (per step): {mean_adjusted_reward:.6f} +/- {std_adjusted_reward:.6f}")
    logging.info(f"Mean Episode PnL: {mean_pnl:.4f} +/- {std_pnl:.4f}")
    logging.info(f"Mean Episode Steps: {mean_steps:.1f}")
    logging.info("--- End Evaluation ---")

    return mean_total_reward, std_total_reward, mean_adjusted_reward, std_adjusted_reward

# --- Script Execution ---

def main():
    # Optional: Enable anomaly detection for debugging NaN gradients (slows training)
    # torch.autograd.set_detect_anomaly(True)

    main_start_time = time.time()
    logging.info(" ====== Starting Training Script ====== ")

    base_log_dir = LOG_DIR
    base_log_dir.mkdir(parents=True, exist_ok=True)  # Ensure base log dir exists

    # --- Check for existing runs and ask user ---
    existing_runs = find_existing_runs(base_log_dir)
    selected_run_dir = None
    resume_model_path = None
    # Make a deep copy of the default config to avoid modifying it globally initially
    run_config = deepcopy(config)

    if existing_runs:
        print("\nFound existing training runs:")
        for i, run_path in enumerate(existing_runs):
            print(f" [{i+1}] {os.path.basename(run_path)}")

        while True:
            try:
                resume_choice = input("Do you want to resume a previous run? (y/n, default n): ").lower().strip()
                if resume_choice == 'y':
                    selection = input(f"Enter the number of the run to resume (1-{len(existing_runs)}): ")
                    idx = int(selection) - 1
                    if 0 <= idx < len(existing_runs):
                        selected_run_dir = existing_runs[idx]
                        logging.info(f"User selected run: {selected_run_dir}")
                        break
                    else:
                        print("Invalid selection.")
                elif resume_choice == 'n' or resume_choice == '':
                    logging.info("User chose not to resume. Will start a new run.")
                    break
                else:
                    print("Invalid input. Please enter 'y' or 'n'.")
            except ValueError:
                print("Invalid input. Please enter a number.")
            except Exception as e:
                 logging.error(f"Error during selection: {e}")
                 logging.info("Defaulting to starting a new run.")
                 break

    # --- Prepare for chosen action (resume or new) ---
    log_dir = None
    model = None

    try:
        if selected_run_dir:
            # --- Load config and find model for resuming ---
            try:
                # Load config from the selected run directory
                run_config = load_config_from_run(selected_run_dir)
                logging.info("Using configuration loaded from selected run.")

                 # Check essential paths from loaded config
                if 'csv_path' not in run_config or not os.path.exists(run_config['csv_path']):
                    logging.error(f"CSV path '{run_config.get('csv_path')}' from loaded config is invalid or missing!")
                    raise FileNotFoundError("Invalid data path in loaded config.")
                if 'env_path' not in run_config or 'env_class' not in run_config:
                     raise ValueError("Environment path or class missing in loaded config.")

                # <<< --- FIX: OVERRIDE total_timesteps --- >>>
                # Get the target from the current script's default config
                current_script_target_timesteps = config.get("total_timesteps")
                if current_script_target_timesteps is not None:
                    loaded_timesteps = run_config.get('total_timesteps', 'N/A')
                    # Only override if the new target is actually different (or if loaded was N/A)
                    if str(loaded_timesteps) != str(current_script_target_timesteps):
                         logging.info(f"Overriding loaded total_timesteps ({loaded_timesteps}) with new target from script: {current_script_target_timesteps}")
                         run_config['total_timesteps'] = int(current_script_target_timesteps) # Ensure it's int
                    else:
                         logging.info(f"Loaded total_timesteps ({loaded_timesteps}) matches current script target. No override needed.")
                         # Still ensure it's an int in run_config
                         run_config['total_timesteps'] = int(current_script_target_timesteps)
                else:
                    logging.warning("Could not find 'total_timesteps' in the current script's default config to override. Using loaded value.")
                    # Ensure loaded value is int if it exists
                    if 'total_timesteps' in run_config:
                        run_config['total_timesteps'] = int(run_config['total_timesteps'])
                # <<< --- END FIX --- >>>

                resume_model_path = find_model_to_load(selected_run_dir)
                if not resume_model_path:
                    logging.warning(f"No suitable model found in {selected_run_dir}. Starting new training instead.")
                    selected_run_dir = None # Fallback to new training
                    run_config = deepcopy(config) # Reset to default config for the new run
                    log_dir = setup_training_dir(base_log_dir, run_config) # Create NEW log dir using default config
                else:
                    log_dir = selected_run_dir # Use the existing directory for logging
            except Exception as e:
                logging.error(f"Failed to prepare for resume from {selected_run_dir}: {e}", exc_info=True)
                logging.info("Attempting to start a new run instead.")
                selected_run_dir = None # Fallback to new training
                run_config = deepcopy(config) # Reset to default config for the new run
                log_dir = setup_training_dir(base_log_dir, run_config) # Create NEW log dir using default config
        else:
             # --- Setup for new training run ---
             # run_config is already the deepcopy of the default config
             log_dir = setup_training_dir(base_log_dir, run_config) # Create new dir and save default config


        # --- Log System Info ---
        if torch.cuda.is_available():
            gpu_mem = torch.cuda.get_device_properties(0).total_memory / 1e9
            logging.info(f"GPU Memory: {gpu_mem:.2f}GB")
        sys_mem = psutil.virtual_memory().total / 1e9
        logging.info(f"System Memory: {sys_mem:.2f}GB")

        # --- Execute Training ---
        # Pass the determined run_config and log_dir, and resume_model_path if applicable
        # The run_config now has the correct total_timesteps whether resuming or starting new
        model, log_dir = train(run_config, log_dir, resume_model_path)

        # --- Evaluation ---
        if model and log_dir:
            logging.info("--- Evaluating Final Model ---")
            try:
                # Import create_test_env AFTER agent creation/loading potentially modified imports
                from rltrader.agents import create_test_env
                # Use the config that was ACTUALLY used for training (run_config)
                eval_episode_len = int(run_config.get("episode_length", 400) * 1.5) # Slightly longer eval episodes
                # Important: Ensure create_test_env correctly handles the model's expected env setup (VecEnv?)
                test_env = create_test_env(run_config, model, episode_length=eval_episode_len)
                evaluate(model, test_env, n_episodes=5)
                # test_env.close() is now called inside evaluate()
            except ImportError as e:
                 logging.error(f"Could not import `create_test_env` from rltrader.agents: {e}. Skipping evaluation.")
            except FileNotFoundError as e: # Catch specific errors from env creation
                 logging.error(f"Evaluation env creation failed (data file?): {e}", exc_info=True)
            except Exception as e:
                logging.error(f"Error during final model evaluation: {e}", exc_info=True)
        else:
            logging.warning("Training did not return a valid model or log directory. Skipping final evaluation.")

    except FileNotFoundError as e:
         logging.error(f"A required file was not found: {e}")
    except ImportError as e:
         logging.error(f"Failed to import necessary modules: {e}", exc_info=True)
    except KeyboardInterrupt:
         logging.warning("Training interrupted by user (KeyboardInterrupt). Attempting to save final state if possible.")
         # Model saving on error/interrupt is handled within train()
    except Exception as e:
         logging.error(f"An unexpected error occurred during the main script execution: {e}", exc_info=True)
    finally:
        total_time_min = (time.time() - main_start_time) / 60
        logging.info(f" ====== Training Script Finished in {total_time_min:.2f} minutes ====== ")


if __name__ == "__main__":
    main()
