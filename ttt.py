#!/usr/bin/env python3
import os
import io # Added for potential future DB optimization
import json
import glob
import shutil
import numpy as np
import pandas as pd
from natsort import natsorted
from datetime import datetime, timedelta
import logging
import paramiko
import argparse
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from sqlalchemy import create_engine, exc as sqlalchemy_exc
from dotenv import load_dotenv
load_dotenv() # Loads variables from .env into environment
# ------------------------
# Configuration Parameters
# ------------------------

# Remote SSH connection settings
REMOTE_HOST = "8.211.140.51"
REMOTE_USER = "root"
REMOTE_PASS = None  # Set password if needed; otherwise assumes key-based auth. Use env vars ideally.
REMOTE_PORT = 22

# Remote base directory for raw orderbooks.
REMOTE_BASE_DIR = "/root/billions_frontend/dataset" # Use POSIX separators

# Defaults for broker, type, and symbol (used for new version structure primarily)
DEFAULT_BROKER = "binance"
DEFAULT_TYPE = "futures"
# DEFAULT_SYMBOL = "ethusdc" # No longer needed as a global default, determined dynamically

# Local directories
HOME_DIR = os.path.expanduser("~")
LOCAL_OUTPUT_BASE_DIR = os.path.join(HOME_DIR, "Documents", "billions_db", "orderbooks")
LOCAL_STAGING_DIR = "/tmp/remote_orderbook_staging" # Staging remains in /tmp
COMBINED_OUTPUT_DIR = os.path.join(LOCAL_OUTPUT_BASE_DIR, "combined")

# Database configuration (PostgreSQL running in Docker)
# !! SECURITY WARNING: Avoid hardcoding credentials. Use environment variables or secrets management. !!
DB_USER = os.getenv('DB_USER', 'billions')
DB_PASS = os.getenv('DB_PASS', 'yourpassword') # Example using env var, fallback to hardcoded
DB_HOST = os.getenv('DB_HOST', 'localhost')
DB_PORT = os.getenv('DB_PORT', '5432')
DB_NAME = os.getenv('DB_NAME', 'billionsdb')
DB_CONN_STRING = f'postgresql://{DB_USER}:{DB_PASS}@{DB_HOST}:{DB_PORT}/{DB_NAME}'

# Report file for logging detailed process summary
LOGS_DIR = os.path.join(HOME_DIR, "Documents", "logs")
REPORT_FILE = os.path.join(LOGS_DIR, "billions_orderbook_processing_report.txt")

# Data Integrity Parameters
MIN_HOURS_FULL_DAY = 23.0  # Minimum hours for a day to be considered "complete"
MAX_HOURS_FULL_DAY = 25.0  # Maximum hours
MAX_GAP_SECONDS = 300      # Maximum allowed gap between consecutive records (5 minutes)
MAX_LEVELS_TO_KEEP = 10    # Max number of bid/ask levels to store in the DataFrame/CSV

# Concurrency
MAX_DOWNLOAD_WORKERS = 4  # Max threads for downloading files
MAX_PROCESS_WORKERS = os.cpu_count() or 4 # Max threads/processes for parsing JSON

# ------------------------
# Logger Setup
# ------------------------
# Ensure log directory exists
try:
    os.makedirs(LOGS_DIR, exist_ok=True)
except OSError as e:
    print(f"Error creating log directory {LOGS_DIR}: {e}")
    # Fallback to current directory? Or exit? For now, just print.

logging.basicConfig(
    level=logging.INFO, # Changed default to INFO, DEBUG is very verbose
    format='%(asctime)s - %(levelname)s - %(threadName)s - %(message)s'
)
# Suppress overly verbose logs from libraries
logging.getLogger("paramiko").setLevel(logging.WARNING)
logging.getLogger("sqlalchemy.engine").setLevel(logging.WARNING)


# ------------------------
# Utility: Write to report log file
# ------------------------
def write_report(msg):
    """Appends a timestamped message to the report file and logs it."""
    timestamp = datetime.now().isoformat()
    log_entry = f"{timestamp} - {msg}\n"
    try:
        with open(REPORT_FILE, "a") as f:
            f.write(log_entry)
    except Exception as e:
        logging.error("Failed to write to report file '%s': %s", REPORT_FILE, e)
    logging.info(msg) # Also log to standard logger

# ------------------------
# SSH/SFTP Helpers
# ------------------------
def _construct_remote_path(*args):
    """Constructs a POSIX-style remote path."""
    # Filter out None or empty parts and join with '/'
    return "/".join(part for part in args if part)

def create_sftp_client(host, port, username, password=None):
    """Creates an SSH client, connects, and returns SFTP client and SSH client."""
    ssh_client = None
    try:
        logging.debug("Initializing SSH client.")
        ssh_client = paramiko.SSHClient()
        ssh_client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
        logging.debug("Connecting to %s:%d as %s...", host, port, username)
        ssh_client.connect(host, port=port, username=username, password=password, timeout=30) # Added timeout
        logging.debug("SSH connection established, opening SFTP session.")
        sftp = ssh_client.open_sftp()
        logging.info("Connected to %s via SSH/SFTP.", host)
        return sftp, ssh_client
    except Exception as e:
        logging.error("Failed to create SFTP client to %s: %s", host, e, exc_info=True)
        if ssh_client:
            try:
                ssh_client.close()
            except Exception as close_e:
                logging.error("Error closing SSH client after connection failure: %s", close_e)
        raise # Re-raise the exception to signal failure

def safe_sftp_listdir(sftp, remote_dir):
    """Safely lists directory contents, handling potential errors."""
    try:
        return sftp.listdir(remote_dir)
    except FileNotFoundError:
        logging.warning("Remote directory not found: %s", remote_dir)
        return []
    except Exception as e:
        logging.error("Error listing remote directory %s: %s", remote_dir, e)
        return []

def download_file_sftp(sftp, remote_file_path, local_file_path):
    """Downloads a single file with existence and size check."""
    try:
        remote_stat = sftp.stat(remote_file_path)
        remote_size = remote_stat.st_size

        if os.path.exists(local_file_path):
            try:
                local_size = os.path.getsize(local_file_path)
                if local_size == remote_size:
                    logging.debug("Skipping download, local file exists and size matches: %s", local_file_path)
                    return local_file_path # Indicate success (already exists)
            except OSError as e:
                logging.warning("Could not get size of local file %s, re-downloading. Error: %s", local_file_path, e)

        logging.debug("Downloading %s -> %s", remote_file_path, local_file_path)
        # Ensure local directory exists before downloading
        os.makedirs(os.path.dirname(local_file_path), exist_ok=True)
        sftp.get(remote_file_path, local_file_path)
        # Verify size after download (optional but good practice)
        local_size_after = os.path.getsize(local_file_path)
        if local_size_after != remote_size:
            logging.warning("Size mismatch after download for %s (Remote: %d, Local: %d). Check network or disk space.",
                            remote_file_path, remote_size, local_size_after)
            # Decide if this should be a fatal error? For now, just warn.
        return local_file_path # Indicate success
    except FileNotFoundError:
        logging.error("Remote file not found: %s", remote_file_path)
        return None
    except Exception as e:
        logging.error("Error downloading file %s: %s", remote_file_path, e)
        # Clean up potentially incomplete local file
        if os.path.exists(local_file_path):
            try:
                os.remove(local_file_path)
            except OSError as rm_e:
                logging.error("Failed to remove incomplete file %s: %s", local_file_path, rm_e)
        return None


def download_remote_files_concurrent(sftp, remote_dir, local_dir, pattern_prefix="orderbook"):
    """
    Downloads files concurrently from remote_dir matching pattern_prefix to local_dir.
    Uses ThreadPoolExecutor for downloads and checks local existence/size.
    Returns a sorted list of local file paths that were successfully downloaded or already existed.
    """
    try:
        logging.debug("Ensuring local directory %s exists.", local_dir)
        os.makedirs(local_dir, exist_ok=True)

        remote_files = safe_sftp_listdir(sftp, remote_dir)
        logging.debug("Remote files in %s: %s", remote_dir, remote_files)

        matching_files = [f for f in remote_files if f.startswith(pattern_prefix)]
        if not matching_files:
            logging.info("No files matching prefix '%s' found in %s.", pattern_prefix, remote_dir)
            return []

        local_files_futures = {}
        downloaded_files = []

        with ThreadPoolExecutor(max_workers=MAX_DOWNLOAD_WORKERS, thread_name_prefix='Downloader') as executor:
            for fname in matching_files:
                remote_file_path = _construct_remote_path(remote_dir, fname)
                local_file_path = os.path.join(local_dir, fname)
                future = executor.submit(download_file_sftp, sftp, remote_file_path, local_file_path)
                local_files_futures[future] = remote_file_path

            for future in as_completed(local_files_futures):
                remote_path = local_files_futures[future]
                try:
                    result_path = future.result()
                    if result_path: # None indicates failure
                        downloaded_files.append(result_path)
                except Exception as e:
                    logging.error("Exception during download task for %s: %s", remote_path, e)

        # Sort the successfully obtained local file paths
        downloaded_files = natsorted(downloaded_files)
        logging.info("Successfully obtained %d files from %s into %s.", len(downloaded_files), remote_dir, local_dir)
        return downloaded_files

    except Exception as e:
        logging.error("General error during concurrent download process for %s: %s", remote_dir, e)
        return []

# ------------------------
# Version Detection and Scanning
# ------------------------
def detect_remote_version(sftp):
    """Detects if the remote data uses the new or old directory structure."""
    try:
        items = safe_sftp_listdir(sftp, REMOTE_BASE_DIR)
        logging.debug("Items in REMOTE_BASE_DIR '%s': %s", REMOTE_BASE_DIR, items)
        # A simple heuristic: if a known broker name exists as a directory, assume new version.
        if DEFAULT_BROKER in items:
            # Further check: see if 'futures' or 'spot' exists under the broker
            try:
                broker_items = safe_sftp_listdir(sftp, _construct_remote_path(REMOTE_BASE_DIR, DEFAULT_BROKER))
                if any(t in broker_items for t in ['futures', 'spot']):
                    write_report("Detected new dataset structure (broker/type/symbol).")
                    return "new"
            except Exception:
                 pass # Fall through to old if checks fail
        write_report("Detected old dataset structure (symbol/date).")
        return "old"
    except Exception as e:
        logging.error("Error detecting remote version: %s. Assuming 'old'.", e)
        return "old" # Default to old structure on error

def scan_available_symbols(sftp, version):
    """
    Scans the remote directory for available symbols based on detected version.
    Returns a list of (broker, type, symbol) tuples.
    """
    symbols = []
    if version == "new":
        broker_dir = _construct_remote_path(REMOTE_BASE_DIR, DEFAULT_BROKER)
        type_dirs = safe_sftp_listdir(sftp, broker_dir)
        logging.debug("Scanning for types in new structure under %s: %s", broker_dir, type_dirs)
        for typee in type_dirs:
            # Basic check if it looks like a type directory (e.g., 'futures', 'spot')
            if typee in ['futures', 'spot']: # Make this configurable if needed
                type_dir_path = _construct_remote_path(broker_dir, typee)
                try:
                    symbol_dirs = safe_sftp_listdir(sftp, type_dir_path)
                    logging.debug("Found symbols under %s: %s", type_dir_path, symbol_dirs)
                    for sym in symbol_dirs:
                        # Add a check if 'sym' looks like a valid symbol directory
                        # e.g., check if it contains date folders, or based on naming pattern
                        # For now, assume all items are symbol directories
                        symbols.append((DEFAULT_BROKER, typee, sym))
                except Exception as e:
                    logging.error("Error scanning symbols in %s: %s", type_dir_path, e)
    else: # Old version
        try:
            items = safe_sftp_listdir(sftp, REMOTE_BASE_DIR)
            logging.debug("Scanning for symbols in old structure under %s: %s", REMOTE_BASE_DIR, items)
            for item in items:
                # Heuristic: lowercase, contains letters, might contain numbers, length > 3?
                if item.islower() and any(c.isalpha() for c in item) and len(item) >= 3:
                     # Check if it's likely a symbol dir by probing for a date folder inside?
                     # This adds extra SFTP calls. Let's assume it's a symbol for now.
                     # We rely on scan_date_folders to filter out non-symbol dirs later.
                    symbols.append((DEFAULT_BROKER, DEFAULT_TYPE, item)) # Use defaults for old version
        except Exception as e:
            logging.error("Error scanning symbols in old version structure: %s", e)

    write_report(f"Found {len(symbols)} potential symbol entries: {symbols}")
    return symbols


def get_remote_symbol_data_path(broker, typee, symbol, version):
    """Constructs the remote path to the symbol's data directory based on version."""
    if version == "new":
        return _construct_remote_path(REMOTE_BASE_DIR, broker, typee, symbol)
    else: # old version
        return _construct_remote_path(REMOTE_BASE_DIR, symbol)

def scan_date_folders(sftp, remote_symbol_dir):
    """
    Scans the remote symbol directory for valid date folders ('DD-Mon-YYYY').
    Returns a sorted list of valid date folder names.
    """
    valid_dates = []
    all_items = safe_sftp_listdir(sftp, remote_symbol_dir)
    logging.debug("Scanning for date folders in %s: Found items %s", remote_symbol_dir, all_items)

    for folder in all_items:
        try:
            # Check if the folder name matches the expected date format
            datetime.strptime(folder, '%d-%b-%Y')
            # Optional: Check if it's actually a directory (requires sftp.stat)
            # remote_path = _construct_remote_path(remote_symbol_dir, folder)
            # if S_ISDIR(sftp.stat(remote_path).st_mode):
            valid_dates.append(folder)
        except ValueError:
            logging.debug("Item '%s' in %s is not a valid date folder name.", folder, remote_symbol_dir)
        except Exception as e:
            logging.warning("Error checking item '%s' in %s: %s", folder, remote_symbol_dir, e)

    valid_dates = natsorted(valid_dates)
    if valid_dates:
        logging.info("Found %d valid date folders in %s.", len(valid_dates), remote_symbol_dir)
    else:
        logging.warning("No valid date folders ('DD-Mon-YYYY') found in %s.", remote_symbol_dir)
    return valid_dates

# ------------------------
# Orderbook Cleaning Functions
# ------------------------

def _parse_orderbook_line(line):
    """
    Parses a single JSONL line, handles potential quote issues, extracts data,
    and formats it into a dictionary. Returns None if parsing fails.
    """
    record = None
    try:
        # Try standard JSON load first
        record = json.loads(line)
    except json.JSONDecodeError:
        try:
            # Fallback: try replacing single quotes (use cautiously)
            logging.debug("Standard JSON parsing failed, trying with single quote replacement.")
            record = json.loads(line.replace("'", '"'))
        except json.JSONDecodeError as e:
            logging.error("Failed to parse line after quote replacement: %s", e)
            logging.debug("Problematic line content: %s", line.strip())
            return None
    except Exception as e:
        logging.error("Unexpected error parsing line: %s", e)
        logging.debug("Problematic line content: %s", line.strip())
        return None

    # Check for essential keys (adjust based on actual data structure)
    if not record or 'datetime' not in record or 'lastUpdateId' not in record or 'bids' not in record or 'asks' not in record:
        logging.warning("Skipping record due to missing essential keys: %s", line.strip())
        return None

    try:
        # Process timestamp
        original_datetime_ts = float(record['datetime'])
        expanded_datetime = datetime.utcfromtimestamp(original_datetime_ts)

        # Prepare output dict
        output = {
            'datetime': original_datetime_ts, # Keep original float timestamp
            'expanded_datetime': expanded_datetime, # Add parsed datetime object
            'lastUpdateId': int(record['lastUpdateId']) # Ensure integer
        }

        # Process bids and asks, padding/truncating to MAX_LEVELS_TO_KEEP
        for side, key in [('bid', 'bids'), ('ask', 'asks')]:
            levels = record[key]
            for i in range(MAX_LEVELS_TO_KEEP):
                level_data = levels[i] if i < len(levels) else (np.nan, np.nan)
                try:
                    price = float(level_data[0])
                except (TypeError, ValueError):
                    price = np.nan
                try:
                    qty = float(level_data[1])
                except (TypeError, ValueError):
                    qty = np.nan

                output[f'{side}{i+1}'] = price
                output[f'{side}qty{i+1}'] = qty

        return output

    except (ValueError, TypeError, KeyError, IndexError) as e:
        logging.error("Error processing fields in record: %s. Record: %s", e, line.strip())
        return None


def load_orderbook_records(filepath):
    """
    Reads a raw orderbook file (JSONL) line by line, parses each line using
    _parse_orderbook_line, and returns a list of valid record dictionaries.
    """
    records = []
    try:
        with open(filepath, 'r') as f:
            for i, line in enumerate(f):
                line = line.strip()
                if not line:
                    continue
                parsed_record = _parse_orderbook_line(line)
                if parsed_record:
                    records.append(parsed_record)
                # Optional: Add logging for skipped lines if needed, but can be verbose
                # else:
                #     logging.debug("Skipped line %d in file %s", i + 1, filepath)
    except FileNotFoundError:
        logging.error("File not found during parsing: %s", filepath)
        return []
    except Exception as e:
        logging.error("Error reading or processing file %s: %s", filepath, e)
        return [] # Return empty list on error
    return records

def create_orderbook_dataframe(records):
    """Converts a list of record dictionaries into a Pandas DataFrame."""
    if not records:
        return pd.DataFrame() # Return empty DataFrame if no records

    try:
        df = pd.DataFrame(records)
        # Ensure correct dtypes (optional but good practice)
        df['expanded_datetime'] = pd.to_datetime(df['expanded_datetime'])
        # Numeric columns could be explicitly cast if needed, but NaN handling usually works
        # Example: df['bid1'] = pd.to_numeric(df['bid1'], errors='coerce')
        return df
    except Exception as e:
        logging.error("Failed to create DataFrame from records: %s", e)
        return pd.DataFrame()

# ------------------------
# Data Coverage Check (Integrated into process_orderbook_date)
# ------------------------
# The old check_full_day_coverage function is removed.
# The check is now performed on the combined daily DataFrame.

# ------------------------
# Processing and Combining Functions
# ------------------------
def process_orderbook_date(broker, typee, symbol, date_str, remote_symbol_dir, sftp, version):
    """
    Downloads, processes, and checks coverage for orderbook files for a given date.
    Returns the local CSV path if successful and data meets basic criteria, else None.
    """
    local_output_dir = os.path.join(LOCAL_OUTPUT_BASE_DIR, broker, typee, symbol, date_str)
    os.makedirs(local_output_dir, exist_ok=True)
    output_csv = os.path.join(local_output_dir, "orderbook.csv")
    coverage_report_path = os.path.join(local_output_dir, "coverage_report.json") # Store coverage info

    # Skip if final CSV and coverage report exist
    if os.path.isfile(output_csv) and os.path.isfile(coverage_report_path):
        try:
            with open(coverage_report_path, 'r') as f:
                report_data = json.load(f)
            write_report(f"Skipping {symbol} on {date_str} (already processed). Coverage: {report_data.get('duration_hours', 'N/A'):.2f}h, Gaps: {report_data.get('num_gaps', 'N/A')}.")
            return output_csv
        except Exception as e:
            logging.warning(f"Found existing CSV for {date_str} but couldn't read coverage report {coverage_report_path}: {e}. Reprocessing.")
            # Clean up potentially inconsistent state before reprocessing
            if os.path.isfile(output_csv): os.remove(output_csv)
            if os.path.isfile(coverage_report_path): os.remove(coverage_report_path)


    # Determine remote date directory path (already constructed relative to symbol dir)
    remote_date_dir = _construct_remote_path(remote_symbol_dir, date_str)

    # Download files concurrently to staging
    local_stage_dir = os.path.join(LOCAL_STAGING_DIR, broker, typee, symbol, date_str)
    try:
        # Clean staging dir for the date first
        if os.path.isdir(local_stage_dir):
            shutil.rmtree(local_stage_dir)
    except Exception as e:
        logging.warning("Could not clean staging directory '%s' before download: %s", local_stage_dir, e)

    downloaded_files = download_remote_files_concurrent(sftp, remote_date_dir, local_stage_dir, pattern_prefix="orderbook")

    if not downloaded_files:
        write_report(f"No valid files downloaded or found locally for {symbol} on {date_str} from {remote_date_dir}. Skipping date.")
        return None

    # Process JSON files concurrently (can use ThreadPoolExecutor as parsing is often I/O bound)
    all_records = []
    with ThreadPoolExecutor(max_workers=MAX_PROCESS_WORKERS, thread_name_prefix='Parser') as executor:
        future_to_fp = {executor.submit(load_orderbook_records, fp): fp for fp in downloaded_files}
        for future in as_completed(future_to_fp):
            fp = future_to_fp[future]
            try:
                records_from_file = future.result()
                if records_from_file:
                    all_records.extend(records_from_file)
                # else: # Log if a file yielded no records (already logged in load_orderbook_records)
                #    logging.warning("No valid records found in file: %s", fp)
            except Exception as e:
                logging.error("Error processing task for file %s: %s", fp, e, exc_info=True)

    if not all_records:
        write_report(f"No valid orderbook records could be parsed for {symbol} on {date_str}. Skipping date.")
        # Clean up staging directory
        try:
            shutil.rmtree(local_stage_dir)
        except Exception as e:
            logging.warning("Could not clean staging dir '%s' after failed processing: %s", local_stage_dir, e)
        return None

    # Create DataFrame and sort
    combined_df = create_orderbook_dataframe(all_records)
    if combined_df.empty:
         write_report(f"DataFrame creation failed for {symbol} on {date_str}. Skipping date.")
         # Clean up staging directory
         try:
            shutil.rmtree(local_stage_dir)
         except Exception as e:
            logging.warning("Could not clean staging dir '%s' after failed DataFrame creation: %s", local_stage_dir, e)
         return None

    combined_df = combined_df.sort_values(by="datetime").reset_index(drop=True)

    # --- Perform Data Coverage and Gap Check ---
    coverage_info = {
        "date": date_str,
        "symbol": symbol,
        "num_records": len(combined_df),
        "start_ts": None, "end_ts": None, "start_dt_utc": None, "end_dt_utc": None,
        "duration_seconds": 0, "duration_hours": 0,
        "is_full_day": False,
        "num_gaps": 0,
        "max_gap_seconds": 0,
        "gaps_details": [] # Store top N gaps maybe? For now, just count and max.
    }

    if not combined_df.empty:
        start_ts = combined_df['datetime'].iloc[0]
        end_ts = combined_df['datetime'].iloc[-1]
        start_dt = datetime.utcfromtimestamp(start_ts)
        end_dt = datetime.utcfromtimestamp(end_ts)
        duration_seconds = end_ts - start_ts
        duration_hours = duration_seconds / 3600.0

        coverage_info.update({
            "start_ts": start_ts, "end_ts": end_ts,
            "start_dt_utc": start_dt.isoformat() + "Z",
            "end_dt_utc": end_dt.isoformat() + "Z",
            "duration_seconds": duration_seconds,
            "duration_hours": duration_hours,
            "is_full_day": (MIN_HOURS_FULL_DAY <= duration_hours <= MAX_HOURS_FULL_DAY)
        })

        # Gap analysis
        if len(combined_df) > 1:
            diffs = combined_df['datetime'].diff()
            large_gaps = diffs[diffs > MAX_GAP_SECONDS]
            coverage_info["num_gaps"] = len(large_gaps)
            if not large_gaps.empty:
                coverage_info["max_gap_seconds"] = large_gaps.max()
                # Optionally store details of large gaps (e.g., timestamps before/after)
                # coverage_info["gaps_details"] = [(combined_df['datetime'].iloc[i-1], combined_df['datetime'].iloc[i], gap) for i, gap in large_gaps.items()]
                # Limit details to avoid huge reports
                gap_indices = large_gaps.index[:10] # Store first 10 gaps info
                coverage_info["gaps_details"] = [
                    {"before_ts": combined_df['datetime'].iloc[i-1], "after_ts": combined_df['datetime'].iloc[i], "gap_sec": gap}
                    for i in gap_indices
                ]


    # Report findings
    report_summary = (
        f"Processed {symbol} on {date_str}. "
        f"Records: {coverage_info['num_records']}. "
        f"Coverage: {coverage_info['start_dt_utc']} to {coverage_info['end_dt_utc']} ({coverage_info['duration_hours']:.2f}h). "
        f"Full Day: {coverage_info['is_full_day']}. "
        f"Gaps > {MAX_GAP_SECONDS}s: {coverage_info['num_gaps']} (Max: {coverage_info['max_gap_seconds']:.1f}s)."
    )
    write_report(report_summary)
    if not coverage_info['is_full_day']:
        write_report(f"WARNING: Data for {symbol} on {date_str} does not meet 'full day' criteria ({MIN_HOURS_FULL_DAY}-{MAX_HOURS_FULL_DAY}h).")
    if coverage_info['num_gaps'] > 0:
        write_report(f"WARNING: Found {coverage_info['num_gaps']} significant gaps in data for {symbol} on {date_str}.")

    # Save results
    try:
        combined_df.to_csv(output_csv, index=False)
        # Save coverage report
        with open(coverage_report_path, 'w') as f:
            json.dump(coverage_info, f, indent=4)
        logging.info("Saved processed data to %s", output_csv)
        logging.info("Saved coverage report to %s", coverage_report_path)
    except Exception as e:
        write_report(f"Error saving results for {symbol} on {date_str}: {e}")
        # Clean up potentially corrupt output files
        if os.path.isfile(output_csv): os.remove(output_csv)
        if os.path.isfile(coverage_report_path): os.remove(coverage_report_path)
        # Also clean staging?
        try:
            shutil.rmtree(local_stage_dir)
        except Exception as clean_e:
            logging.warning("Could not clean staging dir '%s' after save error: %s", local_stage_dir, clean_e)
        return None # Indicate failure

    # Clean up staging directory for the date
    try:
        shutil.rmtree(local_stage_dir)
        logging.debug("Cleaned staging directory: %s", local_stage_dir)
    except Exception as e:
        logging.warning("Could not clean staging dir '%s' after successful processing: %s", local_stage_dir, e)

    return output_csv


def combine_orderbooks(broker, typee, symbol, window_days=3):
    """
    Combines daily processed CSVs into multi-day packs (e.g., 3-day windows).
    Returns a list of paths for the combined CSV files created in this run.
    """
    symbol_daily_dir = os.path.join(LOCAL_OUTPUT_BASE_DIR, broker, typee, symbol)
    if not os.path.isdir(symbol_daily_dir):
        write_report(f"No local daily data directory found for {symbol} at {symbol_daily_dir}. Skipping combination.")
        return []

    # Find valid date folders (containing orderbook.csv)
    valid_date_folders = []
    for d in os.listdir(symbol_daily_dir):
        date_path = os.path.join(symbol_daily_dir, d)
        csv_path = os.path.join(date_path, "orderbook.csv")
        if os.path.isdir(date_path) and os.path.isfile(csv_path):
            try:
                # Verify date format if needed, though folder name implies it
                datetime.strptime(d, '%d-%b-%Y')
                valid_date_folders.append(d)
            except ValueError:
                logging.warning("Folder '%s' in daily dir looks like data but doesn't match date format.", d)

    # Sort dates chronologically
    valid_date_folders = natsorted(valid_date_folders, key=lambda d: datetime.strptime(d, '%d-%b-%Y'))

    if len(valid_date_folders) < window_days:
        logging.info(f"Not enough daily CSVs ({len(valid_date_folders)}) available for {symbol} to create a {window_days}-day window.")
        return []

    newly_combined_files = []
    os.makedirs(COMBINED_OUTPUT_DIR, exist_ok=True)

    # Iterate through possible windows
    for i in range(len(valid_date_folders) - window_days + 1):
        window_dates = valid_date_folders[i : i + window_days]

        # Define combined filename and check if it already exists
        start_date_str = datetime.strptime(window_dates[0], '%d-%b-%Y').strftime('%Y-%m-%d')
        end_date_str = datetime.strptime(window_dates[-1], '%d-%b-%Y').strftime('%Y-%m-%d')
        combined_filename = f"{symbol}_orderbook_{start_date_str}_to_{end_date_str}.csv"
        combined_path = os.path.join(COMBINED_OUTPUT_DIR, combined_filename)

        if os.path.isfile(combined_path):
            logging.debug(f"Combined file already exists: {combined_path}. Skipping window {window_dates}.")
            continue

        # Read and combine DataFrames for the window
        dfs_window = []
        valid_window = True
        for date_str in window_dates:
            csv_path = os.path.join(symbol_daily_dir, date_str, "orderbook.csv")
            try:
                # Add low_memory=False if DtypeWarning occurs, or specify dtypes
                df_daily = pd.read_csv(csv_path, low_memory=False)
                if not df_daily.empty:
                    dfs_window.append(df_daily)
                else:
                    write_report(f"Warning: Daily CSV for {symbol} on {date_str} is empty. Skipping window {window_dates}.")
                    valid_window = False
                    break
            except Exception as e:
                write_report(f"Error reading daily CSV {csv_path} for window {window_dates}: {e}. Skipping window.")
                valid_window = False
                break

        if not valid_window or not dfs_window:
            continue

        # Concatenate, sort, and save
        try:
            combined_df = pd.concat(dfs_window, ignore_index=True)
            # Sort again just in case daily files overlap slightly or order is imperfect
            combined_df = combined_df.sort_values(by="datetime").reset_index(drop=True)

            combined_df.to_csv(combined_path, index=False)
            write_report(f"Combined {symbol} orderbooks for window {window_dates} into {combined_filename}")
            newly_combined_files.append(combined_path)
        except Exception as e:
            write_report(f"Error combining or saving data for window {window_dates}: {e}")
            # Clean up potentially partial combined file
            if os.path.isfile(combined_path):
                try: os.remove(combined_path)
                except OSError as rm_e: logging.error("Failed to remove partial combined file %s: %s", combined_path, rm_e)


    return newly_combined_files

# ------------------------
# Database Loading
# ------------------------
def load_into_db(csv_files, table_name="orderbooks", db_engine=None):
    """
    Loads the provided CSV files into the PostgreSQL database.
    Uses an existing engine if provided, otherwise creates one.
    """
    if not csv_files:
        logging.info("No CSV files provided to load into database.")
        return

    engine = db_engine
    if engine is None:
        try:
            engine = create_engine(DB_CONN_STRING)
            # Test connection
            with engine.connect() as connection:
                logging.info("Database engine created and connection tested successfully.")
        except Exception as e:
            logging.error("Failed to create database engine: %s", e)
            write_report("Database loading skipped due to connection error.")
            return # Cannot proceed without engine

    loaded_count = 0
    failed_count = 0
    for csv_file in csv_files:
        write_report(f"Loading {os.path.basename(csv_file)} into DB table '{table_name}'...")
        try:
            # Read CSV in chunks for potentially large files
            chunk_iter = pd.read_csv(csv_file, chunksize=50000, low_memory=False) # Adjust chunksize as needed
            start_time = time.time()
            total_rows = 0

            for i, chunk in enumerate(chunk_iter):
                # Ensure 'expanded_datetime' is correct type for DB if not already
                chunk['expanded_datetime'] = pd.to_datetime(chunk['expanded_datetime'])
                # Other type conversions if necessary

                chunk.to_sql(table_name, engine, if_exists="append", index=False, method=None) # method=None lets sqlalchemy choose
                total_rows += len(chunk)
                logging.debug(f"Loaded chunk {i+1} ({len(chunk)} rows) for {os.path.basename(csv_file)}")

            end_time = time.time()
            duration = end_time - start_time
            write_report(f"Successfully loaded {total_rows} rows from {os.path.basename(csv_file)} into '{table_name}' in {duration:.2f} seconds.")
            loaded_count += 1

            # Optional: Mark file as loaded (e.g., move to a 'loaded' folder or record in a DB table)
            # shutil.move(csv_file, os.path.join(COMBINED_OUTPUT_DIR, "loaded", os.path.basename(csv_file)))

        except (sqlalchemy_exc.SQLAlchemyError, OSError, pd.errors.ParserError) as e:
            logging.error(f"Failed to load {os.path.basename(csv_file)} into database: {e}", exc_info=True)
            write_report(f"ERROR: Failed loading {os.path.basename(csv_file)}. Reason: {e}")
            failed_count += 1
        except Exception as e: # Catch any other unexpected errors
            logging.error(f"Unexpected error loading {os.path.basename(csv_file)}: {e}", exc_info=True)
            write_report(f"ERROR: Unexpected error loading {os.path.basename(csv_file)}. Reason: {e}")
            failed_count += 1


    write_report(f"Database loading finished. Successfully loaded: {loaded_count} files. Failed: {failed_count} files.")

    # Dispose engine only if it was created within this function
    if db_engine is None and engine is not None:
        try:
            engine.dispose()
            logging.info("Database engine disposed.")
        except Exception as e:
            logging.error("Error disposing database engine: %s", e)


# ------------------------
# Stub for Central Inspection Module
# ------------------------
def trigger_inspection():
    """Placeholder for triggering a central inspection module or process."""
    # This could involve:
    # - Making an API call to another service
    # - Sending a message queue notification
    # - Running another script/process
    write_report("Placeholder: Triggering central inspection module (implementation needed).")
    # Example: time.sleep(2) # Simulate work

# ------------------------
# Pipeline Modes: Once and Continuous
# ------------------------
def run_pipeline_once(interactive=True):
    """
    Runs the processing pipeline once. Handles connection, scanning, processing,
    combining, and loading. Interactive mode prompts for symbol selection.
    """
    sftp = None
    ssh_client = None
    db_engine = None
    start_run_time = time.time()
    write_report("=" * 30 + f" Pipeline Run Started (Mode: {'Interactive' if interactive else 'Batch'}) " + "=" * 30)

    try:
        # --- 1. Connect SFTP ---
        sftp, ssh_client = create_sftp_client(REMOTE_HOST, REMOTE_PORT, REMOTE_USER, REMOTE_PASS)

        # --- 2. Detect Version & Scan Symbols ---
        version = detect_remote_version(sftp)
        available_symbols = scan_available_symbols(sftp, version)
        if not available_symbols:
            write_report("No symbols found or accessible on remote. Exiting run.")
            return

        # --- 3. Select Symbols ---
        selected_symbols_info = []
        if interactive:
            print("\n--- Available Symbols ---")
            for idx, (broker, typee, sym) in enumerate(available_symbols):
                print(f"{idx+1}: Broker: {broker}, Type: {typee}, Symbol: {sym}")
            print("-" * 25)
            while True:
                try:
                    selected_input = input(f"Enter symbol numbers to process (comma-separated), 'all' for all ({len(available_symbols)}), or 'none' to skip: ").strip().lower()
                    if selected_input == 'none':
                        write_report("User chose to skip symbol processing.")
                        selected_symbols_info = []
                        break
                    elif selected_input == 'all':
                        selected_symbols_info = available_symbols
                        write_report(f"User selected all {len(available_symbols)} symbols.")
                        break
                    else:
                        indices = [int(i.strip()) - 1 for i in selected_input.split(",") if i.strip().isdigit()]
                        selected_symbols_info = [available_symbols[i] for i in indices if 0 <= i < len(available_symbols)]
                        if selected_symbols_info:
                             write_report(f"User selected {len(selected_symbols_info)} symbols: {selected_symbols_info}")
                             break
                        else:
                             print("Invalid selection. Please enter valid numbers or 'all'/'none'.")
                except ValueError:
                    print("Invalid input. Please enter numbers separated by commas.")
        else: # Batch mode
            selected_symbols_info = available_symbols
            write_report(f"Batch mode: Processing all {len(available_symbols)} available symbols.")

        if not selected_symbols_info:
            write_report("No symbols selected for processing.")
            # Proceed to potentially trigger inspection even if no symbols processed?
            # trigger_inspection() # Decide if this should run
            return # Exit if nothing to process

        processed_daily_files = {} # Store {symbol: [list_of_daily_csv_paths]}

        # --- 4. Process Each Selected Symbol ---
        for broker, typee, symbol in selected_symbols_info:
            symbol_id = f"{broker}_{typee}_{symbol}" # Unique identifier
            write_report(f"---> Starting processing for Symbol: {symbol_id} <---")
            remote_symbol_dir = get_remote_symbol_data_path(broker, typee, symbol, version)

            # Scan for date folders for this symbol
            date_folders = scan_date_folders(sftp, remote_symbol_dir)
            if not date_folders:
                write_report(f"No date folders found for {symbol_id} at {remote_symbol_dir}. Skipping symbol.")
                continue

            processed_daily_files[symbol_id] = []
            processed_count = 0
            # Process each date folder
            for date_str in date_folders:
                try:
                    daily_csv_path = process_orderbook_date(broker, typee, symbol, date_str, remote_symbol_dir, sftp, version)
                    if daily_csv_path:
                        processed_daily_files[symbol_id].append(daily_csv_path)
                        processed_count += 1
                except Exception as e:
                    # Catch unexpected errors during daily processing
                    logging.error(f"Unexpected error processing {symbol_id} for date {date_str}: {e}", exc_info=True)
                    write_report(f"ERROR: Unexpected failure processing {symbol_id} date {date_str}. See logs for details.")
            write_report(f"Finished daily processing for {symbol_id}. Processed {processed_count}/{len(date_folders)} dates.")


        # --- 5. Combine Daily Files (Optional based on mode/config) ---
        newly_combined_files_all = []
        combine_flag = False
        if interactive:
             combine_choice = input("\nCombine all processed daily orderbooks into multi-day files? (y/n): ").strip().lower()
             if combine_choice == 'y':
                 combine_flag = True
                 write_report("User chose to combine daily files.")
             else:
                 write_report("User chose not to combine daily files.")
        else: # Batch mode assumes combination
            combine_flag = True
            write_report("Batch mode: Proceeding with combining daily files.")

        if combine_flag:
            write_report("--- Starting Combination Phase ---")
            for broker, typee, symbol in selected_symbols_info:
                 symbol_id = f"{broker}_{typee}_{symbol}"
                 if symbol_id in processed_daily_files and processed_daily_files[symbol_id]:
                     write_report(f"Combining files for {symbol_id}...")
                     try:
                         newly_combined = combine_orderbooks(broker, typee, symbol, window_days=3)
                         newly_combined_files_all.extend(newly_combined)
                     except Exception as e:
                         logging.error(f"Unexpected error combining files for {symbol_id}: {e}", exc_info=True)
                         write_report(f"ERROR: Unexpected failure combining files for {symbol_id}. See logs for details.")

                 else:
                     write_report(f"No processed daily files found for {symbol_id} to combine.")
            write_report(f"Combination phase finished. Created {len(newly_combined_files_all)} new combined files.")
        else:
            newly_combined_files_all = [] # Ensure it's empty if skipping combine


        # --- 6. Load Combined Files to DB (Optional based on mode/config) ---
        load_db_flag = False
        if newly_combined_files_all: # Only ask/proceed if there are files to load
            if interactive:
                load_choice = input(f"\nLoad the {len(newly_combined_files_all)} newly combined files into the database? (y/n): ").strip().lower()
                if load_choice == 'y':
                    load_db_flag = True
                    write_report("User chose to load combined files into DB.")
                else:
                    write_report("User chose not to load combined files into DB.")
            else: # Batch mode assumes loading
                load_db_flag = True
                write_report("Batch mode: Proceeding with loading combined files into DB.")

            if load_db_flag:
                 write_report("--- Starting Database Loading Phase ---")
                 try:
                     # Create DB engine once for the loading phase
                     db_engine = create_engine(DB_CONN_STRING)
                     with db_engine.connect() as connection: # Test connection early
                         logging.info("Database engine created and connection tested successfully for loading.")
                     load_into_db(newly_combined_files_all, table_name="orderbooks", db_engine=db_engine)
                 except Exception as e:
                     logging.error(f"Failed to initialize database connection or load data: {e}", exc_info=True)
                     write_report(f"ERROR: Database loading failed. Reason: {e}")
                 finally:
                    if db_engine:
                        try:
                            db_engine.dispose()
                            logging.info("Database engine disposed after loading.")
                        except Exception as e:
                            logging.error("Error disposing database engine after loading: %s", e)
            else:
                 write_report("Skipping database loading phase.")
        else:
             write_report("No new combined files to load into the database.")


        # --- 7. Trigger Inspection ---
        trigger_inspection()

    except (paramiko.AuthenticationException, paramiko.SSHException, TimeoutError) as conn_e:
        logging.critical("SSH/SFTP Connection failed: %s", conn_e, exc_info=True)
        write_report(f"CRITICAL ERROR: Could not establish SSH/SFTP connection: {conn_e}")
        # No point continuing if connection failed
    except Exception as e:
        logging.critical("An unexpected critical error occurred during the pipeline run: %s", e, exc_info=True)
        write_report(f"CRITICAL ERROR: Pipeline run failed unexpectedly. Error: {e}")
        # Log stack trace for debugging

    finally:
        # --- 8. Cleanup ---
        logging.debug("Closing SFTP and SSH client if open.")
        if sftp:
            try: sftp.close()
            except Exception as e: logging.error("Error closing SFTP session: %s", e)
        if ssh_client:
            try: ssh_client.close()
            except Exception as e: logging.error("Error closing SSH client: %s", e)
        write_report("Closed SFTP/SSH connection (if established).")

        # Clean entire staging directory? Or keep for debugging?
        # Let's clean it to avoid buildup. Individual date folders are cleaned after processing.
        try:
            if os.path.isdir(LOCAL_STAGING_DIR):
                 # Check age or be careful if multiple processes might use it
                 # shutil.rmtree(LOCAL_STAGING_DIR)
                 # write_report(f"Cleaned base staging directory: {LOCAL_STAGING_DIR}")
                 pass # Decided against auto-cleaning base staging for now.
        except Exception as e:
            logging.warning("Could not clean base staging directory '%s': %s", LOCAL_STAGING_DIR, e)

        end_run_time = time.time()
        run_duration = end_run_time - start_run_time
        write_report(f"Pipeline Run Finished. Total duration: {timedelta(seconds=run_duration)}")
        write_report("=" * 80 + "\n")


def run_pipeline_continuous(poll_interval):
    """
    Runs the processing pipeline in continuous mode, polling for new data.
    Uses non-interactive mode for run_pipeline_once.
    """
    write_report(f"Starting continuous pipeline mode. Poll interval: {poll_interval} seconds.")
    while True:
        run_start_time = time.time()
        write_report("--- Continuous Mode: Starting new processing cycle ---")
        try:
            # Run the pipeline in non-interactive mode
            run_pipeline_once(interactive=False)
        except Exception as e:
            # Catch errors that might escape run_pipeline_once's final handler
            logging.error("Unhandled error in continuous pipeline loop: %s", e, exc_info=True)
            write_report(f"ERROR: Unhandled exception in continuous cycle: {e}")

        run_end_time = time.time()
        cycle_duration = run_end_time - run_start_time
        write_report(f"--- Continuous Mode: Processing cycle finished in {timedelta(seconds=cycle_duration)} ---")

        # Calculate sleep time, ensuring it's not negative if cycle took longer than interval
        sleep_time = max(0, poll_interval - cycle_duration)
        if sleep_time > 0:
            write_report(f"Sleeping for {timedelta(seconds=sleep_time)} before next cycle.")
            time.sleep(sleep_time)
        else:
            write_report("Processing cycle took longer than poll interval. Starting next cycle immediately.")


# ------------------------
# Main Execution
# ------------------------
def main():
    parser = argparse.ArgumentParser(description="Billions Orderbook Processing Pipeline")
    parser.add_argument("--mode", choices=["once", "continuous"], default="once",
                        help="Operation mode: 'once' (interactive by default) or 'continuous' (batch).")
    parser.add_argument("--batch", action="store_true",
                        help="Run in 'once' mode but non-interactively (batch). Overrides interactive prompts.")
    parser.add_argument("--poll-interval", type=int, default=300,
                        help="Polling interval in seconds for 'continuous' mode (default: 300).")
    parser.add_argument("--debug", action="store_true",
                        help="Enable DEBUG level logging.")

    args = parser.parse_args()

    # Adjust log level if debug flag is set
    if args.debug:
        logging.getLogger().setLevel(logging.DEBUG)
        for handler in logging.getLogger().handlers:
            handler.setLevel(logging.DEBUG)
        logging.debug("DEBUG logging enabled.")

    # Ensure base directories exist
    try:
        os.makedirs(LOCAL_OUTPUT_BASE_DIR, exist_ok=True)
        os.makedirs(LOCAL_STAGING_DIR, exist_ok=True)
        os.makedirs(COMBINED_OUTPUT_DIR, exist_ok=True)
        logging.info("Checked/created necessary local directories.")
    except OSError as e:
        logging.critical(f"Cannot create essential local directories: {e}. Exiting.")
        return # Cannot run without local directories

    if args.mode == "once":
        # Run once, interactive unless --batch is specified
        run_pipeline_once(interactive=not args.batch)
    elif args.mode == "continuous":
        if args.batch:
            logging.warning("--batch flag has no effect in 'continuous' mode (always runs non-interactively).")
        run_pipeline_continuous(args.poll_interval)

if __name__ == "__main__":
    main()