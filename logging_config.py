import logging
import logging.config
import os
import sys # Needed for sys.argv, sys.stdout, sys.stderr

# --- Determine Directories ---
try:
    # __file__ gives the path of the *current* file (logging_setup.py)
    # We want the logs relative to *this* file's location, or a fallback.
    SETUP_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
except NameError:
    SETUP_SCRIPT_DIR = os.path.abspath(".") # Fallback to current working directory

LOG_DIR = os.path.join(SETUP_SCRIPT_DIR, "logs") # Logs stored relative to the setup script

# --- Ensure the central log directory exists ---
if not os.path.exists(LOG_DIR):
    try:
        os.makedirs(LOG_DIR)
        print(f"Created log directory: {LOG_DIR}")
    except OSError as e:
        print(f"Error creating log directory {LOG_DIR}: {e}", file=sys.stderr)
        LOG_DIR = "." # Fallback to current working directory
        print(f"Warning: Log directory creation failed. Using current directory: {LOG_DIR}", file=sys.stderr)

# --- Define BASE LOGGING_CONFIG (filename will be added dynamically) ---
LOGGING_CONFIG = {
    "version": 1,
    "disable_existing_loggers": False,
    "formatters": {
        "standard": {
            "format": "%(asctime)s - %(levelname)-8s - %(name)-15s - %(filename)s:%(lineno)d - %(message)s",
            "datefmt": "%Y-%m-%d %H:%M:%S",
        },
    },
    "handlers": {
        "console": {
            "level": "INFO",
            "class": "logging.StreamHandler",
            "formatter": "standard",
            "stream": "ext://sys.stdout",
        },
        # Renamed handler for clarity
        "file_script_name": {
            "level": "DEBUG",
            "class": "logging.FileHandler", # Using FileHandler, will append by default
            "formatter": "standard",
            "filename": None, # Placeholder - MUST be set before dictConfig
            "encoding": "utf-8",
            # 'mode': 'a' (append) is the default for FileHandler
        },
    },
    "loggers": {
        "": { # Root logger
            "handlers": ["console", "file_script_name"], # Updated handler name
            "level": "DEBUG",
        },
        "httpx": {
            "level": "WARNING"
        }
    },
}

def setup_logging():
    """
    Configures logging using dictConfig, naming the log file after the
    executed script (from sys.argv[0]). Appends to the log on subsequent runs.
    """
    log_filepath = None # Initialize for use in error handling
    try:
        # --- Determine log filename based on the executed script ---
        script_name_base = "unknown_script" # Default fallback
        if hasattr(sys, 'argv') and sys.argv and sys.argv[0]:
            # sys.argv[0] is usually the path to the script executed
            main_script_path = os.path.abspath(sys.argv[0])
            script_name_base_ext = os.path.basename(main_script_path)
            # Get name without extension
            script_name_base = os.path.splitext(script_name_base_ext)[0]
            # Handle cases like running interactively where name might be empty or odd
            if not script_name_base or script_name_base == "-c":
                 script_name_base = "interactive_session"
        else:
             # Fallback if sys.argv is not available or empty
             script_name_base = "unknown_entry_point"

        log_filename = f"{script_name_base}.log"
        log_filepath = os.path.join(LOG_DIR, log_filename)

        # --- Update config with the determined filename ---
        # Ensure the handler name ('file_script_name') matches the key below
        LOGGING_CONFIG['handlers']['file_script_name']['filename'] = log_filepath

        # --- Apply the configuration ---
        logging.config.dictConfig(LOGGING_CONFIG)

        # --- Confirmation Log ---
        confirm_logger = logging.getLogger(__name__)
        actual_log_file = LOGGING_CONFIG['handlers']['file_script_name'].get('filename', 'Unknown')
        confirm_logger.info(f"Logging setup complete. Log file: {actual_log_file} (appending)")

    except Exception as e:
        # More robust fallback logging if setup fails catastrophically
        fallback_log_path = os.path.join(LOG_DIR, "logging_setup_error.log")
        try:
            # Use basicConfig for the fallback - logs the exception too
            logging.basicConfig(level=logging.ERROR, filename=fallback_log_path, filemode='a',
                                format='%(asctime)s - %(levelname)s - Error during dictConfig setup: %(message)s')
            logging.exception(f"CRITICAL: Failed to configure logging using dictConfig! Error: {e}. Check '{fallback_log_path}'. Attempted primary log file: {log_filepath or 'Not determined'}")
            # Also print to stderr as console handler might not be working
            print(f"CRITICAL: Failed to configure logging using dictConfig. Check '{fallback_log_path}'. Error: {e}", file=sys.stderr)
        except Exception as basic_config_e:
            # If even basicConfig fails, just print
             print(f"CRITICAL: Failed to configure logging using dictConfig AND basicConfig failed. DictConfig error: {e}. BasicConfig error: {basic_config_e}", file=sys.stderr)


# --- How to use it ---
# 1. Save this code (e.g., `logging_setup.py`).
# 2. In your main script (e.g., `my_program.py`):
#    import logging
#    from logging_setup import setup_logging
#
#    # Call setup early, before any other logging
#    setup_logging()
#
#    # Get loggers and use them
#    logger = logging.getLogger(__name__) # Good practice
#    logger.info("Starting my program...")
#    # ... rest of your program ...

# Example execution if run directly (will create 'logging_setup.log' or similar)
if __name__ == "__main__":
    print(f"Attempting to set up logging. Log directory target: {LOG_DIR}")
    setup_logging()

    logger = logging.getLogger("main_execution_block") # Example logger name

    logger.debug("This is a DEBUG message (goes to file).")
    logger.info("This is an INFO message (goes to file).")
    logger.warning("This is a WARNING message (goes to console and file).")
    logger.error("This is an ERROR message (goes to console and file).")
    logger.critical("This is a CRITICAL message (goes to console and file).")

    print(f"\nCheck the '{LOG_DIR}' directory for a log file named after this script (e.g., '{os.path.splitext(os.path.basename(sys.argv[0]))[0]}.log').")
    print("Run this script again to see messages appended to the same file.")