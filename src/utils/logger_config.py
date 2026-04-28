import os
import sys
import logging
logging.getLogger("mlflow").setLevel(logging.ERROR)
from datetime import datetime
from dotenv import load_dotenv

# Load environment variables from the src folder
dotenv_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '.env'))

if os.path.exists(dotenv_path):
    load_dotenv(dotenv_path, override=True)
else:
    print(f"Warning: .env file not found at {dotenv_path}")

debug_env = os.getenv("DEBUG")
if debug_env is not None:
    DEBUG = str(debug_env).strip().lower() == "true"
else:
    DEBUG = False

log_dir = os.getenv("LOG_DIR", None)

# Determine project root and logs directory
root_dir = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
logs_dir = log_dir if log_dir else os.path.join(root_dir, "logs")
os.makedirs(logs_dir, exist_ok=True)


def configure_logger(debug_mode=False):
    """
    Configures the logger. If DEBUG is True, sets level to DEBUG, else INFO.
    """
    level = logging.DEBUG if debug_mode else logging.INFO
    log_suffix = "_debug" if debug_mode else ""
    log_filename = datetime.now().strftime(f"QNA_BATCH{log_suffix}_%Y%m%d_%H%M%S.LOG")
    log_path = os.path.join(logs_dir, log_filename)
    
    # Remove all handlers associated with the root logger object.
    for handler in logging.root.handlers[:]:
        logging.root.removeHandler(handler)
    
    # Attach handlers to the named logger
    logger = logging.getLogger("gpd_ai_companion_qna_batch")
    logger.handlers.clear()  # Remove existing handlers from named logger
    logger.propagate = False  # Prevent double logging via root logger
    
    # Create handlers
    file_handler = logging.FileHandler(log_path)
    stream_handler = logging.StreamHandler(sys.stdout)
    
    # Create and set formatter
    formatter = logging.Formatter(
        fmt="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )
    file_handler.setFormatter(formatter)
    stream_handler.setFormatter(formatter)
    
    # Add handlers to the logger
    handlers = [file_handler, stream_handler]
    for h in handlers:
        logger.addHandler(h)
    
    logger.setLevel(level)
    return logger
# omit mlflow warning.
logging.getLogger("mlflow").setLevel(logging.ERROR)
# Create a module-level logger for import
logger = configure_logger(debug_mode=DEBUG)
logger.setLevel(logging.DEBUG if DEBUG else logging.INFO)
logger.info(f"Debug mode is {'enabled' if DEBUG else 'disabled'}")
logger.info(f"Logger configured. Log files will be saved in: {logs_dir}")
