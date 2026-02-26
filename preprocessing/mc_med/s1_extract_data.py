#!/usr/bin/env python3
"""
MCMED Data Processing Main Program - Enhanced Logging Version
Automatically capture output and exceptions for all MCMEDExtractor processing steps
"""

import logging
import sys
from pathlib import Path
from functools import wraps
from typing import Callable, Any
from data_extractor import MCMEDExtractor

# ==================== Logging Decorator ====================
def log_method(func: Callable) -> Callable:
    """Decorator to automatically record method call arguments and results"""
    @wraps(func)
    def wrapper(*args, **kwargs):
        class_name = args[0].__class__.__name__ if args else func.__module__
        logging.info(f"Calling {class_name}.{func.__name__} (Args: {kwargs if kwargs else args[1:]})")
        try:
            result = func(*args, **kwargs)
            logging.info(f"{func.__name__} executed successfully")
            return result
        except Exception as e:
            logging.error(f"{func.__name__} failed: {str(e)}", exc_info=True)
            raise
    return wrapper

# ==================== Logging Configuration ====================
def setup_logging(log_dir = "./preprocessing/mc_med/logs") -> None:
    """Configure dual-output logging system"""
    log_dir = Path(log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)
    
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    
    # File handler
    file_handler = logging.FileHandler(log_dir / f"{Path(__file__).stem}.log", mode="w")
    file_handler.setFormatter(logging.Formatter(
        "%(asctime)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    ))
    
    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(logging.Formatter(
        "%(levelname)s - %(message)s"
    ))
    
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)

# ==================== Main Program ====================
def main() -> None:
    """Enhanced logging main flow"""
    # Configuration parameters
    # config = {
    #     "dataset_dir": "/wujidata/ngk/PPGFounder/datasets/MC-MED/data/waveforms",
    #     "save_dir": "/wujidata/ngk/PPGFounder/datasets/MC-MED/MCMED_Patient_Level",
    #     "num_workers": 16,
    # }

    config = {
        "dataset_dir": "/path/to/datasets/MC-MED/data/waveforms",
        "save_dir": "./preprocessing/mc_med/MCMED_Patient_Level_PPG",
        "num_workers": 16,
    }
    
    try:
        logging.info("=== Start Processing ===")
        logging.info(f"Config: {config}")
        
        # # Dynamically add logging decorator to MCMEDExtractor
        # for name, method in vars(MCMEDExtractor).items():
        #     if callable(method) and not name.startswith("_"):
        #         setattr(MCMEDExtractor, name, log_method(method))
        
        # # Execute processing
        # extractor = MCMEDExtractor(**{k: config[k] for k in ['dataset_dir', 'save_dir']})
        # extractor.extract_whole_dataset_sync_ppg_ecg(
        #     num_workers=config["num_workers"],
        # )

        # Execute processing
        extractor = MCMEDExtractor(**{k: config[k] for k in ['dataset_dir', 'save_dir']})
        extractor.extract_whole_dataset_ppg(
            num_workers=config["num_workers"],
        )
        
        logging.info("=== Processing Completed ===")
    except Exception as e:
        logging.critical(f"Main program failed: {str(e)}", exc_info=True)
        sys.exit(1)

if __name__ == "__main__":
    setup_logging()
    main()