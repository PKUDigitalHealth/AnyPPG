#!/usr/bin/env python3
"""
MESA Data Processing Main Program - Enhanced Logging Version
Automatically capture output and exceptions for all MESAProcessor processing steps
"""

import logging
import sys
from pathlib import Path
from functools import wraps
from typing import Callable, Any
from data_processor import MESAProcessor

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
def setup_logging(log_dir = "./preprocessing/mesa/logs") -> None:
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
    config = {
        "dataset_dir": './preprocessing/mesa/MESA_Patient_Level',
        "seg_duration": 10,
        "overlap_ratio": 0,
        "save_dir": './preprocessing/mesa/Proc_MESA_Patient_Level',
        "num_workers": 32,
        "pattern": '*.npz'
    }
    
    try:
        logging.info("=== Start Processing ===")
        logging.info(f"Config: {config}")
        
        # # Dynamically add logging decorator to MESAProcessor
        # for name, method in vars(MESAProcessor).items():
        #     if callable(method) and not name.startswith("_"):
        #         setattr(MESAProcessor, name, log_method(method))
        
        # Execute processing
        processor = MESAProcessor(**{k: config[k] for k in config.keys() if k != 'pattern'})
        processor.process_whole_dataset(
            pattern=config['pattern']
        )
        
        logging.info("=== Processing Completed ===")
    except Exception as e:
        logging.critical(f"Main program failed: {str(e)}", exc_info=True)
        sys.exit(1)

if __name__ == "__main__":
    setup_logging()
    main()