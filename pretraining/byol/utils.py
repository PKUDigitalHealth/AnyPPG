import logging
import sys
import torch
from torch import optim

def get_logger(log_dir):
    logger = logging.getLogger('train')
    logger.setLevel(logging.INFO)
    
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    
    # File handler
    fh = logging.FileHandler(f'{log_dir}/train.log')
    fh.setLevel(logging.INFO)
    fh.setFormatter(formatter)
    
    # Stream handler
    sh = logging.StreamHandler(sys.stdout)
    sh.setLevel(logging.INFO)
    sh.setFormatter(formatter)
    
    logger.addHandler(fh)
    logger.addHandler(sh)
    return logger

def create_optimizer(name, parameters, lr, weight_decay=0.0, **kwargs):
    if name == 'adam':
        return optim.Adam(parameters, lr=lr, weight_decay=weight_decay, **kwargs)
    elif name == 'adamw':
        return optim.AdamW(parameters, lr=lr, weight_decay=weight_decay, **kwargs)
    elif name == 'sgd':
        return optim.SGD(parameters, lr=lr, weight_decay=weight_decay, **kwargs)
    else:
        raise ValueError(f"Optimizer {name} not supported")
