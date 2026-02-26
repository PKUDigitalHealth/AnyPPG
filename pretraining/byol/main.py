import os
import argparse
import sys
from omegaconf import OmegaConf
from transformers import get_scheduler
from torch.utils.data import DataLoader, ConcatDataset, RandomSampler

# Local imports
from framework import (
    BYOL, 
    PPGEncoderConfig, 
)
from trainer import Trainer
from dataset import ChunkDataset
from utils import get_logger, create_optimizer

def parse_args():
    parser = argparse.ArgumentParser(
        prog='PPG-BYOL',
        description='Pretraining PPG BYOL',
    )
    
    parser.add_argument('--memo', default='default_run', type=str, help='Experiment memo')
    parser.add_argument('--config_path', default=None, type=str, help='Config path (optional)')
    # Dataset args
    parser.add_argument('--data_dir', default='./datasets', type=str, help='Root directory for datasets')
    parser.add_argument('--datasets', nargs='+', default=['pulsedb'], help='List of datasets to use')
    
    # Training args
    parser.add_argument('--save_log_dir', default='./logs', type=str, help='Log directory')
    parser.add_argument('--save_ckpt_dir', default='./checkpoints', type=str, help='Checkpoint directory')
    parser.add_argument('--save_result_dir', default='./results', type=str, help='Result directory')
    parser.add_argument('--save_ckpt_prefix', default='checkpoint', type=str, help='Prefix for checkpoints')
    
    parser.add_argument('--log_interval', default=10, type=int, help='Log interval (steps)')
    parser.add_argument('--save_interval', default=500, type=int, help='Save interval (steps)')
    
    parser.add_argument('--epochs', default=50, type=int, help='Number of epochs')
    parser.add_argument('--batch_size', default=2, type=int, help='Batch size')
    parser.add_argument('--lr', default=1e-4, type=float, help='Learning rate')
    parser.add_argument('--weight_decay', default=1e-4, type=float, help='Weight decay')
    
    parser.add_argument('--resume_from_ckpt', action='store_true', help='Resume from checkpoint')
    parser.add_argument('--ckpt_path', default=None, type=str, help='Checkpoint path to resume from')
    
    parser.add_argument('--train', action='store_true', default=True, help='Run training')
    parser.add_argument('--mixed_precision', default='no', type=str, choices=['no', 'fp16', 'bf16'], help='Mixed precision training')
    
    # Model args
    parser.add_argument('--projection_size', default=128, type=int, help='Projection size')
    parser.add_argument('--hidden_size', default=512, type=int, help='Hidden size for MLP')
    parser.add_argument('--moving_average_decay', default=0.99, type=float, help='EMA decay for target network')
    
    # WandB args
    parser.add_argument('--wandb_project', default='ppg-byol', type=str, help='WandB project name')
    parser.add_argument('--wandb_entity', default=None, type=str, help='WandB entity')
    
    args = parser.parse_args()
    return args

def get_default_config():
    """Returns a default OmegaConf configuration."""
    conf = OmegaConf.create({
        'dataset': {
            'chunk_size': 512, # Match model input
        },
        'dataloader': {
            'num_workers': 8,
            'pin_memory': True,
        },
        'training': {
            'optimizer': 'adamw',
            'scheduler': 'cosine',
            'num_warmup_steps': 5000,
            'gradient_accumulation_steps': 1,
            'clip_grad_norm': True,
            'max_grad_norm': 1.0,
            'seed': 42,
        },
        'model': {
            'projection_size': 256,
            'hidden_size': 1024,
            'moving_average_decay': 0.99,
        }
    })
    return conf

def main():
    args = parse_args()
    
    # Load default config and merge with file config if provided
    cfg = get_default_config()
    if args.config_path and os.path.exists(args.config_path):
        file_cfg = OmegaConf.load(args.config_path)
        cfg = OmegaConf.merge(cfg, file_cfg)
    
    # Merge model args into config
    cfg.model.projection_size = args.projection_size
    cfg.model.hidden_size = args.hidden_size
    cfg.model.moving_average_decay = args.moving_average_decay

    # Merge args into config for logging
    cfg = OmegaConf.merge(cfg, {'Arguments': vars(args)})

    # Dataset Setup
    # Hardcoded dataset paths from base.yaml
    DATASET_PATHS = {
        'pulsedb': './preprocessing/pulsedb/Chunk_PulseDB',
        'mesa': './preprocessing/mesa/Chunk_MESA',
        'hsp': './preprocessing/hsp/Chunk_HSP',
        'mcmed': './preprocessing/mc_med/Chunk_MCMED',
        'cfs': './preprocessing/cfs/Chunk_CFS'
    }

    train_ds_list = []
    valid_ds_list = []
    test_ds_list = []
    
    for ds_name in args.datasets:
        if ds_name not in DATASET_PATHS:
            print(f"Warning: Dataset {ds_name} not defined in DATASET_PATHS. Skipping.")
            continue
            
        ds_path = DATASET_PATHS[ds_name]
        
        # Check if path exists
        if not os.path.exists(ds_path):
             print(f"Warning: Dataset directory {ds_path} not found. Skipping.")
             continue
             
        try:
            # Check for train directory
            if os.path.exists(os.path.join(ds_path, 'train')):
                train_ds_list.append(ChunkDataset(ds_path, split='train', dataset=ds_name, norm_data=True))
            else:
                print(f"Warning: 'train' directory not found in {ds_path}. Skipping.")

            # Optional: Add valid/test if they exist
            if os.path.exists(os.path.join(ds_path, 'valid')):
                valid_ds_list.append(ChunkDataset(ds_path, split='valid', dataset=ds_name, norm_data=True))
            if os.path.exists(os.path.join(ds_path, 'test')):
                test_ds_list.append(ChunkDataset(ds_path, split='test', dataset=ds_name, norm_data=True))
        except Exception as e:
            print(f"Error loading dataset {ds_name}: {e}")

    if not train_ds_list:
        print("No valid datasets found. Exiting.")
        return

    train_ds = ConcatDataset(train_ds_list)
    valid_ds = ConcatDataset(valid_ds_list) if valid_ds_list else None
    test_ds = ConcatDataset(test_ds_list) if test_ds_list else None

    print(f"Train dataset size: {len(train_ds)}")
    if valid_ds: print(f"Valid dataset size: {len(valid_ds)}")

    sampler = RandomSampler(train_ds)
    collate_fn = ChunkDataset.collate_fn
    
    batch_size = args.batch_size
    num_workers = cfg.dataloader.num_workers
    pin_memory = cfg.dataloader.pin_memory

    train_dl = DataLoader(train_ds, batch_size=batch_size, num_workers=num_workers, 
                          pin_memory=pin_memory, sampler=sampler, collate_fn=collate_fn)
    
    valid_dl = DataLoader(valid_ds, batch_size=batch_size, num_workers=num_workers, 
                          pin_memory=pin_memory, collate_fn=collate_fn) if valid_ds else None
    
    test_dl = DataLoader(test_ds, batch_size=batch_size, num_workers=num_workers, 
                         pin_memory=pin_memory, collate_fn=collate_fn) if test_ds else None

    # --- Setup Directories ---
    store_name = f"{args.memo}_{cfg.training.optimizer}_{args.lr}_{batch_size}"
    log_dir = os.path.join(args.save_log_dir, store_name)
    os.makedirs(log_dir, exist_ok=True)
    
    # Save full config
    OmegaConf.save(config=cfg, f=os.path.join(log_dir, 'full_config.yaml'), resolve=True)
    
    logger = get_logger(log_dir)
    logger.info("=== Configuration ===")
    logger.info(OmegaConf.to_yaml(cfg=cfg))

    # --- Model Setup ---
    logger.info("Initializing BYOL Model...")
    ppg_conf = PPGEncoderConfig() # Use defaults or map from cfg if needed
    
    model = BYOL(
        ppg_encoder_config=ppg_conf,
        projection_size=cfg.model.projection_size,
        hidden_size=cfg.model.hidden_size,
        moving_average_decay=cfg.model.moving_average_decay
    )
    
    # --- Optimizer & Scheduler ---
    # Only optimize online parameters
    optimizer = create_optimizer(
        cfg.training.optimizer, 
        filter(lambda p: p.requires_grad, model.parameters()), 
        lr=args.lr, 
        weight_decay=args.weight_decay
    )

    scheduler = get_scheduler(
        name=cfg.training.scheduler, 
        optimizer=optimizer, 
        num_warmup_steps=cfg.training.num_warmup_steps, 
        num_training_steps=args.epochs * len(train_dl)
    )

    # --- Trainer ---
    trainer = Trainer(
        model=model,
        optimizer=optimizer,
        epochs=args.epochs,
        logger=logger,
        store_name=store_name,
        save_ckpt_dir=args.save_ckpt_dir,
        save_result_dir=args.save_result_dir,
        train_loader=train_dl,
        valid_loader=valid_dl,
        test_loader=test_dl,
        log_interval=args.log_interval,
        save_interval=args.save_interval,
        save_ckpt_prefix=args.save_ckpt_prefix,
        scheduler=scheduler,
        resume_from_ckpt=args.resume_from_ckpt,
        gradient_accumulation_steps=cfg.training.gradient_accumulation_steps,
        clip_grad_norm=cfg.training.clip_grad_norm,
        max_grad_norm=cfg.training.max_grad_norm,
        seed=cfg.training.seed,
        ckpt_path=args.ckpt_path,
        wandb_project=args.wandb_project,
        wandb_entity=args.wandb_entity,
        config=OmegaConf.to_container(cfg, resolve=True),
        mixed_precision=args.mixed_precision
    )
    
    if args.train:
        trainer.train()

if __name__ == '__main__':
    main()
