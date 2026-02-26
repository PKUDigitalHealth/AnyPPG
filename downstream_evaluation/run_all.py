import torch.multiprocessing as mp
import numpy as np
import pandas as pd
from pathlib import Path
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import Dict

from extract_emb import process_single_task_on_gpu
from run_eval import run_linear_probe_solver, update_excel, TASK_TYPE_MAP

RESULTS_ROOT = Path('./results')
OUTPUT_EXCEL = RESULTS_ROOT / "LP_Result.xlsx"
EMBED_SAVE_ROOT = './eval_embeddings'

# 模型配置
MODELS = [
    {'name': 'clip', 'ckpt': './eval_models/clip/checkpoint.pth', 'sr': 125, 'duration': 10},    
]

DATA_ROOT_BASE = './preprocessed_datasets'

DATASETS = [
    {
        'root': f'{DATA_ROOT_BASE}/realworldppg',
        'tasks': [
            {'name': 'IdentifyData', 'signal': 'PPG', 'label': 'ID'},
        ]
    },
    {
        'root': f'{DATA_ROOT_BASE}/butppg',
        'tasks': [
            {'name': 'DBPData', 'signal': 'PPG', 'label': 'DBP'},
            {'name': 'HRData',  'signal': 'PPG', 'label': 'HR'},
            {'name': 'SBPData', 'signal': 'PPG', 'label': 'SBP'},
            {'name': 'SQIData', 'signal': 'PPG', 'label': 'Quality'},
        ]
    },
    {
        'root': f'{DATA_ROOT_BASE}/dalia',
        'tasks': [{'name': 'HRData', 'signal': 'PPG', 'label': 'HR'}]
    },
    {
        'root': f'{DATA_ROOT_BASE}/gyro_acc',
        'tasks': [{'name': 'HRData', 'signal': 'PPG', 'label': 'HR'}]
    },
    {
        'root': f'{DATA_ROOT_BASE}/ucibp',
        'tasks': [
            {'name': 'DBPData', 'signal': 'PPG', 'label': 'DBP'},
            {'name': 'SBPData', 'signal': 'PPG', 'label': 'SBP'},
        ]
    },
    {
        'root': f'{DATA_ROOT_BASE}/weasd',
        'tasks': [
            {'name': 'EmotionData', 'signal': 'PPG', 'label': 'Emotion'},
            {'name': 'StressData', 'signal': 'PPG', 'label': 'Stress'},
        ]
    },
    {
        'root': f'{DATA_ROOT_BASE}/deepbeat',
        'tasks': [
            {'name': 'AFData', 'signal': 'PPG', 'label': 'AF'},
        ]           
    },
]

AVAILABLE_GPUS = [0, 1, 2, 3, 4, 5, 6, 7]
BATCH_SIZE = 2056
NUM_WORKERS_PER_LOADER = 8

def worker_process_task(gpu_id: int, job_info: Dict):
    model_cfg = job_info['model']
    ds_cfg = job_info['ds']
    task_cfg = job_info['task']
    
    try:
        process_single_task_on_gpu(
            gpu_id=gpu_id,
            model_cfg=model_cfg,
            ds_cfg=ds_cfg,
            task_cfg=task_cfg,
            batch_size=BATCH_SIZE,
            num_workers=NUM_WORKERS_PER_LOADER,
            save_root=EMBED_SAVE_ROOT
        )
    except Exception as e:
        return {'status': 'failed', 'stage': 'extract', 'error': str(e), 'job': job_info}

    ds_name = Path(ds_cfg['root']).stem
    task_name = task_cfg['name']
    embed_dir = Path(EMBED_SAVE_ROOT) / model_cfg['name'] / ds_name / task_name
    
    task_type = TASK_TYPE_MAP.get(task_name, 'classification')
    
    try:
        metrics, raw_preds = run_linear_probe_solver(embed_dir, task_type)
        
        if metrics is None:
            return {'status': 'failed', 'stage': 'eval', 'error': 'Metrics is None', 'job': job_info}
            
        eval_out_dir = embed_dir / "eval_outputs"
        eval_out_dir.mkdir(exist_ok=True)
        np.savez(
            eval_out_dir / "predictions.npz",
            y_true=raw_preds['y_true'],
            y_pred=raw_preds['y_pred'],
            y_prob=raw_preds['y_prob'] if raw_preds['y_prob'] is not None else []
        )
        
        return {
            'status': 'success',
            'job': job_info,
            'metrics': metrics,
            'dataset': ds_name,
            'task': task_name,
            'task_type': task_type,
            'model_name': model_cfg['name']
        }

    except Exception as e:
        return {'status': 'failed', 'stage': 'eval', 'error': str(e), 'job': job_info}


def main():
    mp.set_start_method('spawn', force=True)
    
    task_queue = []
    for model_cfg in MODELS:
        for ds_cfg in DATASETS:
            for task_cfg in ds_cfg['tasks']:
                task_queue.append({
                    'model': model_cfg,
                    'ds': ds_cfg,
                    'task': task_cfg
                })
    
    print(f"Total tasks: {len(task_queue)}")
    print(f"Available GPUs: {AVAILABLE_GPUS}")
    print(f"Results will be saved to: {OUTPUT_EXCEL}")

    with ProcessPoolExecutor(max_workers=len(AVAILABLE_GPUS)) as executor:
        futures = []
        for i, job in enumerate(task_queue):
            gpu_id = AVAILABLE_GPUS[i % len(AVAILABLE_GPUS)]
            
            f = executor.submit(worker_process_task, gpu_id, job)
            futures.append(f)
        
        for f in tqdm(as_completed(futures), total=len(futures), desc="Overall Progress"):
            result = f.result()
            
            if result['status'] == 'success':
                job = result['job']
                sheet_key = f"{result['dataset']}_{result['task']}"
                
                row = {
                    "Model": result['model_name'],
                    "Dataset": result['dataset'],
                    "Task_Type": result['task_type'],
                    **result['metrics']
                }
                
                try:
                    update_excel(OUTPUT_EXCEL, row, sheet_key)
                    tqdm.write(f"[Success] {result['model_name']} on {sheet_key} -> Excel Updated")
                except Exception as e:
                    tqdm.write(f"[Error] Failed to write Excel: {e}")
            
            else:
                job_desc = f"{result['job']['model']['name']} on {result['job']['ds']['root']} - {result['job']['task']['name']}"
                tqdm.write(f"[Fail] {job_desc} at stage '{result['stage']}': {result['error']}")

if __name__ == '__main__':
    main()
