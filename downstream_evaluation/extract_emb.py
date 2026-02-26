from pathlib import Path
import torch
import numpy as np
import torch.nn as nn
from tqdm import tqdm
from get_model import get_model
from torch.utils.data import TensorDataset, DataLoader
from concurrent.futures import ProcessPoolExecutor, as_completed
from scipy.signal import resample
import torch.multiprocessing as mp
from typing import Tuple, List, Dict
from numpy.typing import NDArray


def read_known_npz_data(path: str, signal_key: str, label_key: str) -> Tuple[NDArray[np.float32], NDArray[np.float32]]:
    data = np.load(path)
    sigs = data[signal_key]
    labels = data[label_key]
    
    if labels.ndim == 0:
        labels = np.array([labels.item()])
    
    return sigs, labels


def extract_split_embeddings(
    model_name: str,
    model: nn.Module,
    dataset_dir: Path,
    data_name: str,
    split: str,
    signal_key: str,
    label_key: str,
    batch_size: int,
    num_workers: int,
    device: torch.device,
    save_dir: Path,
    target_sr: int = 125,
    target_duration: int = 10,
    progress_bar: tqdm = None
):
    target_data_path = dataset_dir / data_name / f'{split}.npz'
    
    sigs, labels = None, None

    if not target_data_path.exists():
        target_data_dir = dataset_dir / data_name / split
        if not target_data_dir.exists():
            if progress_bar: progress_bar.write(f"Skipping {target_data_path}: Not found.")
            return

        data_paths = list(target_data_dir.glob('*.npz'))
        if progress_bar: progress_bar.write(f"Loading {len(data_paths)} files from {target_data_dir}...")
        
        sigs_list, labels_list = [], []
        for p in data_paths:
            s, l = read_known_npz_data(str(p), signal_key, label_key)
            if len(s) > 0:
                s = s.squeeze()
                if len(s) > 0:
                    sigs_list.append(s)
                    labels_list.append(l)
        
        if not sigs_list: return
        sigs = np.concatenate(sigs_list, axis=0)
        labels = np.concatenate(labels_list, axis=0)
    else:
        if progress_bar: progress_bar.write(f"Loading {target_data_path}...")
        sigs, labels = read_known_npz_data(str(target_data_path), signal_key, label_key)

    if sigs.ndim == 2:
        sigs = sigs[:, np.newaxis, :]
    
    target_len = int(target_sr * target_duration)
    current_len = sigs.shape[-1]
    
    if current_len != target_len:
        if progress_bar: progress_bar.write(f"  -> Resampling: {current_len} -> {target_len} (SR: {target_sr}, Dur: {target_duration}s)")
        sigs = resample(sigs, target_len, axis=-1)
    
    tensor_x = torch.from_numpy(sigs).float()
    tensor_y = torch.from_numpy(labels).float()
    
    if tensor_y.ndim == 1:
        tensor_y = tensor_y.unsqueeze(1)

    dataset = TensorDataset(tensor_x, tensor_y)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=False)

    if model_name != 'chronos-2' and model_name != 'chronos-2-synth' and model_name != 'chronos-2-small':
        model.to(device)
        model.eval()
    else:
        model.pipeline.model.eval()
    embeds_all = []
    labels_all = []
    
    
    with torch.no_grad():
        desc = f"Extracting {split}"
        iterator = tqdm(loader, desc=desc, leave=False) if not progress_bar else loader
        
        for x, y in iterator:
            x = x.to(device)
            if model_name != 'chronos-2' and model_name != 'chronos-2-synth' and model_name != 'chronos-2-small' and model_name != 'moment':
                out = model(x)
            elif model_name == 'chronos-2' or model_name == 'chronos-2-synth' or model_name == 'chronos-2-small':
                out = model(x.to('cpu'))
            else:
                out = model(x_enc=x).embeddings
            if out.ndim == 3:
                out = out.mean(dim=1)
            embeds_all.append(out.cpu().numpy())
            labels_all.append(y.numpy())
            
            if progress_bar:
                progress_bar.update(len(x))

    final_embeds = np.concatenate(embeds_all, axis=0)
    final_labels = np.concatenate(labels_all, axis=0)

    # 4. 保存
    save_path = save_dir / f'{split}_embeds.npz'
    save_dir.mkdir(parents=True, exist_ok=True)
    np.savez(save_path, embeds=final_embeds, labels=final_labels)
    if progress_bar: progress_bar.write(f"Saved to {save_path}")


def process_single_task_on_gpu(
    gpu_id: int, 
    model_cfg: Dict, 
    ds_cfg: Dict, 
    task_cfg: Dict, 
    batch_size: int, 
    num_workers: int, 
    save_root: str
):
    """
    单个 GPU 进程的任务函数：加载一个模型，跑一个特定的数据集任务
    """
    device = torch.device(f"cuda:{gpu_id}")
    model_name = model_cfg['name']
    ckpt_path = model_cfg['ckpt']
    model_sr = model_cfg.get('sr', 125)
    model_dur = model_cfg.get('duration', 10)
    
    ds_root = Path(ds_cfg['root'])
    ds_name = ds_root.stem
    task_name = task_cfg['name']
    
    job_desc = f"[GPU {gpu_id}] {model_name[:10]}..|{ds_name}|{task_name}"
    
    if 'chronos' not in model_name:
        model = get_model(model_name, ckpt_path)
        model = model.to(device)
    else:
        model = get_model(model_name, ckpt_path, device)

    save_dir = Path(save_root) / model_name / ds_name / task_name
    
    splits = ['train', 'test']
    
    pbar = tqdm(splits, desc=job_desc, position=gpu_id, leave=False)
    
    for split in pbar:
        extract_split_embeddings(
            model_name=model_name,
            model=model,
            dataset_dir=ds_root,
            data_name=task_name,
            split=split,
            signal_key=task_cfg['signal'],
            label_key=task_cfg['label'],
            batch_size=batch_size,
            num_workers=num_workers,
            device=device,
            save_dir=save_dir,
            target_sr=model_sr,
            target_duration=model_dur,
            progress_bar=None 
        )
            
    pbar.close()


def main():
    mp.set_start_method('spawn', force=True) 

    models = [
        {'name': 'clip', 'ckpt': './eval_models/clip/ckpt.example.pt', 'sr': 125, 'duration': 10}, 
    ]

    data_root_base = './preprocessed_datasets'
    
    datasets = [
        {
            'root': f'{data_root_base}/butppg',
            'tasks': [
                {'name': 'DBPData', 'signal': 'PPG', 'label': 'DBP'},
                {'name': 'HRData',  'signal': 'PPG', 'label': 'HR'},
                {'name': 'SBPData', 'signal': 'PPG', 'label': 'SBP'},
                {'name': 'SQIData', 'signal': 'PPG', 'label': 'Quality'},
            ]
        },
        {
            'root': f'{data_root_base}/dalia',
            'tasks': [{'name': 'HRData', 'signal': 'PPG', 'label': 'HR'}]
        },
        {
            'root': f'{data_root_base}/gyroppgacc',
            'tasks': [{'name': 'HRData', 'signal': 'PPG', 'label': 'HR'}]
        },
        {
            'root': f'{data_root_base}/ucibp',
            'tasks': [
                {'name': 'DBPData', 'signal': 'PPG', 'label': 'DBP'},
                {'name': 'SBPData', 'signal': 'PPG', 'label': 'SBP'},
            ]
        },
        {
            'root': f'{data_root_base}/wesad',
            'tasks': [
                {'name': 'EmotionData', 'signal': 'PPG', 'label': 'Emotion'},
                {'name': 'StressData', 'signal': 'PPG', 'label': 'Stress'},
            ]
        },
        {
            'root': f'{data_root_base}/deepbeat',
            'tasks': [
                {'name': 'AFData', 'signal': 'PPG', 'label': 'AF'},
            ]           
        }
    ]

    BATCH_SIZE = 2056
    NUM_WORKERS_PER_LOADER = 8
    SAVE_ROOT = './eval_embeddings'
    
    AVAILABLE_GPUS = [2, 3, 4, 5, 6, 7]

    print(f"Available GPUs: {AVAILABLE_GPUS}")
    
    task_queue = []
    for model_cfg in models:
        for ds_cfg in datasets:
            for task_cfg in ds_cfg['tasks']:
                task_queue.append({
                    'model': model_cfg,
                    'ds': ds_cfg,
                    'task': task_cfg
                })
    
    print(f"Total tasks: {len(task_queue)}")
    
    with ProcessPoolExecutor(max_workers=len(AVAILABLE_GPUS)) as executor:
        futures = []
        for i, job in enumerate(task_queue):
            gpu_id = AVAILABLE_GPUS[i % len(AVAILABLE_GPUS)]
            
            f = executor.submit(
                process_single_task_on_gpu,
                gpu_id,
                job['model'],
                job['ds'],
                job['task'],
                BATCH_SIZE,
                NUM_WORKERS_PER_LOADER,
                SAVE_ROOT
            )
            futures.append(f)
        
        for _ in tqdm(as_completed(futures), total=len(futures), desc="Overall Progress"):
            pass
        
        for f in futures:
            f.result()

if __name__ == '__main__':
    main()