import numpy as np
import matplotlib.pyplot as plt

import os
import glob
import lmdb
import torch
import pickle
import numpy as np
from functools import reduce
from torch.utils import data 
from scipy.signal import resample


class ChunkDataset(data.Dataset):
    def __init__(self, dataset_dir: str, split='train', dataset='PulseDB', norm_data=True, norm_method='zscore', chunk_size=512):
        self.norm_data = norm_data
        self.norm_method = norm_method
        self.chunk_size = chunk_size
        self.all_subject_data_dirs = glob.glob(f'{os.path.join(dataset_dir, split)}/*.lmdb')
        

    def __getitem__(self, index):
        files, ppg_data, ecg_data = self.load_lmdb_data(self.all_subject_data_dirs[index])
        if self.norm_data:
            ppg_data = self.normalize(ppg_data, self.norm_method)
            ecg_data = self.normalize(ecg_data, self.norm_method)
        # ecg_data = resample(ecg_data, num=1250, axis=-1)
        ppg_data = np.expand_dims(ppg_data, axis=1)
        ecg_data = np.expand_dims(ecg_data, axis=1)
        return files, torch.tensor(ppg_data, dtype=torch.float32), torch.tensor(ecg_data, dtype=torch.float32)
    

    def __len__(self):
        return len(self.all_subject_data_dirs)


    def normalize(self, data: np.ndarray, method='zscore', eps=1e-5):
        assert data.ndim == 2
        assert method in ['zscore', 'minmax']
        if method == 'zscore':
            means = np.mean(data, axis=-1, keepdims=True)
            stds = np.std(data, axis=-1, keepdims=True)
            norm_data = (data - means) / (stds + eps)
        else:
            mins = np.min(data, axis=-1, keepdims=True)
            maxs = np.max(data, axis=-1, keepdims=True)
            norm_data = (data - mins) / (maxs - mins + eps)
        return norm_data


    @staticmethod
    def load_lmdb_data(lmdb_path: str, max_readers: int = 16):
        env = lmdb.open(
            lmdb_path,
            readonly=True,
            lock=False,
            max_readers=max_readers,
        )
        with env.begin() as txn:
            data = [pickle.loads(value) for _, value in txn.cursor()]

        file = np.stack([row[0] for row in data], axis=0)
        ppg_data = np.stack([row[1] for row in data], axis=0)
        ecg_data = np.stack([row[2] for row in data], axis=0)

        return file, ppg_data, ecg_data

    @staticmethod
    def collate_fn(batch):
        files = [item[0] for item in batch]
        ppg_data = [item[1] for item in batch]
        ecg_data = [item[2] for item in batch]
        ppg_data = torch.cat(ppg_data, dim=0)
        ecg_data = torch.cat(ecg_data, dim=0)
        return files, ppg_data, ecg_data