import os
import lmdb
import pickle
import numpy as np
import pandas as pd
from tqdm import tqdm
from functools import partial
from concurrent.futures import ProcessPoolExecutor

def generate_segment_paths(file_paths, num_segs):
    segment_paths = np.empty(sum(num_segs), dtype=object)
    counter = 0
    for file_path, num_seg in zip(file_paths, num_segs):
        for i in range(num_seg):
            segment_paths[counter] = f'{file_path}*{i}'
            counter += 1
    return segment_paths

def save_lmdb_chunk(chunk_attrs, save_dir, split):
    file_paths, chunk_id = chunk_attrs
    save_path = os.path.join(save_dir, split, f'{chunk_id}.lmdb')
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    
    env = lmdb.open(save_path, map_size=2**24, writemap=True)
    try:
        with env.begin(write=True) as txn:
            for file_idx, file_path in enumerate(file_paths):
                try:
                    path_part, index_str = file_path.rsplit('*', 1)
                    index = int(index_str)
                    loaded_data = np.load(path_part, allow_pickle=True)  # Load once
                    data = (
                        file_path,
                        loaded_data['PPG'][index],
                        loaded_data['ECG'][index]
                    )
                    
                    txn.put(
                        key=f"{file_idx:08d}".encode(),
                        value=pickle.dumps(data, protocol=5)  # Protocol 5 is more efficient
                    )
                except Exception as e:
                    print(f"Processing failed {file_path}: {str(e)}")
                    continue
    finally:
        env.close()
    return save_path

if __name__ == '__main__':
    datasplit_path = './preprocessing/hsp/logs/split.csv'
    dataset_dir = './preprocessing/hsp/Proc_HSP_Patient_Level'
    save_dir = './preprocessing/hsp/Chunk_HSP'
    chunk_size = 512
    splits = ['train', 'valid', 'test']
    
    datasplit_df = pd.read_csv(datasplit_path)
    
    for split in splits:
        print(f'Processing {split} data...')
        df = datasplit_df[datasplit_df['Split'] == split]
        file_paths, num_segs = df['Path'].values, df['num_records'].values
        
        segment_paths = generate_segment_paths(file_paths, num_segs)
        np.random.shuffle(segment_paths)
        
        total_samples = len(segment_paths)
        num_full_chunks = total_samples // chunk_size
        full_path_chunks = np.array_split(segment_paths[:num_full_chunks*chunk_size], num_full_chunks)
        residual_path_chunk = segment_paths[num_full_chunks*chunk_size:]
        
        num_workers = 32
        with ProcessPoolExecutor(max_workers=num_workers) as executor:
            results = list(tqdm(
                executor.map(
                    partial(save_lmdb_chunk, save_dir=save_dir, split=split),
                    zip(full_path_chunks, range(num_full_chunks))
                ),
                total=num_full_chunks
            ))

        if len(residual_path_chunk) > 0:
            save_lmdb_chunk(
                (residual_path_chunk, num_full_chunks),
                save_dir=save_dir,
                split=split
            )