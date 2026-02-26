import wfdb
import logging
import numpy as np
from tqdm import tqdm
from pathlib import Path
from datetime import datetime, timedelta
from typing import Union, List, Dict, Any, Optional
from concurrent.futures import ProcessPoolExecutor, as_completed

print = logging.info
class MCMEDExtractor:
    def __init__(self, dataset_dir: str, save_dir: str):
        self.dataset_dir = Path(dataset_dir)
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)

    def get_record_metadata(self, hea_path: Union[str, Path]) -> Optional[Dict[str, Any]]:
        # 并非所有.hea文件的时间戳都精确到毫秒
        try:
            metadata = {}
            hea_path = Path(hea_path).with_suffix('.hea')
            with open(hea_path, 'r') as f:
                header_line = f.readline().strip().split()
                assert len(header_line) == 6
                filename, num_channel, sr, num_points, time_part, date_part = header_line
                record_time = datetime.strptime(f"{date_part} {time_part}", '%d/%m/%Y %H:%M:%S.%f')
            metadata.update({
                'filename': filename,
                'num_channel': int(num_channel),
                'sr': int(sr),
                'num_points': int(num_points),
                'record_time': record_time,
                'record_duration': int(num_points) / int(sr),
                'end_time': record_time + timedelta(seconds=int(num_points) / int(sr))
            })
        except Exception as e:
            print(f'Error get {hea_path} metadata: {e}.')
            return 
        return metadata


    def get_valid_records(self, folder_path: Path) -> List[Path]:
        hea_files, dat_files = map(folder_path.glob, ['*.hea', '*.dat'])
        hea_stems = [hea_file.stem for hea_file in hea_files]
        dat_stems = [dat_file.stem for dat_file in dat_files]
        stems = list(set(hea_stems) & set(dat_stems))
        valid_records = [folder_path.joinpath(stem) for stem in stems]
        return valid_records


    def extract_specified_time_interval_data(
            self,
            data: np.ndarray, 
            start_time: datetime, 
            sr: float, 
            extract_start_time: datetime,
            extract_end_time: datetime
            ) -> np.ndarray:
        start_index = int(round((extract_start_time - start_time).total_seconds() * sr))
        end_index = int(round((extract_end_time - start_time).total_seconds() * sr))
        return data[start_index: end_index]

    def extract_csn_sync_ppg_ecg(self, folder_path: Union[str, Path], save: bool = True) -> Optional[Dict[str, Any]]:
        ecg_dir, ppg_dir = map(Path(folder_path).joinpath, ['II', 'Pleth'])
        if ecg_dir.exists() and ppg_dir.exists():
            ecg_files, ppg_files = map(self.get_valid_records, [ecg_dir, ppg_dir])
            all_shared_ecg_data, all_shared_ppg_data = [], []

            for ecg_file in ecg_files:   
                ecg_metadata = self.get_record_metadata(ecg_file)
                if not ecg_metadata:
                    continue

                ecg_sr, ecg_start_time, ecg_end_time = ecg_metadata['sr'], ecg_metadata['record_time'], ecg_metadata['end_time']

                for ppg_file in ppg_files:
                    ppg_metadata = self.get_record_metadata(ppg_file)
                    if not ppg_metadata: 
                        continue

                    ppg_sr, ppg_start_time, ppg_end_time = ppg_metadata['sr'], ppg_metadata['record_time'], ppg_metadata['end_time']
                    if not (ppg_end_time < ecg_start_time or ppg_start_time > ecg_end_time):
                        ecg_data = wfdb.rdrecord(ecg_file).p_signal.ravel().astype(np.float32) # type: ignore
                        ppg_data = wfdb.rdrecord(ppg_file).p_signal.ravel().astype(np.float32)# type: ignore

                        shared_start_time = max(ppg_start_time, ecg_start_time)
                        shared_end_time = min(ppg_end_time, ecg_end_time)

                        shared_ecg_data = self.extract_specified_time_interval_data(
                            ecg_data, ecg_start_time, ecg_sr, shared_start_time, shared_end_time
                            )
                        shared_ppg_data = self.extract_specified_time_interval_data(
                            ppg_data, ppg_start_time, ppg_sr, shared_start_time, shared_end_time
                            )
                        shared_ecg_data = {**ecg_metadata, **{
                            'ECG': shared_ecg_data,
                            'shared_start_time': shared_start_time,
                            'shared_end_time': shared_end_time
                            }}
                        shared_ppg_data = {**ppg_metadata, **{
                            'PPG': shared_ppg_data,
                            'shared_start_time': shared_start_time,
                            'shared_end_time': shared_end_time
                            }}
                        all_shared_ecg_data.append(shared_ecg_data)
                        all_shared_ppg_data.append(shared_ppg_data)
                        del shared_ecg_data, shared_ppg_data
            if all_shared_ecg_data == [] and all_shared_ppg_data == []:
                print(f'{folder_path}, No matched data')
                return
            else:
                csn = all_shared_ecg_data[0]['filename'].split('_')[0]
                save_path = self.save_dir.joinpath(f'{csn}.npz')

                # 创建保存字典
                save_dict = {}
                
                # 存储ECG数据
                save_dict['ECG'] = np.array([d['ECG'] for d in all_shared_ecg_data], dtype=object)
                save_dict['ECG_metadata'] = np.array([
                    {k: v for k, v in d.items() if k != 'ECG'} 
                    for d in all_shared_ecg_data
                ], dtype=object)
                
                # 存储PPG数据
                save_dict['PPG'] = np.array([d['PPG'] for d in all_shared_ppg_data], dtype=object)
                save_dict['PPG_metadata'] = np.array([
                    {k: v for k, v in d.items() if k != 'PPG'} 
                    for d in all_shared_ppg_data
                ], dtype=object)     

                num_segments = len(all_shared_ecg_data)
                del all_shared_ecg_data, all_shared_ppg_data
                # 保存到文件
                if save:
                    np.savez(save_path, **save_dict)
                    print(f"Saved {save_path} with {num_segments} segments")
                    del save_dict
                else:
                    return save_dict
        else:
            print(f'{folder_path}, No available data')
            return


    def extract_csn_ppg(self, folder_path: Union[str, Path], save: bool = True) -> Optional[Dict[str, Any]]:
        ppg_dir = Path(folder_path).joinpath('Pleth')
        if ppg_dir.exists():
            ppg_files = self.get_valid_records(ppg_dir)
            all_ppg_data = []

            for ppg_file in ppg_files:
                ppg_metadata = self.get_record_metadata(ppg_file)
                if not ppg_metadata: 
                    continue

                ppg_sr, ppg_start_time, ppg_end_time = ppg_metadata['sr'], ppg_metadata['record_time'], ppg_metadata['end_time']
                ppg_data = wfdb.rdrecord(ppg_file).p_signal.ravel().astype(np.float32)# type: ignore

                ppg_data = self.extract_specified_time_interval_data(
                    ppg_data, ppg_start_time, ppg_sr, ppg_start_time, ppg_end_time
                    )
                
                ppg_data = {**ppg_metadata, **{
                    'PPG': ppg_data,
                    'shared_start_time': ppg_start_time,
                    'shared_end_time': ppg_end_time,
                    }}
                all_ppg_data.append(ppg_data)
                del ppg_data
            if all_ppg_data == []:
                print(f'{folder_path}, No matched data')
                return
            else:
                csn = all_ppg_data[0]['filename'].split('_')[0]
                save_path = self.save_dir.joinpath(f'{csn}.npz')

                # 创建保存字典
                save_dict = {}
                
                # 存储PPG数据
                save_dict['PPG'] = np.array([d['PPG'] for d in all_ppg_data], dtype=object)
                save_dict['PPG_metadata'] = np.array([
                    {k: v for k, v in d.items() if k != 'PPG'} 
                    for d in all_ppg_data
                ], dtype=object)     

                num_segments = len(all_ppg_data)
                del all_ppg_data
                # 保存到文件
                if save:
                    np.savez(save_path, **save_dict)
                    print(f"Saved {save_path} with {num_segments} segments")
                    del save_dict
                else:
                    return save_dict
        else:
            print(f'{folder_path}, No available data')
            return
    

    def extract_whole_dataset_sync_ppg_ecg(self, pattern: str = '*/*', num_workers: int = 1):
        waveform_folders = sorted(list(self.dataset_dir.glob(pattern)))
        with ProcessPoolExecutor(max_workers=num_workers) as executor:
            futures = [executor.submit(self.extract_csn_sync_ppg_ecg, folder) for folder in waveform_folders]
            for future in tqdm(as_completed(futures), total=len(futures), desc='Extracting MCMED Data'):
                pass  # 不收集结果


    def extract_whole_dataset_ppg(self, pattern: str = '*/*', num_workers: int = 1):
        waveform_folders = sorted(list(self.dataset_dir.glob(pattern)))
        with ProcessPoolExecutor(max_workers=num_workers) as executor:
            futures = [executor.submit(self.extract_csn_ppg, folder) for folder in waveform_folders]
            for future in tqdm(as_completed(futures), total=len(futures), desc='Extracting MCMED Data'):
                pass  # 不收集结果

if __name__ == '__main__':
    # # 提取同步 PPG 和 ECG 数据
    # dataset_dir = '/wujidata/ngk/PPGFounder/datasets/MC-MED/data/waveforms'
    # save_dir = '/wujidata/ngk/PPGFounder/datasets/MC-MED/MCMED_Patient_Level'
    # extractor = MCMEDExtractor(dataset_dir, save_dir)
    # extractor.extract_whole_dataset_sync_ppg_ecg()

    # 仅提取 PPG 数据
    dataset_dir = '/wujidata/ngk/PPGFounder/datasets/MC-MED/data/waveforms'
    save_dir = '/wujidata/ngk/PPGFounder/datasets/MC-MED/MCMED_Patient_Level_PPG'
    extractor = MCMEDExtractor(dataset_dir, save_dir)
    extractor.extract_whole_dataset_ppg()
    # extractor.extract_csn_sync_ppg_ecg('/data2/2shared/mc-med/data/waveforms/001/98988001')