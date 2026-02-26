import h5py
import numpy as np
from tqdm import tqdm
from pathlib import Path
from typing import List, Dict, Optional
from concurrent.futures import ProcessPoolExecutor, as_completed

class PulseDBExtractor:
    def __init__(
        self,
        dataset_dir: str,
        save_dir: str,
        map_dict: Optional[Dict[str, str]] = None,
    ) -> None:
        """Initialize PulseDB extractor with data paths and channel mapping.
        
        Args:
            dataset_dir: Directory containing mat files
            save_dir: Directory to save extracted NPZ files
            map_dict: Optional mapping from original to target channel names
        """
        self.dataset_dir = Path(dataset_dir)
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
        self.map_dict = map_dict or {}

    @staticmethod
    def read_mat(
        filepath: Path,
        channels: List[str],
        channel_map: Optional[Dict[str, str]] = None,
    ) -> Dict:
        """Read specified channels from a mat file.
        
        Args:
            filepath: Path to mat file
            channels: List of channel names to read
            channel_map: Optional channel name mapping
            
        Returns:
            Dictionary mapping channel names to {'data': ndarray, 'sr': float}
            
        Raises:
            ValueError: If requested channels are missing
            RuntimeError: If file reading fails
        """

        channel_map = channel_map or {ch: ch for ch in channels}
        
        try:
            with h5py.File(str(filepath)) as mat:
                subj_wins = mat['Subj_Wins']
                read_data = {}
                for channel in channels:
                    channel_array = subj_wins[channel][:].ravel()  # type: ignore
                    num_records = len(channel_array)
                    extracted_data_list = []
                    for i in range(num_records):
                        data = mat[channel_array[i]][:] # type: ignore
                        extracted_data_list.append(data)
                    extracted_info_list = np.asarray(extracted_data_list, dtype=np.float32)
                    read_data[channel_map[channel]] = {'data': extracted_info_list, 'sr': 125}
            return read_data
        except Exception as e:
            raise RuntimeError(f"Failed to read {filepath}: {str(e)}")

    def process_file(
            self,
            mat_path: Path,
            channels: List[str],
        ) -> None:
            """Process single mat file and save results."""
            try:
                data = self.read_mat(mat_path, channels, self.map_dict)
                save_path = self.save_dir / f"{mat_path.stem}.npz"
                np.savez(save_path, **data)
            except Exception as e:
                print(f"Error processing {mat_path.name}: {str(e)}")

    def process_all(
        self,
        channels: List[str],
        workers: int = 32,
        pattern: str = "**/*.mat",
    ) -> None:
        """Process all mat files in dataset directory.
        
        Args:
            channels: List of channel names to extract
            workers: Number of parallel workers
            pattern: File pattern to match
        """
        files = list(self.dataset_dir.glob(pattern))
        
        with ProcessPoolExecutor(max_workers=workers) as executor:
            tasks = [
                executor.submit(self.process_file, f, channels)
                for f in files
            ]
            
            for _ in tqdm(
                as_completed(tasks),
                total=len(tasks),
                desc="Processing mat files",
            ):
                pass