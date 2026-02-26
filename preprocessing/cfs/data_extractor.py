import pyedflib
import numpy as np
from tqdm import tqdm
from pathlib import Path
from typing import Dict, List, Optional, Any
from concurrent.futures import ProcessPoolExecutor, as_completed


class CFSExtractor:
    def __init__(
        self,
        dataset_dir: str,
        save_dir: str,
        map_dict: Optional[Dict[str, str]] = None,
    ) -> None:
        """Initialize CFS extractor with data paths and channel mapping.
        
        Args:
            dataset_dir: Directory containing EDF files
            save_dir: Directory to save extracted NPZ files
            map_dict: Optional mapping from original to target channel names
        """
        self.dataset_dir = Path(dataset_dir)
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
        self.map_dict = map_dict or {}

    @staticmethod
    def read_edf(
        filepath: Path,
        channels: List[str],
        channel_map: Optional[Dict[str, str]] = None,
    ) -> Dict:
        """Read specified channels from an EDF file.
        
        Args:
            filepath: Path to EDF file
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
            with pyedflib.EdfReader(str(filepath)) as edf:
                available = edf.getSignalLabels()
                missing = [ch for ch in channels if ch not in available]
                
                if missing:
                    raise ValueError(
                        f"Missing channels {missing} in {filepath.name}. "
                        f"Available: {available}"
                    )
                
                return {
                    channel_map[ch]: {
                        "data": edf.readSignal(edf.getSignalLabels().index(ch)).astype(np.float32),
                        "sr": edf.getSampleFrequency(edf.getSignalLabels().index(ch)),
                    }
                    for ch in channels
                }
                
        except Exception as e:
            raise RuntimeError(f"Failed to read {filepath}: {str(e)}")

    def process_file(
        self,
        edf_path: Path,
        channels: List[str],
    ) -> None:
        """Process single EDF file and save results."""
        try:
            data = self.read_edf(edf_path, channels, self.map_dict)
            save_path = self.save_dir / f"{edf_path.stem}.npz"
            np.savez(save_path, **data)
        except Exception as e:
            print(f"Error processing {edf_path.name}: {str(e)}")

    def process_all(
        self,
        channels: List[str],
        workers: int = 32,
        pattern: str = "*.edf",
    ) -> None:
        """Process all EDF files in dataset directory.
        
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
                desc="Processing EDF files",
            ):
                pass