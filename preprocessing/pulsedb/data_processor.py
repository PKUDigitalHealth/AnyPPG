import mne
import biobss
import logging
import numpy as np
import pandas as pd
from tqdm import tqdm
import neurokit2 as nk
from icecream import ic 
from pathlib import Path
from scipy.signal import resample_poly
from typing import Optional, Tuple, Dict, Any
from numpy.lib.stride_tricks import as_strided
from concurrent.futures import ProcessPoolExecutor, as_completed

print = logging.info

class PulseDBProcessor:
    def __init__(
            self,
            dataset_dir: str, 
            seg_duration: float, 
            overlap_ratio: float, 
            save_dir: str,
            window_duration: float = 0.1, 
            change_thershold: float = 0.001,
            tolerate_threshold: float = 0.25,
            ppg_high_cutoff: float = 8,
            ppg_low_cutoff: float = 0.5,
            ecg_low_cutoff: float = 0.5,
            ecg_high_cutoff: Optional[float] = None,
            peak_loc_delta: float = 1,
            ecg_invert_thre: float = 0.5, 
            ecg_sqi_thre: float = 0.5, 
            ecg_frac_thre: float = 0.75,
            ecg_sqi_method: str = 'zhao2018', 
            ecg_sqi_approach: str = 'simple',
            ppg_correct_peaks: bool = False,
            ppg_sim_thre: float = 0.5, 
            ppg_frac_thre: float = 0.5,
            ppg_quality_check: bool = False,
            ppg_resample_sr: int = 125,
            ecg_resample_sr: int = 500,
            log_dir: str = './logs',
            num_workers: int = 32
            ):
        self.dataset_dir = Path(dataset_dir)
        self.seg_duration = seg_duration
        self.overlap_ratio = overlap_ratio
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
        self.window_duration = window_duration
        self.change_thershold = change_thershold
        self.tolerate_threshold = tolerate_threshold
        self.ppg_high_cutoff = ppg_high_cutoff
        self.ppg_low_cutoff = ppg_low_cutoff
        self.ecg_low_cutoff = ecg_low_cutoff
        self.ecg_high_cutoff = ecg_high_cutoff
        self.peak_loc_delta = peak_loc_delta
        self.ecg_invert_thre = ecg_invert_thre
        self.ecg_sqi_thre  = ecg_sqi_thre
        self.ecg_frac_thre = ecg_frac_thre
        self.ecg_sqi_method = ecg_sqi_method
        self.ecg_sqi_approach = ecg_sqi_approach
        self.ppg_correct_peaks = ppg_correct_peaks
        self.ppg_sim_thre = ppg_sim_thre 
        self.ppg_frac_thre = ppg_frac_thre
        self.ppg_quality_check = ppg_quality_check
        self.ppg_resample_sr = ppg_resample_sr
        self.ecg_resample_sr = ecg_resample_sr
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.num_workers = num_workers

 
    def process_whole_dataset(self, pattern: str = '*.npz'):
        files = sorted(list(self.dataset_dir.glob(pattern)))

        results = {
            'file': [],
            'num_records': [],
            'invert_ratio': [], 
            'valid_ppg_flat_indices': [], 
            'valid_ecg_flat_indices': [],
            'valid_ppg_quality_indices': [], 
            'valid_ecg_quality_indices': []
            }
        with ProcessPoolExecutor(max_workers=self.num_workers) as executor:
            tasks = [
                executor.submit(self.process_individual_data, str(file))
                for file in files
            ]

            for future in tqdm(as_completed(tasks), total=len(tasks), desc='Processing PulseDB data'):
                try:
                    _, _, file_path, num_records, invert_ratio, valid_ppg_flat, valid_ecg_flat, valid_ppg_qual, valid_ecg_qual = future.result()
                    if file_path is not None:
                        results['file'].append(file_path)
                        results['num_records'].append(num_records)
                        results['invert_ratio'].append(invert_ratio)
                        results['valid_ppg_flat_indices'].append(valid_ppg_flat)
                        results['valid_ecg_flat_indices'].append(valid_ecg_flat)
                        results['valid_ppg_quality_indices'].append(valid_ppg_qual)
                        results['valid_ecg_quality_indices'].append(valid_ecg_qual)
                except Exception as e:
                    print(f"[Error] {e}")
                    import traceback
                    traceback.print_exc()

            df = pd.DataFrame(results)
            df.to_csv(self.log_dir.joinpath('log.csv'), index=False)

    def process_individual_data(self, npz_path: str) -> Tuple:
        # loading data
        # print('Reading data...')
        (ppg_data, ppg_sr), (ecg_data, ecg_sr) = self.data_reader(npz_path)
        ppg_data, ecg_data = ppg_data.squeeze(1), ecg_data.squeeze(1)
        # print(ppg_data.shape, ecg_data.shape)
        # print('Reading Finished')

        # segmentation
        # print('Segmenting data...')
        # ppg_data = self.data_segmentor(ppg_data, self.seg_duration, ppg_sr, self.overlap_ratio)
        # ecg_data = self.data_segmentor(ecg_data, self.seg_duration, ecg_sr, self.overlap_ratio)

        if ppg_data is None or ecg_data is None:
            print(f'{npz_path}: No enough length data')
            return npz_path, 'No enough length data', None, None, None, None, None, None, None

        ecg_data, invert_ratios, _ = self.record_ecg_invert_checker(ecg_data, ecg_sr, self.ecg_invert_thre) # type: ignore
        # print(ppg_data.shape, ecg_data.shape)
        # print('Segmentation Finished')

        # NaN and Inf filtering
        # print('Filtering invalid data...')
        # print('PPG_data: ', ppg_data.dtype, ecg_data.dtype, npz_path)
        valid_mask = np.isfinite(ppg_data).all(axis=-1) & np.isfinite(ecg_data).all(axis=-1)
        ppg_data = ppg_data[valid_mask]
        ecg_data = ecg_data[valid_mask]
        # print(ppg_data.shape, ecg_data.shape)
        # print('Filtering Finished')

        # flat detection
        # print('Detecting flat in data...')
        ppg_flat_flag = self.data_flat_detector(ppg_data, ppg_sr, self.window_duration, self.change_thershold, self.tolerate_threshold)
        ecg_flat_flag = self.data_flat_detector(ecg_data, ecg_sr, self.window_duration, self.change_thershold, self.tolerate_threshold)
        valid_ppg_flat_indices = np.where(ppg_flat_flag == True)[0]
        valid_ecg_flat_indices = np.where(ecg_flat_flag == True)[0]
        valid_flat_indices = np.intersect1d(valid_ppg_flat_indices, valid_ecg_flat_indices)

        ppg_data, ecg_data = ppg_data[valid_flat_indices], ecg_data[valid_flat_indices]
        if len(valid_flat_indices) == 0:
            print(f'{npz_path}: No valid data')
            return npz_path, 'No valid data', None, None, None, None, None, None, None
        # print(ppg_data.shape, ecg_data.shape)
        # print('Flat detection finished')

        # filtering
        # print('Filtering data...')
        ppg_data = self.ppg_clean_elgendi_mne(ppg_data, ppg_sr, self.ppg_low_cutoff, self.ppg_high_cutoff)
        ecg_data = self.ecg_clean_nk_mne(ecg_data, ecg_sr, self.ecg_low_cutoff, self.ecg_high_cutoff)
        # print(ppg_data.shape, ecg_data.shape)
        # print('Filtering finished')

        # PPG signal quality assessment
        # print('PPG signal quality check')
        if self.ppg_quality_check:
            ppg_quality_check = self.ppg_quality_checker(ppg_data, ppg_sr, self.peak_loc_delta, self.ppg_correct_peaks, self.ppg_sim_thre, self.ppg_frac_thre)
        else:
            ppg_quality_check = np.asarray([True] * ppg_data.shape[0])
        # print('PPG quality assessment finished')

        # ECG signal quality assessment
        # print('ECG signal quality check')
        ecg_quality_check = self.ecg_quality_checker(npz_path, ecg_data, ecg_sr, self.ecg_sqi_thre, self.ecg_frac_thre, self.ecg_sqi_method, self.ecg_sqi_approach)
        # print('ECG quality assessment finished')
        valid_ppg_quality_indices = np.where(ppg_quality_check == True)[0]
        valid_ecg_quality_indices = np.where(ecg_quality_check == True)[0]
        valid_quality_indices = np.intersect1d(valid_ppg_quality_indices, valid_ecg_quality_indices)

        ppg_data, ecg_data = ppg_data[valid_quality_indices], ecg_data[valid_quality_indices]
        if len(valid_quality_indices) == 0:
            print(f'{npz_path}: No qualified data')
            return npz_path, 'No qualified data', None, None, None, None, None, None, None
        
        # resampling
        ppg_data = self.data_resampler(ppg_data, ppg_sr, self.ppg_resample_sr)
        ecg_data = self.data_resampler(ecg_data, ecg_sr, self.ecg_resample_sr)
        # print(ppg_data.shape, ecg_data.shape)
        # print('Resampling finished')

        # normalization
        # print('Normalization data...')
        ppg_data, ecg_data = self.data_normalizer(ppg_data), self.data_normalizer(ecg_data)
        # print(ppg_data.shape, ecg_data.shape)
        # print('Normalization finished')

        np.savez(self.save_dir.joinpath(f'{Path(npz_path).stem}.npz'), PPG=ppg_data, ECG=ecg_data)
        print(f'{npz_path} has been successfully saved.')

        return ppg_data, ecg_data, npz_path, len(ppg_data), invert_ratios, valid_ppg_flat_indices, valid_ecg_flat_indices, valid_ppg_quality_indices, valid_ecg_quality_indices


    @staticmethod
    def data_reader(npz_path: str):
        data = np.load(npz_path, allow_pickle=True)
        ppg_data, ppg_sr = data['PPG'].item().values()
        ecg_data, ecg_sr = data['ECG'].item().values()
        return (ppg_data, ppg_sr), (ecg_data, ecg_sr)
    

    def data_segmentor(self, data: np.ndarray, seg_duration: float, sr: float, overlap_ratio: float = 0):
        assert data.ndim == 1, f'Expected data dimention is 1, got {data.ndim}'
        window_size = int(seg_duration * sr)
        step_size =int((1 - overlap_ratio) * window_size)
        windows = self.slide_1d_window(data, window_size, step_size)
        return windows

    @staticmethod
    def ppg_clean_elgendi_mne(
        ppg_signal: np.ndarray, 
        sr: float = 125, 
        l_freq: float = 0.5, 
        h_freq: float = 8.0, 
        method: str = 'iir',
        iir_params: Dict[str, Any] = dict(order=3, ftype='butter', output='ba'),
        *args, **kwargs
        ):
        assert ppg_signal.ndim == 2, f'Expected data dimention is 2, got {ppg_signal.ndim}'
        if ppg_signal.ndim == 1:
            ppg_signal = ppg_signal[np.newaxis, :]

        filtered = mne.filter.filter_data(
            ppg_signal.astype(float), 
            sfreq=sr,
            l_freq=l_freq, 
            h_freq=h_freq,
            method=method,
            iir_params=iir_params,
            verbose=False,
            *args, **kwargs
        )
            
        return filtered.astype(np.float32)

    @staticmethod
    def ecg_clean_nk_mne(
        ecg_signal: np.ndarray, 
        sr: float = 500,
        l_freq: float = 0.5,
        h_freq: Optional[float] = None,
        method: str = 'iir',
        iir_params: Dict = dict(order=5, ftype='butter', output='ba'),
        notch_widths: float = 1,
        *args, **kwargs
        ):
        filtered = mne.filter.filter_data(
            ecg_signal.astype(float),
            sfreq=sr,
            l_freq=l_freq,
            h_freq=h_freq,
            method=method,
            iir_params=iir_params,
            verbose=False,
            *args, **kwargs,
        )
        
        # Notch filter for powerline (50Hz)
        filtered = mne.filter.notch_filter(
            filtered,
            Fs=sr,
            freqs=50,
            notch_widths=notch_widths,
            method=method,
            verbose=False
        )
        return filtered.astype(np.float32)

    def data_resampler(self, data: np.ndarray, sr: float, resample_sr: float):
        assert data.ndim == 1 or data.ndim == 2, f'Expected data dimention is 1 or 2, got {data.ndim}'
        resampled_data = resample_poly(data, resample_sr, sr, axis=-1)
        return resampled_data


    def record_ecg_invert_checker(self, ecg_signal: np.ndarray, sampling_rate: float = 500, threshold: float = 0.5):
        """**ECG signal inversion**

        Checks whether an ECG signal is inverted, and if so, corrects for this inversion.
        To automatically detect the inversion, the ECG signal is cleaned, the mean is subtracted,
        and with a rolling window of 2 seconds, the original value corresponding to the maximum
        of the squared signal is taken. If the median of these values is negative, it is
        assumed that the signal is inverted.
        """
        assert ecg_signal.ndim == 1 or ecg_signal.ndim == 2, f'Expected data dimention is 1 or 2, got {ecg_signal.ndim}'
        if ecg_signal.ndim == 1:
            ecg_signal = np.expand_dims(ecg_signal, axis=0)
        

        def _ecg_inverted(ecg_signal, sampling_rate=1000, window_time=2.0):
            """Checks whether an ECG signal is inverted."""
            ecg_cleaned = self.ecg_clean_nk_mne(ecg_signal, sampling_rate)

            # take the median of the original value of the maximum of the squared signal
            # over a window where we would expect at least one heartbeat
            med_max_squared = np.nanmedian(
                _roll_orig_max_squared(ecg_cleaned, window=int(window_time * sampling_rate))
            )
            # if median is negative, assume inverted
            return med_max_squared < 0


        def _roll_orig_max_squared(x, window=2000):
            """With a rolling window, takes the original value corresponding to the maximum of the squared signal."""
            x_rolled = np.lib.stride_tricks.sliding_window_view(x, window, axis=0)
            # https://stackoverflow.com/questions/61703879/in-numpy-how-to-select-elements-based-on-the-maximum-of-their-absolute-values
            shape = np.array(x_rolled.shape)
            shape[-1] = -1
            return np.take_along_axis(x_rolled, np.square(x_rolled).argmax(-1).reshape(shape), axis=-1)

        was_inverted = []
        for item in ecg_signal:
            if _ecg_inverted(item, sampling_rate=int(sampling_rate)):
                was_inverted.append(True)
            else:
                was_inverted.append(False)
        invert_ratio = sum(was_inverted) / len(was_inverted)
        judge = invert_ratio > threshold
        if judge:
            return ecg_signal * -1, invert_ratio, judge
        return ecg_signal, invert_ratio, judge


    def ecg_quality_checker(self, npz_path, data: np.ndarray, sr: float, sqi_thre: float = 0.5, frac_thre: float = 0.75, method: str = 'zhao2018', approach: str = 'simple'):
        assert data.ndim == 1 or data.ndim == 2, f'Expected data dimention is 1 or 2, got {data.ndim}'
        assert method in ['averageQRS', 'zhao2018'], f'Expected signal quality assessment method is averageQRS or zhao2018, got {method}'
        if data.ndim == 1:
            data = np.expand_dims(data, axis=0)
        ecg_sqi = []
        for item in data:
            try:
                item_sqi = nk.ecg_quality(item, sampling_rate=int(sr), method=method, approach=approach)
            except Exception as e:
                # print(f'ECG_QUALITY_CHECKER Error: {e}. {npz_path}')
                item_sqi = np.zeros(data.shape[-1]) if method == 'averageQRS' else 'Unacceptable'
            ecg_sqi.append(item_sqi)
        ecg_sqi = np.asarray(ecg_sqi)
        check_result = []
        if method == 'averageQRS':
            ecg_sqi = (ecg_sqi >= sqi_thre).sum(axis=-1) / ecg_sqi.shape[-1]
            check_result = ecg_sqi >= frac_thre
        if method == 'zhao2018':
            for item_sqi in ecg_sqi:
                if item_sqi == 'Unacceptable':
                    check_result.append(False)
                else:
                    check_result.append(True)
        check_result = np.asarray(check_result)
        return check_result
    
    
    def ppg_quality_checker(self, data: np.ndarray, sr: float, delta: float = 1e-4, correct_peaks: bool = True, sim_thre: float = 0.5, frac_thre: float = 0.5):
        assert data.ndim == 1 or data.ndim == 2, f'Expected data dimention is 1 or 2, got {data.ndim}'
        if data.ndim == 1:
            data = np.expand_dims(data, axis=0)
        check_result = []
        for item in data:
            try:
                info = biobss.ppgtools.ppg_detectpeaks(sig=item, sampling_rate=sr, method='peakdet', delta=delta, correct_peaks=correct_peaks)
                locs_peaks = info['Peak_locs']
                sim, _ = biobss.sqatools.template_matching(item, locs_peaks) # Template matching 
                sim = np.asarray(sim)
                ppg_sqi = (sim >= sim_thre).sum(axis=-1) / sim.shape[-1]
                sqi_flag = ppg_sqi > frac_thre
            except Exception as e:
                # print(f'PPG_QUALITY_CHECKER ERROR: {e}.')
                sqi_flag = False
            check_result.append(sqi_flag)
        check_result = np.asarray(check_result)
        return check_result
    

    def data_flat_detector(self, data: np.ndarray, sr: float, window_duration: float = 0.1, change_threshold: float = 0.005, tolerate_threshold: float = 0.25):
        assert data.ndim == 1 or data.ndim == 2, f'Expected data dimention is 1 or 2, got {data.ndim}'
        if data.ndim == 1:
            data = np.expand_dims(data, axis=0)
        window_size = int(window_duration * sr)
        window_flat_ratios = self.detect_flat_in_windows(data, window_size, change_threshold)
        detect_result = window_flat_ratios <= tolerate_threshold
        return detect_result


    def data_normalizer(self, data: np.ndarray, eps: float = 1e-8):
        mean, std = data.mean(axis=-1, keepdims=True), data.std(axis=-1, keepdims=True)
        normalized_data = (data - mean) / (std + eps)
        return normalized_data


    @staticmethod
    def slide_1d_window(data: np.ndarray, window_size: int, step_size: int):
        """
        Slide a 1D window over the input data to create overlapping windows.

        Args:
            data (np.ndarray): Input 1D array.
            window_size (int): Size of the sliding window.
            step_size (int): Step size for sliding the window.

        Returns:
            np.ndarray: A 2D array where each row is a window of the input data.
        """
        if len(data) < window_size:
            return
        shape = ((len(data) - window_size) // step_size + 1, window_size)
        strides = data.strides[0] * step_size, data.strides[0]
        windows = as_strided(data, shape=shape, strides=strides)
        return windows


    @staticmethod
    def detect_flat_in_windows(windows: np.ndarray, window_size: int = 50, threshold: float = 0.005) -> np.ndarray:
        """
        Detect flat regions in sliding windows of the input data.

        Args:
            windows (np.ndarray): 2D array where each row is a window of the input data.
            window_size (int): Size of the sliding window.
            threshold (float): Threshold for determining flat regions.

        Returns:
            np.ndarray: An array of flat ratios for each window.
        """
        # Compute the absolute difference between consecutive points in each window
        diff = np.abs(np.diff(windows, axis=1))

        # Compute the change rate using a sliding window convolution
        change_rate = np.apply_along_axis(
            lambda x: np.convolve(x, np.ones(window_size - 1), mode='valid') / (window_size - 1),
            axis=1,
            arr=diff
        )
        
        # Determine if each window is flat based on the threshold
        flat_windows = change_rate < threshold
        
        # Compute the ratio of flat regions in each window
        flat_ratios = np.mean(flat_windows, axis=1)
        return flat_ratios