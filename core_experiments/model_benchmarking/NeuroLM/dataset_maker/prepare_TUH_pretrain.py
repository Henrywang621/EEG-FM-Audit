"""
by Wei-Bang Jiang
https://github.com/935963004/NeuroLM
"""

from pathlib import Path
import mne
import pickle
import os
from tqdm import tqdm
from multiprocessing import Pool

standard_1020 = [
    'FP1', 'FPZ', 'FP2', 
    'AF9', 'AF7', 'AF5', 'AF3', 'AF1', 'AFZ', 'AF2', 'AF4', 'AF6', 'AF8', 'AF10', \
    'F9', 'F7', 'F5', 'F3', 'F1', 'FZ', 'F2', 'F4', 'F6', 'F8', 'F10', \
    'FT9', 'FT7', 'FC5', 'FC3', 'FC1', 'FCZ', 'FC2', 'FC4', 'FC6', 'FT8', 'FT10', \
    'T9', 'T7', 'C5', 'C3', 'C1', 'CZ', 'C2', 'C4', 'C6', 'T8', 'T10', \
    'TP9', 'TP7', 'CP5', 'CP3', 'CP1', 'CPZ', 'CP2', 'CP4', 'CP6', 'TP8', 'TP10', \
    'P9', 'P7', 'P5', 'P3', 'P1', 'PZ', 'P2', 'P4', 'P6', 'P8', 'P10', \
    'PO9', 'PO7', 'PO5', 'PO3', 'PO1', 'POZ', 'PO2', 'PO4', 'PO6', 'PO8', 'PO10', \
    'O1', 'OZ', 'O2', 'O9', 'CB1', 'CB2', \
    'IZ', 'O10', 'T3', 'T5', 'T4', 'T6', 'M1', 'M2', 'A1', 'A2', \
    'T1', 'T2', 'I1', 'I2'
]
savePath = Path('/D_data/weibangjiang')
rawDataPath = Path('/D_data/yansen/tuh_data/tuh/edf')
group = rawDataPath.rglob('*.edf')
dump_folder = '../teamdrive/pkl_data'

# preprocessing parameters
l_freq = 0.1
h_freq = 75.0
rsfreq = 200

drop_channels = ['PHOTIC-REF', 'IBI', 'BURSTS', 'SUPPR', 'EEG ROC-REF', 'EEG LOC-REF', 'EEG EKG1-REF', 'EMG-REF', 'EEG C3P-REF', 'EEG C4P-REF', 'EEG SP1-REF', 'EEG SP2-REF', \
                 'EEG LUC-REF', 'EEG RLC-REF', 'EEG RESP1-REF', 'EEG RESP2-REF', 'EEG EKG-REF', 'RESP ABDOMEN-REF', 'ECG EKG-REF', 'PULSE RATE', \
                    'EEG 1X10_LAT_01', '1X10_LAT_02', '1X10_LAT_03', '1X10_LAT_04', '1X10_LAT_05', 'X1', 'PG1', 'PG2']
drop_channels.extend([f'EEG {i}-REF' for i in range(20, 129)])

# channel number * rsfreq

def preprocessing_edf(edfFilePath, l_freq=0.1, h_freq=75.0, sfreq:int=200, drop_channels: list=None):
    # reading edf
    raw = mne.io.read_raw_edf(edfFilePath, preload=False)
    if raw.ch_names[0].split('-')[-1] == 'LE':
        return None, raw.ch_names
    if drop_channels is not None:
        useless_chs = []
        for ch in raw.ch_names:
            if ch.split(' ')[-1].split('-')[0] not in standard_1020:
                useless_chs.append(ch)
        # for ch in drop_channels:
        #     if ch in raw.ch_names:
        #         useless_chs.append(ch)
        raw.drop_channels(useless_chs)

    raw.load_data()
    # filtering
    raw = raw.filter(l_freq=l_freq, h_freq=h_freq, n_jobs=5)
    raw = raw.notch_filter(60.0, n_jobs=5)
    # downsampling
    raw = raw.resample(sfreq, n_jobs=5)
    eegData = raw.get_data(units='uV')

    return eegData, raw.ch_names


def process(cntFile):
    print(f'processing {cntFile.name}')
    eegData, chOrder = preprocessing_edf(cntFile, l_freq, h_freq, rsfreq, drop_channels)
    if chOrder[0].split('-')[-1] == 'LE':
        print('gzp' * 10)
        return

    chOrder = [s.split(' ')[-1].split('-')[0] for s in chOrder]
    eegData = eegData[:, :-10*rsfreq]

    time_length = (1024 // len(chOrder)) * rsfreq
    for i in range(eegData.shape[1] // time_length):
        dump_path = os.path.join(
            dump_folder, cntFile.name.split('.')[0] + "_" + str(i) + ".pkl"
        )
        pickle.dump(
            {"X": eegData[:, i * time_length : (i + 1) * time_length], "ch_names": chOrder},
            open(dump_path, "wb"),
        )

group = [g for g in group]
# split and dump in parallel
with Pool(processes=24) as pool:
    # Use the pool.map function to apply the square function to each element in the numbers list
    pool.map(process, group)
