"""
by Wei-Bang Jiang
https://github.com/935963004/NeuroLM
"""

from pathlib import Path
import h5py
import numpy as np
from pathlib import Path
import mne


class h5Dataset:
    def __init__(self, path:Path, name:str) -> None:
        self.__name = name
        self.__f = h5py.File(path / f'{name}.hdf5', 'a')
    
    def addGroup(self, grpName:str):
        return self.__f.create_group(grpName)
    
    def addDataset(self, grp:h5py.Group, dsName:str, arr:np.array, chunks:tuple):
        return grp.create_dataset(dsName, data=arr, chunks=chunks)
    
    def addAttributes(self, src:'h5py.Dataset|h5py.Group', attrName:str, attrValue):
        src.attrs[f'{attrName}'] = attrValue
    
    def save(self):
        self.__f.close()
    
    @property
    def name(self):
        return self.__name


def preprocessing(cntFilePath, l_freq=0.1, h_freq=75.0, sfreq:int=200):
    # 读取cnt
    raw = mne.io.read_raw_cnt(cntFilePath, preload=True, data_format='int32')
    raw.drop_channels(['M1', 'M2', 'VEO', 'HEO'])
    if 'ECG' in raw.ch_names:
        raw.drop_channels(['ECG'])

    # 滤波
    raw = raw.filter(l_freq=l_freq, h_freq=h_freq)
    raw = raw.notch_filter(50.0)
    # 降采样
    raw = raw.resample(sfreq, n_jobs=5)
    eegData = raw.get_data(units='uV')

    return eegData, raw.ch_names


savePath = Path('/home/liming.zhao/projects/shock/shock-data/h5Data')
rawDataPath = Path('/home/liming.zhao/projects/shock/shock-data/seed-3')
group = rawDataPath.glob('*.cnt')

trialStartTime = [24, 289, 550, 782, 1049, 1260, 1483, 1747, 1993, 2283, 2550, 2812, 3072, 3332, 3598]
trialEndTime = [264, 526, 757, 1023, 1235, 1458, 1722, 1967, 2259, 2525, 2788, 3046, 3308, 3573, 3806]
label = ['H', 'N', 'S', 'S', 'N', 'H', 'S', 'N', 'H', 'H', 'N', 'S', 'N', 'H', 'S']

# 预处理参数
l_freq = 0.1
h_freq = 75.0
rsfreq = 200

#chunks参数, 1s 数据
chunks = (62, rsfreq)

seed3 = h5Dataset(savePath, 'seed-3')
for cntFile in group:
    print(f'processing {cntFile.name}')
    eegData, chOrder = preprocessing(cntFile, l_freq, h_freq, rsfreq)
    eegData = eegData[:, :(trialEndTime[-1]+5)*rsfreq]
    grp = seed3.addGroup(grpName=cntFile.stem)
    dset = seed3.addDataset(grp, 'eeg', eegData, chunks)

    # group 属性，label, trial 属性
    seed3.addAttributes(grp, 'trialStart', trialStartTime)
    seed3.addAttributes(grp, 'trialEnd', trialEndTime)
    seed3.addAttributes(grp, 'label', label)

    # dataset属性，预处理信息
    seed3.addAttributes(dset, 'lFreq', l_freq)
    seed3.addAttributes(dset, 'hFreq', h_freq)
    seed3.addAttributes(dset, 'rsFreq', rsfreq)
    seed3.addAttributes(dset, 'chOrder', chOrder)

seed3.save()
