"""
by Wei-Bang Jiang
https://github.com/935963004/NeuroLM
"""

from torch.utils.data import Dataset
from pathlib import Path
import h5py
import bisect
import torch
from einops import rearrange
import tiktoken
import numpy as np
import pickle
import os
from scipy.signal import resample
from dataset import standard_1020


def get_chans(ch_names):
    chans = []
    for ch_name in ch_names:
        chans.append(standard_1020.index(ch_name))
    return chans


class SEEDDataset(Dataset):
    """读取单个hdf5文件，仅使用范式中途采集的数据，标签内包含范式标签与被试性别，如有需要可以继续往字典中添加"""
    def __init__(self, file_path: Path, window_size: int=200, stride_size: int=1, start_percentage: float=0, end_percentage: float=1, 
                 trial_start_percentage: float=0, trial_end_percentage: float=1, subject_start_percentage: float=0, subject_end_percentage: float=1, 
                 is_instruct: bool=False, is_val: bool=False, eeg_max_len=-1, text_max_len=-1):
        '''
        从路径file_path中提取数据集。

        :param Path file_path: 目标数据路径
        :param int window_size: 单个样本长度
        :param int stride_size: 两个相邻样本间隔
        :param float start_percentage: 数据集中，每个采纳的trial内首个样本在此trial的样本中的百分比索引（包括）。
        :param float end_percentage: 数据集中，每个采纳的trial内末尾样本在此trial的样本中的百分比索引（不包括）。
        :param float trial_start_percentage: 数据集中，采纳的首个trial在此被试的所有trial中的百分比索引（包括）。
        :param float trial_end_percentage: 数据集中，采纳的末个trial在此被试的所有trial中的百分比索引（不包括）。
        :param float subject_start_percentage: 数据集中，采纳的首个被试的百分比索引（包括）。
        :param float subject_end_percentage: 数据集中，采纳的末个被试的百分比索引（不包括）。
        
        比如，数据文件总共10个被试，每个被试有15个trial，每个trial提供100个样本时。取参数为0.2, 0.8, 0.34, 0.67, 0.2, 0.8时，数据集会包括下标为[2, 8)的被试，每个被试的下标为[5, 10)的trial中，每个trial下标为[20, 80)的样本。
        '''
        self.__file_path = file_path
        self.__window_size = window_size
        self.__stride_size = stride_size
        self.__start_percentage = start_percentage
        self.__end_percentage = end_percentage
        self.__trial_start_percentage = trial_start_percentage
        self.__trial_end_percentage = trial_end_percentage
        self.__subject_start_percentage = subject_start_percentage
        self.__subject_end_percentage = subject_end_percentage
        self.__is_instruct = is_instruct
        self.__is_val = is_val
        self.eeg_max_len = eeg_max_len
        self.text_max_len = text_max_len

        self.__file = None
        self.__length = None
        self.__feature_size = None

        self.__subjects = []
        self.__global_idxes = [] # 从第几个样本开始是哪个被试
        self.__local_idxess = [] # 从这个被试的第几个样本开始是哪个trial
        self.__trial_start_idxess = [] # trial开始索引
        self.__genders = []
        self.__labelss = []

        self.__rsFreq = None

        self.__seed_label = {
            'H': 0,
            'N': 1,
            'S': 2
        }
        
        self.__init_dataset()

        if is_instruct:
            enc = tiktoken.get_encoding("gpt2")
            encode = lambda s: enc.encode(s, allowed_special={"<|endoftext|>"})
            # 50257 for [SEP]
            self.__text = {
                0: torch.IntTensor([50257] + encode('Question: Which emotion type does this EEG segment belong to? Answer: Positive <|endoftext|>')),
                2: torch.IntTensor([50257] + encode('Question: Which emotion type does this EEG segment belong to? Answer: Negative <|endoftext|>')),
                1: torch.IntTensor([50257] + encode('Question: Which emotion type does this EEG segment belong to? Answer: Neutral <|endoftext|>'))
            }
            self.__prompt = torch.IntTensor([50257] + encode('Question: Which emotion type does this EEG segment belong to? Answer:'))

    def __init_dataset(self) -> None:
        self.__file = h5py.File(str(self.__file_path), 'r')
        self.__subjects = [i for i in self.__file]

        global_idx = 0
        subject_start_id = int(len(self.__subjects) * self.__subject_start_percentage) # 包括在数据集中的被试开始id
        subject_end_id = int(len(self.__subjects) * self.__subject_end_percentage - 1) # 包括在数据集中的被试结束id
        for subject_id, subject in enumerate(self.__subjects):
            self.__global_idxes.append(global_idx)
            #self.__genders.append(self.__file[subject].attrs['gender'])
            self.__labelss.append(self.__file[subject].attrs['label'])
            self.__rsFreq = self.__file[subject]['eeg'].attrs['rsFreq']

            local_idxes = [] # 当前trial的第一个样本在数据集中的样本索引
            trial_start_idxes = [] # 当前trial在原始数据中的开始位置索引
            trial_starts = self.__file[subject].attrs['trialStart']
            trial_ends = self.__file[subject].attrs['trialEnd']
            local_idx = 0
            if subject_id >= subject_start_id and subject_id <= subject_end_id:
                trial_start_id = int(len(trial_starts) * self.__trial_start_percentage)  # 该被试包括在数据集中的trial开始id
                trial_end_id = int(len(trial_starts) * self.__trial_end_percentage - 1)  # 该被试包括在数据集中的trial结束id
                for trial_id, (trial_start, trial_end) in enumerate(zip(trial_starts, trial_ends)):
                    local_idxes.append(local_idx)

                    if trial_id >= trial_start_id and trial_id <= trial_end_id:
                        trial_len = (trial_end - trial_start + 1) * self.__rsFreq
                        trial_sample_num = (trial_len-self.__window_size) // self.__stride_size + 1
                        start_idx = int(trial_sample_num * self.__start_percentage) * self.__stride_size + trial_start * self.__rsFreq
                        end_idx = int(trial_sample_num * self.__end_percentage - 1) * self.__stride_size + trial_start * self.__rsFreq

                        trial_start_idxes.append(start_idx)
                        local_idx += (end_idx - start_idx) // self.__stride_size + 1
                    else:
                        trial_start_idxes.append(0)

            self.__local_idxess.append(local_idxes)
            self.__trial_start_idxess.append(trial_start_idxes)

            global_idx += local_idx

        self.__length = global_idx

        self.__feature_size = [i for i in self.__file[self.__subjects[0]]['eeg'].shape]
        self.__feature_size[1] = self.__window_size

    @property
    def feature_size(self):
        return self.__feature_size
    
    @property
    def rsfreq(self):
        return self.__rsFreq

    def __len__(self):
        return self.__length

    def __getitem__(self, idx: int):
        # 先确认样本属于哪个被试，再确认样本属于哪个trial
        subject_id = bisect.bisect(self.__global_idxes, idx) - 1
        trial_id = bisect.bisect(self.__local_idxess[subject_id], idx-self.__global_idxes[subject_id]) - 1
        item_start_idx = (idx - self.__global_idxes[subject_id] - self.__local_idxess[subject_id][trial_id]) * self.__stride_size + self.__trial_start_idxess[subject_id][trial_id]
        
        X = self.__file[self.__subjects[subject_id]]['eeg'][:, item_start_idx:item_start_idx+self.__window_size]
        Y = self.__seed_label[self.__labelss[subject_id][trial_id]]

        data = torch.FloatTensor(X / 100)
        time = data.size(1) // 200
        input_time = [i  for i in range(time) for _ in range(data.size(0))]
        data = rearrange(data, 'N (A T) -> (A N) T', T=200)

        ch_names = self.get_ch_names()
        input_chans = list(self.get_ch_names()) * time

        if not self.__is_instruct:
            input_chans = torch.IntTensor(get_chans(input_chans))
            input_time = torch.IntTensor(input_time)

            gpt_mask = torch.tril(torch.ones(data.size(0), data.size(0))).view(1, data.size(0), data.size(0))
            num_chans = len(ch_names)
            for i in range(time):
                gpt_mask[:, i * num_chans:(i + 1) * num_chans,  i * num_chans:(i + 1) * num_chans] = 1
            return data, Y, input_chans, input_time, gpt_mask.bool()
        
        if self.__is_val:
            text = self.__prompt
        else:
            text = self.__text[int(Y)]
            # pad text to text_max_len
            valid_text_len = text.size(0)
            if self.text_max_len > valid_text_len:
                text_pad = torch.full((self.text_max_len,), fill_value=50256)
                text_pad[:valid_text_len] = text
                text = text_pad

        # pad eeg to eeg_max_len
        valid_eeg_len = data.size(0)
        if self.eeg_max_len > data.size(0):
            X_eeg = torch.zeros((self.eeg_max_len, 200))
            X_eeg[:data.size(0)] = data
            eeg_mask = torch.ones(self.eeg_max_len)
            eeg_mask[valid_eeg_len:] = 0

            input_chans.extend(['pad'] * (self.eeg_max_len - data.size(0)))
            input_time.extend([0] * (self.eeg_max_len - data.size(0)))
        else:
            X_eeg = data
            eeg_mask = torch.ones(data.size(0))

        input_chans = torch.IntTensor(get_chans(input_chans))
        input_time = torch.IntTensor(input_time)

        num_tokens = X_eeg.size(0) + text.size(0)
        gpt_mask = torch.tril(torch.ones(num_tokens, num_tokens)).view(1, num_tokens, num_tokens)
        num_chans = len(ch_names)
        for i in range(time):
            gpt_mask[:, i * num_chans:(i + 1) * num_chans,  i * num_chans:(i + 1) * num_chans] = 1
        gpt_mask[:, :, valid_eeg_len:X_eeg.size(0)] = 0
        
        if self.__is_val:
            return X_eeg, text, Y, input_chans, input_time, eeg_mask.bool(), gpt_mask.bool()
        
        Y_text = torch.full_like(text, fill_value=-1)
        prompt_len = self.__prompt.size(0)
        Y_text[prompt_len - 1:valid_text_len - 1] = text[prompt_len:valid_text_len]
        return X_eeg, text, Y_text, input_chans, input_time, eeg_mask.bool(), gpt_mask.bool()

    def free(self) -> None: 
        # TODO 临时方案，目标：减少文件打开次数。查一下flush
        if self.__file:
            self.__file.close()
            self.__file = None
    
    def get_ch_names(self):
        return self.__file[self.__subjects[0]]['eeg'].attrs['chOrder']


class TUABLoader(Dataset):
    # abnormal: 1
    # normal: 0
    def __init__(self, root, files, sampling_rate=200, eeg_max_len=-1, text_max_len=-1, is_instruct=False, is_val=False):
        self.root = root
        self.files = files
        self.default_rate = 200
        self.sampling_rate = sampling_rate
        self.is_instruct = is_instruct
        self.is_val = is_val
        self.eeg_max_len = eeg_max_len
        self.text_max_len = text_max_len

        ch_names = ['EEG FP1', 'EEG FP2-REF', 'EEG F3-REF', 'EEG F4-REF', 'EEG C3-REF', 'EEG C4-REF', 'EEG P3-REF', 'EEG P4-REF', 'EEG O1-REF', 'EEG O2-REF', 'EEG F7-REF', \
                    'EEG F8-REF', 'EEG T3-REF', 'EEG T4-REF', 'EEG T5-REF', 'EEG T6-REF', 'EEG A1-REF', 'EEG A2-REF', 'EEG FZ-REF', 'EEG CZ-REF', 'EEG PZ-REF', 'EEG T1-REF', 'EEG T2-REF']
        self.ch_names = [name.split(' ')[-1].split('-')[0] for name in ch_names]

        if is_instruct:
            enc = tiktoken.get_encoding("gpt2")
            encode = lambda s: enc.encode(s, allowed_special={"<|endoftext|>"})
            # 50257 for [SEP]
            self.text = {
                1: torch.IntTensor([50257] + encode('Question: Is this EEG segment abnormal? Answer: Yes <|endoftext|>')),
                0: torch.IntTensor([50257] + encode('Question: Is this EEG segment abnormal? Answer: No <|endoftext|>'))
            }
            self.prompt = torch.IntTensor([50257] + encode('Question: Is this EEG segment abnormal? Answer:'))

    def __len__(self):
        return len(self.files)

    def __getitem__(self, index):
        sample = pickle.load(open(os.path.join(self.root, self.files[index]), "rb"))
        X = sample["X"]
        Y = sample["y"]

        data = torch.FloatTensor(X / 100)
        time = data.size(1) // 200
        input_time = [i  for i in range(time) for _ in range(data.size(0))]
        data = rearrange(data, 'N (A T) -> (A N) T', T=200)

        ch_names = self.ch_names
        input_chans = list(ch_names) * time

        if not self.is_instruct:
            input_chans = torch.IntTensor(get_chans(input_chans))
            input_time = torch.IntTensor(input_time)

            gpt_mask = torch.tril(torch.ones(data.size(0), data.size(0))).view(1, data.size(0), data.size(0))
            num_chans = len(ch_names)
            for i in range(time):
                gpt_mask[:, i * num_chans:(i + 1) * num_chans,  i * num_chans:(i + 1) * num_chans] = 1
            return data, Y, input_chans, input_time, gpt_mask.bool()
        
        if self.is_val:
            text = self.prompt
        else:
            text = self.text[int(Y)]
            # pad text to text_max_len
            valid_text_len = text.size(0)
            if self.text_max_len > valid_text_len:
                text_pad = torch.full((self.text_max_len,), fill_value=50256)
                text_pad[:valid_text_len] = text
                text = text_pad

        # pad eeg to eeg_max_len
        valid_eeg_len = data.size(0)
        if self.eeg_max_len > data.size(0):
            X_eeg = torch.zeros((self.eeg_max_len, 200))
            X_eeg[:data.size(0)] = data
            eeg_mask = torch.ones(self.eeg_max_len)
            eeg_mask[valid_eeg_len:] = 0

            input_chans.extend(['pad'] * (self.eeg_max_len - data.size(0)))
            input_time.extend([0] * (self.eeg_max_len - data.size(0)))
        else:
            X_eeg = data
            eeg_mask = torch.ones(data.size(0))

        input_chans = torch.IntTensor(get_chans(input_chans))
        input_time = torch.IntTensor(input_time)

        num_tokens = X_eeg.size(0) + text.size(0)
        gpt_mask = torch.tril(torch.ones(num_tokens, num_tokens)).view(1, num_tokens, num_tokens)
        num_chans = len(ch_names)
        for i in range(time):
            gpt_mask[:, i * num_chans:(i + 1) * num_chans,  i * num_chans:(i + 1) * num_chans] = 1
        gpt_mask[:, :, valid_eeg_len:X_eeg.size(0)] = 0
        
        if self.is_val:
            return X_eeg, text, Y, input_chans, input_time, eeg_mask.bool(), gpt_mask.bool()
        
        Y_text = torch.full_like(text, fill_value=-1)
        prompt_len = self.prompt.size(0)
        Y_text[prompt_len - 1:valid_text_len - 1] = text[prompt_len:valid_text_len]
        return X_eeg, text, Y_text, input_chans, input_time, eeg_mask.bool(), gpt_mask.bool()
    
class CCDLoader(Dataset):
    # abnormal: 1
    # normal: 0
    def __init__(self, root, files, sampling_rate=200, eeg_max_len=-1, text_max_len=-1, is_instruct=False, is_val=False):
        self.root = root
        self.files = files
        self.default_rate = 200
        self.sampling_rate = sampling_rate
        self.is_instruct = is_instruct
        self.is_val = is_val
        self.eeg_max_len = eeg_max_len
        self.text_max_len = text_max_len
        self.data_list = []

        # self.ch_names = ['E1', 'E2', 'E3', 'E4', 'E5', 'E6', 'E7', 'E8', 'E9', 'E10', 'E11', 'E12', 'E13', 'E14', 'E15', 'E16', 'E17', 'E18', 'E19', 
        #     'E20', 'E21', 'E22', 'E23', 'E24', 'E25', 'E26', 'E27', 'E28', 'E29', 'E30', 'E31', 'E32', 'E33', 'E34', 'E35', 'E36', 'E37', 
        #     'E38', 'E39', 'E40', 'E41', 'E42', 'E43', 'E44', 'E45', 'E46', 'E47', 'E48', 'E49', 'E50', 'E51', 'E52', 'E53', 'E54', 'E55', 
        #     'E56', 'E57', 'E58', 'E59', 'E60', 'E61', 'E62', 'E63', 'E64', 'E65', 'E66', 'E67', 'E68', 'E69', 'E70', 'E71', 'E72', 'E73', 
        #     'E74', 'E75', 'E76', 'E77', 'E78', 'E79', 'E80', 'E81', 'E82', 'E83', 'E84', 'E85', 'E86', 'E87', 'E88', 'E89', 'E90', 'E91', 
        #     'E92', 'E93', 'E94', 'E95', 'E96', 'E97', 'E98', 'E99', 'E100', 'E101', 'E102', 'E103', 'E104', 'E105', 'E106', 'E107', 'E108', 
        #     'E109', 'E110', 'E111', 'E112', 'E113', 'E114', 'E115', 'E116', 'E117', 'E118', 'E119', 'E120', 'E121', 'E122', 'E123', 'E124', 
        #     'E125', 'E126', 'E127', 'E128']

        # self.ch_names = ['FP1', 'FPZ', 'FP2', 
        # 'AF9', 'AF7', 'AF5', 'AF3', 'AF1', 'AFZ', 'AF2', 'AF4', 'AF6', 'AF8', 'AF10', \
        # 'F9', 'F7', 'F5', 'F3', 'F1', 'FZ', 'F2', 'F4', 'F6', 'F8', 'F10', \
        # 'FT9', 'FT7', 'FC5', 'FC3', 'FC1', 'FCZ', 'FC2', 'FC4', 'FC6', 'FT8', 'FT10', \
        # 'T9', 'T7', 'C5', 'C3', 'C1', 'CZ', 'C2', 'C4', 'C6', 'T8', 'T10', \
        # 'TP9', 'TP7', 'CP5', 'CP3', 'CP1', 'CPZ', 'CP2', 'CP4', 'CP6', 'TP8', 'TP10', \
        # 'P9', 'P7', 'P5', 'P3', 'P1', 'PZ', 'P2', 'P4', 'P6', 'P8', 'P10', \
        # 'PO9', 'PO7', 'PO5', 'PO3', 'PO1', 'POZ', 'PO2', 'PO4', 'PO6', 'PO8', 'PO10', \
        # 'O1', 'OZ', 'O2', 'O9', 'CB1', 'CB2', \
        # 'IZ', 'O10', 'T3', 'T5', 'T4', 'T6', 'M1', 'M2', 'A1', 'A2', \
        # 'CFC1', 'CFC2', 'CFC3', 'CFC4', 'CFC5', 'CFC6', 'CFC7', 'CFC8', \
        # 'CCP1', 'CCP2', 'CCP3', 'CCP4', 'CCP5', 'CCP6', 'CCP7', 'CCP8', \
        # 'T1', 'T2', 'FTT9h', 'TTP7h', 'TPP9h', 'FTT10h', 'TPP8h', 'TPP10h', \
        # "FP1-F7", "F7-T7", "T7-P7", "P7-O1", "FP2-F8", "F8-T8", "T8-P8", "P8-O2", "FP1-F3", "F3-C3", "C3-P3", "P3-O1", "FP2-F4", "F4-C4", "C4-P4", "P4-O2", \
        # 'pad', 'I1', 'I2']

        ch_names = ['EEG FP1', 'EEG FP2-REF', 'EEG F3-REF', 'EEG F4-REF', 'EEG C3-REF', 'EEG C4-REF', 'EEG P3-REF', 'EEG P4-REF', 'EEG O1-REF', 'EEG O2-REF', 'EEG F7-REF', \
                    'EEG F8-REF', 'EEG T3-REF', 'EEG T4-REF', 'EEG T5-REF', 'EEG T6-REF', 'EEG A1-REF', 'EEG A2-REF', 'EEG FZ-REF', 'EEG CZ-REF', 'EEG PZ-REF', 'EEG T1-REF', 'EEG T2-REF']
        self.ch_names = [name.split(' ')[-1].split('-')[0] for name in ch_names]

        if is_instruct:
            enc = tiktoken.get_encoding("gpt2")
            encode = lambda s: enc.encode(s, allowed_special={"<|endoftext|>"})
            # 50257 for [SEP]
            self.text = {
                1: torch.IntTensor([50257] + encode('Question: Is this EEG segment abnormal? Answer: Yes <|endoftext|>')),
                0: torch.IntTensor([50257] + encode('Question: Is this EEG segment abnormal? Answer: No <|endoftext|>'))
            }
            self.prompt = torch.IntTensor([50257] + encode('Question: Is this EEG segment abnormal? Answer:'))

        for fname in self.files:
            sample = pickle.load(open(os.path.join(self.root, fname), "rb"))
            X = sample["segments"]
            Y = sample["correctness"]

            for i in range(len(X)):
                x = X[i]
                        # Convert from volts to microvolts
                        
                x = x * 1e6
                # Z-score normalization per channel
                y = Y[i]
                # print('the shape of the segment is: ' + str(X[i].shape))
                # print('the label is: ' + str(Y[i]))
                y = 0 if y == 'Right' else 1
                self.data_list.append((x, y))

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, index):
        X, Y  = self.data_list[index]
        X = (X - X.mean(axis=1, keepdims=True)) / X.std(axis=1, keepdims=True)
        X = resample(X, 2000, axis=-1)
        X = X[:23]

        data = torch.FloatTensor(X / 100)
        time = data.size(1) // 200
        input_time = [i  for i in range(time) for _ in range(data.size(0))]
        data = rearrange(data, 'N (A T) -> (A N) T', T=200)

        ch_names = self.ch_names
        input_chans = list(ch_names) * time

        if not self.is_instruct:
            input_chans = torch.IntTensor(get_chans(input_chans))
            input_time = torch.IntTensor(input_time)

            gpt_mask = torch.tril(torch.ones(data.size(0), data.size(0))).view(1, data.size(0), data.size(0))
            num_chans = len(ch_names)
            for i in range(time):
                gpt_mask[:, i * num_chans:(i + 1) * num_chans,  i * num_chans:(i + 1) * num_chans] = 1
            return data, Y, input_chans, input_time, gpt_mask.bool()
        
        if self.is_val:
            text = self.prompt
        else:
            text = self.text[int(Y)]
            # pad text to text_max_len
            valid_text_len = text.size(0)
            if self.text_max_len > valid_text_len:
                text_pad = torch.full((self.text_max_len,), fill_value=50256)
                text_pad[:valid_text_len] = text
                text = text_pad

        # pad eeg to eeg_max_len
        valid_eeg_len = data.size(0)
        if self.eeg_max_len > data.size(0):
            X_eeg = torch.zeros((self.eeg_max_len, 200))
            X_eeg[:data.size(0)] = data
            eeg_mask = torch.ones(self.eeg_max_len)
            eeg_mask[valid_eeg_len:] = 0

            input_chans.extend(['pad'] * (self.eeg_max_len - data.size(0)))
            input_time.extend([0] * (self.eeg_max_len - data.size(0)))
        else:
            X_eeg = data
            eeg_mask = torch.ones(data.size(0))

        input_chans = torch.IntTensor(get_chans(input_chans))
        input_time = torch.IntTensor(input_time)

        num_tokens = X_eeg.size(0) + text.size(0)
        gpt_mask = torch.tril(torch.ones(num_tokens, num_tokens)).view(1, num_tokens, num_tokens)
        num_chans = len(ch_names)
        for i in range(time):
            gpt_mask[:, i * num_chans:(i + 1) * num_chans,  i * num_chans:(i + 1) * num_chans] = 1
        gpt_mask[:, :, valid_eeg_len:X_eeg.size(0)] = 0
        
        if self.is_val:
            return X_eeg, text, Y, input_chans, input_time, eeg_mask.bool(), gpt_mask.bool()
        
        Y_text = torch.full_like(text, fill_value=-1)
        prompt_len = self.prompt.size(0)
        Y_text[prompt_len - 1:valid_text_len - 1] = text[prompt_len:valid_text_len]
        return X_eeg, text, Y_text, input_chans, input_time, eeg_mask.bool(), gpt_mask.bool()
    

class TUEVLoader(Dataset):
    # spsw: spike and slow wave
    # gped: generalized periodic epileptiform discharge
    # pled: periodic lateralized epileptiform dischage
    # eyem: eye movement
    # artf: artifact
    # bckg: background
    # 1: spsw
    # 2: gped
    # 3: pled
    # 4: eyem
    # 5: artf
    # 6: bckg
    def __init__(self, root, files, sampling_rate=200, eeg_max_len=-1, text_max_len=-1, is_instruct=False, is_val=False):
        self.root = root
        self.files = files
        self.default_rate = 200
        self.sampling_rate = sampling_rate
        self.is_instruct = is_instruct
        self.is_val = is_val
        self.eeg_max_len = eeg_max_len
        self.text_max_len = text_max_len

        ch_names = ['EEG FP1', 'EEG FP2-REF', 'EEG F3-REF', 'EEG F4-REF', 'EEG C3-REF', 'EEG C4-REF', 'EEG P3-REF', 'EEG P4-REF', 'EEG O1-REF', 'EEG O2-REF', 'EEG F7-REF', \
                    'EEG F8-REF', 'EEG T3-REF', 'EEG T4-REF', 'EEG T5-REF', 'EEG T6-REF', 'EEG A1-REF', 'EEG A2-REF', 'EEG FZ-REF', 'EEG CZ-REF', 'EEG PZ-REF', 'EEG T1-REF', 'EEG T2-REF']
        self.ch_names = [name.split(' ')[-1].split('-')[0] for name in ch_names]

        if is_instruct:
            enc = tiktoken.get_encoding("gpt2")
            encode = lambda s: enc.encode(s, allowed_special={"<|endoftext|>"})
            # 50257 for [SEP]
            self.text = {
                0: torch.IntTensor([50257] + encode('Question: Which event type does this EEG segment belong to? Options: (A) spike and slow wave. (B) generalized periodic epileptiform discharge. (C) periodic lateralized epileptiform discharge. (D) eye movement. (E) artifact. (F) background. Answer: (A) <|endoftext|>')),
                1: torch.IntTensor([50257] + encode('Question: Which event type does this EEG segment belong to? Options: (A) spike and slow wave. (B) generalized periodic epileptiform discharge. (C) periodic lateralized epileptiform discharge. (D) eye movement. (E) artifact. (F) background. Answer: (B) <|endoftext|>')),
                2: torch.IntTensor([50257] + encode('Question: Which event type does this EEG segment belong to? Options: (A) spike and slow wave. (B) generalized periodic epileptiform discharge. (C) periodic lateralized epileptiform discharge. (D) eye movement. (E) artifact. (F) background. Answer: (C) <|endoftext|>')),
                3: torch.IntTensor([50257] + encode('Question: Which event type does this EEG segment belong to? Options: (A) spike and slow wave. (B) generalized periodic epileptiform discharge. (C) periodic lateralized epileptiform discharge. (D) eye movement. (E) artifact. (F) background. Answer: (D) <|endoftext|>')),
                4: torch.IntTensor([50257] + encode('Question: Which event type does this EEG segment belong to? Options: (A) spike and slow wave. (B) generalized periodic epileptiform discharge. (C) periodic lateralized epileptiform discharge. (D) eye movement. (E) artifact. (F) background. Answer: (E) <|endoftext|>')),
                5: torch.IntTensor([50257] + encode('Question: Which event type does this EEG segment belong to? Options: (A) spike and slow wave. (B) generalized periodic epileptiform discharge. (C) periodic lateralized epileptiform discharge. (D) eye movement. (E) artifact. (F) background. Answer: (F) <|endoftext|>')),
            }
            self.prompt = torch.IntTensor([50257] + encode('Question: Which event type does this EEG segment belong to? Options: (A) spike and slow wave. (B) generalized periodic epileptiform discharge. (C) periodic lateralized epileptiform discharge. (D) eye movement. (E) artifact. (F) background. Answer: ('))

    def __len__(self):
        return len(self.files)

    def __getitem__(self, index):
        sample = pickle.load(open(os.path.join(self.root, self.files[index]), "rb"))
        X = sample["signal"]
        Y = int(sample["label"][0] - 1)
        
        data = torch.FloatTensor(X / 100)
        time = data.size(1) // 200
        input_time = [i  for i in range(time) for _ in range(data.size(0))]
        data = rearrange(data, 'N (A T) -> (A N) T', T=200)

        ch_names = self.ch_names
        input_chans = list(ch_names) * time

        if not self.is_instruct:
            input_chans = torch.IntTensor(get_chans(input_chans))
            input_time = torch.IntTensor(input_time)

            gpt_mask = torch.tril(torch.ones(data.size(0), data.size(0))).view(1, data.size(0), data.size(0))
            num_chans = len(ch_names)
            for i in range(time):
                gpt_mask[:, i * num_chans:(i + 1) * num_chans,  i * num_chans:(i + 1) * num_chans] = 1
            return data, Y, input_chans, input_time, gpt_mask.bool()
        
        if self.is_val:
            text = self.prompt
        else:
            text = self.text[int(Y)]
            # pad text to text_max_len
            valid_text_len = text.size(0)
            if self.text_max_len > valid_text_len:
                text_pad = torch.full((self.text_max_len,), fill_value=50256)
                text_pad[:valid_text_len] = text
                text = text_pad

        # pad eeg to eeg_max_len
        valid_eeg_len = data.size(0)
        if self.eeg_max_len > data.size(0):
            X_eeg = torch.zeros((self.eeg_max_len, 200))
            X_eeg[:data.size(0)] = data
            eeg_mask = torch.ones(self.eeg_max_len)
            eeg_mask[valid_eeg_len:] = 0

            input_chans.extend(['pad'] * (self.eeg_max_len - data.size(0)))
            input_time.extend([0] * (self.eeg_max_len - data.size(0)))
        else:
            X_eeg = data
            eeg_mask = torch.ones(data.size(0))

        input_chans = torch.IntTensor(get_chans(input_chans))
        input_time = torch.IntTensor(input_time)

        num_tokens = X_eeg.size(0) + text.size(0)
        gpt_mask = torch.tril(torch.ones(num_tokens, num_tokens)).view(1, num_tokens, num_tokens)
        num_chans = len(ch_names)
        for i in range(time):
            gpt_mask[:, i * num_chans:(i + 1) * num_chans,  i * num_chans:(i + 1) * num_chans] = 1
        gpt_mask[:, :, valid_eeg_len:X_eeg.size(0)] = 0
        
        if self.is_val:
            return X_eeg, text, Y, input_chans, input_time, eeg_mask.bool(), gpt_mask.bool()
        
        Y_text = torch.full_like(text, fill_value=-1)
        prompt_len = self.prompt.size(0) - 1
        Y_text[prompt_len - 1:valid_text_len - 1] = text[prompt_len:valid_text_len]
        return X_eeg, text, Y_text, input_chans, input_time, eeg_mask.bool(), gpt_mask.bool()
    

class TUSLLoader(Dataset):
    def __init__(self, root, files, sampling_rate=200, eeg_max_len=-1, text_max_len=-1, is_instruct=False, is_val=False):
        self.root = root
        self.files = files
        self.default_rate = 200
        self.sampling_rate = sampling_rate
        self.is_instruct = is_instruct
        self.is_val = is_val
        self.eeg_max_len = eeg_max_len
        self.text_max_len = text_max_len

        ch_names = ['EEG FP1', 'EEG FP2-REF', 'EEG F3-REF', 'EEG F4-REF', 'EEG C3-REF', 'EEG C4-REF', 'EEG P3-REF', 'EEG P4-REF', 'EEG O1-REF', 'EEG O2-REF', 'EEG F7-REF', \
                    'EEG F8-REF', 'EEG T3-REF', 'EEG T4-REF', 'EEG T5-REF', 'EEG T6-REF', 'EEG A1-REF', 'EEG A2-REF', 'EEG FZ-REF', 'EEG CZ-REF', 'EEG PZ-REF', 'EEG T1-REF', 'EEG T2-REF']
        self.ch_names = [name.split(' ')[-1].split('-')[0] for name in ch_names]

        if is_instruct:
            enc = tiktoken.get_encoding("gpt2")
            encode = lambda s: enc.encode(s, allowed_special={"<|endoftext|>"})
            # 50257 for [SEP]
            self.text = {
                0: torch.IntTensor([50257] + encode('Question: Which type does this EEG segment belong to? Options: (A) background. (B) seizure. (C) slowing. Answer: (A) <|endoftext|>')),
                1: torch.IntTensor([50257] + encode('Question: Which type does this EEG segment belong to? Options: (A) background. (B) seizure. (C) slowing. Answer: (B) <|endoftext|>')),
                2: torch.IntTensor([50257] + encode('Question: Which type does this EEG segment belong to? Options: (A) background. (B) seizure. (C) slowing. Answer: (C) <|endoftext|>'))
            }
            self.prompt = torch.IntTensor([50257] + encode('Question: Which type does this EEG segment belong to? Options: (A) background. (B) seizure. (C) slowing. Answer: ('))

    def __len__(self):
        return len(self.files)

    def __getitem__(self, index):
        sample = pickle.load(open(os.path.join(self.root, self.files[index]), "rb"))
        X = sample["X"]
        Y = int(sample["y"])

        data = torch.FloatTensor(X / 100)
        time = data.size(1) // 200
        input_time = [i  for i in range(time) for _ in range(data.size(0))]
        data = rearrange(data, 'N (A T) -> (A N) T', T=200)

        ch_names = self.ch_names
        input_chans = list(ch_names) * time

        if not self.is_instruct:
            input_chans = torch.IntTensor(get_chans(input_chans))
            input_time = torch.IntTensor(input_time)

            gpt_mask = torch.tril(torch.ones(data.size(0), data.size(0))).view(1, data.size(0), data.size(0))
            num_chans = len(ch_names)
            for i in range(time):
                gpt_mask[:, i * num_chans:(i + 1) * num_chans,  i * num_chans:(i + 1) * num_chans] = 1
            return data, Y, input_chans, input_time, gpt_mask.bool()
        
        if self.is_val:
            text = self.prompt
        else:
            text = self.text[int(Y)]
            # pad text to text_max_len
            valid_text_len = text.size(0)
            if self.text_max_len > valid_text_len:
                text_pad = torch.full((self.text_max_len,), fill_value=50256)
                text_pad[:valid_text_len] = text
                text = text_pad

        # pad eeg to eeg_max_len
        valid_eeg_len = data.size(0)
        if self.eeg_max_len > data.size(0):
            X_eeg = torch.zeros((self.eeg_max_len, 200))
            X_eeg[:data.size(0)] = data
            eeg_mask = torch.ones(self.eeg_max_len)
            eeg_mask[valid_eeg_len:] = 0

            input_chans.extend(['pad'] * (self.eeg_max_len - data.size(0)))
            input_time.extend([0] * (self.eeg_max_len - data.size(0)))
        else:
            X_eeg = data
            eeg_mask = torch.ones(data.size(0))

        input_chans = torch.IntTensor(get_chans(input_chans))
        input_time = torch.IntTensor(input_time)

        num_tokens = X_eeg.size(0) + text.size(0)
        gpt_mask = torch.tril(torch.ones(num_tokens, num_tokens)).view(1, num_tokens, num_tokens)
        num_chans = len(ch_names)
        for i in range(time):
            gpt_mask[:, i * num_chans:(i + 1) * num_chans,  i * num_chans:(i + 1) * num_chans] = 1
        gpt_mask[:, :, valid_eeg_len:X_eeg.size(0)] = 0
        
        if self.is_val:
            return X_eeg, text, Y, input_chans, input_time, eeg_mask.bool(), gpt_mask.bool()
        
        Y_text = torch.full_like(text, fill_value=-1)
        prompt_len = self.prompt.size(0)
        Y_text[prompt_len - 1:valid_text_len - 1] = text[prompt_len:valid_text_len]
        return X_eeg, text, Y_text, input_chans, input_time, eeg_mask.bool(), gpt_mask.bool()


class HMCLoader(Dataset):
    def __init__(self, root, files, sampling_rate=200, eeg_max_len=-1, text_max_len=-1, is_instruct=False, is_val=False):
        self.root = root
        self.files = files
        self.default_rate = 200
        self.sampling_rate = sampling_rate
        self.is_instruct = is_instruct
        self.is_val = is_val
        self.eeg_max_len = eeg_max_len
        self.text_max_len = text_max_len

        if is_instruct:
            enc = tiktoken.get_encoding("gpt2")
            encode = lambda s: enc.encode(s, allowed_special={"<|endoftext|>"})
            # 50257 for [SEP]
            self.text = {
                0: torch.IntTensor([50257] + encode('Question: Which sleep type does this EEG segment belong to? Options: (A) Wake. (B) NREM-1. (C) NREM-2. (D) NREM-3. (E) REM. Answer: (A) <|endoftext|>')),
                1: torch.IntTensor([50257] + encode('Question: Which sleep type does this EEG segment belong to? Options: (A) Wake. (B) NREM-1. (C) NREM-2. (D) NREM-3. (E) REM. Answer: (B) <|endoftext|>')),
                2: torch.IntTensor([50257] + encode('Question: Which sleep type does this EEG segment belong to? Options: (A) Wake. (B) NREM-1. (C) NREM-2. (D) NREM-3. (E) REM. Answer: (C) <|endoftext|>')),
                3: torch.IntTensor([50257] + encode('Question: Which sleep type does this EEG segment belong to? Options: (A) Wake. (B) NREM-1. (C) NREM-2. (D) NREM-3. (E) REM. Answer: (D) <|endoftext|>')),
                4: torch.IntTensor([50257] + encode('Question: Which sleep type does this EEG segment belong to? Options: (A) Wake. (B) NREM-1. (C) NREM-2. (D) NREM-3. (E) REM. Answer: (E) <|endoftext|>')),
            }
            self.prompt = torch.IntTensor([50257] + encode('Question: Which sleep type does this EEG segment belong to? Options: (A) Wake. (B) NREM-1. (C) NREM-2. (D) NREM-3. (E) REM. Answer: ('))

    def __len__(self):
        return len(self.files)

    def __getitem__(self, index):
        sample = pickle.load(open(os.path.join(self.root, self.files[index]), "rb"))
        X = sample["X"]
        Y = int(sample["y"])

        data = torch.FloatTensor(X / 100)
        time = data.size(1) // 200
        input_time = [i  for i in range(time) for _ in range(data.size(0))]
        data = rearrange(data, 'N (A T) -> (A N) T', T=200)

        ch_names = sample["ch_names"]
        input_chans = list(ch_names) * time

        if not self.is_instruct:
            input_chans = torch.IntTensor(get_chans(input_chans))
            input_time = torch.IntTensor(input_time)

            gpt_mask = torch.tril(torch.ones(data.size(0), data.size(0))).view(1, data.size(0), data.size(0))
            num_chans = len(ch_names)
            for i in range(time):
                gpt_mask[:, i * num_chans:(i + 1) * num_chans,  i * num_chans:(i + 1) * num_chans] = 1
            return data, Y, input_chans, input_time, gpt_mask.bool()
        
        if self.is_val:
            text = self.prompt
        else:
            text = self.text[int(Y)]
            # pad text to text_max_len
            valid_text_len = text.size(0)
            if self.text_max_len > valid_text_len:
                text_pad = torch.full((self.text_max_len,), fill_value=50256)
                text_pad[:valid_text_len] = text
                text = text_pad

        # pad eeg to eeg_max_len
        valid_eeg_len = data.size(0)
        if self.eeg_max_len > data.size(0):
            X_eeg = torch.zeros((self.eeg_max_len, 200))
            X_eeg[:data.size(0)] = data
            eeg_mask = torch.ones(self.eeg_max_len)
            eeg_mask[valid_eeg_len:] = 0

            input_chans.extend(['pad'] * (self.eeg_max_len - data.size(0)))
            input_time.extend([0] * (self.eeg_max_len - data.size(0)))
        else:
            X_eeg = data
            eeg_mask = torch.ones(data.size(0))

        input_chans = torch.IntTensor(get_chans(input_chans))
        input_time = torch.IntTensor(input_time)

        num_tokens = X_eeg.size(0) + text.size(0)
        gpt_mask = torch.tril(torch.ones(num_tokens, num_tokens)).view(1, num_tokens, num_tokens)
        num_chans = len(ch_names)
        for i in range(time):
            gpt_mask[:, i * num_chans:(i + 1) * num_chans,  i * num_chans:(i + 1) * num_chans] = 1
        gpt_mask[:, :, valid_eeg_len:X_eeg.size(0)] = 0
        
        if self.is_val:
            return X_eeg, text, Y, input_chans, input_time, eeg_mask.bool(), gpt_mask.bool()
        
        Y_text = torch.full_like(text, fill_value=-1)
        prompt_len = self.prompt.size(0) - 1
        Y_text[prompt_len - 1:valid_text_len - 1] = text[prompt_len:valid_text_len]
        return X_eeg, text, Y_text, input_chans, input_time, eeg_mask.bool(), gpt_mask.bool()


class WorkloadLoader(Dataset):
    def __init__(self, root, files, sampling_rate=200, eeg_max_len=-1, text_max_len=-1, is_instruct=False, is_val=False):
        self.root = root
        self.files = files
        self.default_rate = 200
        self.sampling_rate = sampling_rate
        self.is_instruct = is_instruct
        self.is_val = is_val
        self.eeg_max_len = eeg_max_len
        self.text_max_len = text_max_len

        if is_instruct:
            enc = tiktoken.get_encoding("gpt2")
            encode = lambda s: enc.encode(s, allowed_special={"<|endoftext|>"})
            # 50257 for [SEP]
            self.text = {
                1: torch.IntTensor([50257] + encode('Question: Is this EEG segment of high workload? Answer: Yes <|endoftext|>')),
                0: torch.IntTensor([50257] + encode('Question: Is this EEG segment of high workload? Answer: No <|endoftext|>')),
            }
            self.prompt = torch.IntTensor([50257] + encode('Question: Is this EEG segment of high workload? Answer:'))

    def __len__(self):
        return len(self.files)

    def __getitem__(self, index):
        sample = pickle.load(open(os.path.join(self.root, self.files[index]), "rb"))
        X = sample["X"]
        Y = int(sample["y"])

        data = torch.FloatTensor(X / 100)
        time = data.size(1) // 200
        input_time = [i  for i in range(time) for _ in range(data.size(0))]
        data = rearrange(data, 'N (A T) -> (A N) T', T=200)

        ch_names = sample["ch_names"]
        input_chans = list(ch_names) * time

        if not self.is_instruct:
            input_chans = torch.IntTensor(get_chans(input_chans))
            input_time = torch.IntTensor(input_time)

            gpt_mask = torch.tril(torch.ones(data.size(0), data.size(0))).view(1, data.size(0), data.size(0))
            num_chans = len(ch_names)
            for i in range(time):
                gpt_mask[:, i * num_chans:(i + 1) * num_chans,  i * num_chans:(i + 1) * num_chans] = 1
            return data, Y, input_chans, input_time, gpt_mask.bool()
        
        if self.is_val:
            text = self.prompt
        else:
            text = self.text[int(Y)]
            # pad text to text_max_len
            valid_text_len = text.size(0)
            if self.text_max_len > valid_text_len:
                text_pad = torch.full((self.text_max_len,), fill_value=50256)
                text_pad[:valid_text_len] = text
                text = text_pad

        # pad eeg to eeg_max_len
        valid_eeg_len = data.size(0)
        if self.eeg_max_len > data.size(0):
            X_eeg = torch.zeros((self.eeg_max_len, 200))
            X_eeg[:data.size(0)] = data
            eeg_mask = torch.ones(self.eeg_max_len)
            eeg_mask[valid_eeg_len:] = 0

            input_chans.extend(['pad'] * (self.eeg_max_len - data.size(0)))
            input_time.extend([0] * (self.eeg_max_len - data.size(0)))
        else:
            X_eeg = data
            eeg_mask = torch.ones(data.size(0))

        input_chans = torch.IntTensor(get_chans(input_chans))
        input_time = torch.IntTensor(input_time)

        num_tokens = X_eeg.size(0) + text.size(0)
        gpt_mask = torch.tril(torch.ones(num_tokens, num_tokens)).view(1, num_tokens, num_tokens)
        num_chans = len(ch_names)
        for i in range(time):
            gpt_mask[:, i * num_chans:(i + 1) * num_chans,  i * num_chans:(i + 1) * num_chans] = 1
        gpt_mask[:, :, valid_eeg_len:X_eeg.size(0)] = 0
        
        if self.is_val:
            return X_eeg, text, Y, input_chans, input_time, eeg_mask.bool(), gpt_mask.bool()
        
        Y_text = torch.full_like(text, fill_value=-1)
        prompt_len = self.prompt.size(0)
        Y_text[prompt_len - 1:valid_text_len - 1] = text[prompt_len:valid_text_len]
        return X_eeg, text, Y_text, input_chans, input_time, eeg_mask.bool(), gpt_mask.bool()
    
def resample_data(data, original_rate=250, target_rate=200):
    """
    Resamples EEG data from original_rate to target_rate.
    Input shape: (Channels, Time)
    """
    if original_rate == target_rate:
        return data
    
    num_samples = data.shape[1]
    new_num_samples = int(num_samples * target_rate / original_rate)
    return resample(data, new_num_samples, axis=1)

def extract_single_window(data, index, window_size, step_size):
    """
    Your custom extraction logic.
    Input data shape: (Trials, Channels, Time)
    """
    # Calculate start/end based on your formula
    start = (index - 1) * step_size
    end = start + window_size
    
    # Validation
    if end > data.shape[2]:
        raise ValueError(f"Window index {index} (samples {start}-{end}) exceeds trial length {data.shape[2]}")
        
    # Return shape: (Trials, Channels, Window_Size)
    return data[:, :, start:end]

# ==========================================
# 2. Main Dataset Class
# ==========================================

class BCICIV2bLoader(Dataset):
    def __init__(self, subject_ids, root_path, is_instruct=False, is_val=False, 
                 eeg_max_len=1024, text_max_len=128):
        
        self.root_path = root_path
        self.subject_ids = subject_ids
        
        # --- Windowing Parameters ---
        self.window_index = 8
        self.window_size = 500  # 2.0 seconds @ 250Hz
        self.step_size = 100
        
        self.original_rate = 250
        self.target_rate = 200  # NeuroLM Requirement
        
        self.is_instruct = is_instruct
        self.is_val = is_val
        self.eeg_max_len = eeg_max_len
        self.text_max_len = text_max_len
        self.ch_names = ['C3', 'CZ', 'C4']

        # 1. Load Data
        print(f"Loading data for subjects: {subject_ids}...")
        self.X, self.y = self.load_subject_data(self.subject_ids)
        print(f"Loaded. X: {self.X.shape}, Y: {self.y.shape}")

        # 2. Dynamic Label Mapping (Fixes the 0.0 accuracy issue)
        unique_labels = np.unique(self.y)
        unique_labels.sort()
        
        print(f"DEBUG: Unique labels found: {unique_labels}")
        
        # Map unique labels to 0 and 1
        if len(unique_labels) >= 2:
            self.label_map = {unique_labels[0]: 0, unique_labels[1]: 1}
        else:
            # Fallback if split has only 1 class
            self.label_map = {unique_labels[0]: 0}
            
        print(f"DEBUG: Applied Label Map: {self.label_map}")

        # 3. Setup Tokenizer
        if is_instruct:
            enc = tiktoken.get_encoding("gpt2")
            encode = lambda s: enc.encode(s, allowed_special={"<|endoftext|>"})
            q_str = 'Question: Is the subject imagining a right hand movement? Answer:'
            
            self.text = {
                0: torch.IntTensor([50257] + encode(q_str + ' No <|endoftext|>')),
                1: torch.IntTensor([50257] + encode(q_str + ' Yes <|endoftext|>'))
            }
            self.prompt = torch.IntTensor([50257] + encode(q_str))

    def load_subject_data(self, subject_ids):
        data_list = []
        labels_list = []
        
        Subj_id_map = {'S01': 1, 'S02': 2, 'S03': 3, 'S04': 4, 'S05': 5, 
                       'S06': 6, 'S07': 7, 'S08': 8, 'S09': 9}

        for subj_str in subject_ids:
            sid = Subj_id_map.get(subj_str, 1)
            
            # Construct paths
            paths = [
                (os.path.join(self.root_path, f"Subj{sid}_1_X.npy"), os.path.join(self.root_path, f"Subj{sid}_1_y.npy")),
                (os.path.join(self.root_path, f"Subj{sid}_2_X.npy"), os.path.join(self.root_path, f"Subj{sid}_2_y.npy"))
            ]

            for x_path, y_path in paths:
                if not os.path.exists(x_path): continue
                
                x_data = np.load(x_path)
                y_data = np.load(y_path)
                
                # Check for NaNs
                x_data = np.nan_to_num(x_data, nan=0.0)

                # Ensure (Trials, Channels, Time)
                if x_data.shape[1] > x_data.shape[2] and x_data.shape[2] == 3:
                     x_data = x_data.transpose(0, 2, 1)

                # Slice the specific window
                x_sliced = extract_single_window(x_data, self.window_index, self.window_size, self.step_size)
                
                data_list.append(x_sliced)
                labels_list.append(y_data)

        if not data_list:
            return np.array([]), np.array([])

        X_all = np.concatenate(data_list, axis=0)
        y_all = np.concatenate(labels_list, axis=0)
        return X_all, y_all

    def __len__(self):
        return len(self.y)

    def __getitem__(self, index):
        # Add this at the very start of __getitem__
        if index == 0:
            print(f"DEBUG CHECK: Is_Val={self.is_val}, Label Found={self.y[index]}")
        X_raw = self.X[index]
        Y_raw = self.y[index]
        
        # 1. Resample to 200Hz (Required for NeuroLM tokenization)
        # 500 samples (250Hz) -> 400 samples (200Hz)
        new_samples = int(X_raw.shape[1] * self.target_rate / self.original_rate)
        X_resampled = resample(X_raw, new_samples, axis=1)
        
        # 2. Skip Normalization (As requested)
        # We assume X_resampled is already in the correct range [-1, 1] or similar
        data = torch.FloatTensor(X_resampled)

        # 3. Rearrange for NeuroLM: (Channels, Time) -> (Seq, 200)
        num_seconds = data.size(1) // 200
        input_time = [i for i in range(num_seconds) for _ in range(data.size(0))]
        data = rearrange(data, 'C (S T) -> (S C) T', T=200)
        
        # 4. Metadata
        ch_names_list = self.ch_names * num_seconds
        input_chans = torch.IntTensor(get_chans(ch_names_list))
        input_time = torch.IntTensor(input_time)
        
        # 5. Masking
        gpt_mask = torch.tril(torch.ones(data.size(0), data.size(0))).view(1, data.size(0), data.size(0))
        num_chans = len(self.ch_names)
        for i in range(num_seconds):
            s, e = i*num_chans, (i+1)*num_chans
            gpt_mask[:, s:e, s:e] = 1

        if not self.is_instruct:
            return data, Y_raw, input_chans, input_time, gpt_mask.bool()

        # 6. Instruct Mode (Using Dynamic Map)
        label_idx = int(Y_raw)
        
        if self.is_val:
            text = self.prompt
        else:
            text = self.text[label_idx]
            # Pad Text
            if self.text_max_len > text.size(0):
                text_pad = torch.full((self.text_max_len,), fill_value=50256)
                text_pad[:text.size(0)] = text
                text = text_pad
        
        # Pad EEG
        valid_eeg_len = data.size(0)
        if self.eeg_max_len > valid_eeg_len:
            X_eeg = torch.zeros((self.eeg_max_len, 200))
            X_eeg[:valid_eeg_len] = data
            eeg_mask = torch.ones(self.eeg_max_len)
            eeg_mask[valid_eeg_len:] = 0
            
            input_chans_pad = torch.cat([input_chans, torch.zeros(self.eeg_max_len - valid_eeg_len, dtype=torch.int)])
            input_time_pad = torch.cat([input_time, torch.zeros(self.eeg_max_len - valid_eeg_len, dtype=torch.int)])
        else:
            X_eeg = data
            eeg_mask = torch.ones(valid_eeg_len)
            input_chans_pad = input_chans
            input_time_pad = input_time

        # Unified Mask
        num_tokens = X_eeg.size(0) + text.size(0)
        full_mask = torch.tril(torch.ones(num_tokens, num_tokens)).view(1, num_tokens, num_tokens)
        for i in range(num_seconds):
            s, e = i*num_chans, (i+1)*num_chans
            full_mask[:, s:e, s:e] = 1
        full_mask[:, :, valid_eeg_len:X_eeg.size(0)] = 0

        if self.is_val:
            return X_eeg, text, label_idx, input_chans_pad, input_time_pad, eeg_mask.bool(), full_mask.bool()

        # Training Labels
        Y_text = torch.full_like(text, fill_value=-1)
        prompt_len = self.prompt.size(0)
        real_len = (text != 50256).sum() if self.text_max_len > 0 else text.size(0)
        Y_text[prompt_len-1 : real_len-1] = text[prompt_len : real_len]
        
        return X_eeg, text, Y_text, input_chans_pad, input_time_pad, eeg_mask.bool(), full_mask.bool()
