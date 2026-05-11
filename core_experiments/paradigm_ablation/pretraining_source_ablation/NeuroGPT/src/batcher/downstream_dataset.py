import os
import pdb
import numpy as np
from batcher.base1 import EEGDataset
from scipy.io import loadmat
from scipy.signal import butter, filtfilt
import pickle
from scipy.signal import resample
import torch


def load_BCI4_2b(subject_ids):

    Subj_id = {'S01': 1, 'S02': 2, 'S03': 3, 'S04': 4, 'S05': 5, 'S06': 6, 'S07': 7, 'S08': 8, 'S09': 9}
    data_subjs = []
    labels_subjs = []

    for subject_id in subject_ids:
        subject_id = Subj_id[subject_id]
        x1 = np.load("/mnt/scratch2/users/xw2336/LLM_Eva/Data/BCICIV_2b/Subj{0}_1_X.npy".format(subject_id))
        x2 = np.load("/mnt/scratch2/users/xw2336/LLM_Eva/Data/BCICIV_2b/Subj{0}_2_X.npy".format(subject_id))

        y1 = np.load("/mnt/scratch2/users/xw2336/LLM_Eva/Data/BCICIV_2b/Subj{0}_1_y.npy".format(subject_id))
        y2 = np.load("/mnt/scratch2/users/xw2336/LLM_Eva/Data/BCICIV_2b/Subj{0}_2_y.npy".format(subject_id))

        x_combined_data = np.concatenate([x1, x2], axis=0)
        y_combined_labels = np.concatenate([y1, y2], axis=0)
        data_subjs.append(x_combined_data)
        labels_subjs.append(y_combined_labels)


    x_combined_data = np.concatenate([data_subjs[i] for i in range(len(data_subjs))], axis=0)
    y_combined_labels = np.concatenate([labels_subjs[i] for i in range(len(labels_subjs))], axis=0)

    return x_combined_data, y_combined_labels

class MotorImageryDataset(EEGDataset):
    def __init__(self, filenames, sample_keys, chunk_len=500, num_chunks=10, ovlp=50, root_path="", gpt_only=True):
        super().__init__(filenames, sample_keys, chunk_len, num_chunks, ovlp, root_path=root_path, gpt_only=gpt_only)

        self.data_all = []
        for fn in self.filenames:
            self.data_all.append(np.load(fn))

        self.mi_types = {769: 'left', 770: 'right',
                         771: 'foot', 772: 'tongue', 1023: 'rejected'} # , 783: 'unknown', 1023: 'rejected'
        # Types of motor imagery
        self.labels_string2int = {'left': 0, 'right': 1,
                         'foot': 2, 'tongue':3 } #, 'unknown': -1
        self.Fs = 250  # 250Hz from original paper
        self.P = np.load("../inputs/tMatrix_value.npy")

        self.trials, self.labels, self.num_trials_per_sub = self.get_trials_all()
        # keys of data ['s', 'etyp', 'epos', 'edur', 'artifacts']

    def __len__(self):
        return sum(self.num_trials_per_sub)

    def __getitem__(self, idx):
        return self.preprocess_sample(self.trials[idx], self.num_chunks, self.labels[idx])

    def map2pret(self, data):
        return np.matmul(self.P, data) # 22x22, 22xTime

    def get_trials_from_single_subj(self, sub_id):
        raw = self.data_all[sub_id]['s'].T
        events_type = self.data_all[sub_id]['etyp'].T
        events_position = self.data_all[sub_id]['epos'].T
        events_duration = self.data_all[sub_id]['edur'].T
        artifacts = self.data_all[sub_id]['artifacts'].T
        # Channel default is C3
        startrial_code = 768
        starttrial_events = events_type == startrial_code
        idxs = [i for i, x in enumerate(starttrial_events[0]) if x]

        trial_labels = self.get_labels(sub_id)

        trials = []
        classes = []
        for j, index in enumerate(idxs):
            try:
                # print(index)
                # type_e = events_type[0, index+1]
                # class_e = self.mi_types[type_e]
                # if type_e == 1023:
                #     continue
                # classes.append(self.labels_string2int[class_e])
                classes.append(trial_labels[j])

                start = events_position[0, index]
                stop = start + events_duration[0, index]
                trial = raw[:22, start+500 : stop-375]
                #add band-pass filter
                # self.bandpass_filter(trial, lowcut=4, highcut=40, fs=250, order=5)
                trials.append(trial)
            except:
                # print("Cannot load trial")
                continue
        return trials, classes

    def get_labels(self, sub_id):
        label_path = self.root_path + "true_labels/"
        base_name = os.path.basename(self.filenames[sub_id])
        sub_name = os.path.splitext(base_name)[0]
        labels = loadmat(label_path + sub_name +".mat")["classlabel"]
        return labels.squeeze() - 1

    def get_trials_all(self):
        trials_all = []
        labels_all = []
        total_num = []
        for sub_id in range(len(self.data_all)):
            trials, labels = self.get_trials_from_single_subj(sub_id)
            total_num.append(len(trials))
            
            trials_all.append(np.array(trials))
            labels_all.append(np.array(labels))
        # reordered_data = self.reorder_channels(np.vstack(trials_all))
        trials_all_arr = np.vstack(trials_all)
        # map to same channel configuration as pretraining
        trials_all_arr = self.map2pret(trials_all_arr)
        return self.normalize(trials_all_arr), np.array(labels_all).flatten(), total_num
    
    # def normalize(self, data):
    #     return (data - np.mean(data)) / np.std(data)
    
    def bandpass_filter(self, data, lowcut, highcut, fs, order=5):
        """
        Apply a bandpass filter to the data.
        
        Parameters:
        - data: The EEG signal
        - lowcut: Low cut-off frequency
        - highcut: High cut-off frequency
        - fs: Sampling rate (frequency)
        - order: Order of the filter
        
        Returns:
        - Filtered data
        """
        nyq = 0.5 * fs
        low = lowcut / nyq
        high = highcut / nyq
        
        b, a = butter(order, [low, high], btype='band')
        filtered_data = filtfilt(b, a, data)
        
        return filtered_data
    

class TUABDataset_PT(EEGDataset):
    def __init__(self, root, filenames, sample_keys, chunk_len=500, num_chunks=4, ovlp=50, root_path="", gpt_only=True):
        super().__init__(filenames, sample_keys, chunk_len, num_chunks, ovlp, root_path=root_path, gpt_only=gpt_only)

        self.root = root
        self.files = filenames

        self.Fs = 250  # 250Hz from original paper
        self.num_chunks = num_chunks
        self.chunk_len = chunk_len
        self.P = np.load("../inputs/tMatrix_value.npy")  # not used for 3 channels

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        sample = pickle.load(open(os.path.join(self.root, self.files[idx]), "rb"))
        X = sample["X"]
        Y = sample["y"]

        # select only 3 channels
        indices = [4, 5, 19]  # example channels
        X_sel = X[indices]
        # X_sel = X_sel[:, :500]
        

        output = self.preprocess_sample(X_sel, self.num_chunks, Y)
        
        # Check if output is empty
        if not output: 
            print(f"WARNING: File {self.files[idx]} resulted in empty output.")
            print(f"Original Shape: {X_sel.shape}")
            print(f"Required approx length for {self.num_chunks} chunks: {self.num_chunks * 500}")
            # You might want to return dummy data or handle this specific error
            # For now, let's see why it's empty
            
        return output

    def map2pret(self, data):
        return np.matmul(self.P, data)  # Not used when using 3 channels
    

# class TUABDataset(EEGDataset):
#     def __init__(self, root, filenames, sample_keys, chunk_len=128, num_chunks=4, ovlp=0, root_path="", gpt_only=True):
#         """
#         Finalized for NeurIPS. Aligns with Neuro-GPT pretraining specs.
#         Args:
#             chunk_len: Must be 128 (1 second at 128Hz).
#             num_chunks: Sequence length (number of tokens).
#         """
#         super().__init__(filenames, sample_keys, chunk_len, num_chunks, ovlp, root_path=root_path, gpt_only=gpt_only)

#         self.root = root
#         self.files = filenames

#         # --- Neuro-GPT Calibration Parameters ---
#         self.target_fs = 128    # Model's pre-trained frequency
#         self.original_fs = 200  # The rate from your make_TUAB.py script
#         self.num_chunks = num_chunks
#         self.chunk_len = chunk_len
#         self.do_normalization = True # Standard for TUH-based models

#     def __len__(self):
#         return len(self.files)

#     def __getitem__(self, idx):
#         # 1. Load data [Channels, Time] from your saved .pkl files
#         file_path = os.path.join(self.root, self.files[idx])
#         sample = pickle.load(open(file_path, "rb"))
#         X = sample["X"]  
#         Y = sample["y"]

#         # 2. Channel Alignment
#         # Neuro-GPT uses the standard 22-channel clinical montage.
#         # Slicing ensures spatial embeddings match the pre-trained weights.
#         X_sel = X[:22, :] 

#         # 3. Call the standardized preprocess_sample
#         output = self.preprocess_sample(X_sel, self.num_chunks, Y)
        
#         # 4. Robust handling for shorter recordings
#         if not output: 
#             return self.__getitem__(np.random.randint(0, len(self.files)))
            
#         return output

#     def map2pret(self, data):
#         return np.matmul(self.P, data)  # Not used when using 3 channels

class TUABDataset(EEGDataset):
    def __init__(self, root, filenames, sample_keys, chunk_len=128, num_chunks=4, ovlp=0, root_path="", gpt_only=True):
        super().__init__(filenames, sample_keys, chunk_len, num_chunks, ovlp, root_path=root_path, gpt_only=gpt_only)

        self.root = root
        self.files = filenames

        # --- Standardized Parameters ---
        self.target_fs = 250    # Model's pre-trained frequency
        self.original_fs = 200  # Confirmed from make_TUAB.py (raw.resample(200))
        self.num_chunks = num_chunks
        self.chunk_len = chunk_len
        self.do_normalization = True 

        # NeuroGPT expected order: Fp1, Fp2, F7, F3, Fz, F4, F8, T1, T3, C3, Cz, C4, T4, T2, T5, P3, Pz, P4, T6, O1, Oz, O2
        # make_TUAB.py order: FP1(0), FP2(1), F3(2), F4(3), C3(4), C4(5), P3(6), P4(7), O1(8), O2(9), 
        # F7(10), F8(11), T3(12), T4(13), T5(14), T6(15), A1(16), A2(17), FZ(18), CZ(19), PZ(20), T1(21), T2(22)
        # Note: Oz is missing in the TUAB generated files.
        self.channel_mapping = [0, 1, 10, 2, 18, 3, 11, 21, 12, 4, 19, 5, 13, 22, 14, 6, 20, 7, 15, 8, -1, 9]

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        file_path = os.path.join(self.root, self.files[idx])
        sample = pickle.load(open(file_path, "rb"))
        X = sample["X"]  
        Y = sample["y"]

        # 1. Byteorder / Endianness fix (Crucial for cross-platform model training)
        if X.dtype.byteorder == '>' or (X.dtype.byteorder == '|' and not X.dtype.str.endswith('f4')):
            X = X.byteswap().newbyteorder('=')
        X = np.array(X, dtype=np.float32, copy=True)

        # 2. Resample to target_fs
        target_length = int(X.shape[-1] * (self.target_fs / self.original_fs))
        X = resample(X, target_length, axis=-1)

        # 3. Channel Alignment & Padding for missing Oz
        X_aligned = np.zeros((22, X.shape[1]), dtype=np.float32)
        for i, source_idx in enumerate(self.channel_mapping):
            if source_idx != -1:
                X_aligned[i, :] = X[source_idx, :]
            # If -1 (Oz), it remains zeros
            
        X_sel = np.ascontiguousarray(X_aligned)

        # 4. Standardized Preprocessing
        output = self.preprocess_sample(X_sel, self.num_chunks, Y)
        
        # 5. Robust handling for shorter recordings
        if not output: 
            return self.__getitem__(np.random.randint(0, len(self.files)))
            
        return output

# class TUEVDataset(EEGDataset):
#     def __init__(self, root, filenames, sample_keys, chunk_len=128, num_chunks=4, ovlp=0, root_path="", gpt_only=True):
#         super().__init__(filenames, sample_keys, chunk_len, num_chunks, ovlp, root_path=root_path, gpt_only=gpt_only)

#         self.root = root
#         self.files = filenames

#         # --- Standardized Parameters ---
#         self.target_fs = 250    # Match the frequency used in your pre-trained model
#         self.original_fs = 200  # Confirmed from make_TUEV.py (Rawdata.resample(200))
#         self.num_chunks = num_chunks
#         self.chunk_len = chunk_len
#         self.do_normalization = True 

#         # NeuroGPT expected order: Fp1, Fp2, F7, F3, Fz, F4, F8, T1, T3, C3, Cz, C4, T4, T2, T5, P3, Pz, P4, T6, O1, Oz, O2
#         # make_TUEV.py order: FP1(0), FP2(1), F3(2), F4(3), C3(4), C4(5), P3(6), P4(7), O1(8), O2(9), 
#         # F7(10), F8(11), T3(12), T4(13), T5(14), T6(15), A1(16), A2(17), FZ(18), CZ(19), PZ(20), T1(21), T2(22)
#         # Note: Oz is missing in the TUEV generated files.
#         self.channel_mapping = [0, 1, 10, 2, 18, 3, 11, 21, 12, 4, 19, 5, 13, 22, 14, 6, 20, 7, 15, 8, -1, 9]

#     def __len__(self):
#         return len(self.files)

#     def __getitem__(self, idx):
#         file_path = os.path.join(self.root, self.files[idx])
#         sample = pickle.load(open(file_path, "rb"))
#         X = sample["signal"]
        
#         # 1. Byteorder / Endianness fix
#         if X.dtype.byteorder == '>' or (X.dtype.byteorder == '|' and not X.dtype.str.endswith('f4')):
#             X = X.byteswap().newbyteorder('=')
#         X = np.array(X, dtype=np.float32, copy=True)
        
#         # 2. Resample to target_fs
#         target_length = int(X.shape[-1] * (self.target_fs / self.original_fs))
#         X = resample(X, target_length, axis=-1)
        
#         # 3. Channel Alignment & Padding for missing Oz
#         X_aligned = np.zeros((22, X.shape[1]), dtype=np.float32)
#         for i, source_idx in enumerate(self.channel_mapping):
#             if source_idx != -1:
#                 X_aligned[i, :] = X[source_idx, :]
#             # If -1 (Oz), it remains zeros
            
#         X_sel = np.ascontiguousarray(X_aligned)
            
#         Y = int(sample["label"][0] - 1)

#         # 4. Standardized Preprocessing
#         output = self.preprocess_sample(X_sel, self.num_chunks, Y)
        
#         # 5. Robust handling for shorter recordings
#         if not output: 
#             return self.__getitem__(np.random.randint(0, len(self.files)))
            
#         return output

class TUEVDataset(EEGDataset):
    def __init__(self, root, filenames, sample_keys, chunk_len=500, num_chunks=4, ovlp=0, root_path="", gpt_only=True):
        super().__init__(filenames, sample_keys, chunk_len, num_chunks, ovlp, root_path=root_path, gpt_only=gpt_only)

        self.root = root
        self.files = filenames

        # --- Re-aligned to your specific Pre-trained Model ---
        self.target_fs = 250    # Pretrained model expects 250Hz
        self.original_fs = 200  # From make_TUEV.py (fs=200)
        self.num_chunks = num_chunks
        self.chunk_len = chunk_len # Set to 500 via bash script
        self.do_normalization = True 

        # Exact 10-20 channel mapping accommodating make_TUEV.py output
        self.channel_mapping = [0, 1, 10, 2, 18, 3, 11, 21, 12, 4, 19, 5, 13, 22, 14, 6, 20, 7, 15, 8, -1, 9]

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        file_path = os.path.join(self.root, self.files[idx])
        sample = pickle.load(open(file_path, "rb"))
        X = sample["signal"]
        
        # 1. Byteorder / Endianness fix
        if X.dtype.byteorder == '>' or (X.dtype.byteorder == '|' and not X.dtype.str.endswith('f4')):
            X = X.byteswap().newbyteorder('=')
        X = np.array(X, dtype=np.float32, copy=True)

        # 2. Safety Fixes for Raw TUEV Amplitudes
        X = np.nan_to_num(X, nan=0.0, posinf=800.0, neginf=-800.0)
        X = np.clip(X, -800.0, 800.0)
        
        # 3. Resample to 250 Hz
        target_length = int(X.shape[-1] * (self.target_fs / self.original_fs))
        X = resample(X, target_length, axis=-1)
        
        # 4. Channel Alignment & Padding for missing Oz
        X_aligned = np.zeros((22, X.shape[1]), dtype=np.float32)
        for i, source_idx in enumerate(self.channel_mapping):
            if source_idx != -1:
                X_aligned[i, :] = X[source_idx, :]
            
        X_sel = np.ascontiguousarray(X_aligned)
            
        # 5. Label Extraction (1-6 to 0-5 mapping)
        Y = int(sample["label"][0] - 1)

        output = self.preprocess_sample(X_sel, self.num_chunks, Y)
        
        if not output: 
            return self.__getitem__(np.random.randint(0, len(self.files)))

        # Add the NEW check right here to catch the all-zero mask
        if torch.sum(output["attention_mask"]) == 0:
            return self.__getitem__(np.random.randint(0, len(self.files)))  
        return output
    
class BCIIV2bDataset(EEGDataset):
    def __init__(self, subject_ids, sample_keys=None, chunk_len=500, num_chunks=10, ovlp=50, root_path="", gpt_only=True):
        super().__init__(subject_ids, sample_keys, chunk_len, num_chunks, ovlp, root_path=root_path, gpt_only=gpt_only)

        self.subject_ids = subject_ids
        self.num_chunks = num_chunks

        self.data_subjs = []
        self.labels_subjs = []

        self.Fs = 250  # 250Hz from original paper
        self.P = np.load("/homes/xw2336/LLM_eva/NeuroGPT/inputs/tMatrix_value.npy")
        self.X, self.y = self.load_subject_data(self.subject_ids)
    
    def load_subject_data(self, subject_ids):
        Subj_id = {'S01': 1, 'S02': 2, 'S03': 3, 'S04': 4, 'S05': 5, 'S06': 6, 'S07': 7, 'S08': 8, 'S09': 9}
        
        for subject_id in subject_ids:
            sub_int = Subj_id[subject_id]

            x1 = np.load(f"/homes/xw2336/data/BCICIV_2b/Subj{sub_int}_1_X.npy")
            x2 = np.load(f"/homes/xw2336/data/BCICIV_2b/Subj{sub_int}_2_X.npy")

            y1 = np.load(f"/homes/xw2336/data/BCICIV_2b/Subj{sub_int}_1_y.npy")
            y2 = np.load(f"/homes/xw2336/data/BCICIV_2b/Subj{sub_int}_2_y.npy")

            x_combined_data = np.concatenate([x1, x2], axis=0)
            y_combined_labels = np.concatenate([y1, y2], axis=0)
            
            self.data_subjs.append(x_combined_data)
            self.labels_subjs.append(y_combined_labels)

        x_combined_data = np.concatenate(self.data_subjs, axis=0)
        y_combined_labels = np.concatenate(self.labels_subjs, axis=0).astype(np.int64)

        # --- STRICT LABEL NORMALIZATION ---
        unique_labels = np.sort(np.unique(y_combined_labels))
        print(f"⚠️ [Dataset Debug] Raw unique labels for {subject_ids}: {unique_labels}")
        
        # If your raw labels are [1, 2], this gracefully maps them to [0, 1].
        # If you see more than 2 unique labels here in your console, you MUST 
        # filter out the artifact trials before training, or increase --num-decoding-classes.
        label_map = {val: idx for idx, val in enumerate(unique_labels)}
        
        y_mapped = np.array([label_map[y] for y in y_combined_labels], dtype=np.int64)
        print(f"⚠️ [Dataset Debug] Mapped labels to: {np.unique(y_mapped)}")
        # -----------------------------------------

        return x_combined_data, y_mapped

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        X = self.X[idx]
        Y = self.y[idx]
        
        # 1. Project 3 channels to 22 channels
        X = self.map2pret(X)
        X = X.astype(np.float32)
        
        # 2. Extract the single integer label
        Y = int(Y)
        
        # CPU-side assertion: Catches the error gracefully before crashing the GPU.
        # Change '2' to match your actual --num-decoding-classes argument.
        if Y >= 4: # Assuming default 4, or change to 2 if doing strictly binary
            raise ValueError(f"Label {Y} exceeds expected number of decoding classes! Check your label_map.")
        
        return self.preprocess_sample(X, self.num_chunks, Y)

    def map2pret(self, data):
        mapped_data = np.zeros((22, data.shape[1]), dtype=np.float32)
        mapped_data[7, :] = data[0, :]  # C3
        mapped_data[9, :] = data[1, :]  # Cz
        mapped_data[11, :] = data[2, :] # C4
        return np.matmul(self.P, mapped_data)

    