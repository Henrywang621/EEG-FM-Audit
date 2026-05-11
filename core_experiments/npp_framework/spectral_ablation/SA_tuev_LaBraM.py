import torch
import torch.nn as nn
import numpy as np
import os
import pickle
import random
from tqdm import tqdm
from collections import OrderedDict
from einops import rearrange
from timm.models import create_model
from scipy.signal import resample

from sklearn.metrics import (
    balanced_accuracy_score, 
    cohen_kappa_score, 
    f1_score
)

import modeling_finetune
import utils

# ==========================================
# 0. SETUP & REPRODUCIBILITY
# ==========================================
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# Physiological Frequency Bands
FREQ_BANDS = {
    'Delta': (1, 4),
    'Theta': (4, 8),
    'Alpha': (8, 13),
    'Beta':  (13, 30),
    'Gamma': (30, 75)
}

standard_1020 = [
    'FP1', 'FPZ', 'FP2', 'AF9', 'AF7', 'AF5', 'AF3', 'AF1', 'AFZ', 'AF2', 'AF4', 'AF6', 'AF8', 'AF10',
    'F9', 'F7', 'F5', 'F3', 'F1', 'FZ', 'F2', 'F4', 'F6', 'F8', 'F10', 'FT9', 'FT7', 'FC5', 'FC3', 'FC1', 
    'FCZ', 'FC2', 'FC4', 'FC6', 'FT8', 'FT10', 'T9', 'T7', 'C5', 'C3', 'C1', 'CZ', 'C2', 'C4', 'C6', 
    'T8', 'T10', 'TP9', 'TP7', 'CP5', 'CP3', 'CP1', 'CPZ', 'CP2', 'CP4', 'CP6', 'TP8', 'TP10', 'P9', 'P7', 
    'P5', 'P3', 'P1', 'PZ', 'P2', 'P4', 'P6', 'P8', 'P10', 'PO9', 'PO7', 'PO5', 'PO3', 'PO1', 'POZ', 
    'PO2', 'PO4', 'PO6', 'PO8', 'PO10', 'O1', 'OZ', 'O2', 'O9', 'CB1', 'CB2', 'IZ', 'O10', 'T3', 'T5', 
    'T4', 'T6', 'M1', 'M2', 'A1', 'A2', 'CFC1', 'CFC2', 'CFC3', 'CFC4', 'CFC5', 'CFC6', 'CFC7', 'CFC8', 
    'CCP1', 'CCP2', 'CCP3', 'CCP4', 'CCP5', 'CCP6', 'CCP7', 'CCP8', 'T1', 'T2', 'FTT9h', 'TTP7h', 
    'TPP9h', 'FTT10h', 'TPP8h', 'TPP10h', "FP1-F7", "F7-T7", "T7-P7", "P7-O1", "FP2-F8", "F8-T8", 
    "T8-P8", "P8-O2", "FP1-F3", "F3-C3", "C3-P3", "P3-O1", "FP2-F4", "F4-C4", "C4-P4", "P4-O2"
]

# ==========================================
# 1. ROBUST TUEV DATA LOADER
# ==========================================
class TUEVLoader(torch.utils.data.Dataset):
    def __init__(self, root, files, sampling_rate=200):
        self.root, self.files = root, files
        self.default_rate, self.sampling_rate = 200, sampling_rate
    def __len__(self): return len(self.files)
    def __getitem__(self, index):
        with open(os.path.join(self.root, self.files[index]), "rb") as f:
            sample = pickle.load(f)
        X = sample["signal"]
        if X.dtype.byteorder == '>' or (X.dtype.byteorder == '|' and X.dtype.str.endswith('f4') == False):
            X = X.byteswap().newbyteorder('=')
        X = np.array(X, dtype=np.float32, copy=True)
        X = np.ascontiguousarray(X)
        if self.sampling_rate != self.default_rate:
            X = resample(X, 5 * self.sampling_rate, axis=-1)
            X = np.ascontiguousarray(X, dtype=np.float32)
        Y = int(sample["label"][0] - 1)
        return torch.as_tensor(X, dtype=torch.float32), Y

def get_input_chans(ch_names):
    input_chans = [0]
    for ch_name in ch_names:
        clean_name = ch_name.split(' ')[-1].split('-')[0].strip()
        input_chans.append(standard_1020.index(clean_name) + 1)
    return torch.tensor(input_chans)

# ==========================================
# 2. SPECTRAL ABLATION & MODEL LOADING
# ==========================================
def apply_spectral_ablation(x, fs, band_name):
    if band_name not in FREQ_BANDS:
        return x
    low_cut, high_cut = FREQ_BANDS[band_name]
    fft_x = torch.fft.rfft(x, dim=-1)
    n = x.shape[-1]
    freqs = torch.fft.rfftfreq(n, d=1/fs).to(x.device)
    mask = ~((freqs >= low_cut) & (freqs <= high_cut))
    return torch.fft.irfft(fft_x * mask.view(1, 1, -1), n=n, dim=-1)

def load_labram_eval_tuev(ckpt_path):
    model = create_model(
        'labram_base_patch200_200', pretrained=False, num_classes=6, # TUEV 6-Class
        drop_path_rate=0.1, use_mean_pooling=True, use_rel_pos_bias=True, 
        use_abs_pos_emb=True, qkv_bias=True, init_values=0.1,
    )
    checkpoint = torch.load(ckpt_path, map_location='cpu', weights_only=False)
    checkpoint_model = checkpoint.get('model_ema', checkpoint.get('model', checkpoint))
    new_dict = OrderedDict()
    for k, v in checkpoint_model.items():
        if k.startswith('student.'): k = k[8:]
        if k.startswith('module.'): k = k[7:]
        new_dict[k] = v
    model.load_state_dict(new_dict, strict=True)
    model.eval()
    return model

# ==========================================
# 3. MAIN EVALUATION LOOP
# ==========================================
def run_analysis():
    DATA_PATH = "/homes/xw2336/data_portal/TUEV/processed/processed_test"
    BASE_DIR = "/homes/xw2336/data_portal/LaBram/checkpoints/retrain_reproduction_tuev"
    SEEDS = [6, 16, 42, 66, 3407]
    BANDS = ['Delta', 'Theta', 'Alpha', 'Beta', 'Gamma']
    FS = 200
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    files = [f for f in os.listdir(DATA_PATH) if f.endswith('.pkl')]
    loader = torch.utils.data.DataLoader(TUEVLoader(DATA_PATH, files), batch_size=64, num_workers=4)

    ch_names_raw = ['EEG FP1-REF', 'EEG FP2-REF', 'EEG F3-REF', 'EEG F4-REF', 'EEG C3-REF', 'EEG C4-REF', 
                    'EEG P3-REF', 'EEG P4-REF', 'EEG O1-REF', 'EEG O2-REF', 'EEG F7-REF', 'EEG F8-REF', 
                    'EEG T3-REF', 'EEG T4-REF', 'EEG T5-REF', 'EEG T6-REF', 'EEG A1-REF', 'EEG A2-REF', 
                    'EEG FZ-REF', 'EEG CZ-REF', 'EEG PZ-REF', 'EEG T1-REF', 'EEG T2-REF']
    input_chans = get_input_chans(ch_names_raw).to(DEVICE)

    # Tracking Structures
    res_orig = {"bacc": [], "f1": [], "kappa": []}
    res_ablated = {band: {"bacc": [], "f1": [], "kappa": []} for band in BANDS}
    res_diff = {band: {"bacc": [], "f1": [], "kappa": []} for band in BANDS}

    for seed in SEEDS:
        print(f"\n🚀 Processing Seed {seed}...")
        set_seed(seed)
        
        ckpt_path = os.path.join(BASE_DIR, f"seed_{seed}", "checkpoint-best.pth")
        if not os.path.exists(ckpt_path):
            ckpt_path = os.path.join(BASE_DIR, "checkpoint-149.pth") # fallback

        model = load_labram_eval_tuev(ckpt_path).to(DEVICE)

        y_true, preds_orig = [], []
        preds_bands = {band: [] for band in BANDS}

        with torch.no_grad():
            for x, y in tqdm(loader, desc=f"Seed {seed} Inference"):
                x = x.to(DEVICE) / 100.0 # Standard Scaling
                y_true.extend(y.numpy())

                # 1. Original Prediction
                x_o = rearrange(x, 'B N (A T) -> B N A T', T=200)
                logits_o = model(x_o, input_chans=input_chans)
                preds_orig.extend(torch.argmax(logits_o, dim=1).cpu().numpy())

                # 2. Ablated Predictions
                for band in BANDS:
                    x_abl = apply_spectral_ablation(x, FS, band)
                    x_abl = rearrange(x_abl, 'B N (A T) -> B N A T', T=200)
                    logits_a = model(x_abl, input_chans=input_chans)
                    preds_bands[band].extend(torch.argmax(logits_a, dim=1).cpu().numpy())

        # Metric Calculation
        yt = np.array(y_true).flatten()
        po = np.array(preds_orig)
        
        m_orig = {
            "bacc": balanced_accuracy_score(yt, po) * 100,
            "f1": f1_score(yt, po, average='weighted') * 100, # Weighted F1 for TUEV
            "kappa": cohen_kappa_score(yt, po)
        }
        for k in res_orig: res_orig[k].append(m_orig[k])

        print(f"\n[Seed {seed} - TUEV Baseline]")
        print(f"BACC: {m_orig['bacc']:.2f}% | F1(W): {m_orig['f1']:.2f}% | Kappa: {m_orig['kappa']:.4f}")
        
        print(f"{'BAND':<15} | {'BACC Diff':<10} | {'F1 Diff':<10} | {'Kappa Diff':<10}")
        print("-" * 55)

        for band in BANDS:
            pa = np.array(preds_bands[band])
            m_a = {
                "bacc": balanced_accuracy_score(yt, pa) * 100,
                "f1": f1_score(yt, pa, average='weighted') * 100,
                "kappa": cohen_kappa_score(yt, pa)
            }
            
            for k in ["bacc", "f1", "kappa"]:
                res_ablated[band][k].append(m_a[k])
                res_diff[band][k].append(m_orig[k] - m_a[k])

            print(f"{band:<15} | {m_orig['bacc']-m_a['bacc']:>9.2f}% | {m_orig['f1']-m_a['f1']:>9.2f}% | {m_orig['kappa']-m_a['kappa']:>10.4f}")

        del model
        torch.cuda.empty_cache()

    # ==========================================
    # 4. FINAL AGGREGATED SUMMARY
    # ==========================================
    metrics_to_print = [
        ("BALANCED ACCURACY (BACC %)", "bacc", "{:.2f}"),
        ("F1 SCORE (WEIGHTED %)", "f1", "{:.2f}"),
        ("COHEN'S KAPPA", "kappa", "{:.4f}")
    ]

    for title, key, fmt in metrics_to_print:
        print("\n" + "="*95)
        print(f"{title} - TUEV AGGREGATED SUMMARY (MEAN ± STD)")
        print("="*95)
        print(f"{'BAND':<15} | {'ORIGINAL':<18} | {'ABLATED':<18} | {'DROP (DIFFERENCE)'}")
        print("-" * 95)

        orig_mean, orig_std = np.mean(res_orig[key]), np.std(res_orig[key])
        print(f"{'BASELINE':<15} | {fmt.format(orig_mean)} ± {fmt.format(orig_std)} | {'-':<18} | {'0.0'}")
        print("-" * 95)

        for band in BANDS:
            a_m, a_s = np.mean(res_ablated[band][key]), np.std(res_ablated[band][key])
            d_m, d_s = np.mean(res_diff[band][key]), np.std(res_diff[band][key])

            row = f"{band:<15} | {fmt.format(orig_mean)} ± {fmt.format(orig_std)} | " \
                  f"{fmt.format(a_m)} ± {fmt.format(a_s)} | {fmt.format(d_m)} ± {fmt.format(d_s)}"
            print(row)

    print("="*95)

if __name__ == "__main__":
    run_analysis()