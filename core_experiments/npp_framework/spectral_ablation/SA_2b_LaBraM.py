import torch
import torch.nn as nn
import numpy as np
import os
import random
from tqdm import tqdm
from collections import OrderedDict
from einops import rearrange
from timm.models import create_model
from sklearn.metrics import balanced_accuracy_score, cohen_kappa_score, f1_score

# Local LaBraM imports
import modeling_finetune
import utils

# ==========================================
# 0. CONFIGURATION & SETUP
# ==========================================
BASE_CHECKPOINT_PATH = "/homes/xw2336/data_portal/LaBram/checkpoints/retrain_reproduction_2b"
SUBJECTS = ['S01', 'S02', 'S03', 'S04', 'S05', 'S06', 'S07', 'S08', 'S09']
CH_NAMES = ['C3', 'CZ', 'C4'] # Specific to BCI IV 2b
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Assume standard BCI Competition IV 2b sampling rate (adjust if you resampled)
FS = 250.0  

# Canonical EEG Frequency Bands
BANDS = {
    'Delta': (0.5, 4),
    'Theta': (4, 8),
    'Alpha': (8, 13),
    'Beta': (13, 30),
    'Gamma': (30, 100) # Up to Nyquist/usable high frequencies
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

def get_input_chans(ch_names):
    input_chans = [0]
    for ch_name in ch_names:
        clean_name = ch_name.split(' ')[-1].split('-')[0].strip().upper()
        input_chans.append(standard_1020.index(clean_name) + 1)
    return torch.tensor(input_chans)

# ==========================================
# 1. SPECTRAL ABLATION LOGIC
# ==========================================
def apply_spectral_ablation(x, fs, band):
    """
    Zeros out the frequencies corresponding to the specified band.
    x expected shape: (Batch, Channels, Time)
    """
    low_freq, high_freq = band
    
    # Transform to frequency domain
    X_fft = torch.fft.rfft(x, dim=-1)
    freqs = torch.fft.rfftfreq(x.shape[-1], d=1/fs, device=x.device)
    
    # Create a boolean mask: True for frequencies outside the band, False for inside
    mask = ~((freqs >= low_freq) & (freqs <= high_freq))
    
    # Reshape mask to broadcast across Batch and Channel dimensions
    mask = mask.view(1, 1, -1)
    
    # Zero out the targeted frequency band
    X_ablated = X_fft * mask
    
    # Inverse transform back to time domain
    return torch.fft.irfft(X_ablated, n=x.shape[-1], dim=-1)

def load_labram_2b(ckpt_path):
    model = create_model(
        'labram_base_patch200_200', pretrained=False, num_classes=1, drop_path_rate=0.1,
        use_mean_pooling=True, use_rel_pos_bias=True, use_abs_pos_emb=True, qkv_bias=True, 
        init_values=0.1, init_scale=0.001
    )
    checkpoint = torch.load(ckpt_path, map_location='cpu', weights_only=False)
    
    if 'model_ema' in checkpoint and checkpoint['model_ema'] is not None:
        checkpoint_model = checkpoint['model_ema']
    else:
        checkpoint_model = checkpoint.get('model', checkpoint)
    
    new_dict = OrderedDict()
    for key, value in checkpoint_model.items():
        if key.startswith('student.'): key = key[8:]
        if key.startswith('module.'): key = key[7:]
        new_dict[key] = value
        
    model.load_state_dict(new_dict, strict=True)
    model.eval()
    return model

# ==========================================
# 2. EVALUATION LOOP
# ==========================================
def run_bci_spectral_ablation():
    input_chans = get_input_chans(CH_NAMES).to(DEVICE)
    
    # Tracking dictionaries
    res_orig = {"bacc": [], "f1": [], "kappa": []}
    res_ablated = {band: {"bacc": [], "f1": [], "kappa": []} for band in BANDS.keys()}

    for sub in SUBJECTS:
        print(f"\n--- Analyzing Fold: {sub} ---")
        ckpt_path = os.path.join(BASE_CHECKPOINT_PATH, f"fold_{sub}", "checkpoint-final.pth")
        
        if not os.path.exists(ckpt_path):
            print(f"Skipping {sub}: Checkpoint not found at {ckpt_path}")
            continue

        train_subs = [s for s in SUBJECTS if s != sub]
        _, test_dataset, _ = utils.prepare_BCIIV2b_dataset(
            (train_subs, [sub], [sub]), window_size=400
        )
        
        loader = torch.utils.data.DataLoader(test_dataset, batch_size=64, shuffle=False)
        model = load_labram_2b(ckpt_path).to(DEVICE)

        y_true, probs_orig = [], []
        probs_ablated = {band: [] for band in BANDS.keys()}

        with torch.no_grad():
            for x, y in tqdm(loader, desc=f"Subject {sub}"):
                # Normalization
                x = x.float().to(DEVICE) / 100 
                
                # --- Baseline Inference ---
                x_in_orig = rearrange(x, 'B N (A T) -> B N A T', T=200)
                with torch.amp.autocast('cuda'):
                    logits_orig = model(x_in_orig, input_chans=input_chans) 
                probs_orig.extend(torch.sigmoid(logits_orig).cpu().numpy())
                
                # --- Spectral Ablation Inference ---
                for band_name, freq_range in BANDS.items():
                    # Apply ablation to the continuous time series (x) BEFORE patching
                    x_abl = apply_spectral_ablation(x, fs=FS, band=freq_range)
                    
                    # Now chop into patches for LaBram
                    x_in_abl = rearrange(x_abl, 'B N (A T) -> B N A T', T=200)
                    
                    with torch.amp.autocast('cuda'):
                        logits_abl = model(x_in_abl, input_chans=input_chans)
                    probs_ablated[band_name].extend(torch.sigmoid(logits_abl).cpu().numpy())

                y_true.extend(y.numpy())

        # Metric Calculation
        yt = np.array(y_true).flatten()
        po = (np.array(probs_orig).flatten() > 0.5).astype(float)
        
        res_orig["bacc"].append(balanced_accuracy_score(yt, po) * 100)
        res_orig["f1"].append(f1_score(yt, po, average='macro') * 100)
        res_orig["kappa"].append(cohen_kappa_score(yt, po))

        for band_name in BANDS.keys():
            pa = (np.array(probs_ablated[band_name]).flatten() > 0.5).astype(float)
            res_ablated[band_name]["bacc"].append(balanced_accuracy_score(yt, pa) * 100)
            res_ablated[band_name]["f1"].append(f1_score(yt, pa, average='macro') * 100)
            res_ablated[band_name]["kappa"].append(cohen_kappa_score(yt, pa))

    # ==========================================
    # 3. REPORTING
    # ==========================================
    print("\n" + "="*80)
    print("BCI IV 2b SPECTRAL ABLATION SUMMARY")
    print("="*80)
    
    # Baseline
    print(f"\n[BASELINE (No Ablation)]")
    print(f"  BACC : {np.mean(res_orig['bacc']):.2f} ± {np.std(res_orig['bacc']):.2f}%")
    print(f"  F1   : {np.mean(res_orig['f1']):.2f} ± {np.std(res_orig['f1']):.2f}%")
    print(f"  Kappa: {np.mean(res_orig['kappa']):.4f} ± {np.std(res_orig['kappa']):.4f}")

    # Ablation Drops
    print(f"\n[ABLATED BANDS: PERFORMANCE & DROP]")
    print(f"{'BAND':<10} | {'BACC':<15} | {'DROP (Δ)':<10} | {'F1':<15} | {'Kappa':<15}")
    print("-" * 75)
    
    orig_bacc_mean = np.mean(res_orig['bacc'])
    
    for band_name in BANDS.keys():
        b_mean = np.mean(res_ablated[band_name]['bacc'])
        b_std = np.std(res_ablated[band_name]['bacc'])
        f_mean = np.mean(res_ablated[band_name]['f1'])
        k_mean = np.mean(res_ablated[band_name]['kappa'])
        
        drop = orig_bacc_mean - b_mean
        
        print(f"{band_name:<10} | {b_mean:>5.2f}±{b_std:>5.2f}% | {-drop:>8.2f}% | {f_mean:>5.2f}% | {k_mean:>6.4f}")

if __name__ == "__main__":
    run_bci_spectral_ablation()