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

# Note: These are local imports from the LaBram repository
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

# ==========================================
# 1. TUEV DATA LOADER (From eval_FR_final.py)
# ==========================================
class TUEVLoader(torch.utils.data.Dataset):
    def __init__(self, root, files, sampling_rate=200):
        self.root = root
        self.files = files
        self.default_rate = 200
        self.sampling_rate = sampling_rate

    def __len__(self):
        return len(self.files)

    def __getitem__(self, index):
        with open(os.path.join(self.root, self.files[index]), "rb") as f:
            sample = pickle.load(f)
        
        X = sample["signal"]
        
        # Byte-order & Memory Continuity Fixes
        if X.dtype.byteorder == '>' or (X.dtype.byteorder == '|' and X.dtype.str.endswith('f4') == False):
            X = X.byteswap().newbyteorder('=')
        
        X = np.array(X, dtype=np.float32, copy=True)
        X = np.ascontiguousarray(X)
        
        if self.sampling_rate != self.default_rate:
            X = resample(X, 5 * self.sampling_rate, axis=-1)
            X = np.ascontiguousarray(X, dtype=np.float32)
            
        # Label handling: 1-indexed to 0-indexed (e.g., 1-6 -> 0-5)
        Y = int(sample["label"][0] - 1)
        X_tensor = torch.as_tensor(X, dtype=torch.float32)
        
        return X_tensor, Y

# ==========================================
# 2. PERTURBATION & MODEL LOADING
# ==========================================
def apply_spatial_preserving_phase_rand(x, phase_seed):
    """Seed-aware phase randomization (Modified from eval_FR_final_tuab.py)"""
    gen = torch.Generator(device=x.device)
    gen.manual_seed(phase_seed)
    
    batch_size, num_channels, time_len = x.shape
    mu_x = x.mean(dim=-1, keepdim=True)
    y = x - mu_x
    Y = torch.fft.rfft(y, dim=-1)
    magnitudes = torch.abs(Y)
    
    # Broadcast single phase across all channels to preserve spatial structure
    random_phases = torch.rand(batch_size, 1, Y.shape[-1], 
                               generator=gen, device=x.device) * 2 * torch.pi
    random_phases[..., 0] = 0.0
    
    Y_surrogate = magnitudes * torch.exp(1j * random_phases)
    return torch.fft.irfft(Y_surrogate, n=time_len, dim=-1) + mu_x

def load_labram_eval_tuev(ckpt_path):
    """Multiclass model loader (6 classes)"""
    model = create_model(
        'labram_base_patch200_200', pretrained=False, num_classes=6, 
        drop_path_rate=0.1, use_mean_pooling=True, use_rel_pos_bias=True, 
        use_abs_pos_emb=True, qkv_bias=True, init_values=0.1,
    )
    checkpoint = torch.load(ckpt_path, map_location='cpu', weights_only=False)
    checkpoint_model = checkpoint.get('model_ema', checkpoint.get('model', checkpoint))
    
    new_dict = OrderedDict()
    for key, value in checkpoint_model.items():
        if key.startswith('student.'): key = key[8:]
        if key.startswith('module.'): key = key[7:]
        new_dict[key] = value
        
    model.load_state_dict(new_dict, strict=True)
    model.eval()
    return model

# ==========================================
# 3. MAIN EVALUATION LOOP
# ==========================================
def run_full_analysis():
    # Paths adjusted for TUEV
    DATA_PATH = "/homes/xw2336/data_portal/TUEV/processed/processed_test"
    BASE_DIR = "/homes/xw2336/data_portal/LaBram/checkpoints/retrain_reproduction_tuev"
    
    # Config matching nested loop strategy
    MODEL_SEEDS = [6, 16, 42, 66, 3407]
    PHASE_SHUFFLE_SEEDS = list(range(100, 110)) 
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Channel mapping logic
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
    ch_names_raw = ['EEG FP1-REF', 'EEG FP2-REF', 'EEG F3-REF', 'EEG F4-REF', 'EEG C3-REF', 'EEG C4-REF', 
                    'EEG P3-REF', 'EEG P4-REF', 'EEG O1-REF', 'EEG O2-REF', 'EEG F7-REF', 'EEG F8-REF', 
                    'EEG T3-REF', 'EEG T4-REF', 'EEG T5-REF', 'EEG T6-REF', 'EEG A1-REF', 'EEG A2-REF', 
                    'EEG FZ-REF', 'EEG CZ-REF', 'EEG PZ-REF', 'EEG T1-REF', 'EEG T2-REF']
    input_chans = [0]
    for name in ch_names_raw:
        clean = name.split(' ')[-1].split('-')[0].strip()
        input_chans.append(standard_1020.index(clean) + 1)
    input_chans = torch.tensor(input_chans).to(DEVICE)

    # Data Setup
    files = [f for f in os.listdir(DATA_PATH) if f.endswith('.pkl')]
    dataset = TUEVLoader(DATA_PATH, files)
    loader = torch.utils.data.DataLoader(dataset, batch_size=64, num_workers=4, shuffle=False)

    metric_keys = ["bacc", "f1_weighted", "kappa"]
    all_results_orig = {k: [] for k in metric_keys}
    all_results_rand = {k: [] for k in metric_keys}

    for m_seed in MODEL_SEEDS:
        print(f"\n--- Processing Model Seed: {m_seed} ---")
        ckpt_path = os.path.join(BASE_DIR, f"seed_{m_seed}", "checkpoint-best.pth")
        
        if not os.path.exists(ckpt_path):
            print(f"⚠️ Checkpoint not found: {ckpt_path}. Skipping.")
            continue

        model = load_labram_eval_tuev(ckpt_path).to(DEVICE)

        # 1. Baseline Run (Original)
        y_true, preds_orig = [], []
        with torch.no_grad():
            for x, y in tqdm(loader, desc=f"Model {m_seed} Baseline"):
                # Scaling by 100.0 is critical for TUEV
                x = x.to(DEVICE) / 100.0
                x_in = rearrange(x, 'B N (A T) -> B N A T', T=200)
                logits = model(x_in, input_chans=input_chans)
                # Multiclass uses argmax
                preds_orig.extend(torch.argmax(logits, dim=1).cpu().numpy())
                y_true.extend(y.numpy())

        y_true = np.array(y_true).flatten()
        p_orig = np.array(preds_orig)
        
        # Calculate Baseline Metrics
        m_bacc = balanced_accuracy_score(y_true, p_orig) * 100
        m_f1 = f1_score(y_true, p_orig, average='weighted') * 100
        m_kappa = cohen_kappa_score(y_true, p_orig)

        # 2. Nested Shuffle Loop (10 per model)
        for p_seed in PHASE_SHUFFLE_SEEDS:
            preds_rand = []
            with torch.no_grad():
                for x, _ in tqdm(loader, desc=f"  Shuffle {p_seed}", leave=False):
                    x = x.to(DEVICE) / 100.0
                    x_rand = apply_spatial_preserving_phase_rand(x, p_seed)
                    x_in_rand = rearrange(x_rand, 'B N (A T) -> B N A T', T=200)
                    logits_rand = model(x_in_rand, input_chans=input_chans)
                    preds_rand.extend(torch.argmax(logits_rand, dim=1).cpu().numpy())

            p_rand = np.array(preds_rand)
            
            # Record paired results
            for k, v in [("bacc", m_bacc), ("f1_weighted", m_f1), ("kappa", m_kappa)]: 
                all_results_orig[k].append(v)
            
            all_results_rand["bacc"].append(balanced_accuracy_score(y_true, p_rand) * 100)
            all_results_rand["f1_weighted"].append(f1_score(y_true, p_rand, average='weighted') * 100)
            all_results_rand["kappa"].append(cohen_kappa_score(y_true, p_rand))

        del model
        torch.cuda.empty_cache()

    # ==========================================
    # 4. FINAL SUMMARY
    # ==========================================
    print("\n" + "="*75)
    print(f"TUEV ANALYSIS SUMMARY (N={len(all_results_orig['bacc'])} Runs)")
    print("="*75)
    
    for label, data in [("ORIGINAL", all_results_orig), ("RANDOMIZED", all_results_rand)]:
        print(f"\n[{label}]")
        for k in metric_keys:
            unit = "%" if "kappa" not in k else ""
            mean, std = np.mean(data[k]), np.std(data[k])
            print(f"  {k.upper():<11}: {mean:.2f} ± {std:.2f}{unit}")

    print("\n[DIFFERENCE (ORIG - RAND)]")
    for k in metric_keys:
        diffs = np.array(all_results_orig[k]) - np.array(all_results_rand[k])
        unit = "%" if "kappa" not in k else ""
        print(f"  {k.upper():<11}: {np.mean(diffs):.2f} ± {np.std(diffs):.2f}{unit}")
    print("="*75)

if __name__ == "__main__":
    run_full_analysis()