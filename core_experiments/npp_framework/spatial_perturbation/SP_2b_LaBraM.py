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
CH_NAMES = ['C3', 'CZ', 'C4'] # Specific to BCI IV 2b[cite: 2]
NOISE_LEVELS = [0.5, 1.0, 2.0] # Based on eval_SP_final.py[cite: 1]
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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

ROI_DEFINITIONS = {
    'Central': ['FC3', 'FC4', 'C3', 'CZ', 'C4', 'C1', 'C2', 'C5', 'C6', 'CP3', 'CPZ', 'CP4', 'CP1', 'CP2']
}

def get_input_chans(ch_names):
    input_chans = [0]
    for ch_name in ch_names:
        clean_name = ch_name.split(' ')[-1].split('-')[0].strip().upper()
        input_chans.append(standard_1020.index(clean_name) + 1)
    return torch.tensor(input_chans)

def get_roi_indices(roi_name, ch_names):
    target_names = ROI_DEFINITIONS.get(roi_name, [])
    indices = []
    for i, name in enumerate(ch_names):
        clean = name.split(' ')[-1].split('-')[0].upper().strip()
        if clean in target_names:
            indices.append(i)
    return indices

def apply_spatial_noise(x, target_indices, noise_lambda=1.0):
    if len(target_indices) == 0: return x
    x_perturbed = x.clone()
    batch_std = x.std()
    noise = torch.randn_like(x_perturbed) * batch_std * noise_lambda
    # B, N, A, T structure: mask the specific channels (N)[cite: 1]
    mask = torch.zeros(x.shape[1], device=x.device).view(1, -1, 1, 1)
    mask[:, target_indices, :, :] = 1.0
    return x_perturbed + (noise * mask)

def load_labram_2b(ckpt_path):
    model = create_model(
        'labram_base_patch200_200', pretrained=False, num_classes=1, drop_path_rate=0.1,
        use_mean_pooling=True, use_rel_pos_bias=True, use_abs_pos_emb=True, qkv_bias=True, 
        init_values=0.1, init_scale=0.001
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
# 2. EVALUATION LOOP
# ==========================================
def run_bci_spectral_analysis():
    input_chans = get_input_chans(CH_NAMES).to(DEVICE)
    target_idx = get_roi_indices('Central', CH_NAMES) # BCI 2b focus[cite: 2]
    
    res_orig = {"bacc": [], "f1": [], "kappa": []}
    # Track results by noise level
    res_pert = {n: {"bacc": [], "f1": [], "kappa": []} for n in NOISE_LEVELS}

    for sub in SUBJECTS:
        print(f"\n--- Analyzing Fold: {sub} ---")
        ckpt_path = os.path.join(BASE_CHECKPOINT_PATH, f"fold_{sub}", "checkpoint-final.pth")
        
        if not os.path.exists(ckpt_path):
            continue

        train_subs = [s for s in SUBJECTS if s != sub]
        _, test_dataset, _ = utils.prepare_BCIIV2b_dataset(
            (train_subs, [sub], [sub]), window_size=400
        )
        
        loader = torch.utils.data.DataLoader(test_dataset, batch_size=64, shuffle=False)
        model = load_labram_2b(ckpt_path).to(DEVICE)

        y_true, probs_orig = [], []
        # Store perturbed probabilities for each noise level
        probs_pert = {n: [] for n in NOISE_LEVELS}

        with torch.no_grad():
            for x, y in tqdm(loader, desc=f"Subject {sub}"):
                x = x.float().to(DEVICE) / 100 
                x_in = rearrange(x, 'B N (A T) -> B N A T', T=200)
                
                with torch.amp.autocast('cuda'):
                    # Original Inference[cite: 2]
                    logits_o = model(x_in, input_chans=input_chans)
                    probs_orig.extend(torch.sigmoid(logits_o).cpu().numpy())
                    
                    # Perturbed Inference per noise level[cite: 1]
                    for n in NOISE_LEVELS:
                        x_p = apply_spatial_noise(x_in, target_idx, noise_lambda=n)
                        logits_p = model(x_p, input_chans=input_chans)
                        probs_pert[n].extend(torch.sigmoid(logits_p).cpu().numpy())
                
                y_true.extend(y.numpy())

        yt = np.array(y_true).flatten()
        
        # Calculate Original Metrics
        po = (np.array(probs_orig).flatten() > 0.5).astype(float)
        res_orig["bacc"].append(balanced_accuracy_score(yt, po) * 100)
        res_orig["f1"].append(f1_score(yt, po, average='macro') * 100)
        res_orig["kappa"].append(cohen_kappa_score(yt, po))

        # Calculate Perturbed Metrics
        for n in NOISE_LEVELS:
            pp = (np.array(probs_pert[n]).flatten() > 0.5).astype(float)
            res_pert[n]["bacc"].append(balanced_accuracy_score(yt, pp) * 100)
            res_pert[n]["f1"].append(f1_score(yt, pp, average='macro') * 100)
            res_pert[n]["kappa"].append(cohen_kappa_score(yt, pp))

    # ==========================================
    # 3. REPORTING
    # ==========================================
    print("\n" + "="*80)
    print("BCI IV 2b SPECTRAL PERTURBATION (CENTRAL ROI) SUMMARY")
    print("="*80)
    print(f"{'METRIC':<10} | {'ORIGINAL':<15} | " + " | ".join([f"NOISE {n}" for n in NOISE_LEVELS]))
    print("-" * 80)
    
    for k in ["bacc", "f1", "kappa"]:
        orig_str = f"{np.mean(res_orig[k]):.2f}±{np.std(res_orig[k]):.2f}"
        pert_strs = [f"{np.mean(res_pert[n][k]):.2f}±{np.std(res_pert[n][k]):.2f}" for n in NOISE_LEVELS]
        print(f"{k.upper():<10} | {orig_str:<15} | " + " | ".join(pert_strs))

if __name__ == "__main__":
    run_bci_spectral_analysis()