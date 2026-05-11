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
PHASE_SHUFFLE_SEEDS = list(range(100, 110))
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

def get_input_chans(ch_names):
    input_chans = [0]
    for ch_name in ch_names:
        clean_name = ch_name.split(' ')[-1].split('-')[0].strip().upper()
        input_chans.append(standard_1020.index(clean_name) + 1)
    return torch.tensor(input_chans)

def apply_spatial_preserving_phase_rand(x, phase_seed):
    gen = torch.Generator(device=x.device)
    gen.manual_seed(phase_seed)
    
    batch_size, num_channels, time_len = x.shape
    mu_x = x.mean(dim=-1, keepdim=True)
    y = x - mu_x
    Y = torch.fft.rfft(y, dim=-1)
    magnitudes = torch.abs(Y)
    
    random_phases = torch.rand(batch_size, 1, Y.shape[-1], 
                               generator=gen, device=x.device) * 2 * torch.pi
    random_phases[..., 0] = 0.0 
    
    Y_surrogate = magnitudes * torch.exp(1j * random_phases)
    return torch.fft.irfft(Y_surrogate, n=time_len, dim=-1) + mu_x

def load_labram_2b(ckpt_path):
    # MATCH engine_for_finetuning1.py EXACTLY: use_abs_pos_emb=True
    model = create_model(
        'labram_base_patch200_200', pretrained=False, num_classes=1, drop_path_rate=0.1,
        use_mean_pooling=True, use_rel_pos_bias=True, use_abs_pos_emb=True, qkv_bias=True, 
        init_values=0.1, init_scale=0.001
    )
    checkpoint = torch.load(ckpt_path, map_location='cpu', weights_only=False)
    
    # Target 'model_ema' first, fallback to 'model'
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
def run_bci_analysis():
    input_chans = get_input_chans(CH_NAMES).to(DEVICE)
    metric_keys = ["bacc", "f1", "kappa"]
    summary_orig = {k: [] for k in metric_keys}
    summary_rand = {k: [] for k in metric_keys}

    for sub in SUBJECTS:
        print(f"\n--- Analyzing Fold: {sub} ---")
        ckpt_path = os.path.join(BASE_CHECKPOINT_PATH, f"fold_{sub}", "checkpoint-final.pth")
        
        if not os.path.exists(ckpt_path):
            print(f"Skipping {sub}: Checkpoint not found at {ckpt_path}")
            continue

        # Correct normalization scaling based ONLY on the 8 training subjects
        train_subs = [s for s in SUBJECTS if s != sub]
        _, test_dataset, _ = utils.prepare_BCIIV2b_dataset(
            (train_subs, [sub], [sub]), window_size=400
        )
        
        loader = torch.utils.data.DataLoader(test_dataset, batch_size=64, shuffle=False)
        model = load_labram_2b(ckpt_path).to(DEVICE)

        # Baseline Evaluation
        y_true, probs_orig = [], []
        with torch.no_grad():
            for x, y in loader:
                # Match evaluate() normalization perfectly
                x = x.float().to(DEVICE) / 100 
                x_in = rearrange(x, 'B N (A T) -> B N A T', T=200)
                
                # Evaluate using Mixed Precision (AMP) to match training precision
                with torch.amp.autocast('cuda'):
                    logits = model(x_in, input_chans=input_chans) 
                
                probs_orig.extend(torch.sigmoid(logits).cpu().numpy())
                y_true.extend(y.numpy())

        y_true = np.array(y_true).flatten()
        pred_orig = (np.array(probs_orig).flatten() > 0.5).astype(float)
        
        m_bacc = balanced_accuracy_score(y_true, pred_orig) * 100
        m_f1 = f1_score(y_true, pred_orig, average='macro') * 100
        m_kappa = cohen_kappa_score(y_true, pred_orig)

        # Phase Randomization Evaluation
        for p_seed in PHASE_SHUFFLE_SEEDS:
            probs_rand = []
            with torch.no_grad():
                for x, _ in loader:
                    # Normalization MUST occur before creating the surrogate
                    x = x.float().to(DEVICE) / 100 
                    x_rand = apply_spatial_preserving_phase_rand(x, p_seed)
                    x_in_rand = rearrange(x_rand, 'B N (A T) -> B N A T', T=200)
                    
                    with torch.amp.autocast('cuda'):
                        logits_rand = model(x_in_rand, input_chans=input_chans)
                    
                    probs_rand.extend(torch.sigmoid(logits_rand).cpu().numpy())

            pred_rand = (np.array(probs_rand).flatten() > 0.5).astype(float)
            
            summary_orig["bacc"].append(m_bacc)
            summary_orig["f1"].append(m_f1)
            summary_orig["kappa"].append(m_kappa)
            summary_rand["bacc"].append(balanced_accuracy_score(y_true, pred_rand) * 100)
            summary_rand["f1"].append(f1_score(y_true, pred_rand, average='macro') * 100)
            summary_rand["kappa"].append(cohen_kappa_score(y_true, pred_rand))

    # ==========================================
    # 3. REPORTING
    # ==========================================
    print("\n" + "="*50)
    print("BCI IV 2b PHASE RANDOMIZATION SUMMARY")
    print("="*50)
    for label, data in [("ORIGINAL", summary_orig), ("RANDOMIZED", summary_rand)]:
        print(f"\n[{label}]")
        for k in metric_keys:
            print(f"  {k.upper():<6}: {np.mean(data[k]):.2f} ± {np.std(data[k]):.2f}")

if __name__ == "__main__":
    run_bci_analysis()