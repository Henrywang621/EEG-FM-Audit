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
        clean_name = ch_name.split(' ')[-1].split('-')[0].strip()
        input_chans.append(standard_1020.index(clean_name) + 1)
    return torch.tensor(input_chans)

class TUABLoader(torch.utils.data.Dataset):
    def __init__(self, root, files):
        self.root = root
        self.files = files
    def __len__(self): return len(self.files)
    def __getitem__(self, index):
        with open(os.path.join(self.root, self.files[index]), "rb") as f:
            sample = pickle.load(f)
        X = sample["X"].astype(np.float32) / 100.0
        return torch.FloatTensor(X), torch.FloatTensor([sample["y"]])

# ==========================================
# 1. PERTURBATION & MODEL LOADING
# ==========================================
def apply_spatial_preserving_phase_rand(x, phase_seed):
    """
    Applies phase randomization while preserving the spatial covariance 
    by broadcasting the same random phase across all channels.
    """
    # Create a local generator to ensure shuffle independence
    gen = torch.Generator(device=x.device)
    gen.manual_seed(phase_seed)
    
    batch_size, num_channels, time_len = x.shape
    mu_x = x.mean(dim=-1, keepdim=True)
    y = x - mu_x
    Y = torch.fft.rfft(y, dim=-1)
    magnitudes = torch.abs(Y)
    
    # Generate random phases (0 to 2*pi)
    random_phases = torch.rand(batch_size, 1, Y.shape[-1], 
                               generator=gen, device=x.device) * 2 * torch.pi
    random_phases[..., 0] = 0.0 # Maintain zero DC component phase
    
    Y_surrogate = magnitudes * torch.exp(1j * random_phases)
    return torch.fft.irfft(Y_surrogate, n=time_len, dim=-1) + mu_x

def load_labram_eval(ckpt_path):
    model = create_model(
        'labram_base_patch200_200', pretrained=False, num_classes=1, drop_path_rate=0.1,
        use_mean_pooling=True, use_rel_pos_bias=True, use_abs_pos_emb=True, 
        qkv_bias=True, init_values=0.1,
    )
    checkpoint = torch.load(ckpt_path, map_location='cpu', weights_only=False)
    
    if 'model_ema' in checkpoint:
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
# 2. MAIN EVALUATION LOOP
# ==========================================
def run_full_analysis():
    # Paths (adjust to your environment)
    DATA_PATH = "/homes/xw2336/data_portal/TUAB/TUAB/processed/test"
    BASE_DIR = "/homes/xw2336/data_portal/LaBram/checkpoints/retrain_reproduction_tuab"
    
    # Config
    MODEL_SEEDS = [6, 16, 42, 66, 3407]
    PHASE_SHUFFLE_SEEDS = list(range(100, 110)) # 10 distinct seeds for shuffling
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Data Setup
    files = [f for f in os.listdir(DATA_PATH) if f.endswith('.pkl')]
    dataset = TUABLoader(DATA_PATH, files)
    loader = torch.utils.data.DataLoader(dataset, batch_size=64, num_workers=4, shuffle=False)

    ch_names = ['EEG FP1', 'EEG FP2-REF', 'EEG F3-REF', 'EEG F4-REF', 'EEG C3-REF', 'EEG C4-REF', 'EEG P3-REF', 'EEG P4-REF', 'EEG O1-REF', 'EEG O2-REF', 'EEG F7-REF', 'EEG F8-REF', 'EEG T3-REF', 'EEG T4-REF', 'EEG T5-REF', 'EEG T6-REF', 'EEG A1-REF', 'EEG A2-REF', 'EEG FZ-REF', 'EEG CZ-REF', 'EEG PZ-REF', 'EEG T1-REF', 'EEG T2-REF']
    input_chans = get_input_chans(ch_names).to(DEVICE)

    metric_keys = ["bacc", "f1", "kappa"]
    all_results_orig = {k: [] for k in metric_keys}
    all_results_rand = {k: [] for k in metric_keys}

    for m_seed in MODEL_SEEDS:
        print(f"\n--- Processing Model Seed: {m_seed} ---")
        ckpt_path = os.path.join(BASE_DIR, f"seed_{m_seed}", "checkpoint-best.pth")
        
        if not os.path.exists(ckpt_path):
            print(f"⚠️ Checkpoint not found: {ckpt_path}. Skipping.")
            continue

        model = load_labram_eval(ckpt_path).to(DEVICE)

        # 1. Run Baseline (Original) once per model
        y_true, probs_orig = [], []
        with torch.no_grad():
            for x, y in tqdm(loader, desc=f"Model {m_seed} Original Baseline"):
                x = x.to(DEVICE)
                x_in = rearrange(x, 'B N (A T) -> B N A T', T=200)
                logits = model(x_in, input_chans=input_chans)
                probs_orig.extend(torch.sigmoid(logits).cpu().numpy())
                y_true.extend(y.numpy())

        y_true = np.array(y_true).flatten()
        p_orig = np.array(probs_orig).flatten()
        pred_orig = (p_orig > 0.5).astype(float)

        # Calculate metrics for the original run
        m_bacc = balanced_accuracy_score(y_true, pred_orig) * 100
        m_f1 = f1_score(y_true, pred_orig) * 100
        m_kappa = cohen_kappa_score(y_true, pred_orig)

        # 2. Run 10 Random Phase Shuffles for THIS model
        for p_seed in PHASE_SHUFFLE_SEEDS:
            probs_rand = []
            with torch.no_grad():
                for x, _ in tqdm(loader, desc=f"  Shuffle Seed {p_seed}", leave=False):
                    x = x.to(DEVICE)
                    x_rand = apply_spatial_preserving_phase_rand(x, p_seed)
                    x_in_rand = rearrange(x_rand, 'B N (A T) -> B N A T', T=200)
                    logits_rand = model(x_in_rand, input_chans=input_chans)
                    probs_rand.extend(torch.sigmoid(logits_rand).cpu().numpy())

            p_rand = np.array(probs_rand).flatten()
            pred_rand = (p_rand > 0.5).astype(float)

            # Record metrics (Paired with the specific model's original performance)
            all_results_orig["bacc"].append(m_bacc)
            all_results_orig["f1"].append(m_f1)
            all_results_orig["kappa"].append(m_kappa)

            all_results_rand["bacc"].append(balanced_accuracy_score(y_true, pred_rand) * 100)
            all_results_rand["f1"].append(f1_score(y_true, pred_rand) * 100)
            all_results_rand["kappa"].append(cohen_kappa_score(y_true, pred_rand))

        del model
        torch.cuda.empty_cache()

    # ==========================================
    # 3. FINAL SUMMARY
    # ==========================================
    print("\n" + "="*75)
    print(f"FINAL ANALYSIS SUMMARY (N={len(all_results_orig['bacc'])} Runs)")
    print(f"5 Models x 10 Phase Randomization Shuffles each")
    print("="*75)
    
    for label, data in [("ORIGINAL (BASELINE)", all_results_orig), ("RANDOMIZED (PHASE)", all_results_rand)]:
        print(f"\n[{label}]")
        for k in metric_keys:
            unit = "%" if k != "kappa" else ""
            mean = np.mean(data[k])
            std = np.std(data[k])
            print(f"  {k.upper():<6}: {mean:.2f} ± {std:.2f}{unit}")

    print("\n[IMPACT OF PHASE RANDOMIZATION (ORIG - RAND)]")
    for k in metric_keys:
        diffs = np.array(all_results_orig[k]) - np.array(all_results_rand[k])
        unit = "%" if k != "kappa" else ""
        print(f"  {k.upper():<6}: {np.mean(diffs):.2f} ± {np.std(diffs):.2f}{unit}")
    print("="*75)

if __name__ == "__main__":
    run_full_analysis()