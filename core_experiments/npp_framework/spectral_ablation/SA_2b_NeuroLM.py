import os
import torch
import numpy as np
from pathlib import Path
from tqdm import tqdm
import tiktoken
from sklearn.metrics import balanced_accuracy_score, cohen_kappa_score, f1_score

# Local NeuroLM imports
from model.model_neurolm import NeuroLM
from model.model import GPTConfig
from utils import prepare_BCI2b_dataset

# ==========================================
# 0. CONFIGURATION & SETUP
# ==========================================
BASE_CHECKPOINT_PATH = "/homes/xw2336/xw2336/NeuroLM/NeuoLM_test/results_BCI2b/default_run/checkpoints/instruction-B_2b4"
DATASET_DIR = "/homes/xw2336/data"
SUBJECTS = ['S01', 'S02', 'S03', 'S04', 'S05', 'S06', 'S07', 'S08', 'S09']

# Assume standard BCI Competition IV 2b sampling rate
FS = 250.0  

BANDS = {
    'Delta': (0.5, 4),
    'Theta': (4, 8),
    'Alpha': (8, 13),
    'Beta': (13, 30),
    'Gamma': (30, 100)
}

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
dtype = 'bfloat16' if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else 'float16'
ptdtype = {'float32': torch.float32, 'bfloat16': torch.bfloat16, 'float16': torch.float16}[dtype]
ctx = torch.amp.autocast(device_type='cuda', dtype=ptdtype)

enc = tiktoken.get_encoding("gpt2")
decode = lambda l: enc.decode(l)

# ==========================================
# 1. SPECTRAL ABLATION & PARSING LOGIC
# ==========================================
def apply_spectral_ablation(x, fs, band):
    """
    Zeros out the frequencies corresponding to the specified band.
    x expected shape: (Batch, Channels, Time)
    """
    low_freq, high_freq = band
    X_fft = torch.fft.rfft(x, dim=-1)
    freqs = torch.fft.rfftfreq(x.shape[-1], d=1/fs, device=x.device)
    
    mask = ~((freqs >= low_freq) & (freqs <= high_freq))
    mask = mask.view(1, 1, -1)
    
    X_ablated = X_fft * mask
    return torch.fft.irfft(X_ablated, n=x.shape[-1], dim=-1)

def get_pred(pred_string):
    s = pred_string.lower()
    if 'yes' in s: return 1
    if 'no' in s: return 0 
    return -1

def load_neurolm(ckpt_path):
    checkpoint = torch.load(ckpt_path, map_location=device, weights_only=False)
    model_args = checkpoint['model_args']
    model = NeuroLM(GPTConfig(**model_args), init_from='gpt2')
    
    state_dict = checkpoint['model']
    unwanted_prefix = '_orig_mod.'
    for k, v in list(state_dict.items()):
        if k.startswith(unwanted_prefix):
            state_dict[k[len(unwanted_prefix):]] = state_dict.pop(k)
            
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    return model

# ==========================================
# 2. EVALUATION LOOP
# ==========================================
def run_neurolm_spectral_ablation():
    res_orig = {"bacc": [], "f1": [], "kappa": []}
    res_ablated = {band: {"bacc": [], "f1": [], "kappa": []} for band in BANDS.keys()}

    for sub in SUBJECTS:
        print(f"\n--- Analyzing Fold: {sub} ---")
        ckpt_path = os.path.join(BASE_CHECKPOINT_PATH, sub, "ckpt-final.pt")
        
        if not os.path.exists(ckpt_path):
            print(f"Skipping {sub}: Checkpoint not found at {ckpt_path}")
            continue

        train_subs = [s for s in SUBJECTS if s != sub]
        _, test_dataset, _ = prepare_BCI2b_dataset(
            root=Path(DATASET_DIR, 'BCICIV_2b'), 
            train_subjects=train_subs,
            val_subjects=[sub],
            test_subjects=[sub],
            is_instruct=True, 
            eeg_max_len=276, 
            text_max_len=80
        )
        
        loader = torch.utils.data.DataLoader(test_dataset, batch_size=64, shuffle=False)
        model = load_neurolm(ckpt_path)

        y_true_orig, probs_orig = [], []
        y_true_ablated = {band: [] for band in BANDS.keys()}
        probs_ablated = {band: [] for band in BANDS.keys()}

        with torch.no_grad():
            for batch in tqdm(loader, desc=f"Subject {sub}"):
                X_eeg, X_text, label, input_chans, input_time, input_mask, gpt_mask = batch
                
                # --- Baseline Inference ---
                X_eeg_orig, X_text = X_eeg.float().to(device), X_text.to(device)
                input_chans, input_time, gpt_mask = input_chans.to(device), input_time.to(device), gpt_mask.to(device)
                if input_mask is not None: input_mask = input_mask.to(device)

                with ctx:
                    text_out = model.generate(X_eeg_orig, X_text, input_chans, input_time, input_mask, eeg_text_mask=gpt_mask, max_new_tokens=5)
                    for i, t in enumerate(text_out[:, 1:]):
                        pred = get_pred(decode(t.tolist()))
                        probs_orig.append(pred)
                        y_true_orig.append(label[i].item())

                # --- Spectral Ablation Inference ---
                for band_name, freq_range in BANDS.items():
                    X_eeg_abl = apply_spectral_ablation(X_eeg_orig, fs=FS, band=freq_range)
                    
                    with ctx:
                        text_out_abl = model.generate(X_eeg_abl, X_text, input_chans, input_time, input_mask, eeg_text_mask=gpt_mask, max_new_tokens=5)
                        for i, t in enumerate(text_out_abl[:, 1:]):
                            pred_abl = get_pred(decode(t.tolist()))
                            probs_ablated[band_name].append(pred_abl)
                            y_true_ablated[band_name].append(label[i].item())

        # Process Baseline Metrics
        valid_idx = [i for i, p in enumerate(probs_orig) if p != -1]
        y_tv = np.array(y_true_orig)[valid_idx]
        p_ov = np.array(probs_orig)[valid_idx]
        
        res_orig["bacc"].append(balanced_accuracy_score(y_tv, p_ov) * 100)
        res_orig["f1"].append(f1_score(y_tv, p_ov, average='macro') * 100)
        res_orig["kappa"].append(cohen_kappa_score(y_tv, p_ov))

        # Process Ablation Metrics
        for band_name in BANDS.keys():
            valid_idx_a = [i for i, p in enumerate(probs_ablated[band_name]) if p != -1]
            y_ta = np.array(y_true_ablated[band_name])[valid_idx_a]
            p_av = np.array(probs_ablated[band_name])[valid_idx_a]
            
            res_ablated[band_name]["bacc"].append(balanced_accuracy_score(y_ta, p_av) * 100)
            res_ablated[band_name]["f1"].append(f1_score(y_ta, p_av, average='macro') * 100)
            res_ablated[band_name]["kappa"].append(cohen_kappa_score(y_ta, p_av))

    # ==========================================
    # 3. REPORTING
    # ==========================================
    print("\n" + "="*80)
    print("NeuroLM BCI IV 2b SPECTRAL ABLATION SUMMARY")
    print("="*80)
    
    print(f"\n[BASELINE (No Ablation)]")
    orig_bacc_mean = np.mean(res_orig['bacc'])
    print(f"  BACC : {orig_bacc_mean:.2f} ± {np.std(res_orig['bacc']):.2f}%")
    print(f"  F1   : {np.mean(res_orig['f1']):.2f} ± {np.std(res_orig['f1']):.2f}%")
    print(f"  Kappa: {np.mean(res_orig['kappa']):.4f} ± {np.std(res_orig['kappa']):.4f}")

    print(f"\n[ABLATED BANDS: PERFORMANCE & DROP]")
    print(f"{'BAND':<10} | {'BACC':<15} | {'DROP (Δ)':<10} | {'F1':<15} | {'Kappa':<15}")
    print("-" * 75)
    
    for band_name in BANDS.keys():
        b_mean = np.mean(res_ablated[band_name]['bacc'])
        b_std = np.std(res_ablated[band_name]['bacc'])
        f_mean = np.mean(res_ablated[band_name]['f1'])
        k_mean = np.mean(res_ablated[band_name]['kappa'])
        drop = orig_bacc_mean - b_mean
        
        print(f"{band_name:<10} | {b_mean:>5.2f}±{b_std:>5.2f}% | {-drop:>8.2f}% | {f_mean:>5.2f}% | {k_mean:>6.4f}")

if __name__ == "__main__":
    run_neurolm_spectral_ablation()