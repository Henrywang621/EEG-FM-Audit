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
# Updated paths based on your command and screenshot
BASE_CHECKPOINT_PATH = "/homes/xw2336/xw2336/NeuroLM/NeuoLM_test/results_BCI2b/default_run/checkpoints/instruction-B_2b4"
DATASET_DIR = "/homes/xw2336/data"
SUBJECTS = ['S01', 'S02', 'S03', 'S04', 'S05', 'S06', 'S07', 'S08', 'S09']
PHASE_SHUFFLE_SEEDS = list(range(100, 110))

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
dtype = 'bfloat16' if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else 'float16'
ptdtype = {'float32': torch.float32, 'bfloat16': torch.bfloat16, 'float16': torch.float16}[dtype]
ctx = torch.amp.autocast(device_type='cuda', dtype=ptdtype)

enc = tiktoken.get_encoding("gpt2")
decode = lambda l: enc.decode(l)

# ==========================================
# 1. PERTURBATION LOGIC & PARSING
# ==========================================
def apply_spatial_preserving_phase_rand(x, phase_seed):
    """Applies phase randomization while preserving spatial covariance."""
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

def get_pred(pred_string):
    """Parses NeuroLM's text generation output into binary classes."""
    s = pred_string.lower()
    if 'yes' in s: return 1 
    if 'no' in s: return 0  
    # Fallback if the model hallucinates an invalid string
    return -1 

def load_neurolm(ckpt_path):
    """Loads the fine-tuned NeuroLM model."""
    checkpoint = torch.load(ckpt_path, map_location=device, weights_only=False)
    
    # Reconstruct the exact model architecture configuration
    model_args = checkpoint['model_args']
    model = NeuroLM(GPTConfig(**model_args), init_from='gpt2')
    
    # Clean up state dict prefixes if wrapped in DDP
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
def run_neurolm_analysis():
    metric_keys = ["bacc", "f1", "kappa"]
    summary_orig = {k: [] for k in metric_keys}
    summary_rand = {k: [] for k in metric_keys}

    for sub in SUBJECTS:
        print(f"\n--- Analyzing Fold: {sub} ---")
        
        # Accessing the ckpt-final.pt inside the fold-specific subdirectory
        ckpt_path = os.path.join(BASE_CHECKPOINT_PATH, sub, "ckpt-final.pt")
        
        if not os.path.exists(ckpt_path):
            print(f"Skipping {sub}: Checkpoint not found at {ckpt_path}")
            continue

        # Isolate LOSO training logic to maintain scaler normalization consistency
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
        
        # Batch size 64 used for rapid evaluation
        loader = torch.utils.data.DataLoader(test_dataset, batch_size=64, shuffle=False)
        model = load_neurolm(ckpt_path)

        # ----------------------------------------
        # A. Baseline Evaluation (Original)
        # ----------------------------------------
        y_true_orig, probs_orig = [], []
        with torch.no_grad():
            for batch in loader:
                X_eeg, X_text, label, input_chans, input_time, input_mask, gpt_mask = batch
                X_eeg, X_text = X_eeg.float().to(device), X_text.to(device)
                input_chans, input_time, gpt_mask = input_chans.to(device), input_time.to(device), gpt_mask.to(device)
                if input_mask is not None: input_mask = input_mask.to(device)

                with ctx:
                    text_out = model.generate(X_eeg, X_text, input_chans, input_time, input_mask, eeg_text_mask=gpt_mask, max_new_tokens=5)
                    
                    for i, t in enumerate(text_out[:, 1:]):
                        decoded_str = decode(t.tolist())
                        pred = get_pred(decoded_str)
                        probs_orig.append(pred)
                        y_true_orig.append(label[i].item())

        # Filter out hallucinated answers (-1) for fair metrics mapping
        valid_idx = [i for i, p in enumerate(probs_orig) if p != -1]
        y_true_valid = np.array(y_true_orig)[valid_idx]
        pred_orig_valid = np.array(probs_orig)[valid_idx]
        
        m_bacc = balanced_accuracy_score(y_true_valid, pred_orig_valid) * 100
        m_f1 = f1_score(y_true_valid, pred_orig_valid, average='macro') * 100
        m_kappa = cohen_kappa_score(y_true_valid, pred_orig_valid)

        # ----------------------------------------
        # B. Phase Randomization Evaluation
        # ----------------------------------------
        for p_seed in PHASE_SHUFFLE_SEEDS:
            probs_rand, y_true_rand = [], []
            with torch.no_grad():
                for batch in loader:
                    X_eeg, X_text, label, input_chans, input_time, input_mask, gpt_mask = batch
                    
                    # Apply perturbation logic directly to the EEG tensor
                    X_eeg = X_eeg.float().to(device)
                    X_eeg_rand = apply_spatial_preserving_phase_rand(X_eeg, p_seed)
                    
                    X_text = X_text.to(device)
                    input_chans, input_time, gpt_mask = input_chans.to(device), input_time.to(device), gpt_mask.to(device)
                    if input_mask is not None: input_mask = input_mask.to(device)

                    with ctx:
                        text_out_rand = model.generate(X_eeg_rand, X_text, input_chans, input_time, input_mask, eeg_text_mask=gpt_mask, max_new_tokens=5)
                        
                        for i, t in enumerate(text_out_rand[:, 1:]):
                            decoded_str = decode(t.tolist())
                            pred = get_pred(decoded_str)
                            probs_rand.append(pred)
                            y_true_rand.append(label[i].item())

            valid_idx_r = [i for i, p in enumerate(probs_rand) if p != -1]
            y_true_r_valid = np.array(y_true_rand)[valid_idx_r]
            pred_rand_valid = np.array(probs_rand)[valid_idx_r]
            
            summary_orig["bacc"].append(m_bacc)
            summary_orig["f1"].append(m_f1)
            summary_orig["kappa"].append(m_kappa)
            
            summary_rand["bacc"].append(balanced_accuracy_score(y_true_r_valid, pred_rand_valid) * 100)
            summary_rand["f1"].append(f1_score(y_true_r_valid, pred_rand_valid, average='macro') * 100)
            summary_rand["kappa"].append(cohen_kappa_score(y_true_r_valid, pred_rand_valid))

    # ==========================================
    # 3. REPORTING
    # ==========================================
    print("\n" + "="*50)
    print("NeuroLM BCI IV 2b PHASE RANDOMIZATION SUMMARY")
    print("="*50)
    for label_str, data in [("ORIGINAL", summary_orig), ("RANDOMIZED", summary_rand)]:
        print(f"\n[{label_str}]")
        for k in metric_keys:
            print(f"  {k.upper():<6}: {np.mean(data[k]):.2f} ± {np.std(data[k]):.2f}")

if __name__ == "__main__":
    run_neurolm_analysis()