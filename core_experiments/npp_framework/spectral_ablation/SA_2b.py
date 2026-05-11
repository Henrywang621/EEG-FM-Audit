import torch
import torch.nn as nn
import numpy as np
import os
import sys
import random
from tqdm import tqdm
from sklearn.metrics import balanced_accuracy_score, f1_score, cohen_kappa_score
from safetensors.torch import load_model

# ==========================================
# 0. NEUROGPT IMPORTS & SETUP
# ==========================================
script_path = os.path.dirname(os.path.realpath(__file__))
sys.path.insert(0, os.path.join(script_path, '../'))

from encoder.conformer_braindecode import EEGConformer
from batcher.downstream_dataset import BCIIV2bDataset 
from decoder.make_decoder import make_decoder
from embedder.make import make_embedder
from decoder.unembedder import make_unembedder
from model import Model

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# Physiological Frequency Bands
FREQ_BANDS = {
    'Delta': (0.5, 4),
    'Theta': (4, 8),
    'Alpha': (8, 13),
    'Beta': (13, 30),
    'Gamma': (30, 100) # Up to Nyquist
}

# Standard BCI IV 2b sampling rate
FS = 250.0

# ==========================================
# 1. SPECTRAL ABLATION & UTILITY LOGIC
# ==========================================
def apply_spectral_ablation(x, fs, band_name):
    """
    Zeros out the frequencies corresponding to the specified band.
    NeuroGPT x expected shape: (Batch, Chunks, Channels, Time)
    """
    if band_name not in FREQ_BANDS:
        return x
        
    low_cut, high_cut = FREQ_BANDS[band_name]
    
    # Transform to frequency domain
    fft_x = torch.fft.rfft(x, dim=-1)
    n = x.shape[-1]
    freqs = torch.fft.rfftfreq(n, d=1/fs).to(x.device)
    
    # Create mask: True for frequencies outside the band, False for inside
    mask = ~((freqs >= low_cut) & (freqs <= high_cut))
    
    # Reshape mask to broadcast across (Batch, Chunks, Channels, Time)
    mask = mask.view(1, 1, 1, -1)
    
    # Zero out the targeted frequency band and inverse transform
    return torch.fft.irfft(fft_x * mask, n=n, dim=-1)

def extract_logits(outputs):
    """Extracts classification logits from NeuroGPT output dictionary."""
    if torch.is_tensor(outputs): return outputs
    if isinstance(outputs, dict):
        for key in ['decoding_logits', 'logits', 'out', 'output', 'y_hat']:
            if key in outputs: return outputs[key]
        return outputs[list(outputs.keys())[0]]
    raise TypeError(f"Unsupported output type: {type(outputs)}")

# ==========================================
# 2. MODEL LOADER (BCI 2B SPECIFIC)
# ==========================================
def load_neurogpt_eval(ckpt_path, device):
    # Standard NeuroGPT decoding config for BCI 2b
    config = {
        "use_encoder": True, "num_decoding_classes": 2, "chunk_len": 500,
        "ft_only_encoder": False,
        "filter_time_length": 25, "pool_time_length": 75,
        "stride_avg_pool": 15, "n_filters_time": 40, "training_style": 'decoding',
        "architecture": 'GPT', "embedding_dim": 1024, "num_hidden_layers_embedding_model": 1,
        "num_hidden_layers_unembedding_model": 1, "dropout": 0.1, "n_positions": 512,
        "num_hidden_layers": 6, "num_attention_heads": 16, "intermediate_dim_factor": 4,
        "hidden_activation": 'gelu_new'
    }
    config["parcellation_dim"] = ((config['chunk_len'] - config['filter_time_length'] + 1 - config['pool_time_length']) // config['stride_avg_pool'] + 1) * config['n_filters_time']

    encoder = EEGConformer(n_outputs=config["num_decoding_classes"], n_chans=22, n_times=config['chunk_len'], is_decoding_mode=config["ft_only_encoder"])
    embedder = make_embedder(training_style=config["training_style"], architecture=config["architecture"], in_dim=config["parcellation_dim"], embed_dim=config["embedding_dim"], num_hidden_layers=config["num_hidden_layers_embedding_model"], dropout=config["dropout"], n_positions=config["n_positions"])
    decoder = make_decoder(architecture=config["architecture"], num_hidden_layers=config["num_hidden_layers"], embed_dim=config["embedding_dim"], num_attention_heads=config["num_attention_heads"], n_positions=config["n_positions"], intermediate_dim_factor=config["intermediate_dim_factor"], hidden_activation=config["hidden_activation"], dropout=config["dropout"])
    unembedder = make_unembedder(embed_dim=config["embedding_dim"], num_hidden_layers=config["num_hidden_layers_unembedding_model"], out_dim=config["parcellation_dim"], dropout=config["dropout"])
    
    model = Model(encoder=encoder, embedder=embedder, decoder=decoder, unembedder=unembedder)
    model.switch_decoding_mode(is_decoding_mode=True, num_decoding_classes=config["num_decoding_classes"])
    
    print(f"Loading weights from: {ckpt_path}")
    if ckpt_path.endswith(".safetensors"):
        load_model(model, ckpt_path, strict=False)
    else:
        state_dict = torch.load(ckpt_path, map_location='cpu', weights_only=True)
        model.load_state_dict(state_dict, strict=False)
        
    model.to(device).eval()
    return model

# ==========================================
# 3. ANALYSIS LOOP (LOSO STYLE)
# ==========================================
def run_neurogpt_spectral_ablation():
    set_seed(42)
    BASE_LOG_DIR = "/homes/xw2336/data_portal/LLM_eva_fast/NeuroGPT/src/log/2b_robustness/weights" 
    ALL_SUBJECTS = ['S01', 'S02', 'S03', 'S04', 'S05', 'S06', 'S07', 'S08', 'S09']
    BANDS = ['Delta', 'Theta', 'Alpha', 'Beta', 'Gamma']
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Tracking metrics across folds
    res_orig = {"bacc": [], "f1": [], "kappa": []}
    res_ablated = {band: {"bacc": [], "f1": [], "kappa": []} for band in BANDS}

    for test_sub in ALL_SUBJECTS:
        print(f"\n--- Evaluating LOSO Fold: Test Subject {test_sub} ---")
        
        ckpt_path = os.path.join(BASE_LOG_DIR, f"fold_{test_sub}", "model_final", "model.safetensors")
        if not os.path.exists(ckpt_path):
            print(f"⚠️ Checkpoint not found at {ckpt_path}. Skipping.")
            continue
        
        test_dataset = BCIIV2bDataset(
            subject_ids=[test_sub], 
            sample_keys=['inputs', 'attention_mask', 'labels'], 
            chunk_len=500, num_chunks=8, ovlp=50, gpt_only=False
        )
        loader = torch.utils.data.DataLoader(test_dataset, batch_size=32, num_workers=4, shuffle=False)
        
        try:
            model = load_neurogpt_eval(ckpt_path, DEVICE)
        except Exception as e:
            print(f"❌ Failed to load model for {test_sub}: {e}")
            continue

        y_true, p_orig = [], []
        p_bands = {band: [] for band in BANDS}

        with torch.no_grad():
            for batch in tqdm(loader, desc=f"Baseline {test_sub}"):
                batch = {k: v.to(DEVICE) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
                inputs = batch['inputs']
                labels = batch['labels'].cpu().numpy().flatten()
                y_true.extend(labels)

                # 1. Baseline Prediction
                logits_o = extract_logits(model(batch))
                preds_o = torch.argmax(logits_o, dim=-1)
                p_orig.extend(preds_o.cpu().numpy().flatten())

                # 2. Spectral Ablation Inference
                for band in BANDS:
                    batch_a = batch.copy()
                    batch_a['inputs'] = apply_spectral_ablation(inputs, FS, band)
                    
                    logits_a = extract_logits(model(batch_a))
                    preds_a = torch.argmax(logits_a, dim=-1)
                    p_bands[band].extend(preds_a.cpu().numpy().flatten())

        # Metric Calculation for this specific Fold
        yt = np.array(y_true)
        po = np.array(p_orig)
        
        res_orig["bacc"].append(balanced_accuracy_score(yt, po) * 100)
        res_orig["f1"].append(f1_score(yt, po, average='macro') * 100)
        res_orig["kappa"].append(cohen_kappa_score(yt, po))

        for band in BANDS:
            pa = np.array(p_bands[band])
            res_ablated[band]["bacc"].append(balanced_accuracy_score(yt, pa) * 100)
            res_ablated[band]["f1"].append(f1_score(yt, pa, average='macro') * 100)
            res_ablated[band]["kappa"].append(cohen_kappa_score(yt, pa))

        del model; torch.cuda.empty_cache()

    # ==========================================
    # 4. FINAL AGGREGATED SUMMARY (ACROSS 9 FOLDS)
    # ==========================================
    if not res_orig["bacc"]:
        print("\nNo results collected. Check paths and checkpoint availability.")
        return

    print("\n" + "="*85)
    print("NEUROGPT BCI IV 2b SPECTRAL ABLATION SUMMARY (9 FOLDS)")
    print("="*85)

    # Baseline Summary
    print(f"\n[BASELINE (No Ablation)]")
    orig_bacc_mean = np.mean(res_orig['bacc'])
    print(f"  BACC : {orig_bacc_mean:.2f} ± {np.std(res_orig['bacc']):.2f}%")
    print(f"  F1   : {np.mean(res_orig['f1']):.2f} ± {np.std(res_orig['f1']):.2f}%")
    print(f"  Kappa: {np.mean(res_orig['kappa']):.4f} ± {np.std(res_orig['kappa']):.4f}")

    # Ablation Drops
    print(f"\n[ABLATED BANDS: PERFORMANCE & DROP]")
    print(f"{'BAND':<10} | {'BACC':<15} | {'DROP (Δ)':<10} | {'F1':<15} | {'Kappa':<15}")
    print("-" * 75)
    
    for band in BANDS:
        b_mean = np.mean(res_ablated[band]['bacc'])
        b_std = np.std(res_ablated[band]['bacc'])
        f_mean = np.mean(res_ablated[band]['f1'])
        k_mean = np.mean(res_ablated[band]['kappa'])
        drop = orig_bacc_mean - b_mean
        
        print(f"{band:<10} | {b_mean:>5.2f}±{b_std:>5.2f}% | {-drop:>8.2f}% | {f_mean:>5.2f}% | {k_mean:>6.4f}")

if __name__ == "__main__":
    run_neurogpt_spectral_ablation()