import torch
import torch.nn as nn
import numpy as np
import os
import sys
import random
from tqdm import tqdm
from sklearn.metrics import balanced_accuracy_score, f1_score, cohen_kappa_score

# ==========================================
# 0. NEUROGPT & REPRODUCIBILITY SETUP
# ==========================================
script_path = os.path.dirname(os.path.realpath(__file__))
sys.path.insert(0, os.path.join(script_path, '../'))

from encoder.conformer_braindecode import EEGConformer
from batcher.downstream_dataset import TUABDataset
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
    'Delta': (1, 4),
    'Theta': (4, 8),
    'Alpha': (8, 13),
    'Beta':  (13, 30),
    'Gamma': (30, 75)
}

# Assume TUAB standard sampling rate for this preprocessing pipeline
FS = 200.0

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
        for key in ['decoding_logits', 'logits', 'out', 'output']:
            if key in outputs: return outputs[key]
        return outputs[list(outputs.keys())[0]]
    raise TypeError(f"Unsupported output type: {type(outputs)}")

# ==========================================
# 2. MODEL LOADER 
# ==========================================
def load_neurogpt_eval(ckpt_path, device):
    config = {
        "use_encoder": True, "num_decoding_classes": 2, "chunk_len": 500,
        "ft_only_encoder": False, "filter_time_length": 25, "pool_time_length": 75,
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
    model.load_state_dict(torch.load(ckpt_path, map_location='cpu', weights_only=True))
    model.to(device).eval()
    return model

# ==========================================
# 3. ANALYSIS LOOP
# ==========================================
def run_analysis():
    DATA_ROOT = "/homes/xw2336/data_portal/TUAB"
    TEST_PATH = os.path.join(DATA_ROOT, 'test/')
    BASE_DIR = "/homes/xw2336/data_portal/LLM_eva_fast/NeuroGPT/pretrained_model/tuab_original" 
    
    MODEL_SEEDS = [42, 3407, 6, 16, 66]
    BANDS = ['Delta', 'Theta', 'Alpha', 'Beta', 'Gamma']
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    test_files = [f for f in os.listdir(TEST_PATH) if f.endswith(".pkl")]
    
    # gpt_only=False ensures inputs are the raw 4D tensor (Batch, Chunks, Channels, Time)
    dataset = TUABDataset(root=TEST_PATH, filenames=test_files, chunk_len=500, num_chunks=2, ovlp=0, gpt_only=False, sample_keys=['inputs', 'attention_mask', 'labels'])
    loader = torch.utils.data.DataLoader(dataset, batch_size=32, num_workers=4, shuffle=False)

    # Tracking Structures
    res_orig = {"bacc": [], "f1": [], "kappa": []}
    res_ablated = {band: {"bacc": [], "f1": [], "kappa": []} for band in BANDS}
    res_diff = {band: {"bacc": [], "f1": [], "kappa": []} for band in BANDS}

    for seed in MODEL_SEEDS:
        print(f"\n🚀 Processing Model Seed {seed}...")
        set_seed(seed)
        
        ckpt_path = os.path.join(BASE_DIR, f"seed_{seed}", "full_weights.pth")
        if not os.path.exists(ckpt_path):
            print(f"⚠️ Checkpoint not found at {ckpt_path}. Skipping.")
            continue
            
        model = load_neurogpt_eval(ckpt_path, DEVICE)

        y_true, p_orig = [], []
        p_bands = {band: [] for band in BANDS}

        with torch.no_grad():
            for batch in tqdm(loader, desc=f"Seed {seed} Inference"):
                batch = {k: v.to(DEVICE) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
                inputs = batch['inputs']
                labels = batch['labels'].cpu().numpy().flatten()
                y_true.extend(labels)

                # 1. Original Baseline Prediction
                logits_o = extract_logits(model(batch))
                p_orig.extend(torch.softmax(logits_o, dim=-1)[:, 1].cpu().numpy())

                # 2. Ablated Predictions
                for band in BANDS:
                    batch_a = batch.copy()
                    batch_a['inputs'] = apply_spectral_ablation(inputs, FS, band)
                    
                    logits_a = extract_logits(model(batch_a))
                    p_bands[band].extend(torch.softmax(logits_a, dim=-1)[:, 1].cpu().numpy())

        # Calculate metrics for this seed
        yt = np.array(y_true)
        po = (np.array(p_orig) > 0.5).astype(float)
        
        m_orig = {
            "bacc": balanced_accuracy_score(yt, po) * 100,
            "f1": f1_score(yt, po) * 100,
            "kappa": cohen_kappa_score(yt, po)
        }
        for k in res_orig: res_orig[k].append(m_orig[k])

        print(f"\n[Seed {seed} - Original Baseline]")
        print(f"BACC: {m_orig['bacc']:.2f}% | F1: {m_orig['f1']:.2f}% | Kappa: {m_orig['kappa']:.4f}")
        
        print(f"{'BAND':<15} | {'BACC Diff':<10} | {'F1 Diff':<10} | {'Kappa Diff':<10}")
        print("-" * 55)

        for band in BANDS:
            pa = (np.array(p_bands[band]) > 0.5).astype(float)
            m_a = {
                "bacc": balanced_accuracy_score(yt, pa) * 100,
                "f1": f1_score(yt, pa) * 100,
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
        ("F1 SCORE (F1 %)", "f1", "{:.2f}"),
        ("COHEN'S KAPPA", "kappa", "{:.4f}")
    ]

    for title, key, fmt in metrics_to_print:
        print("\n" + "="*95)
        print(f"{title} - AGGREGATED SUMMARY (MEAN ± STD)")
        print("="*95)
        print(f"{'BAND':<15} | {'ORIGINAL':<18} | {'ABLATED':<18} | {'DROP (DIFFERENCE)'}")
        print("-" * 95)

        orig_mean = np.mean(res_orig[key])
        orig_std = np.std(res_orig[key])
        print(f"{'BASELINE':<15} | {fmt.format(orig_mean)} ± {fmt.format(orig_std)} | {'-':<18} | {'0.0'}")
        print("-" * 95)

        for band in BANDS:
            a_m = np.mean(res_ablated[band][key])
            a_s = np.std(res_ablated[band][key])
            d_m = np.mean(res_diff[band][key])
            d_s = np.std(res_diff[band][key])

            row = f"{band:<15} | {fmt.format(orig_mean)} ± {fmt.format(orig_std)} | " \
                  f"{fmt.format(a_m)} ± {fmt.format(a_s)} | {fmt.format(d_m)} ± {fmt.format(d_s)}"
            print(row)

    print("="*95)

if __name__ == "__main__":
    run_analysis()