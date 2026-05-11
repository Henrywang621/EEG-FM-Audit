import torch
import torch.nn as nn
import numpy as np
import os
import pickle
import random
import sys
from tqdm import tqdm
from sklearn.metrics import balanced_accuracy_score, f1_score, cohen_kappa_score

# ==========================================
# 0. NEUROGPT IMPORTS & SETUP
# ==========================================
script_path = os.path.dirname(os.path.realpath(__file__))
sys.path.insert(0, os.path.join(script_path, '../'))

from encoder.conformer_braindecode import EEGConformer
from batcher.downstream_dataset import TUEVDataset  # CHANGED: TUEV instead of TUAB
from decoder.make_decoder import make_decoder
from embedder.make import make_embedder
from decoder.unembedder import make_unembedder
from model import Model

# ==========================================
# 1. PERTURBATION & UTILITY LOGIC
# ==========================================
def apply_phase_randomization(x, phase_seed):
    """Applies phase randomization while preserving spatial covariance."""
    gen = torch.Generator(device=x.device)
    gen.manual_seed(phase_seed)
    
    orig_shape = x.shape  # (B, Chunks, C, T)
    x_flat = x.view(-1, orig_shape[2], orig_shape[3]) 
    
    batch_size, num_channels, time_len = x_flat.shape
    mu_x = x_flat.mean(dim=-1, keepdim=True)
    y = x_flat - mu_x
    Y = torch.fft.rfft(y, dim=-1)
    magnitudes = torch.abs(Y)
    
    random_phases = torch.rand(batch_size, 1, Y.shape[-1], 
                               generator=gen, device=x.device) * 2 * torch.pi
    random_phases[..., 0] = 0.0
    
    Y_surrogate = magnitudes * torch.exp(1j * random_phases)
    x_shuffled = torch.fft.irfft(Y_surrogate, n=time_len, dim=-1) + mu_x
    
    return x_shuffled.view(orig_shape)

def extract_logits(outputs):
    """Extracts classification logits from NeuroGPT dictionary."""
    if torch.is_tensor(outputs):
        return outputs
    if isinstance(outputs, dict):
        for key in ['decoding_logits', 'logits', 'out', 'output', 'y_hat']:
            if key in outputs:
                return outputs[key]
        return outputs[list(outputs.keys())[0]]
    raise TypeError(f"Unsupported output type: {type(outputs)}")

# ==========================================
# 2. MODEL LOADER (TUEV SPECIFIC)
# ==========================================
def load_neurogpt_eval(ckpt_path, device):
    # CHANGED: num_decoding_classes = 6 for TUEV
    config = {
        "use_encoder": True, "num_decoding_classes": 6, "chunk_len": 500,
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
    
    print(f"Loading weights from: {ckpt_path}")
    # Using strict=True as TUEV model structure should match local config
    model.load_state_dict(torch.load(ckpt_path, map_location='cpu', weights_only=True), strict=True)
    model.to(device).eval()
    return model

# ==========================================
# 3. ANALYSIS LOOP
# ==========================================
def run_neurogpt_analysis():
    # UPDATED: TUEV Data and Weight Paths
    DATA_ROOT = "/homes/xw2336/data_portal/TUEV/processed"
    TEST_PATH = os.path.join(DATA_ROOT, 'processed_test/')
    BASE_LOG_DIR = "/homes/xw2336/data_portal/LLM_eva_fast/NeuroGPT/pretrained_model/tuev_original" 
    
    MODEL_SEEDS = [42, 3407, 6, 16, 66]
    PHASE_SHUFFLE_SEEDS = list(range(100, 110))
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    test_files = [f for f in os.listdir(TEST_PATH) if f.endswith(".pkl")]
    # CHANGED: Using TUEVDataset
    dataset = TUEVDataset(root=TEST_PATH, filenames=test_files, chunk_len=500, num_chunks=2, ovlp=0, gpt_only=False, sample_keys=['inputs', 'attention_mask', 'labels'])
    loader = torch.utils.data.DataLoader(dataset, batch_size=32, num_workers=4, shuffle=False)

    results = {"orig": {"bacc": [], "f1": [], "kappa": []}, "rand": {"bacc": [], "f1": [], "kappa": []}}

    for m_seed in MODEL_SEEDS:
        print(f"\n--- Model Seed: {m_seed} ---")
        ckpt_path = os.path.join(BASE_LOG_DIR, f"seed_{m_seed}", "full_weights.pth")
        
        if not os.path.exists(ckpt_path):
            print(f"⚠️ Checkpoint not found at {ckpt_path}. Skipping."); continue
        
        model = load_neurogpt_eval(ckpt_path, DEVICE)
        y_true, y_pred_orig = [], []

        # Step A: Baseline
        with torch.no_grad():
            for batch in tqdm(loader, desc=f"Baseline Model {m_seed}"):
                batch = {k: v.to(DEVICE) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
                
                outputs = model(batch)
                logits = extract_logits(outputs)
                
                # CHANGED: Multi-class prediction using argmax
                preds = torch.argmax(logits, dim=-1)
                y_pred_orig.extend(preds.cpu().numpy().flatten())
                y_true.extend(batch['labels'].cpu().numpy().flatten())

        y_true = np.array(y_true)
        y_pred_orig = np.array(y_pred_orig)
        
        # Calculate baseline metrics
        m_bacc = balanced_accuracy_score(y_true, y_pred_orig) * 100
        m_f1 = f1_score(y_true, y_pred_orig, average='weighted') * 100
        m_kappa = cohen_kappa_score(y_true, y_pred_orig)

        # Step B: Randomized
        for p_seed in PHASE_SHUFFLE_SEEDS:
            y_pred_rand = []
            with torch.no_grad():
                for batch in tqdm(loader, desc=f"  Shuffle {p_seed}", leave=False):
                    batch = {k: v.to(DEVICE) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
                    batch_rand = batch.copy()
                    batch_rand['inputs'] = apply_phase_randomization(batch['inputs'], p_seed)
                    
                    outputs_rand = model(batch_rand)
                    logits_rand = extract_logits(outputs_rand)
                    
                    # Multi-class prediction
                    preds_rand = torch.argmax(logits_rand, dim=-1)
                    y_pred_rand.extend(preds_rand.cpu().numpy().flatten())

            y_pred_rand = np.array(y_pred_rand)
            
            # Store original results (repeated for every shuffle to maintain array shape for diff calculation)
            for k, v in [("bacc", m_bacc), ("f1", m_f1), ("kappa", m_kappa)]: 
                results["orig"][k].append(v)
            
            # Store randomized results
            results["rand"]["bacc"].append(balanced_accuracy_score(y_true, y_pred_rand) * 100)
            results["rand"]["f1"].append(f1_score(y_true, y_pred_rand, average='weighted') * 100)
            results["rand"]["kappa"].append(cohen_kappa_score(y_true, y_pred_rand))

        del model; torch.cuda.empty_cache()

    # Final Summary
    print("\n" + "="*75 + "\nNEUROGPT TUEV ROBUSTNESS SUMMARY (50 RUNS)\n" + "="*75)
    for lbl, data in [("ORIGINAL", results["orig"]), ("RANDOMIZED", results["rand"])]:
        print(f"[{lbl}]")
        for k in ["bacc", "f1", "kappa"]:
            print(f"  {k.upper():<6}: {np.mean(data[k]):.2f} ± {np.std(data[k]):.2f}{'%' if k!='kappa' else ''}")
    
    print("\n[DIFFERENCE (ORIG - RAND)]")
    for k in ["bacc", "f1", "kappa"]:
        diffs = np.array(results["orig"][k]) - np.array(results["rand"][k])
        print(f"  {k.upper():<6}: {np.mean(diffs):.2f} ± {np.std(diffs):.2f}{'%' if k!='kappa' else ''}")

if __name__ == "__main__":
    run_neurogpt_analysis()