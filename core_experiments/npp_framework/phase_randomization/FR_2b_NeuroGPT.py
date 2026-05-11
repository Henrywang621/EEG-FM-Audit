import torch
import torch.nn as nn
import numpy as np
import os
import sys
from tqdm import tqdm
from sklearn.metrics import balanced_accuracy_score, f1_score, cohen_kappa_score
from safetensors.torch import load_model  # For .safetensors

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
    if torch.is_tensor(outputs):
        return outputs
    if isinstance(outputs, dict):
        for key in ['decoding_logits', 'logits', 'out', 'output', 'y_hat']:
            if key in outputs:
                return outputs[key]
        return outputs[list(outputs.keys())[0]]
    raise TypeError(f"Unsupported output type: {type(outputs)}")

# ==========================================
# 2. MODEL LOADER (FIXED)
# ==========================================
def load_neurogpt_eval(ckpt_path, device):
    # Standard NeuroGPT decoding config for BCI 2b
    config = {
        "use_encoder": True, "num_decoding_classes": 2, "chunk_len": 500,
        "ft_only_encoder": False, # Usually False for full GPT finetuning
        "filter_time_length": 25, "pool_time_length": 75,
        "stride_avg_pool": 15, "n_filters_time": 40, "training_style": 'decoding',
        "architecture": 'GPT', "embedding_dim": 1024, "num_hidden_layers_embedding_model": 1,
        "num_hidden_layers_unembedding_model": 1, "dropout": 0.1, "n_positions": 512,
        "num_hidden_layers": 6, "num_attention_heads": 16, "intermediate_dim_factor": 4,
        "hidden_activation": 'gelu_new'
    }
    config["parcellation_dim"] = ((config['chunk_len'] - config['filter_time_length'] + 1 - config['pool_time_length']) // config['stride_avg_pool'] + 1) * config['n_filters_time']

    # Initialize Components
    encoder = EEGConformer(n_outputs=config["num_decoding_classes"], n_chans=22, n_times=config['chunk_len'], is_decoding_mode=config["ft_only_encoder"])
    embedder = make_embedder(training_style=config["training_style"], architecture=config["architecture"], in_dim=config["parcellation_dim"], embed_dim=config["embedding_dim"], num_hidden_layers=config["num_hidden_layers_embedding_model"], dropout=config["dropout"], n_positions=config["n_positions"])
    decoder = make_decoder(architecture=config["architecture"], num_hidden_layers=config["num_hidden_layers"], embed_dim=config["embedding_dim"], num_attention_heads=config["num_attention_heads"], n_positions=config["n_positions"], intermediate_dim_factor=config["intermediate_dim_factor"], hidden_activation=config["hidden_activation"], dropout=config["dropout"])
    unembedder = make_unembedder(embed_dim=config["embedding_dim"], num_hidden_layers=config["num_hidden_layers_unembedding_model"], out_dim=config["parcellation_dim"], dropout=config["dropout"])
    
    model = Model(encoder=encoder, embedder=embedder, decoder=decoder, unembedder=unembedder)
    
    # 1. Switch to decoding mode BEFORE loading to create the Model-level decoding_head
    model.switch_decoding_mode(is_decoding_mode=True, num_decoding_classes=config["num_decoding_classes"])
    
    print(f"Loading weights from: {ckpt_path}")
    
    # 2. Use strict=False to handle discrepancies between Encoder-head and Model-head weights
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
def run_neurogpt_analysis():
    # PATHS
    BASE_LOG_DIR = "/homes/xw2336/data_portal/LLM_eva_fast/NeuroGPT/src/log/2b_robustness/weights" 
    ALL_SUBJECTS = ['S01', 'S02', 'S03', 'S04', 'S05', 'S06', 'S07', 'S08', 'S09']
    PHASE_SHUFFLE_SEEDS = list(range(100, 110))
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    results = {"orig": {"bacc": [], "f1": [], "kappa": []}, "rand": {"bacc": [], "f1": [], "kappa": []}}

    for test_sub in ALL_SUBJECTS:
        print(f"\n--- Evaluating LOSO Fold: Test Subject {test_sub} ---")
        
        ckpt_path = os.path.join(BASE_LOG_DIR, f"fold_{test_sub}", "model_final", "model.safetensors")
        
        if not os.path.exists(ckpt_path):
            print(f"⚠️ Checkpoint not found at {ckpt_path}. Skipping."); continue
        
        test_dataset = BCIIV2bDataset(
            subject_ids=[test_sub], 
            sample_keys=['inputs', 'attention_mask', 'labels'], 
            chunk_len=500, num_chunks=8, ovlp=50, gpt_only=False
        )
        loader = torch.utils.data.DataLoader(test_dataset, batch_size=32, num_workers=4, shuffle=False)
        
        try:
            model = load_neurogpt_eval(ckpt_path, DEVICE)
        except Exception as e:
            print(f"❌ Failed to load model for {test_sub}: {e}"); continue

        y_true, y_pred_orig = [], []

        # Baseline Prediction
        with torch.no_grad():
            for batch in tqdm(loader, desc=f"Baseline {test_sub}"):
                batch = {k: v.to(DEVICE) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
                logits = extract_logits(model(batch))
                preds = torch.argmax(logits, dim=-1)
                y_pred_orig.extend(preds.cpu().numpy().flatten())
                y_true.extend(batch['labels'].cpu().numpy().flatten())

        y_true, y_pred_orig = np.array(y_true), np.array(y_pred_orig)
        m_bacc = balanced_accuracy_score(y_true, y_pred_orig) * 100
        m_f1 = f1_score(y_true, y_pred_orig, average='macro') * 100
        m_kappa = cohen_kappa_score(y_true, y_pred_orig)

        # Phase Randomization
        for p_seed in PHASE_SHUFFLE_SEEDS:
            y_pred_rand = []
            with torch.no_grad():
                for batch in tqdm(loader, desc=f"  Shuffle {p_seed}", leave=False):
                    batch = {k: v.to(DEVICE) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
                    batch_rand = batch.copy()
                    batch_rand['inputs'] = apply_phase_randomization(batch['inputs'], p_seed)
                    
                    logits_rand = extract_logits(model(batch_rand))
                    preds_rand = torch.argmax(logits_rand, dim=-1)
                    y_pred_rand.extend(preds_rand.cpu().numpy().flatten())

            y_pred_rand = np.array(y_pred_rand)
            
            for k, v in [("bacc", m_bacc), ("f1", m_f1), ("kappa", m_kappa)]: 
                results["orig"][k].append(v)
            
            results["rand"]["bacc"].append(balanced_accuracy_score(y_true, y_pred_rand) * 100)
            results["rand"]["f1"].append(f1_score(y_true, y_pred_rand, average='macro') * 100)
            results["rand"]["kappa"].append(cohen_kappa_score(y_true, y_pred_rand))

        del model; torch.cuda.empty_cache()

    # Final Summary Output
    print("\n" + "="*75 + "\nNEUROGPT BCI IV 2b ROBUSTNESS SUMMARY (9 FOLDS x 10 SHUFFLES)\n" + "="*75)
    if not results["orig"]["bacc"]:
        print("No results collected. Check paths and checkpoint availability.")
        return

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