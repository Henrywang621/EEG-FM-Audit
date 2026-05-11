import torch
import torch.nn as nn
import numpy as np
import os
import sys
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
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True

# ==========================================
# 1. PERTURBATION & ROI LOGIC
# ==========================================
# NeuroGPT typically pads to 22 channels. We identify the BCI 2b specific channels.
STANDARD_CH_NAMES = ['EEG FP1', 'EEG FP2', 'EEG F3', 'EEG F4', 'EEG C3', 'EEG C4', 'EEG P3', 'EEG P4', 'EEG O1', 'EEG O2', 
                     'EEG F7', 'EEG F8', 'EEG T3', 'EEG T4', 'EEG T5', 'EEG T6', 'EEG A1', 'EEG A2', 'EEG FZ', 'EEG CZ', 
                     'EEG PZ', 'EEG T1', 'EEG T2']

ROI_DEFINITIONS = {
    'Central': ['C3', 'CZ', 'C4'] # BCI IV 2b specific electrodes
}

def get_roi_indices(roi_name, ch_names):
    target_names = ROI_DEFINITIONS.get(roi_name, [])
    indices = []
    for i, name in enumerate(ch_names):
        clean = name.split(' ')[-1].upper().strip()
        if clean in target_names:
            indices.append(i)
    return indices

def apply_spatial_noise(x, target_indices, noise_lambda=1.0):
    if len(target_indices) == 0: return x
    x_perturbed = x.clone()
    batch_std = x.std()
    noise = torch.randn_like(x_perturbed) * batch_std * noise_lambda
    
    # Mask for the channel dimension (index 2 in B, Chunks, C, T)
    mask = torch.zeros(x.shape[2], device=x.device).view(1, 1, -1, 1)
    mask[:, :, target_indices, :] = 1.0
    return x_perturbed + (noise * mask)

def extract_logits(outputs):
    if torch.is_tensor(outputs): return outputs
    if isinstance(outputs, dict):
        for key in ['decoding_logits', 'logits', 'out', 'output', 'y_hat']:
            if key in outputs: return outputs[key]
        return outputs[list(outputs.keys())[0]]
    return outputs

# ==========================================
# 2. MODEL LOADER (BCI 2B SPECIFIC)
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

    # Keep n_chans=22 as expected by the pre-trained Conformer weights
    encoder = EEGConformer(n_outputs=config["num_decoding_classes"], n_chans=22, n_times=config['chunk_len'], is_decoding_mode=config["ft_only_encoder"])
    embedder = make_embedder(training_style=config["training_style"], architecture=config["architecture"], in_dim=config["parcellation_dim"], embed_dim=config["embedding_dim"], num_hidden_layers=config["num_hidden_layers_embedding_model"], dropout=config["dropout"], n_positions=config["n_positions"])
    decoder = make_decoder(architecture=config["architecture"], num_hidden_layers=config["num_hidden_layers"], embed_dim=config["embedding_dim"], num_attention_heads=config["num_attention_heads"], n_positions=config["n_positions"], intermediate_dim_factor=config["intermediate_dim_factor"], hidden_activation=config["hidden_activation"], dropout=config["dropout"])
    unembedder = make_unembedder(embed_dim=config["embedding_dim"], num_hidden_layers=config["num_hidden_layers_unembedding_model"], out_dim=config["parcellation_dim"], dropout=config["dropout"])
    
    model = Model(encoder=encoder, embedder=embedder, decoder=decoder, unembedder=unembedder)
    
    # Switch to decoding mode BEFORE loading weights
    model.switch_decoding_mode(is_decoding_mode=True, num_decoding_classes=config["num_decoding_classes"])
    
    # Handle .safetensors format
    if ckpt_path.endswith(".safetensors"):
        load_model(model, ckpt_path, strict=False)
    else:
        state_dict = torch.load(ckpt_path, map_location='cpu', weights_only=True)
        model.load_state_dict(state_dict, strict=False)
        
    model.to(device).eval()
    return model

# ==========================================
# 3. ANALYSIS LOOP
# ==========================================
def run_analysis():
    set_seed(42)
    BASE_LOG_DIR = "/homes/xw2336/data_portal/LLM_eva_fast/NeuroGPT/src/log/2b_robustness/weights" 
    ALL_SUBJECTS = ['S01', 'S02', 'S03', 'S04', 'S05', 'S06', 'S07', 'S08', 'S09']
    
    REGIONS = ['Central'] # Only analyzing Central region due to BCI 2b 3-channel layout
    NOISE_LEVELS = [0.5, 1.0, 2.0]
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    res_orig = {"bacc": [], "f1": [], "kappa": []}
    res_pert = {roi: {n: {"bacc": [], "f1": [], "kappa": []} for n in NOISE_LEVELS} for roi in REGIONS}

    for test_sub in ALL_SUBJECTS:
        print(f"\n--- Evaluating LOSO Fold: Test Subject {test_sub} ---")
        
        ckpt_path = os.path.join(BASE_LOG_DIR, f"fold_{test_sub}", "model_final", "model.safetensors")
        if not os.path.exists(ckpt_path):
            print(f"⚠️ Checkpoint not found at {ckpt_path}. Skipping.")
            continue
        
        # Load dataset with gpt_only=False to ensure 4D inputs
        test_dataset = BCIIV2bDataset(
            subject_ids=[test_sub], 
            sample_keys=['inputs', 'attention_mask', 'labels'], 
            chunk_len=500, num_chunks=8, ovlp=50, gpt_only=False
        )
        loader = torch.utils.data.DataLoader(test_dataset, batch_size=32, num_workers=4, shuffle=False)
        
        model = load_neurogpt_eval(ckpt_path, DEVICE)
        y_true, y_pred_orig = [], []
        p_pert = {roi: {n: [] for n in NOISE_LEVELS} for roi in REGIONS}

        with torch.no_grad():
            for batch in tqdm(loader, desc=f"Fold {test_sub}"):
                batch = {k: v.to(DEVICE) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
                inputs = batch['inputs']
                labels = batch['labels'].cpu().numpy().flatten()
                y_true.extend(labels)

                # Baseline Inference
                logits_o = extract_logits(model(batch))
                preds_o = torch.argmax(logits_o, dim=-1)
                y_pred_orig.extend(preds_o.cpu().numpy().flatten())

                # Perturbation Inference (Central Region)
                for roi in REGIONS:
                    target_idx = get_roi_indices(roi, STANDARD_CH_NAMES)
                    for n in NOISE_LEVELS:
                        batch_p = batch.copy()
                        batch_p['inputs'] = apply_spatial_noise(inputs, target_idx, noise_lambda=n)
                        
                        logits_p = extract_logits(model(batch_p))
                        preds_p = torch.argmax(logits_p, dim=-1)
                        p_pert[roi][n].extend(preds_p.cpu().numpy().flatten())

        yt = np.array(y_true)
        po = np.array(y_pred_orig)
        
        # Track metrics per fold
        res_orig["bacc"].append(balanced_accuracy_score(yt, po) * 100)
        res_orig["f1"].append(f1_score(yt, po, average='macro') * 100)
        res_orig["kappa"].append(cohen_kappa_score(yt, po))

        for roi in REGIONS:
            for n in NOISE_LEVELS:
                pp = np.array(p_pert[roi][n])
                res_pert[roi][n]["bacc"].append(balanced_accuracy_score(yt, pp) * 100)
                res_pert[roi][n]["f1"].append(f1_score(yt, pp, average='macro') * 100)
                res_pert[roi][n]["kappa"].append(cohen_kappa_score(yt, pp))

        del model; torch.cuda.empty_cache()

    # Final Aggregated Reporting
    print("\n" + "="*85 + "\nNEUROGPT BCI IV 2b SPATIAL PERTURBATION SUMMARY (9 FOLDS)\n" + "="*85)
    
    orig_b = res_orig["bacc"]
    orig_f = res_orig["f1"]
    orig_k = res_orig["kappa"]
    print(f"\n[BASELINE SUMMARY]")
    print(f"  BACC : {np.mean(orig_b):.2f} ± {np.std(orig_b):.2f}%")
    print(f"  F1   : {np.mean(orig_f):.2f} ± {np.std(orig_f):.2f}%")
    print(f"  Kappa: {np.mean(orig_k):.3f} ± {np.std(orig_k):.3f}")

    for roi in REGIONS:
        print(f"\n[REGION: {roi}]")
        print(f"{'Noise':<8} | {'BACC':<15} | {'F1':<15} | {'Kappa':<10}")
        print("-" * 55)
        for n in NOISE_LEVELS:
            b = res_pert[roi][n]["bacc"]
            f = res_pert[roi][n]["f1"]
            k = res_pert[roi][n]["kappa"]
            print(f"{n:<8} | {np.mean(b):.2f}±{np.std(b):.2f} | {np.mean(f):.2f}±{np.std(f):.2f} | {np.mean(k):.3f}±{np.std(k):.3f}")

if __name__ == "__main__":
    run_analysis()