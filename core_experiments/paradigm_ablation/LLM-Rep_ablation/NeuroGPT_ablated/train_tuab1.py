#!/usr/bin/env python3
import torch
import os
import argparse
import json
import time
import numpy as np
import pandas as pd
from datetime import datetime
from numpy import random
from torch import manual_seed

'''
CUDA_VISIBLE_DEVICES=2,5 nohup torchrun --nproc_per_node=2 --master_port=29533 train_tuab1.py \
--pretrained-model none \
--ft-only-encoder True \
--training-style decoding \
--run-name "NeuroGPT_Strict_LLMRep_Ablation" \
--log-dir "results_tuab_llmrep_ablation/" \
--per-device-training-batch-size 32 \
--per-device-validation-batch-size 32 > llmrep_tuab.log 2>&1 &
'''

# Set environment variables before other imports
os.environ["WANDB_DISABLED"] = "true"
import sys
from sklearn.metrics import balanced_accuracy_score, f1_score, cohen_kappa_score

# Ensure the local src directory is in the path
script_path = os.path.dirname(os.path.realpath(__file__))
sys.path.insert(0, os.path.join(script_path, '../'))

from encoder.conformer_braindecode import EEGConformer
from batcher.downstream_dataset import TUABDataset
from decoder.make_decoder import make_decoder
from embedder.make import make_embedder
from trainer.make import make_trainer
from decoder.unembedder import make_unembedder
from model import Model

def compute_metrics(eval_pred):
    """Custom metric computation for Balanced Accuracy, F1, and Kappa"""
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    return {
        "balanced_accuracy": balanced_accuracy_score(labels, predictions),
        "f1": f1_score(labels, predictions, average='weighted'),
        "kappa": cohen_kappa_score(labels, predictions)
    }

def make_model(model_config):
    """Builds the model architecture with correct dimensions"""
    if model_config["use_encoder"]:
        encoder = EEGConformer(
            n_outputs=model_config["num_decoding_classes"], 
            n_chans=22, 
            n_times=model_config['chunk_len'], 
            is_decoding_mode=model_config["ft_only_encoder"]
        )
        model_config["parcellation_dim"] = ((model_config['chunk_len'] - model_config['filter_time_length'] + 1 - model_config['pool_time_length']) // model_config['stride_avg_pool'] + 1) * model_config['n_filters_time']
    else:
        encoder = None
        model_config["parcellation_dim"] = model_config["chunk_len"] * 22

    embedder = make_embedder(
        training_style=model_config["training_style"], 
        architecture=model_config["architecture"],
        in_dim=model_config["parcellation_dim"], 
        embed_dim=model_config["embedding_dim"],
        num_hidden_layers=model_config["num_hidden_layers_embedding_model"], 
        dropout=model_config["dropout"], 
        n_positions=model_config["n_positions"]
    )
    
    decoder = make_decoder(
        architecture=model_config["architecture"], 
        num_hidden_layers=model_config["num_hidden_layers"],
        embed_dim=model_config["embedding_dim"], 
        num_attention_heads=model_config["num_attention_heads"],
        n_positions=model_config["n_positions"], 
        intermediate_dim_factor=model_config["intermediate_dim_factor"],
        hidden_activation=model_config["hidden_activation"], 
        dropout=model_config["dropout"]
    )

    unembedder = make_unembedder(
        embed_dim=model_config["embedding_dim"], 
        num_hidden_layers=model_config["num_hidden_layers_unembedding_model"],
        out_dim=model_config["parcellation_dim"], 
        dropout=model_config["dropout"]
    ) if model_config["embedding_dim"] != model_config["parcellation_dim"] else None

    model = Model(encoder=encoder, embedder=embedder, decoder=decoder, unembedder=unembedder)
    
    # =========================================================================
    # CRITICAL FIX FOR STRICT LLM-REP ABLATION
    # This activates the bypass switch in model.py. If --ft-only-encoder is True, 
    # it extracts the Conformer and ignores the GPT/Decoder (Figure 2b).
    # =========================================================================
    model.switch_ft_mode(ft_encoder_only=model_config["ft_only_encoder"])
    
    if model_config["pretrained_model"] is not None and model_config["pretrained_model"] != 'none':
        path = model_config["pretrained_model"]
        if os.path.isdir(path):
            path = os.path.join(path, 'full_weights.pth')
        print(f"Loading pretrained weights from {path}")
        model.load_state_dict(torch.load(path, map_location='cpu'), strict=False)
    
    if model_config["training_style"] == 'decoding':
        model.switch_decoding_mode(is_decoding_mode=True, num_decoding_classes=model_config["num_decoding_classes"])
    
    return model

def run_single_seed_training(config, seed):
    """Handles the training pipeline for one specific seed"""
    is_main_process = not torch.distributed.is_initialized() or torch.distributed.get_rank() == 0
    
    if is_main_process:
        print(f"\n" + "="*60)
        print(f"STARTING TRAINING RUN: SEED {seed}")
        print("="*60)
    
    current_config = config.copy() 
    
    random.seed(seed)
    manual_seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    
    original_log_dir = current_config["log_dir"]
    seed_log_dir = os.path.join(original_log_dir, f"seed_{seed}")
    current_config["log_dir"] = seed_log_dir
    
    if is_main_process:
        os.makedirs(seed_log_dir, exist_ok=True)

    files_path = "/homes/xw2336/xw2336B/processed"
    train_path = os.path.join(files_path, 'train/')
    val_path = os.path.join(files_path, 'val/')
    test_path = os.path.join(files_path, 'test/')

    train_files = [f for f in os.listdir(train_path) if f.endswith(".pkl")]
    val_files = [f for f in os.listdir(val_path) if f.endswith(".pkl")]
    test_files = [f for f in os.listdir(test_path) if f.endswith(".pkl")]

    dataset_kwargs = {
        "chunk_len": current_config["chunk_len"],
        "num_chunks": current_config["num_chunks"],
        "ovlp": current_config["chunk_ovlp"],
        "gpt_only": not current_config["use_encoder"],
        "sample_keys": ['inputs', 'attention_mask', 'labels']
    }

    train_dataset = TUABDataset(root=train_path, filenames=train_files, **dataset_kwargs)
    validation_dataset = TUABDataset(root=val_path, filenames=val_files, **dataset_kwargs)
    test_dataset = TUABDataset(root=test_path, filenames=test_files, **dataset_kwargs)

    def model_init(trial=None):
        return make_model(current_config)

    start_time = time.time()

    trainer = make_trainer(
        model_init=model_init,
        training_style=current_config["training_style"],
        run_name=f"{current_config['run_name']}_seed_{seed}",
        output_dir=seed_log_dir,
        train_dataset=train_dataset, 
        validation_dataset=validation_dataset,
        per_device_train_batch_size=current_config["per_device_training_batch_size"],
        per_device_eval_batch_size=current_config["per_device_validation_batch_size"],
        dataloader_num_workers=current_config["num_workers"],
        optim=current_config["optim"],
        learning_rate=current_config["learning_rate"],
        weight_decay=current_config["weight_decay"],
        num_train_epochs=current_config["num_train_epochs"], 
        eval_strategy='epoch',
        save_strategy="epoch",
        metric_for_best_model="balanced_accuracy",
        load_best_model_at_end=True,
        compute_metrics=compute_metrics,
        max_steps=current_config["training_steps"],
        logging_steps=current_config["log_every_n_steps"],
        seed=seed,
        fp16=current_config["fp16"]
    )

    trainer.train()
    training_duration = time.time() - start_time
    
    # 1. Sync all processes before saving
    if torch.distributed.is_initialized():
        torch.distributed.barrier()
        
    # 2. Only main process writes to disk
    if is_main_process:
        save_path = os.path.join(seed_log_dir, 'model_best')
        os.makedirs(save_path, exist_ok=True)
        trainer.save_model(save_path)
        torch.save(trainer.model.state_dict(), os.path.join(save_path, 'full_weights.pth'))
        print(f"Evaluating Best Model for Seed {seed} on Test Set...")

    # 3. Sync all processes before evaluating
    if torch.distributed.is_initialized():
        torch.distributed.barrier()

    # 4. ALL RANKS MUST EVALUATE to stay perfectly synced
    raw_test_results = trainer.evaluate(test_dataset)
    
    # 5. Sync after evaluation
    if torch.distributed.is_initialized():
        torch.distributed.barrier()
        
    # 6. Only Rank 0 returns the metrics to append to the DataFrame
    if is_main_process:
        test_results = {}
        for k, v in raw_test_results.items():
            new_key = k.replace("eval_", "").replace("test_", "")
            test_results[new_key] = v
            
        test_results['training_time_sec'] = training_duration
        test_results['seed'] = seed
        return test_results
    
    return None 

def get_args():
    parser = argparse.ArgumentParser(description='run model training')
    parser.add_argument('--dst-data-path', default="../../tuab_egg_npz/", type=str)
    parser.add_argument('--pretrained-model', type=str, default='none')
    parser.add_argument('--embedding-dim', default=1024, type=int)
    parser.add_argument('--num-hidden-layers-embedding-model', default=1, type=int)
    parser.add_argument('--num-hidden-layers-unembedding-model', default=1, type=int)
    parser.add_argument('--architecture', default='GPT', type=str)
    parser.add_argument('--num-hidden-layers', default=6, type=int)
    parser.add_argument('--num-attention-heads', default=16, type=int)
    parser.add_argument('--intermediate-dim-factor', default=4, type=int)
    parser.add_argument('--hidden-activation', default='gelu_new', type=str)
    parser.add_argument('--training-style', default='decoding', type=str)
    parser.add_argument('--num-decoding-classes', default=2, type=int)
    parser.add_argument('--training-steps', default=10000, type=int)
    parser.add_argument('--num_train_epochs', default=20, type=int)
    parser.add_argument('--per-device-training-batch-size', default=32, type=int)
    parser.add_argument('--per-device-validation-batch-size', default=32, type=int)
    parser.add_argument('--optim', default='adamw_hf', type=str)
    parser.add_argument('--learning-rate', default=1e-4, type=float)
    parser.add_argument('--weight-decay', default=0.1, type=float)
    parser.add_argument('--dropout', default=0.1, type=float)
    parser.add_argument('--log-dir', default='results/', type=str)
    parser.add_argument('--log-every-n-steps', default=1000, type=int)
    parser.add_argument('--run-name', default='none', type=str)
    parser.add_argument('--fp16', default='True', type=str)
    parser.add_argument('--num-workers', default=8, type=int)
    parser.add_argument('--n-positions', default=512, type=int)
    parser.add_argument('--chunk_len', default=500, type=int)
    parser.add_argument('--num_chunks', default=2, type=int)
    parser.add_argument('--chunk_ovlp', default=0, type=int)
    parser.add_argument('--use-encoder', default='True', type=str)
    parser.add_argument('--ft-only-encoder', default='False', type=str)
    parser.add_argument('--filter-time-length', default=25, type=int)
    parser.add_argument('--pool-time-length', default=75, type=int)
    parser.add_argument('--stride-avg-pool', default=15, type=int)
    parser.add_argument('--n-filters-time', default=40, type=int)
    return parser

if __name__ == '__main__':
    SEEDS = [42, 3407, 6, 16, 66]
    parser = get_args()
    args = parser.parse_args()
    config = vars(args)
    
    for arg in config:
        if config[arg] in {'True', 'False'}: config[arg] = config[arg] == 'True'
        elif config[arg] == 'none': config[arg] = None

    all_seed_results = []
    
    for s in SEEDS:
        res = run_single_seed_training(config, s)
        if res is not None:
            all_seed_results.append(res)

    if not torch.distributed.is_initialized() or torch.distributed.get_rank() == 0:
        df = pd.DataFrame(all_seed_results)
        
        b_acc_key = 'balanced_accuracy'
        f1_key = 'f1'
        kappa_key = 'kappa'
        
        if b_acc_key in df.columns:
            b_acc = df[b_acc_key].values
            f1 = df[f1_key].values
            kappa = df[kappa_key].values
            
            print("\n" + "="*50)
            print(f"FINAL ROBUSTNESS RESULTS ({len(SEEDS)} SEEDS)")
            print("="*50)
            print(f"Balanced Accuracy: {np.mean(b_acc):.4f} ± {np.std(b_acc):.4f}")
            print(f"F1 Score:          {np.mean(f1):.4f} ± {np.std(f1):.4f}")
            print(f"Cohen's Kappa:     {np.mean(kappa):.4f} ± {np.std(kappa):.4f}")
            print("="*50)
            
            os.makedirs(args.log_dir, exist_ok=True)
            summary_path = os.path.join(args.log_dir, 'final_robustness_summary.csv')
            df.to_csv(summary_path, index=False)
            print(f"Results saved to: {summary_path}")
        else:
            print("Error: Metrics not found in results. Check if compute_metrics is being called.")
            print("Found columns:", df.columns.tolist())