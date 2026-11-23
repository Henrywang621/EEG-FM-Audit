#!/usr/bin/env python3

"""
train_seed.py

Training of NeuroGPT on SEED-IV using strict Leave-One-Subject-Out CV.
Saves the absolute full PyTorch state_dict for guaranteed reproducibility.
"""
from batcher.downstream_dataset import SeedIVDataset
import torch
from sklearn.metrics import balanced_accuracy_score, f1_score, cohen_kappa_score, roc_auc_score
import os
import argparse
from typing import Dict
import json
from datetime import datetime
from numpy import random
import pandas as pd
import numpy as np
from encoder.conformer_braindecode import EEGConformer
from torch import manual_seed
import sys

from utils import cv_split_bci, read_threshold_sub
script_path = os.path.dirname(os.path.realpath(__file__))
sys.path.insert(0, os.path.join(script_path, '../'))

from batcher.base2 import EEGDataset
from decoder.make_decoder import make_decoder
from embedder.make import make_embedder
from trainer.make import make_trainer
from trainer.base import Trainer
from decoder.unembedder import make_unembedder

os.environ["WANDB_DISABLED"] = "true"
# CUDA_VISIBLE_DEVICES=1 nohup /homes/xw2336/miniconda3/envs/EEG26pytorch/bin/python -u pretrain_seed.py   --num-epochs 100   --training-style CSM   --use-encoder True   --log-dir ./log/seed_iv_pretrain_scratch   --run-name neurogpt_seed_pretraining_scratch   > seed_iv_pretrain_scratch.log 2>&1 & disown

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    
    if isinstance(logits, tuple):
        logits = logits[0]
        
    predictions = np.argmax(logits, axis=-1)
    
    acc = np.mean(predictions == labels)
    bal_acc = balanced_accuracy_score(labels, predictions)
    f1 = f1_score(labels, predictions, average='macro')
    kappa = cohen_kappa_score(labels, predictions)
    
    return {
        "accuracy": acc,
        "balanced_accuracy": bal_acc,
        "f1": f1,
        "kappa": kappa
    }

def train(config: Dict=None):
    if config['do_train']:
        os.makedirs(config["log_dir"], exist_ok=True)
        config_filepath = os.path.join(config["log_dir"], 'train_config.json')
        
        with open(config_filepath, 'w') as f:
            json.dump(config, f, indent=2)

        config["resume_from"] = None

    if config['set_seed']:
        random.seed(config["seed"])
        manual_seed(config["seed"])
        torch.cuda.manual_seed_all(config["seed"])

    # --- DATASET SETUP FOR SEED-IV LOSO CV ---
    train_dataset = SeedIVDataset(
        subject_ids=config['train_subs'], 
        sample_keys=['inputs', 'attention_mask', 'labels'], 
        chunk_len=config["chunk_len"], 
        num_chunks=config["num_chunks"], 
        ovlp=config["chunk_ovlp"], 
        root_path=config["train_data_path"],
        gpt_only=not config["use_encoder"]
    )
    
    validation_dataset = None
    
    test_dataset = SeedIVDataset(
        subject_ids=config['test_subs'], 
        sample_keys=['inputs', 'attention_mask', 'labels'], 
        chunk_len=config["chunk_len"], 
        num_chunks=config["num_chunks"], 
        ovlp=config["chunk_ovlp"], 
        root_path=config["train_data_path"],
        gpt_only=not config["use_encoder"]
    )

    def model_init(params: Dict=None):
        model_config = dict(config)
        if params is not None:
            model_config |= params
        return make_model(model_config)

    trainer = make_trainer(
        model_init=model_init,
        training_style=config["training_style"],
        run_name=config["run_name"],
        output_dir=config["log_dir"],
        train_dataset=train_dataset,
        validation_dataset=validation_dataset,
        per_device_train_batch_size=config["per_device_training_batch_size"],
        per_device_eval_batch_size=config["per_device_validation_batch_size"],
        dataloader_num_workers=config["num_workers"],
        optim=config["optim"],
        num_train_epochs=config["num_epochs"],
        learning_rate=config["learning_rate"],
        evaluation_strategy='no',
        save_strategy="no", 
        eval_strategy="no",
        weight_decay=config["weight_decay"],
        adam_beta1=config["adam_beta_1"],
        adam_beta2=config["adam_beta_2"],
        adam_epsilon=config["adam_epsilon"],
        max_grad_norm=config["max_grad_norm"],
        lr_scheduler_type=config["lr_scheduler"],
        warmup_ratio=config["warmup_ratio"],
        logging_steps=config["log_every_n_steps"],
        seed=config["seed"] if config['set_seed'] else np.random.choice(range(1, 100000)),
        fp16=config["fp16"],
        deepspeed=config["deepspeed"],
        compute_metrics=compute_metrics,
        load_best_model_at_end=False 
    )

    if config['do_train']:
        trainer.train(resume_from_checkpoint=config["resume_from"])
        
        # Explicit Full State Dict Save for Absolute Reproducibility
        print(f"💾 Saving complete model state dict to {config['log_dir']} ...")
        torch.save(trainer.model.state_dict(), os.path.join(config["log_dir"], 'pytorch_model_full.pth'))
        trainer.save_model(os.path.join(config["log_dir"], 'hf_model_final'))

    # Evaluate Final Model
    print(f"\nRunning final evaluation on held-out subject: {config['test_subs'][0]} ...")
    test_prediction = trainer.predict(test_dataset)
    test_metrics = test_prediction.metrics
    
    metrics_df = pd.DataFrame([test_metrics])
    metrics_csv_path = os.path.join(config["log_dir"], 'test_metrics.csv')
    metrics_df.to_csv(metrics_csv_path, index=False)
    
    np.save(os.path.join(config["log_dir"], 'test_predictions.npy'), test_prediction.predictions)
    np.save(os.path.join(config["log_dir"], 'test_label_ids.npy'), test_prediction.label_ids)

    return test_metrics


def make_model(model_config: Dict=None):
    if model_config["use_encoder"] == True:
        chann_coords = None
        encoder = EEGConformer(n_outputs=model_config["num_decoding_classes"], n_chans=22, n_times=model_config['chunk_len'], ch_pos=chann_coords, is_decoding_mode=model_config["ft_only_encoder"])
        model_config["parcellation_dim"] = ((model_config['chunk_len'] - model_config['filter_time_length'] + 1 - model_config['pool_time_length']) // model_config['stride_avg_pool'] + 1) * model_config['n_filters_time']
    else:
        encoder = None
        model_config["parcellation_dim"] = model_config["chunk_len"] * 22

    embedder = make_embedder(
        training_style=model_config["training_style"], architecture=model_config["architecture"],
        in_dim=model_config["parcellation_dim"], embed_dim=model_config["embedding_dim"],
        num_hidden_layers=model_config["num_hidden_layers_embedding_model"], dropout=model_config["dropout"], n_positions=model_config["n_positions"]
    )
    decoder = make_decoder(
        architecture=model_config["architecture"], num_hidden_layers=model_config["num_hidden_layers"],
        embed_dim=model_config["embedding_dim"], num_attention_heads=model_config["num_attention_heads"],
        n_positions=model_config["n_positions"], intermediate_dim_factor=model_config["intermediate_dim_factor"],
        hidden_activation=model_config["hidden_activation"], dropout=model_config["dropout"]
    )

    if model_config["embedding_dim"] != model_config["parcellation_dim"]:
        unembedder = make_unembedder(embed_dim=model_config["embedding_dim"], num_hidden_layers=model_config["num_hidden_layers_unembedding_model"], out_dim=model_config["parcellation_dim"], dropout=model_config["dropout"])
    else:
        unembedder = None

    from model import Model
    model = Model(encoder=encoder, embedder=embedder, decoder=decoder, unembedder=unembedder)

    if model_config["ft_only_encoder"]: model.switch_ft_mode(ft_encoder_only=True)
    if model_config["training_style"] == 'decoding': model.switch_decoding_mode(is_decoding_mode=True, num_decoding_classes=model_config["num_decoding_classes"])
    if model_config["pretrained_model"] is not None and model_config["pretrained_model"] != 'none':
        pretrained_path = model_config["pretrained_model"]
        print(f"Loading full model weights from: {pretrained_path}")
        
        if str(pretrained_path).endswith('.safetensors'):
            from safetensors.torch import load_model
            load_model(model, pretrained_path, strict=False)
        else:
            model.load_state_dict(torch.load(pretrained_path, map_location='cpu'), strict=False)

    # Freeze Logic (Omitted for brevity, but same as train_2b.py)
    if model_config["freeze_embedder"]:
        for param in model.embedder.parameters(): param.requires_grad = False
    if model_config["freeze_decoder"]:
        for param in model.decoder.parameters(): param.requires_grad = False
    if model_config["freeze_encoder"]:
        for name, param in model.encoder.named_parameters():
            if 'fc.' not in name and 'final_layer' not in name: param.requires_grad = False

    return model

def get_config(args: argparse.Namespace=None) -> Dict:
    parser = argparse.ArgumentParser(description='Run SEED-IV Model Training')
    
    # Model & Optimization Args
    parser.add_argument('--train-data-path', type=str, default='/homes/yy2541/henry/data/SEED_IV')
    parser.add_argument('--pretrained-model', type=str, default='none')
    parser.add_argument('--architecture', type=str, default='GPT', choices=('GPT', 'PretrainedGPT2'))
    parser.add_argument('--training-style', type=str, default='decoding', choices=('CSM', 'CSM_causal', 'decoding'))
    parser.add_argument('--num-decoding-classes', type=int, default=4)
    parser.add_argument('--num-epochs', type=int, default=50)
    parser.add_argument('--per-device-training-batch-size', type=int, default=64)
    parser.add_argument('--per-device-validation-batch-size', type=int, default=64)
    parser.add_argument('--optim', type=str, default='adamw_hf')
    parser.add_argument('--learning-rate', type=float, default=1e-4)
    parser.add_argument('--weight-decay', type=float, default=0.1)
    parser.add_argument('--log-dir', type=str, default='results/models/seed_iv_downstream')
    parser.add_argument('--run-name', type=str, default='NeuroGPT_SEEDIV')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--set-seed', choices=('True', 'False'), default='True')
    
    # Architecture internals
    parser.add_argument('--embedding-dim', type=int, default=1024)
    parser.add_argument('--num-hidden-layers', type=int, default=4)
    parser.add_argument('--num-attention-heads', type=int, default=-1)
    parser.add_argument('--intermediate-dim-factor', type=int, default=4)
    parser.add_argument('--n-positions', type=int, default=512)
    parser.add_argument('--dropout', type=float, default=0.1)
    
    # EEG Specifics
    parser.add_argument('--chunk_len', type=int, default=500)
    parser.add_argument('--num_chunks', type=int, default=4)
    parser.add_argument('--chunk_ovlp', type=int, default=0)
    parser.add_argument('--use-encoder', choices=('True', 'False'), default='True')

    # EEG Conformer specific arguments
    parser.add_argument('--filter-time-length', type=int, default=25)
    parser.add_argument('--pool-time-length', type=int, default=75)
    parser.add_argument('--stride-avg-pool', type=int, default=15)
    parser.add_argument('--n-filters-time', type=int, default=40)
    parser.add_argument('--num-encoder-layers', type=int, default=6)
    
    # Boilers
    parser.add_argument('--freeze-embedder', choices=('True', 'False'), default='False')
    parser.add_argument('--freeze-decoder', choices=('True', 'False'), default='False')
    parser.add_argument('--freeze-encoder', choices=('True', 'False'), default='False')
    parser.add_argument('--ft-only-encoder', choices=('True', 'False'), default='False')
    parser.add_argument('--do-train', choices=('True', 'False'), default='True')
    parser.add_argument('--fp16', choices=('True', 'False'), default='True')
    parser.add_argument('--num-workers', type=int, default=8)
    parser.add_argument('--lr-scheduler', type=str, default='linear')
    parser.add_argument('--warmup-ratio', type=float, default=0.01)
    parser.add_argument('--max-grad-norm', type=float, default=1.0)
    parser.add_argument('--adam-beta-1', type=float, default=0.9)
    parser.add_argument('--adam-beta-2', type=float, default=0.999)
    parser.add_argument('--adam-epsilon', type=float, default=1e-8)
    parser.add_argument('--log-every-n-steps', type=int, default=100)
    parser.add_argument('--deepspeed', type=str, default="none")
    parser.add_argument('--num-hidden-layers-embedding-model', type=int, default=1)
    parser.add_argument('--num-hidden-layers-unembedding-model', type=int, default=1)
    parser.add_argument('--hidden-activation', type=str, default='gelu_new')

    args = parser.parse_args()
    
    if args.num_attention_heads == -1:
        args.num_attention_heads = args.embedding_dim // 64

    args.run_name += f"_{datetime.now().strftime('%Y-%m-%d_%H')}"
    config = vars(args)

    for arg in config:
        if config[arg] in {'True', 'False'}:
            config[arg] = config[arg] == 'True'
        elif config[arg] == 'none':
            config[arg] = None

    return config

if __name__ == '__main__':
    base_config = get_config()
    
    all_subjects = list(range(1, 16))
    all_fold_metrics = []

    for test_sub in all_subjects:
        train_subs = [sub for sub in all_subjects if sub != test_sub]
        
        print("\n" + "="*60)
        print(f"🚀 STRICT LOOCV: FOLD {test_sub} / 15")
        print(f"Training Subjects: {train_subs}")
        print(f"Test Subject:      {test_sub}")
        print("="*60 + "\n")
        
        fold_config = base_config.copy()
        fold_config['train_subs'] = train_subs
        fold_config['test_subs'] = [test_sub]
        fold_config['log_dir'] = os.path.join(base_config['log_dir'], f"fold_subject_{test_sub}")
        
        fold_metrics = train(fold_config)
        
        if fold_metrics is not None:
            print(f"\n✅ RESULTS FOR TEST SUBJECT {test_sub}:")
            print(f"   Accuracy:          {fold_metrics.get('test_accuracy', 0):.4f}")
            print(f"   Balanced Accuracy: {fold_metrics.get('test_balanced_accuracy', 0):.4f}")
            print(f"   F1 Score (Macro):  {fold_metrics.get('test_f1', 0):.4f}")
            print(f"   Kappa:             {fold_metrics.get('test_kappa', 0):.4f}\n")
            
            all_fold_metrics.append(fold_metrics)

        import gc
        gc.collect()
        torch.cuda.empty_cache()

    if all_fold_metrics:
        print("\n" + "🌟"*30)
        print("FINAL STRICT LOOCV AVERAGES (15 FOLDS)")
        print("🌟"*30)
        
        avg_metrics = {}
        for key in all_fold_metrics[0].keys():
            if isinstance(all_fold_metrics[0][key], (int, float)):
                avg_metrics[key] = np.mean([m[key] for m in all_fold_metrics if key in m])
                
        print(f"Average Accuracy:          {avg_metrics.get('test_accuracy', 0):.4f}")
        print(f"Average Balanced Accuracy: {avg_metrics.get('test_balanced_accuracy', 0):.4f}")
        print(f"Average F1 Score (Macro):  {avg_metrics.get('test_f1', 0):.4f}")
        print(f"Average Cohen's Kappa:     {avg_metrics.get('test_kappa', 0):.4f}")
        print("🌟"*30 + "\n")