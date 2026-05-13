#!/usr/bin/env python3

import torch
import os
import argparse
import json
import gc
from datetime import datetime
import numpy as np
from numpy import random
import pandas as pd
import sys
from typing import Dict

# Ensure imports work from source directory
script_path = os.path.dirname(os.path.realpath(__file__))
sys.path.insert(0, os.path.join(script_path, '../'))

from batcher.downstream_dataset import TUABDataset
from encoder.conformer_braindecode import EEGConformer
from torch import manual_seed
from utils import cv_split_bci, read_threshold_sub
from batcher.base1 import EEGDataset
from decoder.make_decoder import make_decoder
from embedder.make import make_embedder
from trainer.make import make_trainer
from trainer.base import Trainer
from decoder.unembedder import make_unembedder

# Environment settings for stability
os.environ["WANDB_DISABLED"] = "true"
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

def train(config: Dict = None) -> Trainer:
    """Model training according to config with optimized memory hygiene."""
    
    if config is None:
        config = get_config()

    if config['do_train']:
        os.makedirs(config["log_dir"], exist_ok=True)
        resume_path = str(config["resume_from"]) if config["resume_from"] is not None else None
        
        if resume_path is not None:
            config_filepath = os.path.join(resume_path, 'train_config.json')
            if os.path.isfile(config_filepath):
                with open(config_filepath, 'r') as f:
                    config = json.load(f)
            
            checkpoints = [
                int(p.split('checkpoint-')[1])
                for p in os.listdir(resume_path)
                if 'checkpoint-' in p and os.path.isdir(os.path.join(resume_path, p))
            ]
            if checkpoints:
                last_checkpoint = max(checkpoints)
                config["resume_from"] = os.path.join(resume_path, f'checkpoint-{last_checkpoint}')
        else:
            config_filepath = os.path.join(config["log_dir"], 'train_config.json')
            with open(config_filepath, 'w') as f:
                json.dump(config, f, indent=2)
            config["resume_from"] = None

    # Dataset path setup
    files_path = "/homes/xw2336/xw2336B/processed"
    train_path = os.path.join(files_path, 'train/')
    val_path = os.path.join(files_path, 'val/')
    test_path = os.path.join(files_path, 'test/')

    train_files = [f for f in os.listdir(train_path) if f.endswith(".pkl")]
    val_files = [f for f in os.listdir(val_path) if f.endswith(".pkl")]
    test_files = [f for f in os.listdir(test_path) if f.endswith(".pkl")]

    # Dataset instantiation
    train_dataset = TUABDataset(
        root=train_path, filenames=train_files, 
        sample_keys=['inputs', 'attention_mask', 'labels'],
        chunk_len=config["chunk_len"], num_chunks=config["num_chunks"], 
        ovlp=config["chunk_ovlp"], gpt_only=not config["use_encoder"]
    )
    
    validation_dataset = TUABDataset(
        root=val_path, filenames=val_files, 
        sample_keys=['inputs', 'attention_mask', 'labels'],
        chunk_len=config["chunk_len"], num_chunks=config["num_chunks"], 
        ovlp=config["chunk_ovlp"], gpt_only=not config["use_encoder"]
    )

    test_dataset = None
    if config["training_style"] == "decoding":
        test_dataset = TUABDataset(
            root=test_path, filenames=test_files, 
            sample_keys=['inputs', 'attention_mask', 'labels'],
            chunk_len=config["chunk_len"], num_chunks=config["num_chunks"], 
            ovlp=config["chunk_ovlp"], gpt_only=not config["use_encoder"]
        )

    def model_init(params: Dict = None):
        model_config = dict(config)
        if params is not None:
            model_config |= params
        return make_model(model_config)

    # Memory clear before starting the trainer
    gc.collect()
    torch.cuda.empty_cache()

    trainer = make_trainer(
        model_init=model_init,
        training_style=config["training_style"],
        run_name=config["run_name"],
        output_dir=config["log_dir"],
        train_dataset=train_dataset,
        validation_dataset=validation_dataset,
        per_device_train_batch_size=config["per_device_training_batch_size"],
        per_device_eval_batch_size=config["per_device_validation_batch_size"],
        dataloader_num_workers=min(config["num_workers"], 4), 
        optim=config["optim"],
        num_train_epochs=20,
        learning_rate=config["learning_rate"],
        evaluation_strategy='epoch',
        save_strategy="epoch",
        weight_decay=config["weight_decay"],
        adam_beta1=config["adam_beta_1"],
        adam_beta2=config["adam_beta_2"],
        adam_epsilon=config["adam_epsilon"],
        max_grad_norm=config["max_grad_norm"],
        lr_scheduler_type=config["lr_scheduler"],
        warmup_ratio=config["warmup_ratio"],
        max_steps=config["training_steps"],
        save_steps=config["training_steps"] * 2 if config["training_style"] == "decoding" else config["log_every_n_steps"],
        logging_steps=config["log_every_n_steps"],
        eval_steps=config["eval_every_n_steps"],
        seed=config["seed"] if config['set_seed'] else np.random.choice(range(1, 100000)),
        fp16=config["fp16"],
        deepspeed=config["deepspeed"] if config["deepspeed"] != "none" else None,
    )

    if config['do_train']:
        try:
            trainer.train(resume_from_checkpoint=config["resume_from"])
        finally:
            gc.collect()
            torch.cuda.empty_cache()

        save_path = os.path.join(config["log_dir"], 'model_final')
        trainer.save_model(save_path)

        # Optimization: Move state dict to CPU before saving
        final_weights = {k: v.cpu() for k, v in trainer.model.state_dict().items()}
        torch.save(final_weights, os.path.join(save_path, 'full_weights.pth'))
        print(f"✅ Weights saved to {save_path}/full_weights.pth")

    if test_dataset is not None:
        test_prediction = trainer.predict(test_dataset)
        metrics_df = pd.DataFrame([test_prediction.metrics])
        metrics_df.to_csv(os.path.join(config["log_dir"], 'test_metrics.csv'), index=False)
        
        np.save(os.path.join(config["log_dir"], 'test_predictions.npy'), test_prediction.predictions)
        np.save(os.path.join(config["log_dir"], 'test_label_ids.npy'), test_prediction.label_ids)

    return trainer

def make_model(model_config: Dict = None):
    """Builds the full BCI Model from configuration."""
    if model_config["use_encoder"]:
        encoder = EEGConformer(
            n_outputs=model_config["num_decoding_classes"], n_chans=22, 
            n_times=model_config['chunk_len'], ch_pos=None, 
            is_decoding_mode=model_config["ft_only_encoder"]
        )
        model_config["parcellation_dim"] = (
            (model_config['chunk_len'] - model_config['filter_time_length'] + 1 - model_config['pool_time_length']) 
            // model_config['stride_avg_pool'] + 1
        ) * model_config['n_filters_time']
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

    unembedder = None
    if model_config["embedding_dim"] != model_config["parcellation_dim"]:
        unembedder = make_unembedder(
            embed_dim=model_config["embedding_dim"],
            num_hidden_layers=model_config["num_hidden_layers_unembedding_model"],
            out_dim=model_config["parcellation_dim"],
            dropout=model_config["dropout"],
        )

    from model import Model
    model = Model(encoder=encoder, embedder=embedder, decoder=decoder, unembedder=unembedder)

    if model_config["ft_only_encoder"]:
        model.switch_ft_mode(ft_encoder_only=True)

    if model_config["training_style"] == 'decoding':
        model.switch_decoding_mode(is_decoding_mode=True, num_decoding_classes=model_config["num_decoding_classes"])

    if model_config["pretrained_model"] is not None and model_config["pretrained_model"] != "none":
        full_pth_path = os.path.join(model_config["pretrained_model"], 'full_weights.pth')
        if os.path.exists(full_pth_path):
            model.load_state_dict(torch.load(full_pth_path, map_location='cpu'))
        elif model_config["pretrained_model"].endswith('.pth'):
            model.load_state_dict(torch.load(model_config["pretrained_model"], map_location='cpu'))
        else:
            model.from_pretrained(model_config["pretrained_model"])

    # Handle freezing
    if model_config["freeze_embedder"]:
        for p in model.embedder.parameters(): p.requires_grad = False
    if model_config["freeze_decoder"]:
        for p in model.decoder.parameters(): p.requires_grad = False
    if model_config["freeze_encoder"] and model.encoder is not None:
        for name, p in model.encoder.named_parameters():
            if 'fc.' not in name and 'final_layer' not in name: p.requires_grad = False
    if model_config["freeze_unembedder"] and unembedder is not None:
        for p in model.unembedder.parameters(): p.requires_grad = False

    return model

def get_config(args: argparse.Namespace = None) -> Dict:
    if args is None:
        args = get_args().parse_args()

    if args.smoke_test == "True":
        args.per_device_training_batch_size = 2
        args.per_device_validation_batch_size = 2
        args.training_steps = 2
        args.log_every_n_steps = 1

    if args.num_attention_heads == -1:
        args.num_attention_heads = args.embedding_dim // 64

    if args.run_name == 'none':
        args.run_name = f"{args.architecture}_L{args.num_hidden_layers}_H{args.num_attention_heads}_{datetime.now().strftime('%Y-%m-%d_%H')}"

    config = vars(args)
    for arg in config:
        if config[arg] == 'True': config[arg] = True
        elif config[arg] == 'False': config[arg] = False
        elif config[arg] == 'none': config[arg] = None

    return config

def get_args() -> argparse.ArgumentParser:
    """Get command line arguments"""
    parser = argparse.ArgumentParser(description='run model training')

    # Data pipeline settings:
    parser.add_argument('--train-data-path', default='', type=str)
    parser.add_argument('--dst-data-path', default="../../bci2a_egg_npz/", type=str)
    parser.add_argument('--parcellation-dim', default=1024, type=int)
    parser.add_argument('--pretrained-model', type=str, default='none')

    # Embedder settings:    
    parser.add_argument('--embedding-dim', default=1024, type=int)
    parser.add_argument('--num-hidden-layers-embedding-model', default=1, type=int)
    parser.add_argument('--freeze-embedder', default='False', choices=('True', 'False'), type=str)

    # UnEmbedder settings:
    parser.add_argument('--num-hidden-layers-unembedding-model', default=1, type=int)
    parser.add_argument('--freeze-unembedder', default='False', choices=('True', 'False'), type=str)

    # Decoder settings:
    parser.add_argument('--architecture', default='GPT', choices=('GPT', 'PretrainedGPT2'), type=str)
    parser.add_argument('--num-hidden-layers', default=4, type=int)
    parser.add_argument('--num-attention-heads', default=-1, type=int)
    parser.add_argument('--intermediate-dim-factor', default=4, type=int)
    parser.add_argument('--hidden-activation', default='gelu_new', type=str)
    parser.add_argument('--freeze-decoder', default='False', choices=('True', 'False'), type=str)
    parser.add_argument('--freeze-decoder-without-pooler-heads', default='False', choices=('True', 'False'), type=str)

    # Trainer settings:
    parser.add_argument('--resume-from', type=str, default='none')
    parser.add_argument('--training-style', default='CSM_causal', choices=('CSM', 'CSM_causal', 'decoding'), type=str)
    parser.add_argument('--decoding-target', default='none', type=str)
    parser.add_argument('--num-decoding-classes', default=4, type=int)
    parser.add_argument('--training-steps', default=60000, type=int)
    parser.add_argument('--validation-steps', default=1000, type=int)
    parser.add_argument('--test-steps', default=1000, type=int)
    parser.add_argument('--per-device-training-batch-size', default=16, type=int)
    parser.add_argument('--per-device-validation-batch-size', default=16, type=int)
    
    # MISSING OPTIMIZER KEY RESTORED HERE:
    parser.add_argument('--optim', default='adamw_hf', type=str)
    
    parser.add_argument('--learning-rate', default=1e-4, type=float)
    parser.add_argument('--warmup-ratio', default=0.01, type=float)
    parser.add_argument('--weight-decay', default=0.1, type=float)
    parser.add_argument('--adam-beta-1', default=0.9, type=float)
    parser.add_argument('--adam-beta-2', default=0.999, type=float)
    parser.add_argument('--adam-epsilon', default=1e-8, type=float)
    parser.add_argument('--max-grad-norm', default=1.0, type=float)
    parser.add_argument('--lr-scheduler', default='linear', type=str)
    parser.add_argument('--dropout', default=0.1, type=float)
    
    # Logging settings:
    parser.add_argument('--log-dir', default='results/models/upstream', type=str)
    parser.add_argument('--log-every-n-steps', default=1000, type=int)
    parser.add_argument('--run-name', type=str, default='none')
    parser.add_argument('--wandb-mode', default='disabled', type=str)
    parser.add_argument('--wandb-project-name', default='learning-from-brains', type=str)

    # Other settings:
    parser.add_argument('--seed', default=1234, type=int)
    parser.add_argument('--set-seed', default='True', choices=('True', 'False'), type=str)
    parser.add_argument('--fp16', default='True', choices=('True', 'False'), type=str)
    parser.add_argument('--deepspeed', default="none", type=str)
    
    # MISSING LOCAL_RANK RESTORED HERE (Crucial for torchrun):
    parser.add_argument('--local_rank', default=-1, type=int)
    
    parser.add_argument('--num-workers', default=8, type=int)
    parser.add_argument('--plot-model-graph', default="False", choices=('True', 'False'), type=str)
    parser.add_argument('--smoke-test', default="False", choices=("True", "False"), type=str)
    parser.add_argument('--bold-dummy-mode', default='False', choices=('True', 'False'), type=str)
    parser.add_argument('--do-train', default='True', choices=('True', 'False'), type=str)
    parser.add_argument('--n-positions', default=512, type=int)
    
    # EEG settings
    parser.add_argument('--chunk_len', default=500, type=int)
    parser.add_argument('--num_chunks', default=8, type=int)
    parser.add_argument('--chunk_ovlp', default=50, type=int)
    parser.add_argument('--sampling_rate', default=250, type=int)
    parser.add_argument('--fold_i', default=0, type=int)
    parser.add_argument('--use-encoder', default='True', choices=('True', 'False'), type=str)
    parser.add_argument('--do-normalization', default='True', choices=('True', 'False'), type=str)
    parser.add_argument('--filter-time-length', default=25, type=int)
    parser.add_argument('--pool-time-length', default=75, type=int)
    parser.add_argument('--stride-avg-pool', default=15, type=int)
    parser.add_argument('--n-filters-time', default=40, type=int)
    parser.add_argument('--num-encoder-layers', default=6, type=int)
    parser.add_argument('--eval_every_n_steps', default=200, type=int)
    parser.add_argument('--freeze-encoder', default='False', choices=('True', 'False'), type=str)
    parser.add_argument('--ft-only-encoder', default='False', choices=('True', 'False'), type=str)

    return parser

if __name__ == '__main__':
    train()
