#!/usr/bin/env python3

"""
train.py

Training of models on given data using strict Leave-One-Subject-Out CV.
No early stopping, zero data leakage.
"""
from batcher.downstream_dataset import MotorImageryDataset, TUABDataset, BCIIV2bDataset, load_BCI4_2b, TUEVDataset
import torch
from sklearn.metrics import balanced_accuracy_score, f1_score, cohen_kappa_score
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


def compute_metrics(eval_pred):
    """Calculates balanced accuracy, F1, and Kappa for testing."""
    logits, labels = eval_pred
    
    if isinstance(logits, tuple):
        logits = logits[0]
        
    predictions = np.argmax(logits, axis=-1)
    
    bal_acc = balanced_accuracy_score(labels, predictions)
    f1 = f1_score(labels, predictions, average='macro')
    kappa = cohen_kappa_score(labels, predictions)
    
    return {
        "balanced_accuracy": bal_acc,
        "f1": f1,
        "kappa": kappa
    }


def train(config: Dict=None):
    """Model training according to config for a specific CV fold."""
    
    if config['do_train']:
        os.makedirs(config["log_dir"], exist_ok=True)
        config_filepath = os.path.join(config["log_dir"], 'train_config.json')
        
        with open(config_filepath, 'w') as f:
            json.dump(config, f, indent=2)

        config["resume_from"] = None

    assert config["training_style"] in {'CSM', 'CSM_causal', 'decoding'}, f'{config["training_style"]} is not supported.'
    assert config["architecture"] in {'GPT', 'PretrainedGPT2'}, f'{config["architecture"]} is not supported.'
    
    if config['set_seed']:
        random.seed(config["seed"])
        manual_seed(config["seed"])

    # --- DATASET SETUP FOR PURE LOSO CV ---
    if config["training_style"] == 'decoding':
        
        # 1. Load all 8 subjects for pure training (No internal validation split)
        train_dataset = BCIIV2bDataset(
            subject_ids=config['train_subs'], 
            sample_keys=['inputs', 'attention_mask', 'labels'], 
            chunk_len=config["chunk_len"], 
            num_chunks=config["num_chunks"], 
            ovlp=config["chunk_ovlp"], 
            gpt_only=not config["use_encoder"]
        )
        
        # 2. No validation dataset to ensure absolute zero leakage
        validation_dataset = None
        
        # 3. Load the 1 held-out subject strictly for the final test
        test_dataset = BCIIV2bDataset(
            subject_ids=config['test_subs'], 
            sample_keys=['inputs', 'attention_mask', 'labels'], 
            chunk_len=config["chunk_len"], 
            num_chunks=config["num_chunks"], 
            ovlp=config["chunk_ovlp"], 
            gpt_only=not config["use_encoder"]
        )
        
    else:
        train_dataset = None
        validation_dataset = None
        test_dataset = None

    def model_init(params: Dict=None):
        model_config = dict(config)
        if params is not None:
            model_config |= params
        return make_model(model_config)

    # Use a fixed epoch approach
    num_fixed_epochs = 50 

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
        num_train_epochs=num_fixed_epochs,
        learning_rate=config["learning_rate"],
        
        # Disabled evaluation during training to ensure zero interaction with test data
        evaluation_strategy='no',
        save_strategy="no", # Only save the final model to save disk space
        eval_strategy="no",
        
        weight_decay=config["weight_decay"],
        adam_beta1=config["adam_beta_1"],
        adam_beta2=config["adam_beta_2"],
        adam_epsilon=config["adam_epsilon"],
        max_grad_norm=config["max_grad_norm"],
        lr_scheduler_type=config["lr_scheduler"],
        warmup_ratio=config["warmup_ratio"],
        max_steps=config["training_steps"],
        logging_steps=config["log_every_n_steps"],
        seed=config["seed"] if config['set_seed'] else np.random.choice(range(1, 100000)),
        fp16=config["fp16"],
        deepspeed=config["deepspeed"],
        
        # Only compute metrics at the very end
        compute_metrics=compute_metrics,
        load_best_model_at_end=False # We take the final epoch's weights
    )

    if config['do_train']:
        trainer.train(resume_from_checkpoint=config["resume_from"])
        trainer.save_model(os.path.join(config["log_dir"], 'model_final'))

    # Evaluate the final model on the strictly held-out test subject
    test_metrics = None
    if test_dataset is not None:
        print("\nRunning final evaluation on held-out test subject...")
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
        # Ensures mapping to the 22-channel standard
        encoder = EEGConformer(n_outputs=model_config["num_decoding_classes"], n_chans=22, n_times=model_config['chunk_len'], ch_pos=chann_coords, is_decoding_mode=model_config["ft_only_encoder"])
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

    if model_config["embedding_dim"] != model_config["parcellation_dim"]:
        unembedder = make_unembedder(
            embed_dim=model_config["embedding_dim"],
            num_hidden_layers=model_config["num_hidden_layers_unembedding_model"],
            out_dim=model_config["parcellation_dim"],
            dropout=model_config["dropout"],
        )
    else:
        unembedder = None

    from model import Model
    model = Model(
        encoder=encoder,
        embedder=embedder,
        decoder=decoder,
        unembedder=unembedder
    )

    if model_config["ft_only_encoder"]:
        model.switch_ft_mode(ft_encoder_only=False)

    if model_config["training_style"] == 'decoding':
        model.switch_decoding_mode(
            is_decoding_mode=True,
            num_decoding_classes=model_config["num_decoding_classes"]
        )

    if model_config["pretrained_model"] is not None:
        model.from_pretrained(model_config["pretrained_model"])

    if model_config["freeze_embedder"]:
        for param in model.embedder.parameters():
            param.requires_grad = False

    if model_config["freeze_decoder"]:
        for param in model.decoder.parameters():
            param.requires_grad = False

    if model_config["freeze_encoder"]:
        for name, param in model.encoder.named_parameters():
            if 'fc.' in name or 'final_layer' in name:
                continue
            else:
                param.requires_grad = False

    if 'freeze_decoder_without_pooler_heads' in model_config and model_config["freeze_decoder_without_pooler_heads"]:
        for name, param in model.decoder.named_parameters():
            if 'pooler_layer' in name or 'decoding_head' in name or 'is_next_head' in name:
                continue
            else:
                param.requires_grad = False

    if model_config["freeze_unembedder"] and unembedder is not None:
        for param in model.unembedder.parameters():
            param.requires_grad = False

    return model


def get_config(args: argparse.Namespace=None) -> Dict:
    if args is None:
        args = get_args().parse_args()

    if args.smoke_test == "True":
        args.per_device_training_batch_size =  2
        args.per_device_validation_batch_size = 2
        args.training_steps = 2
        args.validation_steps = 2
        args.test_steps = 2
        args.log_every_n_steps = 1

    if args.num_attention_heads == -1:
        assert (args.embedding_dim%64) == 0, f'embedding-dim needs be be multiple of 64 (currently: {args.embedding_dim})' 
        args.num_attention_heads = args.embedding_dim//64

    if args.run_name == 'none':
        args.run_name = f'{args.architecture}'
        if args.architecture != 'LinearBaseline':
            if 'Pretrained' not in args.architecture:
                args.run_name += f'_lrs-{args.num_hidden_layers}'
                args.run_name += f'_hds-{args.num_attention_heads}'
            args.run_name += f'_ChunkLen-{args.chunk_len}'
            args.run_name += f'_NumChunks-{args.num_chunks}'
            args.run_name += f'_ovlp-{args.chunk_ovlp}'
        else:
            args.run_name += f'_train-{args.training_style}'

        args.run_name += f"_{datetime.now().strftime('%Y-%m-%d_%H')}"

    if args.training_style == "decoding":
        args.run_name += '-' + str(args.fold_i)

    if args.smoke_test == "True":
        args.run_name = f'smoke-test_{args.run_name}'

    args.log_dir = os.path.join(args.log_dir, args.run_name)
    args.wandb_mode = args.wandb_mode if args.wandb_mode in {'online', 'offline'} and args.local_rank in {-1, 0} else "disabled"
    
    config = vars(args)

    for arg in config:
        if config[arg] in {'True', 'False'}:
            config[arg] = config[arg] == 'True'
        elif config[arg] == 'none':
            config[arg] = None
        elif 'subjects_per_dataset' in arg:
            config[arg] = None if config[arg] == -1 else config[arg]

    return config


def get_args() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description='run model training')
    
    parser.add_argument('--train-data-path', metavar='DIR', default='', type=str)
    parser.add_argument('--dst-data-path', metavar='DIR', default="../../bci2a_egg_npz/", type=str)
    parser.add_argument('--parcellation-dim', metavar='INT', default=1024, type=int)
    parser.add_argument('--pretrained-model', metavar='DIR', type=str, default='none')
    parser.add_argument('--embedding-dim', metavar='INT', default=1024, type=int)
    parser.add_argument('--num-hidden-layers-embedding-model', metavar='INT', default=1, type=int)
    parser.add_argument('--freeze-embedder', metavar='BOOL', default='False', choices=('True', 'False'), type=str)
    parser.add_argument('--num-hidden-layers-unembedding-model', metavar='INT', default=1, type=int)
    parser.add_argument('--freeze-unembedder', metavar='BOOL', default='False', choices=('True', 'False'), type=str)
    parser.add_argument('--architecture', metavar='STR', default='GPT', choices=('GPT', 'PretrainedGPT2'), type=str)
    parser.add_argument('--num-hidden-layers', metavar='INT', default=4, type=int)
    parser.add_argument('--num-attention-heads', metavar='INT', default=-1, type=int)
    parser.add_argument('--intermediate-dim-factor', metavar='INT', default=4, type=int)
    parser.add_argument('--hidden-activation', metavar='STR', default='gelu_new', choices=('gelu', 'gelu_new', 'relu', 'silu'), type=str)
    parser.add_argument('--freeze-decoder', metavar='BOOL', default='False', choices=('True', 'False'), type=str)
    parser.add_argument('--freeze-decoder-without-pooler-heads', metavar='BOOL', default='False', choices=('True', 'False'), type=str)
    parser.add_argument('--resume-from', metavar='DIR', type=str, default='none')
    parser.add_argument('--training-style', metavar='STR', default='CSM_causal', choices=('CSM', 'CSM_causal', 'decoding'), type=str)
    parser.add_argument('--decoding-target', metavar='STR', default='none', type=str)
    parser.add_argument('--num-decoding-classes', metavar='INT', default=2, type=int)
    parser.add_argument('--training-steps', metavar='INT', default=60000, type=int)
    parser.add_argument('--validation-steps', metavar='INT', default=1000, type=int)
    parser.add_argument('--test-steps', metavar='INT', default=1000, type=int)
    parser.add_argument('--per-device-training-batch-size', metavar='INT', default=16, type=int)
    parser.add_argument('--per-device-validation-batch-size', metavar='INT', default=16, type=int)
    parser.add_argument('--optim', metavar='STR', default='adamw_hf', type=str)
    parser.add_argument('--learning-rate', metavar='FLOAT', default=1e-4, type=float)
    parser.add_argument('--warmup-ratio', metavar='FLOAT', default=0.01, type=float)
    parser.add_argument('--weight-decay', metavar='FLOAT', default=0.1, type=float)
    parser.add_argument('--adam-beta-1', metavar='FLOAT', default=0.9, type=float)
    parser.add_argument('--adam-beta-2', metavar='FLOAT', default=0.999, type=float)
    parser.add_argument('--adam-epsilon', metavar='FLOAT', default=1e-8, type=float)
    parser.add_argument('--max-grad-norm', metavar='FLOAT', default=1.0, type=float)
    parser.add_argument('--lr-scheduler', metavar='STR', default='linear', choices=('linear', 'constant_with_warmup', 'none'), type=str)
    parser.add_argument('--dropout', metavar='FLOAT', default=0.1, type=float)
    parser.add_argument('--log-dir', metavar='DIR', type=str, default='results/models/upstream')
    parser.add_argument('--log-every-n-steps', metavar='INT', default=1000, type=int)
    parser.add_argument('--run-name', metavar='STR', type=str, default='none')
    parser.add_argument('--wandb-mode', metavar='STR', choices=('online', 'offline', 'disabled'), default='disabled')
    parser.add_argument('--wandb-project-name', metavar='STR', type=str, default='learning-from-brains')
    parser.add_argument('--seed', metavar='INT', default=1234, type=int)
    parser.add_argument('--set-seed', metavar='BOOL', choices=('True', 'False'), default='True', type=str)
    parser.add_argument('--fp16', metavar='BOOL', choices=('True', 'False'), default='True')
    parser.add_argument('--deepspeed', metavar='DIR', default="none", type=str)
    parser.add_argument('--local_rank', metavar='INT', default=-1, type=int)
    parser.add_argument('--num-workers', metavar='INT', default=8, type=int)
    parser.add_argument('--plot-model-graph', metavar='BOOL', default="False", type=str, choices=('True', 'False'))
    parser.add_argument('--smoke-test', metavar='BOOL', default="False", type=str, choices=("True", "False"))
    parser.add_argument('--bold-dummy-mode', metavar='BOOL', default='False', type=str, choices=('True', 'False'))
    parser.add_argument('--do-train', metavar='BOOL', default='True', type=str, choices=('True', 'False'))
    parser.add_argument('--n-positions', metavar='INT', default=512, type=int)
    
    ## EEG settings
    parser.add_argument('--chunk_len', default=500, type=int)
    parser.add_argument('--num_chunks', default=8, type=int)
    parser.add_argument('--chunk_ovlp', default=50, type=int)
    parser.add_argument('--sampling_rate', default=250, type=int)
    parser.add_argument('--fold_i', default=0, type=int)
    parser.add_argument('--use-encoder', metavar='BOOL', default='True', type=str, choices=('True', 'False'))
    parser.add_argument('--do-normalization', metavar='BOOL', default='True', type=str, choices=('True', 'False'))
    parser.add_argument('--filter-time-length', metavar='INT', default=25, type=int)
    parser.add_argument('--pool-time-length', metavar='INT', default=75, type=int)
    parser.add_argument('--stride-avg-pool', metavar='INT', default=15, type=int)
    parser.add_argument('--n-filters-time', metavar='INT', default=40, type=int)
    parser.add_argument('--num-encoder-layers', metavar='INT', default=6, type=int)
    parser.add_argument('--eval_every_n_steps', default=200, type=int)
    parser.add_argument('--freeze-encoder', metavar='BOOL', default='False', choices=('True', 'False'), type=str)
    parser.add_argument('--ft-only-encoder', metavar='BOOL', default='False', choices=('True', 'False'), type=str)

    return parser


if __name__ == '__main__':
    # Parse base configurations
    base_config = get_config()
    
    # Define the 9 subjects in BCI-IV 2b
    all_subjects = ['S01', 'S02', 'S03', 'S04', 'S05', 'S06', 'S07', 'S08', 'S09']
    
    all_fold_metrics = []

    # Leave-One-Subject-Out (LOSO) CV Loop
    for i, test_sub in enumerate(all_subjects):
        # 1 subject for test, 8 for pure training
        train_subs = [sub for sub in all_subjects if sub != test_sub]
        
        print("\n" + "="*60)
        print(f"🚀 STRICT LOSO CV: FOLD {i+1} / {len(all_subjects)}")
        print(f"Training Subjects: {train_subs}")
        print(f"Test Subject:      {test_sub}")
        print("="*60 + "\n")
        
        # Create a fold-specific configuration
        fold_config = base_config.copy()
        fold_config['train_subs'] = train_subs
        fold_config['test_subs'] = [test_sub]
        
        # Isolate logging directory to prevent checkpoint overlap
        fold_config['log_dir'] = os.path.join(base_config['log_dir'], f"fold_{test_sub}")
        
        # Run training and return test metrics
        fold_metrics = train(fold_config)
        
        if fold_metrics is not None:
            # Print fold results
            print(f"\n✅ RESULTS FOR TEST SUBJECT {test_sub}:")
            print(f"   Balanced Accuracy: {fold_metrics.get('test_balanced_accuracy', 0):.4f}")
            print(f"   F1 Score (Macro):  {fold_metrics.get('test_f1', 0):.4f}")
            print(f"   Kappa:             {fold_metrics.get('test_kappa', 0):.4f}\n")
            
            all_fold_metrics.append(fold_metrics)

        import gc
        gc.collect()
        torch.cuda.empty_cache()

    # --- CALCULATE & PRINT FINAL CV AVERAGES ---
    if all_fold_metrics:
        print("\n" + "🌟"*30)
        print("FINAL STRICT LOSO CV AVERAGES (9 FOLDS)")
        print("🌟"*30)
        
        avg_metrics = {}
        for key in all_fold_metrics[0].keys():
            if isinstance(all_fold_metrics[0][key], (int, float)):
                avg_metrics[key] = np.mean([m[key] for m in all_fold_metrics if key in m])
                
        print(f"Average Balanced Accuracy: {avg_metrics.get('test_balanced_accuracy', 0):.4f}")
        print(f"Average F1 Score (Macro):  {avg_metrics.get('test_f1', 0):.4f}")
        print(f"Average Cohen's Kappa:     {avg_metrics.get('test_kappa', 0):.4f}")
        print("🌟"*30 + "\n")