# --------------------------------------------------------
# LaBraM fine-tuning and evaluation script for SEED-IV (LOOCV)
# ---------------------------------------------------------

import argparse
import datetime
import numpy as np
import time
import torch
import torch.backends.cudnn as cudnn
import json
import os
import gc

from pathlib import Path
from collections import OrderedDict
from timm.models import create_model
from timm.loss import LabelSmoothingCrossEntropy
from timm.utils import ModelEma
from optim_factory import create_optimizer, get_parameter_groups, LayerDecayValueAssigner

from engine_for_finetuning1 import train_one_epoch, evaluate
from utils import NativeScalerWithGradNormCount as NativeScaler
import utils
import modeling_finetune

'''
CUDA_VISIBLE_DEVICES=2,3,6 nohup /homes/xw2336/miniconda3/envs/EEG26pytorch/bin/python -u run_class_finetuning_seedIV.py   --output_dir ./checkpoints/retrain_reproduction_seed_iv/   --log_dir ./log/retrain_reproduction_seed_iv   --model labram_base_patch200_200   --finetune ./checkpoints/labram-base.pth   --batch_size 256   --lr 1e-5   --epochs 150   --dataset SEED_IV   --input_size 800   --abs_pos_emb   --seed 0   > seed_iv_run.log 2>&1 & disown

'''

def get_args():
    parser = argparse.ArgumentParser('LaBraM fine-tuning and evaluation script for SEED-IV', add_help=False)
    parser.add_argument('--batch_size', default=64, type=int)
    parser.add_argument('--epochs', default=50, type=int)
    parser.add_argument('--update_freq', default=1, type=int)
    parser.add_argument('--save_ckpt_freq', default=10, type=int)

    parser.add_argument('--model', default='labram_base_patch200_200', type=str, metavar='MODEL')
    parser.add_argument('--qkv_bias', action='store_true')
    parser.add_argument('--disable_qkv_bias', action='store_false', dest='qkv_bias')
    parser.set_defaults(qkv_bias=True)
    parser.add_argument('--rel_pos_bias', action='store_true')
    parser.add_argument('--disable_rel_pos_bias', action='store_false', dest='rel_pos_bias')
    parser.set_defaults(rel_pos_bias=True)
    parser.add_argument('--abs_pos_emb', action='store_true')
    parser.set_defaults(abs_pos_emb=False)
    parser.add_argument('--layer_scale_init_value', default=0.1, type=float)

    # SEED-IV sequences are 800 length (4 seconds @ 200Hz). 
    # Perfectly divides by LaBraM's 200 patch size into 4 tokens.
    parser.add_argument('--input_size', default=800, type=int, help='EEG input size') 

    parser.add_argument('--drop', type=float, default=0.1, metavar='PCT')
    parser.add_argument('--attn_drop_rate', type=float, default=0.0, metavar='PCT')
    parser.add_argument('--drop_path', type=float, default=0.1, metavar='PCT')

    parser.add_argument('--disable_eval_during_finetuning', action='store_true', default=False)
    parser.add_argument('--model_ema', action='store_true', default=False)
    parser.add_argument('--model_ema_decay', type=float, default=0.9999)
    parser.add_argument('--model_ema_force_cpu', action='store_true', default=False)

    parser.add_argument('--opt', default='adamw', type=str, metavar='OPTIMIZER')
    parser.add_argument('--opt_eps', default=1e-8, type=float, metavar='EPSILON')
    parser.add_argument('--opt_betas', default=None, type=float, nargs='+', metavar='BETA')
    parser.add_argument('--clip_grad', type=float, default=None, metavar='NORM')
    parser.add_argument('--momentum', type=float, default=0.9, metavar='M')
    parser.add_argument('--weight_decay', type=float, default=0.05)
    parser.add_argument('--weight_decay_end', type=float, default=None)

    parser.add_argument('--lr', type=float, default=5e-4, metavar='LR')
    parser.add_argument('--layer_decay', type=float, default=0.9)
    parser.add_argument('--warmup_lr', type=float, default=1e-6, metavar='LR')
    parser.add_argument('--min_lr', type=float, default=1e-6, metavar='LR')
    parser.add_argument('--warmup_epochs', type=int, default=5, metavar='N')
    parser.add_argument('--warmup_steps', type=int, default=-1, metavar='N')

    parser.add_argument('--smoothing', type=float, default=0.1)

    parser.add_argument('--finetune', default='')
    parser.add_argument('--model_key', default='model|module', type=str)
    parser.add_argument('--model_prefix', default='', type=str)
    parser.add_argument('--model_filter_name', default='gzp', type=str)
    parser.add_argument('--init_scale', default=0.001, type=float)
    parser.add_argument('--use_mean_pooling', action='store_true')
    parser.set_defaults(use_mean_pooling=True)
    parser.add_argument('--use_cls', action='store_false', dest='use_mean_pooling')
    parser.add_argument('--disable_weight_decay_on_rel_pos_bias', action='store_true', default=False)

    parser.add_argument('--nb_classes', default=4, type=int)

    parser.add_argument('--output_dir', default='')
    parser.add_argument('--log_dir', default=None)
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--seed', default=42, type=int)
    parser.add_argument('--resume', default='')
    parser.add_argument('--auto_resume', action='store_true')
    parser.add_argument('--no_auto_resume', action='store_false', dest='auto_resume')
    parser.set_defaults(auto_resume=True)

    parser.add_argument('--save_ckpt', action='store_true')
    parser.add_argument('--no_save_ckpt', action='store_false', dest='save_ckpt')
    parser.set_defaults(save_ckpt=True)

    parser.add_argument('--start_epoch', default=0, type=int, metavar='N')
    parser.add_argument('--eval', action='store_true')
    parser.add_argument('--dist_eval', action='store_true', default=False)
    parser.add_argument('--num_workers', default=5, type=int)
    parser.add_argument('--pin_mem', action='store_true')
    parser.add_argument('--no_pin_mem', action='store_false', dest='pin_mem')
    parser.set_defaults(pin_mem=True)

    parser.add_argument('--world_size', default=1, type=int)
    parser.add_argument('--local_rank', default=-1, type=int)
    parser.add_argument('--dist_on_itp', action='store_true')
    parser.add_argument('--dist_url', default='env://')

    parser.add_argument('--enable_deepspeed', action='store_true', default=False)
    parser.add_argument('--dataset', default='SEED_IV', type=str)
    parser.add_argument('--test_subject', default='1', type=str, help='Holdout subject for the current fold')

    known_args, _ = parser.parse_known_args()

    if known_args.enable_deepspeed:
        try:
            import deepspeed
            parser = deepspeed.add_config_arguments(parser)
            ds_init = deepspeed.initialize
        except:
            print("Please 'pip install deepspeed==0.4.0'")
            exit(0)
    else:
        ds_init = None

    return parser.parse_args(), ds_init


def get_models(args):
    model = create_model(
        args.model,
        pretrained=False,
        num_classes=args.nb_classes,
        drop_rate=args.drop,
        drop_path_rate=args.drop_path,
        attn_drop_rate=args.attn_drop_rate,
        drop_block_rate=None,
        use_mean_pooling=args.use_mean_pooling,
        init_scale=args.init_scale,
        use_rel_pos_bias=args.rel_pos_bias,
        use_abs_pos_emb=args.abs_pos_emb,
        init_values=args.layer_scale_init_value,
        qkv_bias=args.qkv_bias,
    )
    return model


def get_dataset(args):
    if args.dataset == 'SEED_IV':
        all_subjects = list(range(1, 16))
        test_sub = int(args.test_subject)
        train_subs = [sub for sub in all_subjects if sub != test_sub]
        test_subs = [test_sub]
        
        train_dataset, val_dataset, test_dataset = utils.prepare_SeedIV_dataset(
            (train_subs, test_subs, test_subs), 
            base_path="/homes/yy2541/Henry1_parkin/SEED_IV/"
        )
        
        # 62 Channels matching SEED-IV exactly mapped to LaBraM's standard_1020 names
        ch_names = [
            'FP1', 'FPZ', 'FP2', 'AF3', 'AF4', 'F7', 'F5', 'F3', 'F1', 'FZ', 'F2', 'F4', 'F6', 'F8', 
            'FT7', 'FC5', 'FC3', 'FC1', 'FCZ', 'FC2', 'FC4', 'FC6', 'FT8', 'T7', 'C5', 'C3', 'C1', 
            'CZ', 'C2', 'C4', 'C6', 'T8', 'TP7', 'CP5', 'CP3', 'CP1', 'CPZ', 'CP2', 'CP4', 'CP6', 
            'TP8', 'P7', 'P5', 'P3', 'P1', 'PZ', 'P2', 'P4', 'P6', 'P8', 'PO7', 'PO5', 'PO3', 'POZ', 
            'PO4', 'PO6', 'PO8', 'CB1', 'O1', 'OZ', 'O2', 'CB2'
        ]
        
        args.nb_classes = 4
        # Note: AUC is excluded for multi-class to prevent shape mismatches in older pyhealth versions,
        # but you can add 'roc_auc' back if your pyhealth supports multi-class OVR.
        metrics = ["accuracy", "balanced_accuracy", "f1_macro", "cohen_kappa"]

    return train_dataset, test_dataset, val_dataset, ch_names, metrics


def main(args, ds_init):
    if ds_init is not None:
        utils.create_ds_config(args)

    device = torch.device(args.device)

    seed = args.seed + utils.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)

    cudnn.benchmark = True

    dataset_train, dataset_test, dataset_val, ch_names, metrics = get_dataset(args)

    if args.disable_eval_during_finetuning:
        dataset_val = None
        dataset_test = None

    num_tasks = utils.get_world_size()
    global_rank = utils.get_rank()
    sampler_train = torch.utils.data.DistributedSampler(
        dataset_train, num_replicas=num_tasks, rank=global_rank, shuffle=True
    )
    if args.dist_eval:
        sampler_val = torch.utils.data.DistributedSampler(
            dataset_val, num_replicas=num_tasks, rank=global_rank, shuffle=False)
        sampler_test = torch.utils.data.DistributedSampler(
            dataset_test, num_replicas=num_tasks, rank=global_rank, shuffle=False)
    else:
        sampler_val = torch.utils.data.SequentialSampler(dataset_val)
        sampler_test = torch.utils.data.SequentialSampler(dataset_test)

    if global_rank == 0 and args.log_dir is not None:
        os.makedirs(args.log_dir, exist_ok=True)
        log_writer = utils.TensorboardLogger(log_dir=args.log_dir)
    else:
        log_writer = None

    data_loader_train = torch.utils.data.DataLoader(
        dataset_train, sampler=sampler_train,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=args.pin_mem,
        drop_last=True,
    )

    if dataset_val is not None:
        data_loader_val = torch.utils.data.DataLoader(
            dataset_val, sampler=sampler_val,
            batch_size=int(1.5 * args.batch_size),
            num_workers=args.num_workers,
            pin_memory=args.pin_mem,
            drop_last=False
        )
        data_loader_test = torch.utils.data.DataLoader(
            dataset_test, sampler=sampler_test,
            batch_size=int(1.5 * args.batch_size),
            num_workers=args.num_workers,
            pin_memory=args.pin_mem,
            drop_last=False
        )
    else:
        data_loader_val = None
        data_loader_test = None

    model = get_models(args)
    patch_size = model.patch_size
    args.window_size = (1, args.input_size // patch_size)
    args.patch_size = patch_size

    if args.finetune:
        checkpoint = torch.load(args.finetune, map_location='cpu', weights_only=False)
        checkpoint_model = None
        for model_key in args.model_key.split('|'):
            if model_key in checkpoint:
                checkpoint_model = checkpoint[model_key]
                break
        if checkpoint_model is None:
            checkpoint_model = checkpoint
        if (checkpoint_model is not None) and (args.model_filter_name != ''):
            all_keys = list(checkpoint_model.keys())
            new_dict = OrderedDict()
            for key in all_keys:
                if key.startswith('student.'):
                    new_dict[key[8:]] = checkpoint_model[key]
            checkpoint_model = new_dict

        state_dict = model.state_dict()
        for k in ['head.weight', 'head.bias']:
            if k in checkpoint_model and checkpoint_model[k].shape != state_dict[k].shape:
                del checkpoint_model[k]

        all_keys = list(checkpoint_model.keys())
        for key in all_keys:
            if "relative_position_index" in key:
                checkpoint_model.pop(key)

        utils.load_state_dict(model, checkpoint_model, prefix=args.model_prefix)

    model.to(device)
    model_ema = None
    if args.model_ema:
        model_ema = ModelEma(
            model,
            decay=args.model_ema_decay,
            device='cpu' if args.model_ema_force_cpu else '',
            resume='')

    model_without_ddp = model
    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)

    total_batch_size = args.batch_size * args.update_freq * utils.get_world_size()
    num_training_steps_per_epoch = len(dataset_train) // total_batch_size

    num_layers = model_without_ddp.get_num_layers()
    if args.layer_decay < 1.0:
        assigner = LayerDecayValueAssigner(list(args.layer_decay ** (num_layers + 1 - i) for i in range(num_layers + 2)))
    else:
        assigner = None

    skip_weight_decay_list = model.no_weight_decay()
    if args.disable_weight_decay_on_rel_pos_bias:
        for i in range(num_layers):
            skip_weight_decay_list.add("blocks.%d.attn.relative_position_bias_table" % i)

    if args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu], find_unused_parameters=True)
        model_without_ddp = model.module

    optimizer = create_optimizer(
        args, model_without_ddp, skip_list=skip_weight_decay_list,
        get_num_layer=assigner.get_layer_id if assigner is not None else None, 
        get_layer_scale=assigner.get_scale if assigner is not None else None)
    loss_scaler = NativeScaler()

    lr_schedule_values = utils.cosine_scheduler(
        args.lr, args.min_lr, args.epochs, num_training_steps_per_epoch,
        warmup_epochs=args.warmup_epochs, warmup_steps=args.warmup_steps,
    )
    if args.weight_decay_end is None:
        args.weight_decay_end = args.weight_decay
    wd_schedule_values = utils.cosine_scheduler(
        args.weight_decay, args.weight_decay_end, args.epochs, num_training_steps_per_epoch)

    if args.nb_classes == 1:
        criterion = torch.nn.BCEWithLogitsLoss()
    elif args.smoothing > 0.:
        criterion = LabelSmoothingCrossEntropy(smoothing=args.smoothing)
    else:
        criterion = torch.nn.CrossEntropyLoss()

    utils.auto_load_model(
        args=args, model=model, model_without_ddp=model_without_ddp,
        optimizer=optimizer, loss_scaler=loss_scaler, model_ema=model_ema)
            
    max_accuracy = 0.0
    final_test_stats = None 

    for epoch in range(args.start_epoch, args.epochs):
        if args.distributed:
            data_loader_train.sampler.set_epoch(epoch)
        if log_writer is not None:
            log_writer.set_step(epoch * num_training_steps_per_epoch * args.update_freq)
            
        train_stats = train_one_epoch(
            model, criterion, data_loader_train, optimizer,
            device, epoch, loss_scaler, args.clip_grad, model_ema,
            log_writer=log_writer, start_steps=epoch * num_training_steps_per_epoch,
            lr_schedule_values=lr_schedule_values, wd_schedule_values=wd_schedule_values,
            num_training_steps_per_epoch=num_training_steps_per_epoch, update_freq=args.update_freq, 
            ch_names=ch_names, is_binary=args.nb_classes == 1
        )
        
        # --- MODIFICATION 1: Commented out periodic epoch saving ---
        # if args.output_dir and args.save_ckpt:
        #     utils.save_model(
        #         args=args, model=model, model_without_ddp=model_without_ddp, optimizer=optimizer,
        #         loss_scaler=loss_scaler, epoch=epoch, model_ema=model_ema, save_ckpt_freq=args.save_ckpt_freq)
            
        if data_loader_val is not None:
            val_stats = evaluate(data_loader_val, model, device, header='Val:', ch_names=ch_names, metrics=metrics, is_binary=args.nb_classes == 1)
            test_stats = evaluate(data_loader_test, model, device, header='Test:', ch_names=ch_names, metrics=metrics, is_binary=args.nb_classes == 1)
            final_test_stats = test_stats 
            
            if max_accuracy < val_stats["accuracy"]:
                max_accuracy = val_stats["accuracy"]
                
                # --- MODIFICATION 2: Commented out "best" model saving ---
                if args.output_dir and args.save_ckpt:
                    utils.save_model(
                        args=args, model=model, model_without_ddp=model_without_ddp, optimizer=optimizer,
                        loss_scaler=loss_scaler, epoch="best", model_ema=model_ema)

            if log_writer is not None:
                for key, value in test_stats.items():
                    if key in ['accuracy', 'balanced_accuracy', 'f1', 'f1_weighted', 'pr_auc', 'roc_auc', 'cohen_kappa', 'loss']:
                        log_writer.update(**{key: value}, head="test", step=epoch)
                
            log_stats = {**{f'train_{k}': v for k, v in train_stats.items()},
                         **{f'val_{k}': v for k, v in val_stats.items()},
                         **{f'test_{k}': v for k, v in test_stats.items()},
                         'epoch': epoch,
                         'n_parameters': n_parameters}
        else:
            log_stats = {**{f'train_{k}': v for k, v in train_stats.items()},
                         'epoch': epoch,
                         'n_parameters': n_parameters}

        if args.output_dir and utils.is_main_process():
            if log_writer is not None:
                log_writer.flush()
            with open(os.path.join(args.output_dir, "log.txt"), mode="a", encoding="utf-8") as f:
                f.write(json.dumps(log_stats) + "\n")

    # --- THIS REMAINS: Saves the final model at the end of the entire loop ---
    # if args.output_dir and args.save_ckpt:
    #     utils.save_model(
    #         args=args, model=model, model_without_ddp=model_without_ddp, optimizer=optimizer,
    #         loss_scaler=loss_scaler, epoch="final", model_ema=model_ema)

    cleaned_stats = {}
    if final_test_stats is not None:
        for k, v in final_test_stats.items():
            cleaned_stats[k] = v.item() if hasattr(v, 'item') else v

    del model, model_without_ddp, optimizer, data_loader_train, data_loader_test, data_loader_val
    gc.collect()
    torch.cuda.empty_cache()

    return cleaned_stats


if __name__ == '__main__':
    opts, ds_init = get_args()
    utils.init_distributed_mode(opts)

    if opts.dataset == 'SEED_IV':
        all_subjects = list(range(1, 16))
        all_fold_metrics = []

        base_output_dir = opts.output_dir
        base_log_dir = opts.log_dir

        for test_sub in all_subjects:
            if utils.is_main_process():
                print(f"\n{'='*60}")
                print(f"🚀 STRICT LOOCV: FOLD {test_sub} / 15")
                print(f"Test Subject:      {test_sub}")
                print(f"{'='*60}\n")
            
            opts.test_subject = str(test_sub)
            
            if base_output_dir:
                opts.output_dir = os.path.join(base_output_dir, f"fold_subject_{test_sub}")
                Path(opts.output_dir).mkdir(parents=True, exist_ok=True)
            if base_log_dir:
                opts.log_dir = os.path.join(base_log_dir, f"fold_subject_{test_sub}")

            # Execution
            fold_metrics = main(opts, ds_init)

            # --- MEMORY CLEANUP (Correctly placed to clear each fold) ---
            gc.collect()
            torch.cuda.empty_cache()
            # -------------------------------------------------------------

            if fold_metrics and utils.is_main_process():
                # FIX: Look for 'f1_macro' to match your dataset definition
                f1_val = fold_metrics.get('f1_macro', fold_metrics.get('f1', 0))
                print(f"\n✅ RESULTS FOR TEST SUBJECT {test_sub}:")
                print(f"   Accuracy:          {fold_metrics.get('accuracy', 0):.4f}")
                print(f"   Balanced Accuracy: {fold_metrics.get('balanced_accuracy', 0):.4f}")
                print(f"   F1 Score:          {f1_val:.4f}")
                print(f"   Kappa:             {fold_metrics.get('cohen_kappa', 0):.4f}\n")
                all_fold_metrics.append(fold_metrics)

        if utils.is_main_process() and all_fold_metrics:
            print("\n" + "🌟"*30)
            print("FINAL STRICT LOOCV AVERAGES (15 FOLDS)")
            print("🌟"*30)
            
            avg_metrics = {}
            for key in all_fold_metrics[0].keys():
                values = [m[key] for m in all_fold_metrics if key in m]
                if values:
                    avg_metrics[key] = np.mean(values)
            
            def get_m(d, *names):
                for n in names:
                    if n in d: return d[n]
                return 0.0

            print(f"Average Accuracy:          {get_m(avg_metrics, 'accuracy'):.4f}")
            print(f"Average Balanced Accuracy: {get_m(avg_metrics, 'balanced_accuracy'):.4f}")
            # FIX: Included 'f1_macro' here so the final aggregation pulls the correct data
            print(f"Average F1 Score:          {get_m(avg_metrics, 'f1_macro', 'f1', 'f1_weighted'):.4f}")
            print(f"Average Cohen's Kappa:     {get_m(avg_metrics, 'cohen_kappa'):.4f}")
            print("🌟"*30 + "\n")

    else:
        if opts.output_dir:
            Path(opts.output_dir).mkdir(parents=True, exist_ok=True)
        main(opts, ds_init)