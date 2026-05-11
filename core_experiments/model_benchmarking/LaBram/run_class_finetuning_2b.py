# --------------------------------------------------------
# Large Brain Model for Learning Generic Representations with Tremendous EEG Data in BCI
# Run with: CUDA_VISIBLE_DEVICES=5 python eval_stage2_EEGNet.py
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

def get_args():
    parser = argparse.ArgumentParser('LaBraM fine-tuning and evaluation script for EEG classification', add_help=False)
    parser.add_argument('--batch_size', default=64, type=int)
    parser.add_argument('--epochs', default=30, type=int)
    parser.add_argument('--update_freq', default=1, type=int)
    parser.add_argument('--save_ckpt_freq', default=5, type=int)

    parser.add_argument('--robust_test', default=None, type=str)
    
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

    parser.add_argument('--input_size', default=400, type=int, help='EEG input size') 

    parser.add_argument('--drop', type=float, default=0.0, metavar='PCT')
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
    parser.add_argument('--reprob', type=float, default=0.25, metavar='PCT')
    parser.add_argument('--remode', type=str, default='pixel')
    parser.add_argument('--recount', type=int, default=1)
    parser.add_argument('--resplit', action='store_true', default=False)

    parser.add_argument('--finetune', default='')
    parser.add_argument('--model_key', default='model|module', type=str)
    parser.add_argument('--model_prefix', default='', type=str)
    parser.add_argument('--model_filter_name', default='gzp', type=str)
    parser.add_argument('--init_scale', default=0.001, type=float)
    parser.add_argument('--use_mean_pooling', action='store_true')
    parser.set_defaults(use_mean_pooling=True)
    parser.add_argument('--use_cls', action='store_false', dest='use_mean_pooling')
    parser.add_argument('--disable_weight_decay_on_rel_pos_bias', action='store_true', default=False)

    parser.add_argument('--nb_classes', default=0, type=int)

    parser.add_argument('--output_dir', default='')
    parser.add_argument('--log_dir', default=None)
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--seed', default=0, type=int)
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
    parser.add_argument('--dataset', default='TUAB', type=str)
    parser.add_argument('--test_subject', default='S01', type=str, help='Holdout subject for the current fold')

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
    if args.dataset == 'TUAB':
        train_dataset, test_dataset, val_dataset = utils.prepare_TUAB_dataset("/home/u5s/henrywang.u5s/THUA/v3.0.1/edf/processed")
        ch_names = ['EEG FP1', 'EEG FP2-REF', 'EEG F3-REF', 'EEG F4-REF', 'EEG C3-REF', 'EEG C4-REF', 'EEG P3-REF', 'EEG P4-REF', 'EEG O1-REF', 'EEG O2-REF', 'EEG F7-REF', 'EEG F8-REF', 'EEG T3-REF', 'EEG T4-REF', 'EEG T5-REF', 'EEG T6-REF', 'EEG A1-REF', 'EEG A2-REF', 'EEG FZ-REF', 'EEG CZ-REF', 'EEG PZ-REF', 'EEG T1-REF', 'EEG T2-REF']
        ch_names = [name.split(' ')[-1].split('-')[0] for name in ch_names]
        args.nb_classes = 1
        metrics = ["pr_auc", "roc_auc", "accuracy", "balanced_accuracy"]

    elif args.dataset == 'TUEV':
        train_dataset, test_dataset, val_dataset = utils.prepare_TUEV_dataset("/home/u5s/henrywang.u5s/TUEV/v2.0.1/edf/processed")
        ch_names = ['EEG FP1-REF', 'EEG FP2-REF', 'EEG F3-REF', 'EEG F4-REF', 'EEG C3-REF', 'EEG C4-REF', 'EEG P3-REF', 'EEG P4-REF', 'EEG O1-REF', 'EEG O2-REF', 'EEG F7-REF', 'EEG F8-REF', 'EEG T3-REF', 'EEG T4-REF', 'EEG T5-REF', 'EEG T6-REF', 'EEG A1-REF', 'EEG A2-REF', 'EEG FZ-REF', 'EEG CZ-REF', 'EEG PZ-REF', 'EEG T1-REF', 'EEG T2-REF']
        args.nb_classes = 6
        metrics = ["pr_auc", "roc_auc", "accuracy", "balanced_accuracy"]

    elif args.dataset == 'BCIIV2b':
        all_subjects = ['S01', 'S02', 'S03', 'S04', 'S05', 'S06', 'S07', 'S08', 'S09']
        train_subs = [sub for sub in all_subjects if sub != args.test_subject]
        test_subs = [args.test_subject]
        
        train_dataset, test_dataset, val_dataset = utils.prepare_BCIIV2b_dataset(
            (train_subs, test_subs, test_subs), 
            window_size=args.input_size
        )
        
        ch_names = ['C3', 'CZ', 'C4']
        ch_names = [name.split(' ')[-1].split('-')[0] for name in ch_names]
        args.nb_classes = 1
        metrics = ["pr_auc", "roc_auc", "accuracy", "balanced_accuracy", "cohen_kappa", "f1"]

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

    if True:  
        num_tasks = utils.get_world_size()
        global_rank = utils.get_rank()
        sampler_train = torch.utils.data.DistributedSampler(
            dataset_train, num_replicas=num_tasks, rank=global_rank, shuffle=True
        )
        if args.dist_eval:
            if len(dataset_val) % num_tasks != 0:
                print('Warning: Enabling distributed evaluation with an eval dataset not divisible by process number.')
            sampler_val = torch.utils.data.DistributedSampler(
                dataset_val, num_replicas=num_tasks, rank=global_rank, shuffle=False)
            sampler_test = torch.utils.data.DistributedSampler(
                dataset_test, num_replicas=num_tasks, rank=global_rank, shuffle=False)
        else:
            sampler_val = torch.utils.data.SequentialSampler(dataset_val)
            sampler_test = torch.utils.data.SequentialSampler(dataset_test)
    else:
        sampler_train = torch.utils.data.RandomSampler(dataset_train)
        sampler_val = torch.utils.data.SequentialSampler(dataset_val)

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
        
        if args.output_dir and args.save_ckpt:
            utils.save_model(
                args=args, model=model, model_without_ddp=model_without_ddp, optimizer=optimizer,
                loss_scaler=loss_scaler, epoch=epoch, model_ema=model_ema, save_ckpt_freq=args.save_ckpt_freq)
            
        if data_loader_val is not None:
            val_stats = evaluate(data_loader_val, model, device, header='Val:', ch_names=ch_names, metrics=metrics, is_binary=args.nb_classes == 1)
            test_stats = evaluate(data_loader_test, model, device, header='Test:', ch_names=ch_names, metrics=metrics, is_binary=args.nb_classes == 1)
            final_test_stats = test_stats 
            
            if max_accuracy < val_stats["accuracy"]:
                max_accuracy = val_stats["accuracy"]
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

    if args.output_dir and args.save_ckpt:
        utils.save_model(
            args=args, model=model, model_without_ddp=model_without_ddp, optimizer=optimizer,
            loss_scaler=loss_scaler, epoch="final", model_ema=model_ema)

    # CRITICAL FIX: Convert Tensors to Python floats before returning for averaging logic
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

    if opts.dataset == 'BCIIV2b':
        all_subjects = ['S01', 'S02', 'S03', 'S04', 'S05', 'S06', 'S07', 'S08', 'S09']
        all_fold_metrics = []

        base_output_dir = opts.output_dir
        base_log_dir = opts.log_dir

        for i, test_sub in enumerate(all_subjects):
            if utils.is_main_process():
                print(f"\n{'='*60}")
                print(f"🚀 STRICT LOSO CV: FOLD {i+1} / {len(all_subjects)}")
                print(f"Test Subject:      {test_sub}")
                print(f"{'='*60}\n")
            
            opts.test_subject = test_sub
            
            if base_output_dir:
                opts.output_dir = os.path.join(base_output_dir, f"fold_{test_sub}")
                Path(opts.output_dir).mkdir(parents=True, exist_ok=True)
            if base_log_dir:
                opts.log_dir = os.path.join(base_log_dir, f"fold_{test_sub}")

            fold_metrics = main(opts, ds_init)

            if fold_metrics and utils.is_main_process():
                # Flexible key retrieval for F1
                f1_val = fold_metrics.get('f1', fold_metrics.get('f1_weighted', 0))
                print(f"\n✅ RESULTS FOR TEST SUBJECT {test_sub}:")
                print(f"   Balanced Accuracy: {fold_metrics.get('balanced_accuracy', 0):.4f}")
                print(f"   F1 Score:          {f1_val:.4f}")
                print(f"   Kappa:             {fold_metrics.get('cohen_kappa', 0):.4f}\n")
                all_fold_metrics.append(fold_metrics)

        if utils.is_main_process() and all_fold_metrics:
            print("\n" + "🌟"*30)
            print("FINAL STRICT LOSO CV AVERAGES (9 FOLDS)")
            print("🌟"*30)
            
            avg_metrics = {}
            for key in all_fold_metrics[0].keys():
                values = [m[key] for m in all_fold_metrics if key in m]
                if values:
                    avg_metrics[key] = np.mean(values)
            
            # Helper to check multiple common names for the same metric
            def get_m(d, *names):
                for n in names:
                    if n in d: return d[n]
                return 0.0

            print(f"Average Balanced Accuracy: {get_m(avg_metrics, 'balanced_accuracy'):.4f}")
            print(f"Average F1 Score:          {get_m(avg_metrics, 'f1', 'f1_weighted'):.4f}")
            print(f"Average Cohen's Kappa:     {get_m(avg_metrics, 'cohen_kappa'):.4f}")
            print("🌟"*30 + "\n")

    else:
        if opts.output_dir:
            Path(opts.output_dir).mkdir(parents=True, exist_ok=True)
        main(opts, ds_init)