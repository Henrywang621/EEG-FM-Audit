import argparse
import datetime
import numpy as np
import time
import torch
import torch.backends.cudnn as cudnn
import os

from pathlib import Path
from timm.models import create_model
from timm.loss import LabelSmoothingCrossEntropy
from timm.utils import ModelEma
from optim_factory import create_optimizer, get_parameter_groups, LayerDecayValueAssigner

from engine_for_finetuning1 import train_one_epoch, evaluate
from utils import NativeScalerWithGradNormCount as NativeScaler
import utils

# IMPORT THE NEW ABLATED MODEL FILE
import ablated_neural_transformer 

def get_args():
    parser = argparse.ArgumentParser('Supervised Baseline (Ablated) for LaBraM', add_help=False)
    parser.add_argument('--batch_size', default=64, type=int)
    parser.add_argument('--epochs', default=50, type=int) 
    parser.add_argument('--update_freq', default=1, type=int)

    parser.add_argument('--model', default='ablated_labram_base_patch200_200', type=str, metavar='MODEL')
    parser.add_argument('--qkv_bias', action='store_true')
    parser.add_argument('--disable_qkv_bias', action='store_false', dest='qkv_bias')
    parser.set_defaults(qkv_bias=True)
    parser.add_argument('--rel_pos_bias', action='store_true')
    parser.add_argument('--disable_rel_pos_bias', action='store_false', dest='rel_pos_bias')
    parser.set_defaults(rel_pos_bias=True)
    parser.add_argument('--abs_pos_emb', action='store_true')
    parser.set_defaults(abs_pos_emb=False)
    parser.add_argument('--layer_scale_init_value', default=0.1, type=float)
    parser.add_argument('--input_size', default=200, type=int)

    parser.add_argument('--drop', type=float, default=0.1, metavar='PCT')
    parser.add_argument('--attn_drop_rate', type=float, default=0.1, metavar='PCT')
    parser.add_argument('--drop_path', type=float, default=0.1, metavar='PCT')

    parser.add_argument('--model_ema', action='store_true', default=False)
    parser.add_argument('--model_ema_decay', type=float, default=0.9999)

    parser.add_argument('--opt', default='adamw', type=str)
    parser.add_argument('--weight_decay', type=float, default=0.05)
    parser.add_argument('--weight_decay_end', type=float, default=None)
    parser.add_argument('--lr', type=float, default=5e-4)
    parser.add_argument('--layer_decay', type=float, default=1.0)
    parser.add_argument('--warmup_epochs', type=int, default=5)
    parser.add_argument('--warmup_steps', type=int, default=-1)
    parser.add_argument('--min_lr', type=float, default=1e-6)
    parser.add_argument('--smoothing', type=float, default=0.1)

    parser.add_argument('--init_scale', default=0.001, type=float)
    parser.add_argument('--use_mean_pooling', action='store_true')
    parser.set_defaults(use_mean_pooling=True)
    parser.add_argument('--disable_weight_decay_on_rel_pos_bias', action='store_true', default=False)

    parser.add_argument('--nb_classes', default=0, type=int)
    parser.add_argument('--output_dir', default='./ablated_output', help='path where to save')
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--seed', default=0, type=int)
    parser.add_argument('--num_workers', default=10, type=int)
    parser.add_argument('--pin_mem', action='store_true')
    parser.set_defaults(pin_mem=True)

    parser.add_argument('--world_size', default=1, type=int)
    parser.add_argument('--local_rank', default=-1, type=int)
    parser.add_argument('--dist_on_itp', action='store_true')
    parser.add_argument('--dist_url', default='env://')
    parser.add_argument('--dataset', default='TUEV', type=str)

    return parser.parse_known_args()[0]

def get_models(args):
    model = create_model(
        args.model,
        pretrained=False,
        num_classes=args.nb_classes,
        drop_rate=args.drop,
        drop_path_rate=args.drop_path,
        attn_drop_rate=args.attn_drop_rate,
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
        train_dataset, test_dataset, val_dataset = utils.prepare_TUAB_dataset("/homes/xw2336/xw2336B/processed")
        ch_names = ['FP1', 'FP2', 'F3', 'F4', 'C3', 'C4', 'P3', 'P4', 'O1', 'O2', 'F7', 'F8', 'T3', 'T4', 'T5', 'T6', 'A1', 'A2', 'FZ', 'CZ', 'PZ', 'T1', 'T2']
        args.nb_classes = 1
        metrics = ["pr_auc", "roc_auc", "accuracy", "balanced_accuracy", "cohen_kappa", "f1"]
    elif args.dataset == 'TUEV':
        train_dataset, test_dataset, val_dataset = utils.prepare_TUEV_dataset("/homes/xw2336/data_portal/TUEV/processed")
        ch_names = ['FP1', 'FP2', 'F3', 'F4', 'C3', 'C4', 'P3', 'P4', 'O1', 'O2', 'F7', 'F8', 'T3', 'T4', 'T5', 'T6', 'A1', 'A2', 'FZ', 'CZ', 'PZ', 'T1', 'T2']
        args.nb_classes = 6
        metrics = ["accuracy", "balanced_accuracy", "cohen_kappa", "f1_weighted"]
    return train_dataset, test_dataset, val_dataset, ch_names, metrics

def main_worker(args):
    # REMOVED: utils.init_distributed_mode(args) - Now handled in __main__
    device = torch.device(args.device)

    seed = args.seed + utils.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)
    cudnn.benchmark = False
    cudnn.deterministic = True

    dataset_train, dataset_test, dataset_val, ch_names, metrics = get_dataset(args)

    sampler_train = torch.utils.data.RandomSampler(dataset_train)
    sampler_val = torch.utils.data.SequentialSampler(dataset_val)
    sampler_test = torch.utils.data.SequentialSampler(dataset_test)

    data_loader_train = torch.utils.data.DataLoader(
        dataset_train, sampler=sampler_train, batch_size=args.batch_size,
        num_workers=args.num_workers, pin_memory=args.pin_mem, drop_last=True,
    )
    data_loader_val = torch.utils.data.DataLoader(
        dataset_val, sampler=sampler_val, batch_size=int(1.5 * args.batch_size),
        num_workers=args.num_workers, pin_memory=args.pin_mem, drop_last=False
    )
    data_loader_test = torch.utils.data.DataLoader(
        dataset_test, sampler=sampler_test, batch_size=int(1.5 * args.batch_size),
        num_workers=args.num_workers, pin_memory=args.pin_mem, drop_last=False
    )

    model = get_models(args)
    model.to(device)
    model_without_ddp = model

    total_batch_size = args.batch_size * args.update_freq * utils.get_world_size()
    num_training_steps_per_epoch = len(dataset_train) // total_batch_size

    skip_weight_decay_list = model.no_weight_decay()
    optimizer = create_optimizer(args, model_without_ddp, skip_list=skip_weight_decay_list)
    loss_scaler = NativeScaler()

    lr_schedule_values = utils.cosine_scheduler(
        args.lr, args.min_lr, args.epochs, num_training_steps_per_epoch,
        warmup_epochs=args.warmup_epochs, warmup_steps=args.warmup_steps,
    )
    wd_schedule_values = utils.cosine_scheduler(
        args.weight_decay, args.weight_decay if args.weight_decay_end is None else args.weight_decay_end, 
        args.epochs, num_training_steps_per_epoch)

    if args.nb_classes == 1:
        criterion = torch.nn.BCEWithLogitsLoss()
    elif args.smoothing > 0.:
        criterion = LabelSmoothingCrossEntropy(smoothing=args.smoothing)
    else:
        criterion = torch.nn.CrossEntropyLoss()

    print(f"Start from-scratch training (Ablated) for {args.epochs} epochs")
    start_time = time.time()
    
    max_val_bacc = 0.0
    best_test_stats_for_seed = None 
    
    for epoch in range(0, args.epochs):
        train_stats = train_one_epoch(
            model, criterion, data_loader_train, optimizer,
            device, epoch, loss_scaler, None, None,
            log_writer=None, start_steps=epoch * num_training_steps_per_epoch,
            lr_schedule_values=lr_schedule_values, wd_schedule_values=wd_schedule_values,
            num_training_steps_per_epoch=num_training_steps_per_epoch, update_freq=args.update_freq, 
            ch_names=ch_names, is_binary=args.nb_classes == 1
        )
        
        val_stats = evaluate(data_loader_val, model, device, header='Val:', ch_names=ch_names, metrics=metrics, is_binary=args.nb_classes == 1)
        test_stats = evaluate(data_loader_test, model, device, header='Test:', ch_names=ch_names, metrics=metrics, is_binary=args.nb_classes == 1)
        
        print(f"Epoch {epoch} - Val Balanced Acc: {val_stats['balanced_accuracy']:.4f} | Test Balanced Acc: {test_stats['balanced_accuracy']:.4f}")
        
        if max_val_bacc < val_stats["balanced_accuracy"]:
            max_val_bacc = val_stats["balanced_accuracy"]
            best_test_stats_for_seed = test_stats
            
            utils.save_model(
                args=args, model=model, model_without_ddp=model_without_ddp, optimizer=optimizer,
                loss_scaler=loss_scaler, epoch="best_ablated", model_ema=None)

        print(f'Current Max Val Balanced Accuracy: {max_val_bacc:.4f}')

    total_time = time.time() - start_time
    print('Training time {}'.format(str(datetime.timedelta(seconds=int(total_time)))))
    return best_test_stats_for_seed

if __name__ == '__main__':
    base_opts = get_args()
    
    # Initialize distributed mode ONCE for the whole process
    utils.init_distributed_mode(base_opts)
    
    seeds = [42, 3407, 6, 16, 66]
    aggregate_results = {'balanced_accuracy': [], 'f1_score': [], 'cohen_kappa': []}

    for seed in seeds:
        print(f"\n{'='*50}\nStarting Supervised Ablation Run with Seed: {seed}\n{'='*50}")
        
        # Deep copy base_opts or create new with shared distributed info
        opts = get_args()
        opts.seed = seed
        
        # Manually carry over distributed settings initialized by base_opts
        opts.rank = base_opts.rank
        opts.world_size = base_opts.world_size
        opts.gpu = base_opts.gpu
        opts.distributed = base_opts.distributed
        
        opts.output_dir = os.path.join(base_opts.output_dir, f"ablated_seed_{seed}")
        Path(opts.output_dir).mkdir(parents=True, exist_ok=True)
        
        best_test_stats = main_worker(opts)
        
        if best_test_stats is not None:
            aggregate_results['balanced_accuracy'].append(best_test_stats.get('balanced_accuracy', 0))
            aggregate_results['cohen_kappa'].append(best_test_stats.get('cohen_kappa', 0))
            f1_val = best_test_stats.get('f1_weighted', best_test_stats.get('f1', 0))
            aggregate_results['f1_score'].append(f1_val)

    # Cleanup after all seeds are finished
    if torch.distributed.is_initialized():
        torch.distributed.destroy_process_group()

    print("\n\n" + "="*55)
    print("FINAL ABLATION PERFORMANCE ACROSS 5 SEEDS")
    print("="*55)
    
    if len(aggregate_results['balanced_accuracy']) > 0:
        print(f"Balanced Accuracy: {np.mean(aggregate_results['balanced_accuracy']):.4f} ± {np.std(aggregate_results['balanced_accuracy']):.4f}")
        print(f"F1 Score:          {np.mean(aggregate_results['f1_score']):.4f} ± {np.std(aggregate_results['f1_score']):.4f}")
        print(f"Cohen's Kappa:     {np.mean(aggregate_results['cohen_kappa']):.4f} ± {np.std(aggregate_results['cohen_kappa']):.4f}")