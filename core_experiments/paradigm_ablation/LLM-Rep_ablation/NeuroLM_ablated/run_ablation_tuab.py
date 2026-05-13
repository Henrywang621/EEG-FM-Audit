import os
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.distributed as dist
import gc
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed import init_process_group
from torch.utils.data import DataLoader, DistributedSampler
from sklearn.metrics import balanced_accuracy_score, f1_score, cohen_kappa_score

# NeuroLM Specific Imports
from model.model import GPTConfig
from utils import cosine_scheduler
from ablated_model import SupervisedNeuroLM
from downstream_dataset import TUABLoader 

# DDP Configuration
master_process = None; device = None; dtype = None; ctx = None
ddp_rank = None; ddp_local_rank = None; ddp_world_size = None

def init_ddp():
    global ctx, master_process, ddp_rank, ddp_local_rank, ddp_world_size, device, dtype
    if not dist.is_initialized():
        init_process_group(backend='nccl')
    ddp_rank = int(os.environ['RANK'])
    ddp_local_rank = int(os.environ['LOCAL_RANK'])
    ddp_world_size = int(os.environ['WORLD_SIZE'])
    device = f'cuda:{ddp_local_rank}'
    torch.cuda.set_device(device)
    master_process = (ddp_rank == 0)
    dtype = 'bfloat16' if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else 'float16'
    ptdtype = {'float32': torch.float32, 'bfloat16': torch.bfloat16, 'float16': torch.float16}[dtype]
    ctx = torch.amp.autocast(device_type='cuda', dtype=ptdtype)

@torch.no_grad()
def evaluate_supervised(model, loader):
    model.eval()
    all_preds, all_labels = [], []
    for batch in loader:
        X_eeg, _, target, chans, t_steps, eeg_mask, _ = [
            b.to(device, non_blocking=True) if isinstance(b, torch.Tensor) else b for b in batch
        ]
        
        with ctx:
            logits = model(X_eeg.float(), chans, t_steps, eeg_mask)
            preds = torch.argmax(logits, dim=1)
            
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(target.cpu().numpy())
        
    model.train()
    return {
        "balanced_accuracy": balanced_accuracy_score(all_labels, all_preds),
        "f1": f1_score(all_labels, all_preds, average='weighted'),
        "kappa": cohen_kappa_score(all_labels, all_preds)
    }

def main(args):
    init_ddp()
    seeds = [int(s) for s in args.seeds.split(',')]
    all_seed_results = []

    for seed in seeds:
        if master_process: print(f"\n>>> LLM-Rep TUAB Ablation Start - Seed: {seed}")
        torch.manual_seed(seed + ddp_rank)
        np.random.seed(seed + ddp_rank)
        
        # 1. Model Setup
        model_args = dict(n_layer=12, n_head=12, n_embd=768, block_size=args.block_size, bias=True, dropout=0.1)
        model = SupervisedNeuroLM(GPTConfig(**model_args), num_classes=2).to(device)
        model = DDP(model, device_ids=[ddp_local_rank], find_unused_parameters=True)
        raw_model = model.module

        # 2. Data Preparation
        train_path = os.path.join(args.dataset_dir, 'train/')
        val_path = os.path.join(args.dataset_dir, 'val/')
        test_path = os.path.join(args.dataset_dir, 'test/')

        train_files = sorted([f for f in os.listdir(train_path) if f.endswith(".pkl")])
        val_files = sorted([f for f in os.listdir(val_path) if f.endswith(".pkl")])
        test_files = sorted([f for f in os.listdir(test_path) if f.endswith(".pkl")])

        dataset_train = TUABLoader(train_path, train_files, is_instruct=True, is_val=True, 
                                   eeg_max_len=args.eeg_max_len, text_max_len=args.text_max_len)
        dataset_val = TUABLoader(val_path, val_files, is_instruct=True, is_val=True, 
                                 eeg_max_len=args.eeg_max_len, text_max_len=args.text_max_len)
        dataset_test = TUABLoader(test_path, test_files, is_instruct=True, is_val=True, 
                                  eeg_max_len=args.eeg_max_len, text_max_len=args.text_max_len)
        
        sampler = DistributedSampler(dataset_train, num_replicas=ddp_world_size, rank=ddp_rank, shuffle=True, seed=seed)
        train_loader = DataLoader(dataset_train, sampler=sampler, batch_size=args.eeg_batch_size, num_workers=4, pin_memory=True)
        val_loader = DataLoader(dataset_val, batch_size=args.eeg_batch_size, shuffle=False, num_workers=4)
        test_loader = DataLoader(dataset_test, batch_size=args.eeg_batch_size, shuffle=False, num_workers=4)

        # 3. Optimization Strategy
        optimizer = raw_model.configure_optimizers(args.weight_decay, args.learning_rate, (0.9, 0.95), 'cuda')
        criterion = nn.CrossEntropyLoss()
        scaler = torch.amp.GradScaler('cuda', enabled=(dtype == 'float16'))
        lr_schedule = cosine_scheduler(args.learning_rate, args.min_lr, args.epochs, len(train_loader))

        best_val_bacc = -1.0
        best_test_metrics = {}

        # 4. Supervised Loop
        for epoch in range(args.epochs):
            train_loader.sampler.set_epoch(epoch)
            for step, batch in enumerate(train_loader):
                it = epoch * len(train_loader) + step
                for pg in optimizer.param_groups: pg['lr'] = lr_schedule[it]

                X_eeg, _, label, chans, t_steps, eeg_mask, _ = [
                    b.to(device) if isinstance(b, torch.Tensor) else b for b in batch
                ]

                with ctx:
                    logits = model(X_eeg.float(), chans, t_steps, eeg_mask)
                    loss = criterion(logits, label.long()) / args.gradient_accumulation_steps
                
                scaler.scale(loss).backward()

                if (step + 1) % args.gradient_accumulation_steps == 0:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    scaler.step(optimizer); scaler.update(); optimizer.zero_grad(set_to_none=True)

            # 5. Evaluation
            val_res = evaluate_supervised(raw_model, val_loader)
            if val_res['balanced_accuracy'] > best_val_bacc:
                best_val_bacc = val_res['balanced_accuracy']
                best_test_metrics = evaluate_supervised(raw_model, test_loader)
                if master_process:
                    print(f"Seed {seed} | Ep {epoch} | Best Val BACC: {best_val_bacc:.4f} | Test BACC: {best_test_metrics['balanced_accuracy']:.4f}")

        all_seed_results.append(best_test_metrics)
        
        # Inter-seed memory cleanup
        del model, optimizer, train_loader, dataset_train; gc.collect(); torch.cuda.empty_cache()
        dist.barrier()

    # 6. Final Aggregate Reporting
    if master_process:
        print("\n" + "="*50)
        print(f"FINAL TUAB ABLATION RESULTS ({len(seeds)} SEEDS)")
        print("="*50)
        for m in ['balanced_accuracy', 'f1', 'kappa']:
            vals = [r[m] for r in all_seed_results if m in r]
            print(f"{m:<20}: {np.mean(vals):.4f} ± {np.std(vals):.4f}")
        print("="*50)

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_dir', type=str, required=True, help='Path to root dir containing train/val/test folders')
    parser.add_argument('--out_dir', type=str, default="results_tuab_ablation/")
    parser.add_argument('--seeds', type=str, default="42,3407,6,16,66")
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--eeg_batch_size', type=int, default=16)
    parser.add_argument('--gradient_accumulation_steps', type=int, default=32)
    parser.add_argument('--learning_rate', type=float, default=5e-4)
    parser.add_argument('--min_lr', type=float, default=5e-5)
    parser.add_argument('--weight_decay', type=float, default=1e-1)
    parser.add_argument('--block_size', type=int, default=1024)
    parser.add_argument('--eeg_max_len', type=int, default=1024)
    parser.add_argument('--text_max_len', type=int, default=80)
    return parser.parse_args()

if __name__ == '__main__':
    main(get_args())