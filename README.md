# A Code Repository for EEG-FM-Audit: A Systematic Evaluation and Analysis Pipeline for EEG Foundation Models


## 🧠 Evaluated four SOTA Foundation Models (FMs) and their source code

| methods | title | author |  year | code |  
| ------ | ------ | ------ | ------ | ------ |
| NeuroGPT | Neuro-GPT: Towards A Foundation Model for EEG. [[Paper link (published in ISBI 2024)]](https://ieeexplore.ieee.org/document/10635453) | Cui et al. | 2024 | [Code](https://github.com/wenhui0206/neurogpt) |  
| LaBraM | Large Brain Model for Learning Generic Representations with Tremendous EEG Data in BCI. [[Paper link (published in ICLR 2024)]](https://openreview.net/forum?id=QzTpTRVtrP) | Jiang et al. | 2024 | [Code](https://github.com/935963004/LaBraM) |  
| EEGPT | EEGPT: Pretrained Transformer for Universal and Reliable Representation of EEG Signals. [[Paper link (published in NeurIPS 2024)]](https://openreview.net/forum?id=lvS2b8CjG5) | Wang et al. | 2024 | [Code](https://github.com/BINE022/EEGPT) |  
| NeuroLM | NeuroLM: A Universal Multi-task Foundation Model for Bridging the Gap between Language and EEG Signals. [[Paper link (published in ICLR 2025)]](https://openreview.net/forum?id=lvS2b8CjG5) | Jiang et al. | 2025 | [Code](https://github.com/935963004/NeuroLM) |  

---

## 🧬 Supervised baseline models and their source code


| Model | year |code |
| :--- | :--- | :--- |
| **EEGNet** | 2018 | [Code](https://github.com/aliasvishnu/EEGNet) |
| **TS-SEFFNet** | 2021 | [Code](https://github.com/LianghuiGuo/TS-SEFFNet) |
| **CSPNet** | 2024 | [Code](https://braindecode.org/stable/index.html) |
| **CTNet** | 2024 | [Code](https://github.com/snailpt/CTNet) |
| **MSCFormer** | 2025 | [Code](https://github.com/snailpt/MSCFormer) |


---

## 🖥️ Hardware Specifications

Our experiments were conducted across two high-performance servers to ensure computational consistency:

| Server | Configuration | Memory per Unit |
| :--- | :--- | :--- |
| **Server A** | 4 × NVIDIA A100 | 80 GB |
| **Server B** | 8 × NVIDIA A5000 | 24 GB |

---



## 🗄️ Public Datasets Used

We utilize the [BCI Competition IV 2b][1], the [TUAB Dataset][2], and the [TUEV Dataset][3] in our experiments.

---

[1]: https://www.bbci.de/competition/iv/
[2]: https://isip.piconepress.com/projects/nedc/data/tuh_eeg/tuh_eeg_abnormal/
[3]: https://isip.piconepress.com/projects/nedc/data/tuh_eeg/tuh_eeg_events/


###  📁 Data Preparation & Preprocessing


```text
data/
├── TUAB/
├── TUEV/
└── BCICIV_2b/
```

We use the same processing methods to process TUAB and TUEV datasets as [LaBraM](https://openreview.net/forum?id=QzTpTRVtrP). The related data processing scripts can be found [here](https://github.com/935963004/LaBraM) (Click).

For BCI Competition IV 2b, you can run the script below from your terminal to process this dataset.

```bash
# Create the environment from the environment.yml file under the directory (configs)
python Preprocess_BCIIV2b.py --data-dir /path/to/BCICIV_2b_gdf --output-dir ./BCICIV_2b
```

Please place the downloaded TUAB, TUEV, and BCI Competition IV 2b datasets under data/TUAB, data/TUEV, and data/BCICIV_2b, respectively.

---


## 🛠️ Environment Setup

To ensure computational consistency across the high-performance servers (NVIDIA A100 and A5000) described above, please follow these steps to configure your environment.

### 1. Prerequisites
* **Conda:** Ensure you have [Miniconda](https://docs.conda.io/en/latest/miniconda.html) or [Anaconda](https://www.anaconda.com/) installed.
* **CUDA:** Ensure your system has NVIDIA drivers compatible with the CUDA version specified in the environment file.

### 2. Installation
Clone the repository and create the virtual environment using the provided `.yml` file. This file contains all necessary dependencies, including PyTorch and specialized libraries for BCI research.

```bash
# Create the environment from the environment.yml file under the directory (configs)
conda env create -f environment.yml
```

### 3. Environment Activation
Activate the installed Conda environment before using it.

```bash
conda activate EEG26pytorch
```

<br>

# ⚙️ ASHA-Driven Baseline Benchmarking

For example, to run the ASHA-driven script evaluating CSPNet on the TUAB dataset:

```bash
CUDA_VISIBLE_DEVICES=0 nohup python stage1_CSPNet_tuab.py > CSPNet_tuab1.log 2>&1 & disown

CUDA_VISIBLE_DEVICES=0 nohup python stage2_CSPNet_tuab.py > CSPNet_tuab2.log 2>&1 & disown 
```

Note: Before running the scripts, please update the dataset paths (e.g., TUAB_ROOT) to match your local directory structure. To evaluate different datasets or baseline models (i.e., CTNet, EEGNet, MSCFormer, and TS-SEFFNet), use the corresponding scripts provided in model_benchmarking/baselines_names instead of the scripts above.

---
<br>

# ⚙️ EEG-FMs Evaluation


### ⚡ Quick Start: Example Evaluation
To avoid redundancy, we provide a standard execution template. For example, to fine-tune and evaluate LaBraM on the TUAB dataset, run:


```bash
CUDA_VISIBLE_DEVICES=1 nohup python -u run_class_finetuning_tuab_tuev.py \
   --output_dir ./checkpoints/retrain_reproduction_tuab/ \
   --log_dir ./log/retrain_reproduction_tuab  \
   --model labram_base_patch200_200 \
   --finetune ./checkpoints/labram-base.pth  \
   --batch_size 64  \
   --lr 5e-4  \
   --epochs 50   \
   --dataset TUAB   \
   --abs_pos_emb  \
   > debug_run.log 2>&1 & 
```
<br>
🔄 Testing Other Datasets

To evaluate the other models across other datasets, simply swap the arguments in the execution template above based on the tables below.

| Target Dataset | Executable Script | `--dataset` Argument |
| :--- | :--- | :--- |
| **TUEV** | `run_class_finetuning_tuab_and_tuev.py` | `TUEV` |
| **BCICIV_2b** | `run_class_finetuning_2b.py` | `BCIIV2b` |
---
<br>

🔄 Testing Other Foundation Models

To evaluate other foundation models, similar commands can be executed. Below is an example for fine-tuning and evaluating **NeuroGPT** on the **TUAB** dataset:

```bash
CUDA_VISIBLE_DEVICES=2 nohup python -u train_tuab.py \
   --num_train_epochs 50 \
   --training-steps 10000 \
   --log-dir ./log/tuab \
   --run-name neurogpt_repro \
   --pretrained-model NeuroGPT/pretrained_model/pytorch_model.bin \
   > neurogpt_tuab.log 2>&1 &
```

Below is an example for fine-tuning and evaluating **NeuroLM** on the **TUAB** dataset:

```bash
export WANDB_MODE=offline && \
export CUDA_VISIBLE_DEVICES=0,1,2,5,6 && \
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True && \
torchrun --nnodes=1 --nproc_per_node=4 --master_port=29505 NeuroLM/train_instruction_tuab.py \
  --dataset_dir "/homes/xw2336/data" \
  --out_dir "NeuroLM/results_TUAB/default_run" \
  --NeuroLM_path "NeuroLM/checkpoints/NeuroLM-B.pt" \
  --gradient_accumulation_steps 1 \
  > training_tuab.log 2>&1 & disown
```

**Note:** Please replace the file paths with your actual local paths. You can download the pretrained model (e.g., `NeuroLM-B.pt`) from the repositories linked in the [Evaluated four SOTA Foundation Models](#fm-section) section at the beginning of this README.


---
<br>

# ⚙️ Paradigm-Level Ablation Studies

### 1. Pretraining-Source Ablation

For example, to conduct this ablation for NeuroGPT on the TUAB dataset:

```
CUDA_VISIBLE_DEVICES=0 nohup 

python -u train_tuab.py \
  --training-style generative \
  --pretrained-model none \
  --num_train_epochs 100 \
  --training-steps 20000 \
  --per-device-training-batch-size 16 \
  --per-device-validation-batch-size 16 \
  --log-dir ./log/tuab_pretraining \
  --run-name neurogpt_pretrain_tuab \
  > tuab_pretrain.log 2>&1 &
```

