# A Code Repository for EEG-FM-Audit: A Systematic Evaluation and Analysis Pipeline for EEG Foundation Models


## 🧠 Evaluated four SOTA Foundation Models (FMs) and their source code

| methods | title | author |  year | code |  
| ------ | ------ | ------ | ------ | ------ |
| NeuroGPT | Neuro-GPT: Towards A Foundation Model for EEG. [[Paper link (published in ISBI 2024)]](https://ieeexplore.ieee.org/document/10635453) | Cui et al. | 2024 | [Code](https://github.com/wenhui0206/neurogpt) |  
| LaBraM | Large Brain Model for Learning Generic Representations with Tremendous EEG Data in BCI. [[Paper link (published in ICLR 2024)]](https://openreview.net/forum?id=QzTpTRVtrP) | Jiang et al. | 2024 | [Code](https://github.com/935963004/LaBraM) |  
| EEGPT | EEGPT: Pretrained Transformer for Universal and Reliable Representation of EEG Signals. [[Paper link (published in NeurIPS 2025)]](https://openreview.net/forum?id=lvS2b8CjG5) | Wang et al. | 2024 | [Code](https://github.com/BINE022/EEGPT) |  
| NeuroLM | NeuroLM: A Universal Multi-task Foundation Model for Bridging the Gap between Language and EEG Signals. [[Paper link (published in ICLR 2025)]](https://openreview.net/forum?id=lvS2b8CjG5) | Jiang et al. | 2025 | [Code](https://github.com/935963004/NeuroLM) |  


### 🧬 Supervised baseline models and their source code


| Model | year |code |
| :--- | :--- | :--- |
| **EEGNet** | 2018 | [Code](https://github.com/aliasvishnu/EEGNet) |
| **TS-SEFFNet** | 2021 | [Code](https://github.com/LianghuiGuo/TS-SEFFNet) |
| **CSPNet** | 2024 | [Code](https://braindecode.org/stable/index.html) |
| **CTNet** | 2024 | [Code](https://github.com/snailpt/CTNet) |
| **MSCFormer** | 2025 | [Code](https://github.com/snailpt/MSCFormer) |


---

### 🖥️ Hardware Specifications

Our experiments were conducted across two high-performance servers to ensure computational consistency:

| Server | Configuration | Memory per Unit |
| :--- | :--- | :--- |
| **Server A** | 4 × NVIDIA A100 | 80 GB |
| **Server B** | 8 × NVIDIA A5000 | 24 GB |

---



### 🗄️ Public Datasets Used

We utilize the [BCI Competition IV 2b][1], the [TUAB Dataset][2], and the [TUEV Dataset][3] in our experiments.

---

[1]: https://www.bbci.de/competition/iv/
[2]: https://isip.piconepress.com/projects/nedc/data/tuh_eeg/tuh_eeg_abnormal/
[3]: https://isip.piconepress.com/projects/nedc/data/tuh_eeg/tuh_eeg_events/


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