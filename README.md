# Effect of Collective Communication Algorithms on LLM Training

> **Group-12** | MT25014 · MT25064 · MT25052 · MT25092  
> Anand · Arjun Prasad · Videh Sharma · Yash Verma

---

## 📌 Overview

Large Language Models (LLMs) contain billions to trillions of parameters and cannot be trained on a single GPU. This project investigates how **collective communication algorithms** — specifically **Ring-AllReduce** and **NVLS-Tree-AllReduce** — affect the performance of distributed LLM training.

Rather than running expensive real-hardware experiments, we use **[SimAI](https://github.com/aliyun/SimAI)** — a high-fidelity LLM training simulator with ~98.1% accuracy compared to real systems — to evaluate performance across varying configurations.

---

## 🎯 Objectives

1. **Algorithm Comparison** — Benchmark Ring-AllReduce vs. NVLS-Tree-AllReduce across varying message sizes, GPU counts, and network bandwidths.
2. **Optimal Operating Conditions** — Identify parameter combinations where each algorithm performs best.
3. **Parallelism Strategy Analysis** — Study how Data Parallelism (DP), Tensor Parallelism (TP), and Pipeline Parallelism (PP) influence algorithm selection.

---

## 🧠 Background

### Why Simulation?

Training GPT-3 required ~34 days on 1,024 GPUs. Even minor communication inefficiencies can cost days of training time. Real-cluster experimentation at this scale is cost-prohibitive, which motivates simulation-based evaluation.

### Collective Communication Primitives

| Operation | Description | Use Case |
|-----------|-------------|----------|
| **AllReduce** | Aggregates gradients across all GPUs (sum/average), then distributes result | Data Parallelism |
| **AllGather** | Each GPU shares its tensor with all others | Tensor Parallelism |
| **P2P Send/Recv** | Direct peer-to-peer tensor transfer | Pipeline Parallelism |

### Algorithms Under Study

- **Ring-AllReduce** — GPUs arranged in a logical ring; data circulates step-by-step. Scales well with increasing GPU count.
- **NVLS-Tree-AllReduce** — GPUs arranged in a tree; gradients reduce from leaves to root, then broadcast back. Lower latency for small messages.

### Parallelism Strategies

- **Data Parallelism (DP)** — Each GPU holds a full model copy and trains on different data batches.
- **Tensor Parallelism (TP)** — Individual layer tensors are split across GPUs.
- **Pipeline Parallelism (PP)** — DNN layers are distributed across GPU nodes.

---

## 🛠️ Simulator: SimAI

SimAI is a unified simulation framework for LLM training (presented at **NSDI 2025**).

### Architecture Components

```
Input Description
    └── Hardware config (GPU count, bandwidth, topology)
    
Workload Generator (SimAI-WG)
    ├── Module: Embedding → Transformer × N → Loss
    ├── Sub-modules: Self-Attention + MLP + Comm Ops
    └── Parallelisation: TP Comm / PP Comm / DP Comm

Execution Engine
    ├── Computation Simulator (GEMM, GeLU, LayerNorm)
    ├── Communication Simulator (AllReduce, P2P, Send/Recv)
    └── Scheduler (execution order & dependency resolution)
    
Simulation Speedup
    ├── Kernel time modelling (no real execution)
    ├── Parallel event simulation
    └── Trace compression
```

### Key SimAI Features

- **Framework Hijacking** — Modifies training frameworks so a single host emulates a large GPU cluster.
- **NCCL Modification** — Custom NCCL build enables pipeline parallelism within the simulated environment.
- **~98.1% accuracy** vs. real hardware systems.

---

## 🔬 Experimental Parameters

| Parameter | Values Explored |
|-----------|----------------|
| Number of GPUs | 2, 4, 8, 16, 32, ... |
| Message Size | Small → Large (gradient tensor sizes) |
| Network Bandwidth | Varied (NVLink / InfiniBand configs) |
| Collective Algorithm | Ring-AllReduce, NVLS-Tree-AllReduce |
| Parallelism Strategy | DP, TP, PP |

---

## 📊 Metrics Evaluated

- **Training Time** — End-to-end step time across configurations
- **Communication Latency** — Time spent in collective communication ops
- **Communication Efficiency** — Useful data transferred vs. total communication overhead
- **Scalability** — Performance degradation/improvement as GPU count increases

---

## 🚀 Getting Started

### Prerequisites

```bash
# Clone the repository
git clone https://github.com/Vyash2002/Effect_Of_Collective_Communication_Algorithms_on_LLM_Training
cd Effect_Of_Collective_Communication_Algorithms_on_LLM_Training
```

### Install SimAI

Follow the official SimAI setup instructions:
```bash
# Refer to the SimAI repository for installation
# https://github.com/aliyun/SimAI
```

### Running Simulations

```bash
# Configure your simulation parameters
# Edit the input description file with desired:
#   - GPU count
#   - Network bandwidth
#   - Collective algorithm (ring / NVLS-tree)
#   - Parallelism strategy

# Run the simulator
python run_simulation.py --algorithm ring --gpus 8 --message-size 1GB
python run_simulation.py --algorithm NVLS-tree --gpus 8 --message-size 1GB
```

> ⚠️ Refer to the project-specific scripts in the repo for exact invocation commands.

---

## 📁 Repository Structure

```
.
├── configs/               # Simulation configuration files
│   ├── ring_allreduce/    # Ring-AllReduce experiment configs
│   └── tree_allreduce/    # NVLS-Tree-AllReduce experiment configs
├── workloads/             # Generated workload traces
├── results/               # Simulation output and logs
├── analysis/              # Scripts for parsing and plotting results
├── report/                # Project report (PDF)
└── README.md
```

---

## 📖 References

1. Wang et al., *"SimAI: Unifying Architecture Design and Performance Tuning for Large-Scale LLM Training"*, NSDI 2025.
2. NVIDIA Collective Communications Library (NCCL) — https://developer.nvidia.com/nccl
3. Shoeybi et al., *Megatron-LM*, 2020.
4. ASTRA-sim — https://astra-sim.github.io/tutorials/micro-2024
5. Thakur et al., *"Optimization of Collective Communication Operations in MPICH"*, IJHPCA 2005.

---

## 👥 Team

| Name | Roll No. |
|------|----------|
| Anand | MT25014 |
| Arjun Prasad | MT25064 |
| Videh Sharma | MT25052 |
| Yash Verma | MT25092 |

---

<p align="center">
  <i>Built as part of a graduate systems course project exploring distributed ML infrastructure.</i>
</p> 
