# CS3602 Project: ClusterFusion for Pythia-2.8B

This project implements a CUDA-accelerated decoder layer for EleutherAI Pythia-2.8B model, focusing on the **MLP Down Projection** computation.

## Supported Model

| Model | Architecture | Status |
|-------|--------------|--------|
| **Pythia-2.8B** | GPT-NeoX | ✅ Optimized |

## What's Accelerated

The following operation is implemented in a CUDA kernel:

| Operation | Status | Description |
|-----------|--------|-------------|
| LayerNorm | PyTorch | Normalization layers |
| QKV Projection | PyTorch | Query, Key, Value computation |
| RoPE | PyTorch | Rotary Position Embedding |
| Flash Decoding | PyTorch | Attention mechanism |
| Output Projection | PyTorch | Attention output |
| MLP Up + GELU | PyTorch | First MLP layer with activation |
| **MLP Down** | ✅ CUDA | Second MLP layer (matmul + residual) |

## Performance Results

Benchmarked on NVIDIA RTX 5090 (sm_120), batch=1:

| Metric | PyTorch | ClusterFusion | Speedup |
|--------|---------|---------------|---------|
| MLP Down per layer | 0.052 ms | 0.048 ms | **1.08x** |

Note: The MLP Down projection is a memory-bound operation (loading 26.2M weight parameters), limiting the achievable speedup.

## Environment

- Python 3.13 (conda), NVIDIA GPU with `sm_120` compute capability
- CUDA 12.8+ user-space wheels via PyTorch cu130 index

## Quick Start

```bash
# Create environment
conda create -n nlp_project python=3.13 -y
conda activate nlp_project

# Core DL stack (cu130 wheels)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu130

# Kernel + HF stack
pip install flashinfer-python
pip install transformers accelerate datasets

# ClusterFusion build
pip install -e .

# Test (use HF mirror for model download if needed)
export HF_ENDPOINT=https://hf-mirror.com
python tests/test_mlp_only.py
```

## API Usage

```python
import clusterfusion
import torch.nn.functional as F

# Compute attention and MLP Up in PyTorch first
attn_output = pytorch_attention(hidden_states, ...)
mlp_up = F.linear(post_ln, mlp_up_weight, mlp_up_bias)
mlp_intermediate = F.gelu(mlp_up)

# CUDA kernel: MLP Down + Residual
output = clusterfusion.pythia_2b8_mlp_only(
    hidden_states,      # [1, hidden_dim] - original input for residual
    attn_output,        # [1, hidden_dim] - attention output for residual
    mlp_intermediate,   # [ffn_dim] - MLP Up + GELU output
    mlp_down_weight,    # [hidden_dim, ffn_dim] transposed
    mlp_down_bias       # [hidden_dim]
)
# output = hidden_states + attn_output + MLP_Down(mlp_intermediate)
```

## Key Optimizations

1. **TMA Weight Loading**: Hardware-accelerated tensor memory access
2. **Cluster-level Reduction**: Efficient cross-block accumulation
3. **Fused Residual**: Combined MLP output with residual connection

## Model Configuration

| Parameter | Value |
|-----------|-------|
| Hidden Size | 2560 |
| Attention Heads | 32 |
| Head Dimension | 80 |
| FFN Dimension | 10240 |
| Layers | 32 |

## Files

| File | Description |
|------|-------------|
| `include/5090/pythia_2b8/kernel_mlp.cuh` | CUDA kernel implementation |
| `include/5090/pythia_2b8/pythia_mlp_dispatch.cu` | Kernel dispatch |
| `tests/test_mlp_only.py` | Test and benchmark |

## Requirements

- Python 3.13+ (conda recommended)
- PyTorch 2.0+ with CUDA (cu130 wheels)
- NVIDIA GPU with `sm_120` compute capability (RTX 5090 / Blackwell)
- CUDA 12.8+
- flashinfer-python

## Citation

```bibtex
@misc{luo2025clusterfusion,
      title={ClusterFusion: Expanding Operator Fusion Scope for LLM Inference},
      author={Xinhao Luo et al.},
      year={2025},
      eprint={2508.18850},
      archivePrefix={arXiv}
}
```
