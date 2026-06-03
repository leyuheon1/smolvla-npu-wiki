# SmolVLA on AMD NPU

This wiki explains how `smolvla_npu.py` implements SmolVLA on AMD Ryzen AI NPU using IRON.


```text
Original SmolVLA / PyTorch behavior
        ↓
static shapes and padding required by IRON
        ↓
NPU operator calls
```

Start here:

- [`smolvla_npu.py` Guide](smolvla_npu.md): the main map of the file
- [Policy Wrapper](smolvla_npu/policy-wrapper.md): how LeRobot calls into the NPU model
- [Expert Denoise Path](smolvla_npu/expert-denoise.md): the repeated hot path
- [Attention](smolvla_npu/attention.md): even self-attention and odd cross-attention layers
- [MLP](smolvla_npu/mlp.md): expert feed-forward block
- [Profiling](smolvla_npu/profiling.md): profiling
