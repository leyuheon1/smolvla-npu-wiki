# `smolvla_npu.py`

`IRON/iron/applications/smolvla_npu3/smolvla_npu.py` is a single-file SmolVLA inference port for AMD NPU. It mirrors the original LeRobot SmolVLA policy, but replaces many PyTorch tensor operations with fixed-shape IRON operators.

## Model Flow

```text
SmolVLA policy
└── VLAFlowMatching
    ├── prefix path
    │   └── SmolVLM side
    │       ├── image encoder / vision tower
    │       ├── language token embedding / text side
    │       ├── state projection
    │       └── prefix KV-cache fill
    │
    └── action denoise path
        └── Expert side
            ├── action + timestep suffix embedding
            ├── 16 expert transformer layers
            │   ├── attention
            │   │   ├── even layers: self-attn over prefix + action suffix
            │   │   └── odd layers: cross-attn to prefix cache
            │   └── MLP
            └── final RMSNorm + action_out projection
```


