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

## File Sections

| Area | Page | What It Explains |
|---|---|---|
| Shared NPU helpers | [Shared Helpers](smolvla_npu/shared-helpers.md) | padding, BF16 buffers, GEMM/add/RMS wrappers |
| Vision prefix | [Vision Prefix](smolvla_npu/vision-prefix.md) | image embedding and vision tower |
| State prefix | [State Prefix](smolvla_npu/state-prefix.md) | robot state projection |
| Text prefix cache | [Text Prefix Cache](smolvla_npu/text-prefix-cache.md) | text/prefix transformer cache fill |
| Action suffix | [Action Suffix](smolvla_npu/action-suffix.md) | action + timestep embedding |
| Expert denoise | [Expert Denoise](smolvla_npu/expert-denoise.md) | 16-layer expert stack and final action output |
| Attention | [Attention](smolvla_npu/attention.md) | self-attention, cross-attention, RoPE, scores, softmax, context |
| MLP | [MLP](smolvla_npu/mlp.md) | residual, RMSNorm, gate/up, SiLU-mul, down projection |
| Wrappers | [Policy Wrapper](smolvla_npu/policy-wrapper.md) | LeRobot `SmolVLAPolicy` integration |
| Profiling | [Profiling](smolvla_npu/profiling.md) | timing probes and bottleneck interpretation |

## Execution Order

At LIBERO runtime, the main call path is:

```text
SmolVLANPUPolicy.sample_actions
  -> NPUVLAFlowMatchingModule.sample_actions
  -> VLAFlowMatching.sample_actions
     -> embed_prefix
        -> SmolVLMWithExpertModel.embed_image
        -> SmolVLMWithExpertModel.embed_language_tokens
        -> run_state_projection_npu
        -> SmolVLMWithExpertModel.forward
           -> run_text_layer_npu
     -> denoise_step repeated for num_steps
        -> embed_suffix
           -> run_embed_suffix_npu
        -> run_denoise_npu
           -> run_prefix_self_attention_npu or run_cross_attention_npu
           -> run_layer_mlp_npu
           -> run_action_out_npu
```

## Main Classes

| Class | Role |
|---|---|
| `SmolVLANPUPolicy` | LeRobot policy shell that loads NPU weights and exposes the standard policy API. |
| `NPUVLAFlowMatchingModule` | `torch.nn.Module` adapter around the plain NPU implementation. |
| `VLAFlowMatching` | Top-level SmolVLA flow-matching model: prefix path plus action denoise loop. |
| `SmolVLMWithExpertModel` | SmolVLM-side facade for image embedding, language embedding, and prefix KV-cache fill. |
| `NPUConfig` | Static subset of SmolVLA config expected by this NPU port. |
| `EmbedOps`, `TextStackOps`, `DenoiseOps` | Dataclasses that hold compiled fixed-shape IRON operators. |
| `NPUProfile` | Optional timing accumulator for LIBERO profiling. |

## How To Read The File

Read it in this order:

1. Shared helpers: understand padding and fixed-shape NPU calls.
2. `VLAFlowMatching.sample_actions`: understand the whole policy flow.
3. Prefix path: image/text/state and KV cache.
4. Expert denoise path: repeated action generation hot path.
5. Attention and MLP pages: understand the expensive inner blocks.
