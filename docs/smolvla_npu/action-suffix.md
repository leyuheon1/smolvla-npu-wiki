# Action Suffix Path

The action suffix path embeds the current noisy action chunk and timestep before the expert transformer denoises it.

## Flow

```text
noisy action chunk + timestep
  -> action projection
  -> timestep sinusoidal embedding
  -> action/time MLP
  -> suffix embeddings
```

## `EmbedOps`

Dataclass holding compiled IRON operators for suffix embedding.

Includes:

- action projection GEMM
- action/time MLP GEMMs
- bias add
- SiLU activation

## `compile_embed_ops`

Compiles the fixed-shape NPU operators for action suffix embedding.

## `run_embed_suffix_npu`

Runs the NPU suffix embedding path.

Original PyTorch role:

```text
embed_suffix(noisy_actions, timestep)
```

Detailed flow:

```text
noisy action
  -> action_in_proj
  -> concatenate with timestep embedding
  -> action_time_mlp_in
  -> SiLU
  -> action_time_mlp_out
  -> suffix embeddings
```

The result becomes the initial hidden state for the expert denoise path.

## `run_sample_loop_npu`

Standalone helper for running the denoise loop outside the LeRobot policy wrapper.

It repeatedly calls:

```text
run_embed_suffix_npu
run_denoise_npu
x_t = x_t + dt * v_t
```

This is useful for trace probes and debugging.
