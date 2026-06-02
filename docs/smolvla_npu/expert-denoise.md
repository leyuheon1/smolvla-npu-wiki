# Expert Denoise Path

The expert denoise path is the repeated action-generation hot path.

## High-Level Flow

```text
suffix embeddings
  -> 16 expert transformer layers
  -> final RMSNorm
  -> action_out projection
  -> velocity v_t
```

In the full policy, this happens once per flow-matching step. With `num_steps=10`, the expert stack runs 10 times per action chunk.

## `DenoiseOps`

Dataclass holding compiled IRON operators for the expert path.

Includes:

- weighted RMSNorm
- residual add
- Q/K/V projection GEMMs
- RoPE
- attention score GEMMs
- attention context GEMMs
- softmax
- MLP gate/up/down GEMMs
- fused `SiLU(gate) * up`
- final action output projection

## `compile_denoise_ops`

Compiles all NPU operators needed by the expert transformer.

This is expensive at startup, but after compilation the operators can be reused for each denoise step.

## `make_small_gemm`

Builds smaller one-column GEMMs used by attention score/context operations.

Why this exists:

Attention score/context matrices have smaller inner dimensions than full hidden projection GEMMs, so they use a different fixed GEMM configuration.

## `run_denoise_npu`

Runs the full expert stack.

Layer pattern:

```text
for layer_idx in 0..15:
    if layer_idx is even:
        run_prefix_self_attention_npu
    else:
        run_cross_attention_npu
    run_layer_mlp_npu

run_action_out_npu
```

Even layers attend over prefix plus action suffix. Odd layers attend from action suffix into the prefix cache.

## `run_action_out_npu`

Projects the final hidden states into action velocity.

Flow:

```text
hidden
  -> final RMSNorm
  -> action_out GEMM
  -> bias add
  -> v_t
```

The policy then updates the flow-matching sample:

```text
x_t = x_t + dt * v_t
```

## Why This Path Is Important

The expert denoise path dominates repeated runtime because it runs:

```text
num_steps * 16 expert layers
```

With `num_steps=10`, that means 160 expert layers per action chunk.
