# Expert Attention

The expert alternates between two attention modes.

```text
even layers: self-attention over prefix + action suffix
odd layers:  cross-attention from action suffix to prefix cache
```

## Shared Attention Pieces

Both paths use:

- RMSNorm
- Q projection
- K/V projections
- RoPE
- score GEMM
- mask/softmax
- context GEMM
- output projection

## `run_rope`

Runs IRON's `RoPE` operator.

RoPE rotates Q/K head vectors using the angle table produced by `rope_angles` or `rope_angles_with_offset`.

## `run_softmax`

Runs IRON softmax over attention scores.

## `run_scaled_softmax`

Runs score scaling plus softmax.

Original PyTorch equivalent:

```python
softmax(scores / sqrt(head_dim))
```

## `run_prefix_self_attention_npu`

Runs even-layer expert attention.

Input:

```text
action suffix hidden
prefix key cache
prefix value cache
```

Flow:

```text
hidden
  -> RMSNorm
  -> Q/K/V projections for suffix
  -> RoPE on suffix Q/K
  -> concatenate prefix K/V with suffix K/V
  -> Q @ K^T score GEMMs
  -> prefix padding mask + causal suffix mask
  -> scaled softmax
  -> softmax @ V context GEMMs
  -> output projection
```

The special part is that suffix tokens can see prefix tokens and earlier suffix tokens, but not future suffix tokens.

## `run_cross_attention_npu`

Runs odd-layer expert attention.

Input:

```text
action suffix hidden
prefix key cache
prefix value cache
```

Flow:

```text
hidden
  -> RMSNorm
  -> Q projection from suffix
  -> K/V projection from prefix cache
  -> RoPE on Q
  -> Q @ K^T score GEMMs
  -> prefix padding mask
  -> scaled softmax
  -> softmax @ V context GEMMs
  -> output projection
```

Unlike self-attention, this path does not create suffix K/V. It attends into the prefix cache.

## Why Attention Is Expensive

The current implementation launches many small NPU GEMMs for score and context computation.

That is why attention is a major candidate for custom fusion:

```text
Q/K/V projection
  -> RoPE
  -> score
  -> mask
  -> softmax
  -> context
  -> output projection
```

Keeping more of this chain resident on the NPU would reduce host synchronization.
