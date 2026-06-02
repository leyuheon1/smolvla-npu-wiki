# Shared Helpers

These functions provide the shape and runtime glue used throughout the NPU port.

## Constants

The file defines static dimensions such as:

```text
TEXT_HIDDEN = 960
TEXT_HIDDEN_PAD = 1024
E_HIDDEN = 720
E_HIDDEN_PAD = 768
E_Q_OUT = 960
E_Q_OUT_PAD = 1024
E_PREFIX_LEN = 256
E_SEQ_PAD = 64
E_GEMM_M = 256
E_MLP_HIDDEN = 2048
```

The important idea is that SmolVLA's natural tensor sizes are often not directly legal for the compiled NPU kernels. The code pads tensors to fixed shapes that IRON GEMM and elementwise operators can compile for.

## `sinusoidal_pos_embedding`

Creates the timestep embedding used by the action suffix path.

Original role:

```text
timestep -> sinusoidal embedding -> action/time MLP
```

Why it stays in Python:

The timestep embedding is small compared with the transformer blocks. It produces a CPU tensor that is then packed into the suffix embedding input for NPU work.

## `repeat_bias`

Expands a linear bias vector into a matrix with one copy per row.

Original PyTorch behavior:

```python
linear(x, weight, bias)
```

NPU porting idea:

IRON GEMM computes `x @ W`. Bias is added by a separate `ElementwiseAdd`, so the bias must be repeated into a full matrix matching the GEMM output.

## `padded_weight`

Transposes and pads a PyTorch linear weight for IRON GEMM.

Original PyTorch weight layout:

```text
(out_features, in_features)
```

IRON GEMM layout:

```text
(K, N)
```

So this helper does:

```text
weight.T -> BF16 -> zero-padded matrix
```

This helper is one of the most important porting bridges in the file.

## `make_gemm`

Compiles a fixed-shape IRON `GEMM` operator.

Every call creates an operator for one static matrix shape:

```text
[M, K] @ [K, N] -> [M, N]
```

The model must pad activations and weights to those shapes before launching the NPU kernel.

## `run_gemm`

Runs one compiled GEMM on the NPU.

Flow:

```text
torch tensor
  -> XRTTensor
  -> NPU GEMM
  -> XRTTensor output
  -> torch tensor clone
```

This is used by most projections: Q/K/V, output projection, MLP gate/up/down, state projection, and action projection.

## `run_linear_add`

Runs:

```text
GEMM -> bias add
```

It is the NPU replacement for a PyTorch `Linear` layer with bias.

## `run_add`

Runs an NPU elementwise add.

Used for residual connections and bias additions.

## `run_rms`

Runs weighted RMSNorm on the NPU.

Used in expert layers and final action output.

## `rope_angles` and `rope_angles_with_offset`

Build RoPE angle tables for the IRON `RoPE` operator.

Difference:

- `rope_angles`: starts at position 0
- `rope_angles_with_offset`: starts after the prefix, so suffix positions line up with the cached prefix length
