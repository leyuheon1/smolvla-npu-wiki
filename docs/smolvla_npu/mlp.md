# Expert MLP

The expert MLP runs after attention in every expert layer.

## Flow

```text
hidden + attention output
  -> RMSNorm
  -> gate projection
  -> up projection
  -> SiLU(gate) * up
  -> down projection
  -> residual add
```

## `run_layer_mlp_npu`

Runs the MLP block for one expert layer.

Detailed steps:

1. Pad `hidden` and `attn_out` to the compiled hidden shape.
2. Run residual add:

   ```text
   after_attn = hidden + attn_out
   ```

3. Run post-attention RMSNorm.
4. Pad normalized hidden to `E_HIDDEN_PAD`.
5. Run gate projection GEMM.
6. Run up projection GEMM.
7. Run fused `SiLU(gate) * up`.
8. Run down projection GEMM.
9. Add the MLP output back to `after_attn`.

## `SiLUMul`

Custom/fused elementwise operator imported from:

```text
iron.operators.silu_mul.op
```

It computes:

```text
SiLU(gate) * up
```

This matches the gated MLP style used by the expert.

## Why The MLP Matters

The MLP runs once per expert layer. With 16 layers and 10 denoise steps:

```text
16 * 10 = 160 MLP blocks per action chunk
```

Even if each MLP block is not the single largest component, the repeated launch overhead matters.

## Fusion Target

A natural fusion target is:

```text
residual + RMSNorm + gate/up + SiLU-mul + down + residual
```

The goal would be to reduce intermediate host synchronization and keep activations on the NPU longer.
