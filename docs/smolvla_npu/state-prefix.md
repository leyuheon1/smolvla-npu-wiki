# State Prefix Path

The state prefix path converts the robot state vector into one prefix token.

## Flow

```text
robot state
  -> padded state vector
  -> NPU projection
  -> state embedding token
```

## `compile_state_ops`

Compiles the fixed-shape NPU operators for state projection.

The state path is small compared with image and expert denoise, but it still needs to produce a token with the same hidden size as the rest of the prefix stream.

## `run_state_projection_npu`

Runs the state projection.

Original PyTorch role:

```text
self.state_proj(state)
```

NPU porting idea:

The state vector is padded to the compiled GEMM input shape, projected on the NPU, and returned as a prefix embedding.

Called by:

```text
VLAFlowMatching.embed_prefix
```
