# Policy Wrapper

These classes connect the NPU implementation to LeRobot's policy/evaluation interface.

## `NPUConfig`

Small static config subset expected by the NPU port.

Important assumptions:

- `chunk_size = 50`
- `max_action_dim = 32`
- `num_steps = 10`
- cache is enabled
- image special tokens are disabled

The NPU implementation relies on these assumptions because the IRON operators are compiled for fixed shapes.

## `make_att_2d_masks`

Local copy of LeRobot's attention mask construction.

It builds the prefix attention mask used while filling the KV cache.

## `SmolVLMWithExpertModel`

Facade for the SmolVLM side.

### `compile`

Creates the model wrapper from the loaded weights.

### `embed_image`

Runs the image path:

```text
image -> run_image_embedding_npu -> image tokens
```

### `embed_language_tokens`

Looks up language token embeddings.

This is still host-side embedding lookup, because it is small and direct.

### `forward`

Used for prefix KV-cache fill.

It runs text layers and returns `past_key_values` for the expert denoise path.

## `VLAFlowMatching`

Top-level action-generation model.

### `compile`

Creates the NPU version of the flow-matching model.

### `sample_noise`

Creates the initial Gaussian action noise.

### `embed_prefix`

Builds the prefix sequence:

```text
image tokens + language tokens + state token
```

### `embed_suffix`

Builds action suffix embeddings from noisy actions and timestep.

### `denoise_step`

Runs one flow-matching denoise step:

```text
suffix embedding -> expert stack -> action velocity
```

### `sample_actions`

Full policy action generation:

```text
prefix embedding/cache
  -> 10 denoise steps
  -> action chunk
```

## `SmolVLANPUPolicy`

LeRobot-compatible policy class.

It:

1. Loads the safetensors checkpoint.
2. Validates that the config matches the fixed-shape NPU port.
3. Creates `NPUVLAFlowMatchingModule`.

## `NPUVLAFlowMatchingModule`

Thin `torch.nn.Module` adapter.

It forwards LeRobot calls to the plain NPU implementation:

```text
sample_noise
sample_actions
```

## `main`, `run_trace_probe`, and `compare`

These are debugging/probe utilities.

- `main`: CLI entry point for a local trace probe
- `run_trace_probe`: runs the NPU model and compares against saved reference tensors
- `compare`: computes max/mean/relative error
