# CLI Probe

The bottom of `smolvla_npu.py` contains utilities for local correctness probing.

## `compare`

Computes error metrics between the NPU output and a reference tensor.

Returns:

```text
max_abs
mean_abs
rel_mean
reference_mean_abs
```

## `run_trace_probe`

Runs the single-file NPU model against a saved trace.

It checks:

- prefix embedding error
- final action error

This is useful before running LIBERO because it isolates model correctness from environment rollout behavior.

## `main`

CLI entry point for the trace probe.

Typical use:

```bash
python IRON/iron/applications/smolvla_npu3/smolvla_npu.py \
  --checkpoint ../checkpoints/lerobot_smolvla_base/model.safetensors \
  --trace iron/applications/smolvla_npu/traces/smolvla_reference_trace.pt
```
