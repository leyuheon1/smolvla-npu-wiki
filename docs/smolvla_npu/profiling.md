# Profiling

The file includes optional timing probes for understanding where LIBERO rollout time goes.

## `NPUProfile`

Timing accumulator used by the benchmark script.

It records:

- call count
- total time
- mean time
- max time

The rows are sorted by total time.

## `configure_npu_profile`

Enables or disables profiling and chooses output settings.

Called by:

```text
run_libero_benchmark.py
```

## `dump_npu_profile`

Prints the final summary and writes JSON output.

## `profile_section`

Context manager for coarse wall-clock sections.

Example sections:

```text
wall.sample_actions
wall.embed_prefix
wall.prefix_cache_fill
wall.denoise_loop
wall.denoise_step
wall.run_denoise_npu
```

## Running The LIBERO Profiler

```bash
cd /home/johnjdoe/Desktop/my/project/amdvla/smol
source /home/johnjdoe/miniforge3/bin/activate iron312
source /opt/xilinx/xrt/setup.sh

OUT=outputs/eval/smolvla_npu_profile_3min
mkdir -p "$OUT"

PYTHONPATH=$PWD/IRON:$PYTHONPATH timeout 190s \
  python IRON/iron/applications/smolvla_npu3/run_libero_benchmark.py \
    --output-dir "$OUT" \
    --profile-npu \
    --profile-every 10 \
    --profile-top-k 30 \
  2>&1 | tee "$OUT/run.log"
```

Output:

```text
$OUT/npu_profile.json
```

## Important Rows

| Row | Meaning |
|---|---|
| `wall.sample_actions` | full action chunk wall time |
| `wall.embed_prefix` | image/text/state prefix work |
| `wall.prefix_cache_fill` | SmolVLM prefix KV-cache fill |
| `wall.denoise_loop` | all flow-matching denoise steps |
| `wall.run_denoise_npu` | expert stack wall time |
| `denoise_step.mlp` | MLP total per denoise step |
| `denoise_step.scores` | attention score GEMM work |
| `denoise_step.context` | attention context GEMM work |
| `denoise_step.qkv` | Q/K/V projection work |
| `primitive.gemm` | all low-level GEMM launches combined |

## Why App-Level Profiling Comes First

AMD/XRT and AIE trace tools can measure low-level kernel behavior, especially when C++ kernels use `event0()` and `event1()`.

But first we need to know which model function causes the most repeated launches. The app-level profiler answers:

```text
Which model block is dragging the LIBERO rollout?
```

After that, vendor trace tools can answer:

```text
Why is this specific NPU kernel slow?
```
