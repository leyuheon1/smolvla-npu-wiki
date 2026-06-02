# Text Prefix Cache Path

The text prefix cache path fills the prefix KV cache used later by the action expert.

## Flow

```text
image tokens + language tokens + state token
  -> text/expert prefix transformer layers
  -> key/value cache for each layer
```

## `TextStackOps`

Dataclass holding compiled operators for the text prefix stack.

It groups together the NPU kernels needed by text-layer execution: normalization, projections, attention, MLP, and residual operators.

## `compile_text_ops`

Compiles the NPU operators used by `run_text_layer_npu`.

This is the prefix-cache equivalent of `compile_denoise_ops`.

## `run_text_layer_npu`

Runs one prefix transformer layer and produces cache tensors.

Typical layer structure:

```text
hidden
  -> input norm
  -> Q/K/V projections
  -> RoPE
  -> attention
  -> output projection
  -> residual
  -> post-attention norm
  -> MLP
  -> residual
  -> key/value cache
```

Why this matters:

The expert denoise path does not recompute the whole prefix every flow step. It uses the cached prefix keys and values produced here.

## `SmolVLMWithExpertModel.forward`

This method is used for prefix cache fill.

Supported path:

```text
inputs_embeds = [prefix_embs, None]
fill_kv_cache = True
use_cache = True
```

It loops through the text layers, calls `run_text_layer_npu`, pads each layer's key/value cache to `E_PREFIX_LEN`, and returns:

```text
past_key_values[layer_idx]["key_states"]
past_key_values[layer_idx]["value_states"]
```

Those cached tensors are consumed by the expert denoise attention layers.
