# Vision Prefix Path

The vision prefix path ports the image side of SmolVLM. It turns input images into prefix embeddings that later become part of the KV cache used by the action expert.

## Flow

```text
image
  -> patch embedding
  -> vision transformer layers
  -> post layer norm
  -> connector
  -> image tokens in text hidden size
```

## `compile_vision_ops`

Compiles the IRON operators needed by the vision tower.

Main operator types:

- patch embedding GEMM
- layer norm
- Q/K/V projection GEMMs
- attention score/context GEMMs
- softmax
- MLP GEMMs
- connector GEMMs

This function does not run inference. It prepares fixed-shape NPU kernels.

## `repeat_vector`

Repeats a vector into a padded matrix.

Used for bias/position style data that must match an NPU operator's matrix shape.

## `padded_position_embedding`

Pads position embeddings to match the fixed vision sequence shape.

## `run_layer_norm_affine`

Runs LayerNorm-style affine normalization for the vision path.

Original PyTorch role:

```text
hidden -> layer norm -> normalized hidden
```

## `run_linear`

Runs a GEMM plus optional bias for the vision path.

It is similar in spirit to `run_linear_add`, but is used in the vision/text prefix code path.

## `run_patch_embedding_npu`

Converts image patches into hidden vectors.

Original PyTorch equivalent:

```text
Conv2d patch embedding or patch projection
```

NPU porting idea:

The image is unfolded/arranged into patch vectors, then a GEMM projects each patch into the vision hidden dimension.

## `run_vision_layer_npu`

Runs one vision transformer layer.

Layer structure:

```text
LayerNorm
  -> Q/K/V projections
  -> attention score
  -> softmax
  -> context
  -> output projection
  -> residual
  -> LayerNorm
  -> MLP
  -> residual
```

This mirrors the original SmolVLM vision transformer, but each large operation is mapped to an IRON operator.

## `run_post_layernorm_npu`

Runs the final normalization after the vision transformer stack.

## `pixel_shuffle`

Rearranges the vision hidden states before the connector.

This is still a tensor-layout operation, not a learned projection.

## `run_connector_npu`

Projects vision outputs into the text/expert hidden space.

The connector is what lets image tokens join the same prefix stream as language and state tokens.

## `run_image_embedding_npu`

Top-level image embedding function.

It calls:

```text
run_patch_embedding_npu
  -> run_vision_layer_npu repeated
  -> run_post_layernorm_npu
  -> run_connector_npu
```

Called by:

```text
SmolVLMWithExpertModel.embed_image
```
