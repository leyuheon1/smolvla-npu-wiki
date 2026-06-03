# `smolvla_npu.py`

`IRON/iron/applications/smolvla_npu3/smolvla_npu.py` is a single-file SmolVLA inference port for AMD NPU. It mirrors the original LeRobot SmolVLA policy, but replaces many PyTorch tensor operations with fixed-shape IRON operators.

## Model Flow

The flowchart for my implementation is like the following.


```mermaid
graph TD
    %% Top Level Interfaces
    subgraph Interfaces [Policy & Module Interfaces]
        A[SmolVLANPUPolicy] -->|Wraps| B[NPUVLAFlowMatchingModule]
        B -->|Instantiates| C[VLAFlowMatching]
    end

    %% Prefix Processing
    subgraph Prefix [Prefix Processing / Context Embedding]
        C -->|embed_prefix| D[SmolVLMWithExpertModel]
        C -->|embed_prefix| E[run_state_projection_npu]

        D -->|embed_image| F[Vision Encoder Path]
        F --> F1[run_patch_embedding_npu]
        F --> F2[run_vision_layer_npu x12]

        D -->|forward| G[Text Model Prefix Cache]
        G --> G1[run_text_layer_npu x16]
    end

    %% Action Denoising
    subgraph Denoising [Action Suffix & Denoising Loop]
        C -->|embed_suffix| H[run_embed_suffix_npu]
        C -->|denoise_step| I[run_denoise_npu]

        I --> I1[run_prefix_self_attention_npu]
        I --> I2[run_cross_attention_npu]
        I --> I3[run_layer_mlp_npu]
        I --> J[run_action_out_npu]
    end

    %% Hardware Primitives
    subgraph Hardware [AMD IRON Hardware Primitives]
        subgraph Math [Math]
            OpGemm[GEMM]
            OpAdd[ElementwiseAdd]
            OpMul[ElementwiseMul]
        end
        subgraph Norm [Normalization]
            OpRMS[RMSNorm]
            OpLN[LayerNorm]
        end
        subgraph Act [Activations]
            OpGelu[GELU]
            OpSilu[SiLU]
            OpSiluMul[SiLUMul]
        end
        subgraph Attn [Attention]
            OpRope[RoPE]
            OpSoft[Softmax]
        end
    end

    %% Connections from high-level Python paths to low-level Hardware Ops
    F2 -.->|Executes| OpGemm
    F2 -.->|Executes| OpAdd
    F2 -.->|Executes| OpMul
    F2 -.->|Executes| OpLN
    F2 -.->|Executes| OpGelu
    F2 -.->|Executes| OpRope
    F2 -.->|Executes| OpSoft
    G1 -.->|Executes| OpGemm
    G1 -.->|Executes| OpAdd
    G1 -.->|Executes| OpMul
    G1 -.->|Executes| OpRMS
    G1 -.->|Executes| OpSilu
    G1 -.->|Executes| OpRope
    G1 -.->|Executes| OpSoft
    I3 -.->|Executes| OpGemm
    I3 -.->|Executes| OpAdd
    I3 -.->|Executes| OpRMS
    I3 -.->|Executes| OpSiluMul
    H -.->|Executes| OpGemm
    H -.->|Executes| OpAdd
    H -.->|Executes| OpSilu
```
