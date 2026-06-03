# SmolVLA on AMD NPU

This wiki goes in depth about how my project implements SmolVLA on AMD Ryzen AI NPU using IRON. Also it contains installation/replication instructions.

For the codebase, please visit [GitHub](https://github.com/leyuheon1/cse145code). Currently the repository is private. Please let me know if you need access to the repository.


```text
Original SmolVLA / PyTorch behavior
        ↓
static shapes and padding required by IRON
        ↓
NPU operator calls
```



- [`smolvla_npu.py` Guide](smolvla_npu.md): the main map of the file
