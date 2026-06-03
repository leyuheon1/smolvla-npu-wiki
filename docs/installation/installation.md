
To properly reproduce the result from this project, you need to install the necessary environments for IRON, Pytorch and ROCm. Please follow the installation guide correctly for correct reproduction of the results.

## Prerequisites

All experiments for this project were conducted on Fedora Linux 43 KDE using an AMD Ryzen AI MAX+ 395 platform. You have to upgrade your kernel to 6.19+ for the following guide to work. I used miniforge conda to mangage python environments.

## Running Pytorch on iGPU for Strix Halo

To reproduce the iGPU result from this project, you should be able to run Pytorch on iGPU for Strix Halo Architecture. To do so, please follow the following steps : 



2. Run the following commands
```
sudo dnf install -y kernel-headers gcc make
```
3. create conda environment
```
conda create --name <name> python=3.11 -y
```
4. Install ROCm from TheRock
```
pip install --index-url https://rocm.nightlies.amd.com/v2/gfx1151/ "rocm[libraries,devel]"
```
5. Install Pytorch from TheRock
```
pip install --index-url https://rocm.nightlies.amd.com/v2/gfx1151/ torch torchaudio torchvision



## Installing environment for IRON

The official AMD IRON setup instructions target Ubuntu 24.04. The following instructions describe how to set up IRON on Fedora 43 using Miniforge Conda. The setup is separated into two parts:

* **System-level setup**: install and build XRT, the XDNA driver, firmware, and `pyxrt`.
* **Conda-level setup**: create a Python environment and install the IRON Python dependencies.

The system-level setup only needs to be done once per machine, unless you reinstall Fedora, intentionally update XRT/XDNA, switch to a different Python minor version for `pyxrt`, or a kernel update breaks DKMS. The Conda-level setup is what you repeat for each new IRON environment.

---

### System-level setup

Disable Secure Boot in BIOS before starting.

Then check the kernel and required kernel options:

```bash
uname -r
mokutil --sb-state || true
grep -E 'CONFIG_AMD_IOMMU=|CONFIG_DRM_ACCEL=' /boot/config-$(uname -r)
```

Expected:

```text
SecureBoot disabled
CONFIG_AMD_IOMMU=y
CONFIG_DRM_ACCEL=y
```

Choose a build directory. This guide uses variables so the path can be changed freely:

```bash
export WORKDIR="$HOME/src/amd-iron-fedora-build"
export XDNA_REPO="$WORKDIR/xdna-driver"
export IRON_REPO="$WORKDIR/IRON"
```

Install matching kernel development files:

```bash
sudo dnf upgrade --refresh -y
sudo dnf install -y kernel-devel-$(uname -r) kernel-headers
```

Install system build dependencies:

```bash
sudo dnf install -y \
  git jq curl wget dnf-plugins-core \
  gcc gcc-c++ make cmake ninja-build clang lld \
  dkms pciutils \
  boost-devel boost-filesystem boost-program-options boost-static \
  elfutils-devel elfutils-libs gnutls-devel gtest-devel json-glib-devel \
  libcurl-devel libdrm-devel libffi-devel libjpeg-turbo-devel \
  libstdc++-static libtiff-devel libuuid-devel libyaml-devel \
  lm_sensors ncurses-devel opencl-headers \
  openssl-devel perl pkgconf-pkg-config protobuf-compiler protobuf-devel \
  python3-devel python3-pip rapidjson-devel rpm-build strace \
  systemd-devel systemtap-sdt-devel unzip zlib-static \
  pybind11-devel python3-pybind11 \
  rocm-hip-devel
```

XRT’s OpenCL/XOCL code expects `ocl_icd.h`, so install Fedora’s `ocl-icd-devel` package:

```bash
sudo dnf install -y ocl-icd ocl-icd-devel --allowerasing
```

This may replace Fedora’s Khronos `OpenCL-ICD-Loader` package. That is expected.

Verify OpenCL headers and libraries:

```bash
rpm -ql ocl-icd-devel | grep 'ocl_icd.h\|libOpenCL.so'
ldconfig -p | grep -i opencl
```

Create a Conda build environment for XRT. This is used so XRT builds `pyxrt` for Python 3.12 instead of Fedora’s system Python:

```bash
conda create -n xrt-build-py312 -c conda-forge python=3.12 pip setuptools wheel cmake ninja pybind11 -y
conda activate xrt-build-py312
python --version
```

Clone the XDNA driver repository and its XRT submodule:

```bash
mkdir -p "$WORKDIR"
git clone https://github.com/amd/xdna-driver.git "$XDNA_REPO"
cd "$XDNA_REPO"
git submodule update --init --recursive
```

Run the dependency helper:

```bash
cd "$XDNA_REPO"
sudo ./tools/amdxdna_deps.sh
```

If it fails because `redhat-lsb` conflicts with `lsb_release`, patch that dependency out and rerun the helper:

```bash
sed -i '/redhat-lsb \\/d' xrt/src/runtime_src/tools/scripts/xrtdeps.sh
sudo ./tools/amdxdna_deps.sh
```

Build XRT for Conda Python 3.12:

```bash
conda activate xrt-build-py312
cd "$XDNA_REPO/xrt/build"

./build.sh clean || true

export VIRTUAL_ENV="$CONDA_PREFIX"
export CMAKE_PREFIX_PATH="$CONDA_PREFIX:${CMAKE_PREFIX_PATH:-}"

./build.sh -npu -opt -noctest \
  -cmake-flags "-DPython3_FIND_VIRTUALENV=ONLY -DPython3_EXECUTABLE=$CONDA_PREFIX/bin/python3 -DPython3_ROOT_DIR=$CONDA_PREFIX -DPython3_INCLUDE_DIR=$CONDA_PREFIX/include/python3.12 -DPython3_LIBRARY=$CONDA_PREFIX/lib/libpython3.12.so -DPYTHON_EXECUTABLE=$CONDA_PREFIX/bin/python3 -DOpenCL_LIBRARY=/usr/lib64/libOpenCL.so -DOpenCL_INCLUDE_DIR=/usr/include"
```

During the CMake configuration step, check that it uses Python 3.12, not Fedora’s system Python:

```text
Virtual environment detected, using Python3: .../envs/xrt-build-py312/bin/python3
Python libs version: 3.12.x
```

After the build completes, verify that `pyxrt` was built for Python 3.12:

```bash
cd "$XDNA_REPO/xrt/build/Release"
find . -name 'pyxrt*.so' -print
```

Expected:

```text
pyxrt.cpython-312-x86_64-linux-gnu.so
```

Install the XRT RPMs:

```bash
cd "$XDNA_REPO/xrt/build/Release"
sudo dnf install -y ./*.rpm
```

Test XRT and `pyxrt`:

```bash
conda activate xrt-build-py312
source /opt/xilinx/xrt/setup.sh

which xrt-smi
xrt-smi --help
python -c "import pyxrt; print('pyxrt OK')"
```

Build and install the XDNA plugin/driver:

```bash
conda activate xrt-build-py312
source /opt/xilinx/xrt/setup.sh

cd "$XDNA_REPO/build"
./build.sh -release -j "$(nproc)"
```

Install the generated XDNA plugin RPM:

```bash
cd "$XDNA_REPO/build/Release"
ls -lh *.rpm
sudo dnf install -y ./*.rpm
```

Set permissions and reboot:

```bash
sudo usermod -aG render "$USER"

sudo mkdir -p /etc/security/limits.d
sudo tee /etc/security/limits.d/99-amdxdna.conf > /dev/null <<'EOF'
* soft memlock unlimited
* hard memlock unlimited
EOF

sudo reboot
```

After reboot, validate the system-level setup:

```bash
conda activate xrt-build-py312
source /opt/xilinx/xrt/setup.sh

groups | tr ' ' '\n' | grep '^render$'
ulimit -l

lsmod | grep amdxdna || true
xrt-smi examine
xrt-smi validate

python -c "import pyxrt; print('pyxrt OK')"
```

Expected results:

* the user is in the `render` group
* `ulimit -l` reports `unlimited`
* the `amdxdna` kernel module is loaded
* `xrt-smi examine` detects the NPU
* `xrt-smi validate` passes
* `pyxrt OK` prints successfully

At this point, the system-level setup is complete.

---

### Conda-level setup for IRON

Do this for each Conda environment where you want to use IRON.

Create a Python 3.12 Conda environment:

```bash
conda create -n iron312 -c conda-forge python=3.12 pip setuptools wheel cmake ninja pybind11 -y
conda activate iron312
```

Source XRT in the shell:

```bash
source /opt/xilinx/xrt/setup.sh
python -c "import pyxrt; print('pyxrt OK')"
```

Clone IRON:

```bash
mkdir -p "$WORKDIR"
git clone https://github.com/amd/IRON.git "$IRON_REPO"
cd "$IRON_REPO"
```

Patch the known `llvm-aie` pin mismatch if it is present in `requirements.txt`:

```bash
cp requirements.txt requirements.txt.bak

sed -i 's/llvm-aie==21\.0\.0\.2026050601+2c363ce/llvm-aie==21.0.0.2026050601+2c363cef/' requirements.txt
```

Install IRON’s Python dependencies:

```bash
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

Test the installation:

```bash
source /opt/xilinx/xrt/setup.sh
cd "$IRON_REPO"

python - <<'PY'
import pyxrt
import iron
print("pyxrt OK")
print("iron OK")
PY

pytest -q ./iron/operators/axpy/ -s
```

If the `axpy` test passes, the IRON Conda environment is working.

Optional broader test:

```bash
pytest iron/operators/ -m "not extensive" -v
```

For each new terminal session, run:

```bash
conda activate iron312
source /opt/xilinx/xrt/setup.sh
cd "$IRON_REPO"
```

Only repeat the Conda-level setup for new Python environments. Do not repeat the full XRT/XDNA system setup unless the system-level installation needs to be rebuilt.





## Reproducing the project results

Please clone the repository first.

```bash
git clone https://github.com/leyuheon1/cse145code
```

This repository only contains the IRON-based NPU implementation. The LeRobot
and LIBERO repositories are external dependencies and should be cloned
separately. The benchmark script in this repository plugs the NPU SmolVLA policy
into LeRobot's existing LIBERO evaluator.

The expected workspace layout is:

```text
project-root/
|-- IRON/        # this repository
|-- lerobot/     # external LeRobot checkout
|-- LIBERO/      # external LIBERO checkout
`-- outputs/     # benchmark outputs
```

The paths above match the defaults used by
[`run_libero_benchmark.py`](iron/applications/smolvla_npu3/run_libero_benchmark.py).
If your checkouts are somewhere else, pass `--libero-root`, `--checkpoint`, and
`--output-dir` explicitly.

### 1. Install the external benchmark dependencies

Set up IRON first by following the environment instructions above. Then install
LeRobot and LIBERO in the same Conda environment used for the NPU run:

```bash
conda activate iron312
source /opt/xilinx/xrt/setup.sh

export PROJECT_ROOT=/path/to/project-root
cd "$PROJECT_ROOT"

git clone https://github.com/huggingface/lerobot.git lerobot
git clone https://github.com/Lifelong-Robot-Learning/LIBERO.git LIBERO

cd lerobot
git checkout b9ff99b84208661bfa7c8f7399bbe505cefdb7bc

cd "$PROJECT_ROOT/LIBERO"
git checkout 8f1084e3132a39270c3a13ebe37270a43ece2a01
python - <<'PY'
from pathlib import Path

path = Path("libero/libero/benchmark/__init__.py")
text = path.read_text()
text = text.replace(
    "init_states = torch.load(init_states_path)",
    "init_states = torch.load(init_states_path, weights_only=False)",
)
path.write_text(text)
PY

cd "$PROJECT_ROOT/lerobot"
python -m pip install --no-deps -e .
python -m pip install robosuite bddl future
```

The LIBERO patch is needed with newer PyTorch versions because `torch.load`
defaults changed. Without it, LIBERO init-state loading can fail before the
policy is evaluated.

The `--no-deps` install is intentional for this project environment. IRON's
`mlir-aie` dependency requires `numpy<2.0`, while current LeRobot metadata asks
for `numpy>=2.0`. Installing LeRobot without dependency resolution avoids
breaking the IRON environment after it has already been configured.

Verify that the environment can import the required packages:

```bash
cd "$PROJECT_ROOT"
python - <<'PY'
import pyxrt
import iron
import lerobot
import robosuite
import bddl

print("pyxrt OK")
print("iron:", iron.__file__)
print("lerobot:", lerobot.__file__)
print("LIBERO dependencies OK")
PY
```

### 2. Download or provide the SmolVLA LIBERO checkpoint

The benchmark script defaults to the Hugging Face cache path for the
`pepijn223/smolvla_libero` checkpoint:

```text
$HOME/.cache/huggingface/hub/models--pepijn223--smolvla_libero/snapshots/31d453f7edd78c839a8bbc39744a292686daf0de
```

If the checkpoint is stored somewhere else, pass it with:

```bash
--checkpoint /path/to/checkpoint_directory
```

The checkpoint directory must contain both `config.json` and
`model.safetensors`.

### 3. Run the NPU LIBERO Spatial benchmark (10 tasks version)

This is the reproduced project result: LIBERO Spatial, 10 tasks, 1 rollout per
task, 10 total rollouts, `num_steps=10`, and `n_action_steps=1`.

```bash
cd "$PROJECT_ROOT"
conda activate iron312
source /opt/xilinx/xrt/setup.sh

mkdir -p outputs/eval/smolvla_npu_libero_spatial_n1

python IRON/iron/applications/smolvla_npu3/run_libero_benchmark.py \
  --output-dir outputs/eval/smolvla_npu_libero_spatial_n1 \
  2>&1 | tee outputs/eval/smolvla_npu_libero_spatial_n1/run.log
```

The result is written to:

```text
outputs/eval/smolvla_npu_libero_spatial_n1/eval_info.json
```

On my machine, this run produced:

| Backend | Episodes | Success | Total Eval Time | Avg Time / Episode |
|---|---:|---:|---:|---:|
| NPU | 10 | 90.0% | 20864.46 s | 2086.45 s |


### 4. Compare with the iGPU baseline

The iGPU baseline should be run in an environment where PyTorch can see the
Strix Halo iGPU through ROCm. The baseline uses the official LeRobot
`SmolVLAPolicy` with the same checkpoint and the same LIBERO Spatial benchmark.

The project iGPU result was:

| Backend | Episodes | Success | Total Eval Time | Avg Time / Episode |
|---|---:|---:|---:|---:|
| iGPU | 10 | 90.0% | 458.03 s | 45.80 s |

The NPU implementation matched the iGPU smoke success rate on this run
(`90.0%` vs `90.0%`). 
More detailed benchmark notes are in
[`libero_benchmark_result.md`](iron/applications/smolvla_npu3/libero_benchmark_result.md),
and command usage details are in
[`libero_benchmark_usage.md`](iron/applications/smolvla_npu3/libero_benchmark_usage.md).

### 5. Optional paper-style Spatial run

The paper-style Spatial protocol uses 10 rollouts per task, for 100 total
Spatial rollouts. At the current NPU speed this takes a very long time:

```bash
cd "$PROJECT_ROOT"
conda activate iron312
source /opt/xilinx/xrt/setup.sh

mkdir -p outputs/eval/smolvla_npu_libero_spatial_paper_style

python IRON/iron/applications/smolvla_npu3/run_libero_benchmark.py \
  --n-episodes 10 \
  --output-dir outputs/eval/smolvla_npu_libero_spatial_paper_style \
  2>&1 | tee outputs/eval/smolvla_npu_libero_spatial_paper_style/run.log
```

For day-to-day debugging and grading reproduction, use the 10-episode smoke
benchmark first.
