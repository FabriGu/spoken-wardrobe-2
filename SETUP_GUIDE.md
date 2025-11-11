# Setup Guide: Spoken Wardrobe 2

Complete guide for setting up the project on a fresh Mac or PC.

---

## Table of Contents

1. [Prerequisites](#prerequisites)
2. [Mac Setup (Apple Silicon)](#mac-setup-apple-silicon)
3. [PC Setup (Windows + NVIDIA GPU)](#pc-setup-windows--nvidia-gpu)
4. [Manual Patches](#manual-patches)
5. [Verify Installation](#verify-installation)
6. [Troubleshooting](#troubleshooting)

---

## Prerequisites

### All Platforms

- **Python 3.11.x** (REQUIRED - tested with 3.11.5)
  - Download from: https://www.python.org/downloads/
  - Verify: `python --version` or `python3 --version`

- **Git**
  - Download from: https://git-scm.com/downloads
  - Verify: `git --version`

- **Disk Space**: At least 10GB free (for AI models)

### Mac-Specific

- macOS 12.3+ (for Metal Performance Shaders)
- Apple Silicon (M1/M2/M3) OR Intel Mac with GPU

### PC-Specific

- Windows 10/11 64-bit
- NVIDIA GPU with 6GB+ VRAM (recommended)
  - Check: Run `nvidia-smi` in Command Prompt
- CUDA Toolkit 11.8 or 12.1
  - Download: https://developer.nvidia.com/cuda-downloads
- cuDNN 8.6+
  - Download: https://developer.nvidia.com/cudnn (requires NVIDIA account)

---

## Mac Setup (Apple Silicon)

### Step 1: Clone Repository

```bash
git clone https://github.com/yourusername/spoken_wardrobe_2.git
cd spoken_wardrobe_2
```

### Step 2: Create Virtual Environment

```bash
python3 -m venv venv
source venv/bin/activate
```

You should see `(venv)` in your terminal prompt.

### Step 3: Upgrade pip

```bash
pip install --upgrade pip setuptools wheel
```

### Step 4: Install Core Dependencies

```bash
pip install -r requirements-core.txt
```

**Expected time:** ~2-3 minutes

**Note:** This installs a patched version of chumpy from git (fixes getargs error)

### Step 5: Install Mac-Specific Dependencies

```bash
pip install -r requirements-mac.txt
```

**Expected time:** ~5-10 minutes (downloads PyTorch, TensorFlow, etc.)

**Common Issues:**
- If TensorFlow installation fails, ensure you're using Python 3.11 (not 3.12)
- If you see "metal backend not available", install tensorflow-metal separately:
  ```bash
  pip install tensorflow-metal==1.1.0
  ```

### Step 6: Apply Manual Patch (CRITICAL!)

**File to edit:** `venv/lib/python3.11/site-packages/tfjs_graph_converter/util.py`

**Change line 30:**
```python
# BEFORE (will cause error):
return np.bool

# AFTER (correct):
return np.bool_
```

**Quick command to apply patch:**
```bash
sed -i '' 's/np\.bool$/np.bool_/g' venv/lib/python3.11/site-packages/tfjs_graph_converter/util.py
```

Or manually open the file in a text editor and make the change.

### Step 7: Install External Dependencies

#### TripoSR (Image-to-3D Mesh)

```bash
cd external
git clone https://github.com/VAST-AI-Research/TripoSR.git
cd TripoSR
pip install -r requirements.txt
cd ../..
```

### Step 8: Verify Installation

```bash
python -c "import torch; print(f'PyTorch: {torch.__version__}'); print(f'MPS available: {torch.backends.mps.is_available()}')"
python -c "import tensorflow as tf; print(f'TensorFlow: {tf.__version__}')"
python -c "import mediapipe; print('MediaPipe OK')"
python -c "import cv2; print('OpenCV OK')"
```

**Expected output:**
```
PyTorch: 2.1.0
MPS available: True
TensorFlow: 2.15.0
MediaPipe OK
OpenCV OK
```

---

## PC Setup (Windows + NVIDIA GPU)

### Step 1: Install CUDA Toolkit

1. Download CUDA 11.8 from: https://developer.nvidia.com/cuda-11-8-0-download-archive
2. Run installer, select "Express" installation
3. Verify installation:
   ```cmd
   nvcc --version
   ```

### Step 2: Install cuDNN

1. Download cuDNN 8.6+ from: https://developer.nvidia.com/cudnn (requires account)
2. Extract ZIP file
3. Copy files to CUDA directory:
   - Copy `bin\*.dll` → `C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v11.8\bin\`
   - Copy `include\*.h` → `C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v11.8\include\`
   - Copy `lib\x64\*.lib` → `C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v11.8\lib\x64\`

### Step 3: Clone Repository

```cmd
git clone https://github.com/yourusername/spoken_wardrobe_2.git
cd spoken_wardrobe_2
```

### Step 4: Create Virtual Environment

```cmd
python -m venv venv
venv\Scripts\activate
```

You should see `(venv)` in your command prompt.

### Step 5: Upgrade pip

```cmd
python -m pip install --upgrade pip setuptools wheel
```

### Step 6: Install Core Dependencies

```cmd
pip install -r requirements-core.txt
```

**Note:** This installs a patched version of chumpy from git (fixes getargs error)

### Step 7: Install PyTorch with CUDA

**DO NOT use requirements file for PyTorch on PC!**

For CUDA 11.8:
```cmd
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

For CUDA 12.1:
```cmd
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

**Verify CUDA:**
```cmd
python -c "import torch; print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0))"
```

Should print:
```
True
NVIDIA GeForce RTX 3060  (or your GPU name)
```

### Step 8: Install PC-Specific Dependencies

```cmd
pip install -r requirements-pc.txt
```

### Step 9: Install Intel RealSense SDK (For RealSense Testing)

1. Download from: https://github.com/IntelRealSense/librealsense/releases
2. Install `Intel.RealSense.SDK-WIN10-<version>.exe`
3. Reboot PC
4. Verify with RealSense Viewer: `C:\Program Files (x86)\Intel RealSense SDK 2.0\Intel RealSense Viewer.exe`

### Step 10: Apply Manual Patch (CRITICAL!)

**File to edit:** `venv\Lib\site-packages\tfjs_graph_converter\util.py`

**Change line 30:**
```python
# BEFORE:
return np.bool

# AFTER:
return np.bool_
```

**PowerShell command to apply patch:**
```powershell
(Get-Content venv\Lib\site-packages\tfjs_graph_converter\util.py) -replace 'np\.bool$', 'np.bool_' | Set-Content venv\Lib\site-packages\tfjs_graph_converter\util.py
```

### Step 11: Install External Dependencies

#### TripoSR

```cmd
cd external
git clone https://github.com/VAST-AI-Research/TripoSR.git
cd TripoSR
pip install -r requirements.txt
cd ..\..
```

### Step 12: Verify Installation

```cmd
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"
python -c "import tensorflow as tf; print(f'TensorFlow: {tf.__version__}')"
python -c "import pyrealsense2 as rs; print('RealSense OK')"
```

---

## Manual Patches

### tfjs-graph-converter NumPy Compatibility Fix

**Why needed:** tfjs-graph-converter was written for NumPy 1.x, which used `np.bool`. NumPy 1.20+ removed this alias and requires `np.bool_`.

**Location:**
- Mac: `venv/lib/python3.11/site-packages/tfjs_graph_converter/util.py`
- PC: `venv\Lib\site-packages\tfjs_graph_converter\util.py`

**Fix:**

Find line 30:
```python
return np.bool
```

Change to:
```python
return np.bool_
```

**How to verify patch was applied:**

```python
python -c "from tfjs_graph_converter import util; print('Patch OK')"
```

If you see `AttributeError: module 'numpy' has no attribute 'bool'`, the patch was not applied.

---

## Verify Installation

### Test 1: Basic Imports

```bash
python -c "
import torch
import tensorflow as tf
import mediapipe as mp
import cv2
import numpy as np
import trimesh
import websockets
print('✓ All core imports successful')
"
```

### Test 2: GPU Detection

**Mac:**
```bash
python -c "
import torch
print(f'MPS (Mac GPU): {torch.backends.mps.is_available()}')
"
```

**PC:**
```bash
python -c "
import torch
print(f'CUDA: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'GPU: {torch.cuda.get_device_name(0)}')
    print(f'VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB')
"
```

### Test 3: Run Basic Test

**Mac (OAK-D):**
```bash
python tests/test_oakd_blazepose_basic.py
```

**PC (RealSense):**
```bash
python real_sense_test/test_realsense_basic.py
```

---

## Troubleshooting

### NumPy Version Conflicts

**Error:** `ImportError: numpy.core.multiarray failed to import`

**Fix:**
```bash
pip uninstall numpy -y
pip install numpy==1.23.5
```

### TensorFlow Import Errors (Mac)

**Error:** `ImportError: DLL load failed` or `Symbol not found`

**Fix:**
1. Uninstall TensorFlow:
   ```bash
   pip uninstall tensorflow tensorflow-macos tensorflow-metal -y
   ```

2. Reinstall in correct order:
   ```bash
   pip install tensorflow-macos==2.15.0
   pip install tensorflow-metal==1.1.0
   ```

### PyTorch CUDA Not Detected (PC)

**Error:** `torch.cuda.is_available()` returns `False`

**Fixes:**

1. Verify CUDA installation:
   ```cmd
   nvcc --version
   nvidia-smi
   ```

2. Reinstall PyTorch with explicit CUDA version:
   ```cmd
   pip uninstall torch torchvision torchaudio -y
   pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
   ```

3. Check environment variables:
   - `CUDA_PATH` should point to `C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v11.8`
   - Add to PATH: `%CUDA_PATH%\bin`

### BodyPix/tfjs-graph-converter Errors

**Error:** `AttributeError: module 'numpy' has no attribute 'bool'`

**Fix:** Apply manual patch (see [Manual Patches](#manual-patches))

**Error:** `ModuleNotFoundError: No module named 'tfjs_graph_converter.api'`

**Fix:**
```bash
pip uninstall tfjs-graph-converter -y
pip install tfjs-graph-converter==1.3.0
# Then apply manual patch again
```

### RealSense Camera Not Detected (PC)

**Error:** `RuntimeError: No device connected`

**Fixes:**

1. Check USB connection (USB 3.0 required)
2. Run `rs-enumerate-devices` to verify camera is detected
3. Reinstall RealSense SDK
4. Check Device Manager → Imaging Devices → Intel RealSense should be listed

### Out of Memory Errors

**Error:** `RuntimeError: CUDA out of memory` or `MPS out of memory`

**Fixes:**

1. Reduce batch size in AI generation (`num_inference_steps`)
2. Lower resolution (640x480 instead of 1280x720)
3. Close other GPU-intensive apps
4. For Stable Diffusion, use:
   ```python
   generator.num_inference_steps = 10  # Lower from 50
   ```

---

## Next Steps

After successful installation:

1. **Test the pipeline:**
   ```bash
   # Mac
   python tests/test_a_rigged_clothing.py

   # PC
   python real_sense_test/test_realsense_rigged_clothing_pc.py
   ```

2. **Generate clothing:**
   ```bash
   # Mac
   python tests/day2_oakd_sd_integration.py

   # PC
   python real_sense_test/test_realsense_sd_integration_pc.py
   ```

3. **Read the documentation:**
   - `CLAUDE.md` - Project overview and architecture
   - `real_sense_test/README.md` - RealSense testing guide
   - `docs/` - Detailed technical documentation

---

## Getting Help

If you encounter issues not covered here:

1. Check existing issues: https://github.com/yourusername/spoken_wardrobe_2/issues
2. Create new issue with:
   - Platform (Mac/PC)
   - Python version (`python --version`)
   - Error message (full traceback)
   - Steps to reproduce

---

**Last Updated:** 2025-01-06
