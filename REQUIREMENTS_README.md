# Requirements Installation Guide

This project has **platform-specific dependencies** that must be installed in the correct order to avoid conflicts.

---

## Quick Start

### Mac (Apple Silicon)

```bash
# 1. Create and activate virtual environment
python3 -m venv venv
source venv/bin/activate

# 2. Upgrade pip
pip install --upgrade pip

# 3. Install in this order:
pip install -r requirements-core.txt  # Includes patched chumpy from git
pip install -r requirements-mac.txt

# 4. Apply manual patch (CRITICAL!)
sed -i '' 's/np\.bool$/np.bool_/g' venv/lib/python3.11/site-packages/tfjs_graph_converter/util.py

# 5. Install TripoSR
cd external
git clone https://github.com/VAST-AI-Research/TripoSR.git
cd TripoSR && pip install -r requirements.txt && cd ../..

# 6. Verify
python -c "import torch; print(f'MPS: {torch.backends.mps.is_available()}')"
```

### PC (Windows + NVIDIA GPU)

```cmd
:: 1. Install CUDA 11.8 and cuDNN first (see SETUP_GUIDE.md)

:: 2. Create and activate virtual environment
python -m venv venv
venv\Scripts\activate

:: 3. Upgrade pip
python -m pip install --upgrade pip

:: 4. Install core dependencies (includes patched chumpy)
pip install -r requirements-core.txt

:: 5. Install PyTorch with CUDA (SPECIAL COMMAND!)
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

:: 6. Install PC dependencies (excluding PyTorch)
pip install -r requirements-pc.txt

:: 7. Apply manual patch (CRITICAL!)
powershell -Command "(Get-Content venv\Lib\site-packages\tfjs_graph_converter\util.py) -replace 'np\.bool$', 'np.bool_' | Set-Content venv\Lib\site-packages\tfjs_graph_converter\util.py"

:: 8. Install RealSense SDK (download from Intel website)

:: 9. Install TripoSR
cd external
git clone https://github.com/VAST-AI-Research/TripoSR.git
cd TripoSR
pip install -r requirements.txt
cd ..\..

:: 10. Verify
python -c "import torch; print(f'CUDA: {torch.cuda.is_available()}')"
```

---

## File Descriptions

### requirements-core.txt
**Cross-platform core dependencies**

Contains libraries that work identically on Mac and PC:
- NumPy 1.23.5 (CRITICAL version - do not upgrade!)
- OpenCV
- MediaPipe
- Trimesh
- WebSockets
- etc.

### requirements-mac.txt
**Mac-specific (Apple Silicon M1/M2/M3)**

Contains:
- PyTorch with MPS (Metal) support
- tensorflow-macos + tensorflow-metal
- OAK-D Pro dependencies (depthai)
- Stable Diffusion libraries

**Key points:**
- Uses `tensorflow-macos` (NOT regular `tensorflow`)
- Requires manual tfjs-graph-converter patch

### requirements-pc.txt
**PC-specific (Windows + NVIDIA GPU)**

Contains:
- Regular TensorFlow with CUDA support
- Intel RealSense dependencies
- PC-specific versions

**Key points:**
- PyTorch MUST be installed separately (see commands above)
- Requires CUDA Toolkit and cuDNN pre-installed
- Requires manual tfjs-graph-converter patch

### Legacy Files (Deprecated)

- `requirements.txt` - Old core dependencies (use requirements-core.txt instead)
- `requirements-ai.txt` - Old AI dependencies (now split into mac/pc)
- `requirements-oakd.txt` - Now integrated into requirements-mac.txt

---

## Dependency Conflict Notes

Based on testing, these versions are **locked** to avoid conflicts:

### NumPy: 1.23.5
- ✅ Compatible with PyTorch 2.1.0
- ✅ Compatible with TensorFlow 2.15.0
- ❌ NumPy 2.x breaks both PyTorch and TensorFlow
- ❌ NumPy 1.24+ has API changes that break tfjs-graph-converter

### TensorFlow: 2.15.0
- ✅ Works with NumPy 1.23.5
- ❌ TensorFlow 2.19.0 has JAX dependency conflicts
- ✅ Python 3.11 compatible

### ML-dtypes: 0.2.0
- Downgraded from 0.5.3
- Required for NumPy 1.23.5 compatibility

### Protobuf: 4.25.8
- Downgraded from 5.29.5
- Required for TensorFlow 2.15.0

---

## Manual Patches Required

### 1. Chumpy (Installed via Git - Automatic)

**Status:** ✅ Handled automatically by requirements-core.txt

**What:** Patched version of chumpy for getargs compatibility

**Why needed:** The PyPI version (0.70) has a bug: `getargs() got an unexpected keyword argument`

**Solution:** Install from specific git commit that fixes this:
```bash
pip install git+https://github.com/mattloper/chumpy@9b045ff5d6588a24a0bab52c83f032e2ba433e17
```

This is **automatically installed** when you run `pip install -r requirements-core.txt`

**Verify:**
```bash
python -c "import chumpy; print(f'Chumpy version: {chumpy.__version__}')"
```

### 2. tfjs-graph-converter (CRITICAL - Manual Fix Required!)

**File:** `venv/lib/python3.11/site-packages/tfjs_graph_converter/util.py` (Mac)
**File:** `venv\Lib\site-packages\tfjs_graph_converter\util.py` (PC)

**Line 30:**
```python
# CHANGE THIS:
return np.bool

# TO THIS:
return np.bool_
```

**Why:** NumPy 1.20+ deprecated `np.bool` in favor of `np.bool_`. The library hasn't been updated yet.

**Mac command:**
```bash
sed -i '' 's/np\.bool$/np.bool_/g' venv/lib/python3.11/site-packages/tfjs_graph_converter/util.py
```

**PC command:**
```powershell
(Get-Content venv\Lib\site-packages\tfjs_graph_converter\util.py) -replace 'np\.bool$', 'np.bool_' | Set-Content venv\Lib\site-packages\tfjs_graph_converter\util.py
```

---

## Verification Steps

After installation, run these tests:

### 1. Check Core Libraries
```bash
python -c "import numpy; print(f'NumPy: {numpy.__version__}')"
python -c "import cv2; print(f'OpenCV: {cv2.__version__}')"
python -c "import mediapipe; print('MediaPipe OK')"
python -c "import trimesh; print('Trimesh OK')"
```

**Expected:**
```
NumPy: 1.23.5
OpenCV: 4.12.0
MediaPipe OK
Trimesh OK
```

### 2. Check AI Libraries

**Mac:**
```bash
python -c "import torch; print(f'PyTorch: {torch.__version__}, MPS: {torch.backends.mps.is_available()}')"
python -c "import tensorflow as tf; print(f'TensorFlow: {tf.__version__}')"
```

**PC:**
```bash
python -c "import torch; print(f'PyTorch: {torch.__version__}, CUDA: {torch.cuda.is_available()}')"
python -c "import tensorflow as tf; print(f'TensorFlow: {tf.__version__}')"
```

### 3. Check Platform-Specific

**Mac (OAK-D):**
```bash
python -c "import depthai; print('OAK-D OK')"
```

**PC (RealSense):**
```bash
python -c "import pyrealsense2; print('RealSense OK')"
```

### 4. Check Manual Patch Applied
```bash
python -c "from tfjs_graph_converter import util; print('tfjs patch OK')"
```

If this fails with `AttributeError: module 'numpy' has no attribute 'bool'`, the patch was NOT applied.

---

## Common Installation Errors

### Error: "No module named 'torch'"
**Cause:** PyTorch not installed or wrong installation method
**Fix (PC):**
```bash
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

### Error: "numpy.core.multiarray failed to import"
**Cause:** NumPy version mismatch
**Fix:**
```bash
pip uninstall numpy -y
pip install numpy==1.23.5
```

### Error: "AttributeError: module 'numpy' has no attribute 'bool'"
**Cause:** Manual patch not applied
**Fix:** See [Manual Patches](#manual-patches-required) section

### Error: "getargs() got an unexpected keyword argument"
**Cause:** Wrong chumpy version (PyPI instead of patched git version)
**Fix:**
```bash
pip uninstall chumpy -y
pip install git+https://github.com/mattloper/chumpy@9b045ff5d6588a24a0bab52c83f032e2ba433e17
```

### Error: "CUDA not available" (PC)
**Cause:** PyTorch installed without CUDA support
**Fix:**
```bash
pip uninstall torch torchvision torchaudio -y
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

---

## Updating Dependencies

**WARNING:** Do not run `pip install --upgrade` on any package without testing!

The dependency versions are carefully balanced. Upgrading one package may break others.

**Safe updates:**
- Minor version updates (e.g., 1.23.5 → 1.23.6) are usually safe
- Patch updates (e.g., 2.15.0 → 2.15.1) are usually safe

**Risky updates:**
- NumPy 1.x → 2.x (will break everything)
- TensorFlow 2.15 → 2.19+ (JAX conflicts)
- PyTorch major versions

---

## Need Help?

See **SETUP_GUIDE.md** for detailed platform-specific installation instructions.

For troubleshooting, check the [Troubleshooting](#common-installation-errors) section or create a GitHub issue.

---

**Last Updated:** 2025-01-06
