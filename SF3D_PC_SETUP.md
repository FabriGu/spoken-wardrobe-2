# SF3D on PC (NVIDIA GPU) Setup Guide

## Differences from Mac

### 1. GPU Backend
- **Mac**: MPS or CPU
- **PC**: CUDA (NVIDIA GPU)

### 2. Installation Differences

#### Check CUDA Version First
```bash
nvidia-smi
# Look for "CUDA Version: X.X"
```

#### Install PyTorch for CUDA
```bash
# On PC, install PyTorch with CUDA support
# Example for CUDA 12.1:
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# For CUDA 11.8:
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

### 3. Running SF3D on PC

**NO CPU mode needed!**

```bash
# On PC - just run normally (will auto-detect CUDA)
python tests/sf3d_test_1_basic.py image.png

# High quality settings (GPU can handle it)
python tests/sf3d_test_1_basic.py image.png \
    --texture-resolution 2048 \
    --remesh-option quad \
    --target-vertex-count 15000
```

### 4. Performance Expectations

| Setting | Mac CPU | PC GPU (RTX 3060+) |
|---------|---------|-------------------|
| Generation | ~50s | **0.5-2s** |
| Texture Res | 512-1024 | 2048-4096 |
| Memory | Works | 6-8GB VRAM |

### 5. Environment Variables

**Mac**:
```bash
export SF3D_USE_CPU=1  # Required
export PYTORCH_ENABLE_MPS_FALLBACK=1
```

**PC**:
```bash
# No special variables needed!
# Just run the scripts normally
```

### 6. File Transfer

Transfer these to PC:
- `external/stable-fast-3d/` (entire directory)
- `tests/sf3d_test_1_basic.py`
- `tests/sf3d_test_2_viewer.py`
- `tests/sf3d_viewer_textured.html`
- `SF3D_README.md`

### 7. PC Installation Steps

```bash
# 1. Clone/copy your repo to PC
# 2. Create venv
python -m venv venv
source venv/bin/activate  # Linux: source venv/bin/activate
                           # Windows: venv\Scripts\activate

# 3. Install PyTorch for CUDA (check your CUDA version first!)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# 4. Install other requirements
pip install transformers==4.42.3 diffusers huggingface-hub rembg pillow numpy

# 5. Install SF3D packages (NO OpenMP needed on PC)
pip install ./external/stable-fast-3d/texture_baker/
pip install ./external/stable-fast-3d/uv_unwrapper/

# 6. Install SF3D dependencies
cd external/stable-fast-3d
pip install -r requirements.txt
cd ../..

# 7. Login to Hugging Face
huggingface-cli login
```

### 8. Test on PC

```bash
# Should auto-detect CUDA and run FAST
python tests/sf3d_test_1_basic.py generated_meshes/1761618888/generated_clothing.png \
    --texture-resolution 2048 \
    --remesh-option quad
```

Expected output:
```
Device: cuda
✓ Generation completed in 0.8 seconds  # vs 50s on Mac!
Peak GPU Memory: 4,500 MB
```

---

## Key Takeaways

| Aspect | Mac | PC |
|--------|-----|-----|
| Speed | 50s (CPU) | **0.5-2s (GPU)** |
| Quality | Low-medium | **High** |
| Setup | OpenMP + MPS fallback | Simple CUDA install |
| Limitations | Memory issues on MPS | None (if 8GB+ VRAM) |
