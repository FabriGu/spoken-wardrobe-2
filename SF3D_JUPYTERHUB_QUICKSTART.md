# SF3D JupyterHub API - Quick Start Guide

## Why This Approach?

ComfyUI-3D-Pack installation failed due to **CUDA version mismatch**:
- School GPU has CUDA 11.8
- ComfyUI's PyTorch was compiled with CUDA 12.8
- You don't have permissions to rebuild the environment

**Solution**: Run your own SF3D API server in JupyterHub where you have full control!

## Architecture

```
Your Mac                          School GPU (JupyterHub)
┌────────────────┐               ┌──────────────────────┐
│ Image          │   Upload      │ FastAPI Server       │
│ ↓              │──────────>    │ (Running in notebook)│
│ API Request    │   HTTP POST   │ ↓                    │
│ ↓              │<────────      │ SF3D Model           │
│ 3D Mesh (GLB)  │   Download    │ ↓                    │
└────────────────┘               │ Generated Mesh       │
                                 └──────────────────────┘
```

**Benefits**:
- ✅ You control the environment (install whatever you need)
- ✅ Uses school GPU (same 2x RTX 6000 Ada)
- ✅ Simple FastAPI endpoint (same pattern as ComfyUI)
- ✅ No permission issues
- ✅ Can use any SF3D version/implementation

## Setup Steps

### Step 1: Upload Notebook to JupyterHub

1. Log into JupyterHub: `http://itp-ml.itp.tsoa.nyu.edu` (or wherever your JupyterHub is)
2. Upload `sf3d_api_server.ipynb` from this project
3. Open the notebook

### Step 2: Install Dependencies (First Time Only)

Run **Cell 1** in the notebook to install:
```python
!pip install -q fastapi uvicorn[standard] python-multipart
!pip install -q sf3d
!pip install -q torch torchvision --index-url https://download.pytorch.org/whl/cu118
!pip install -q pillow numpy trimesh
```

This installs PyTorch with **CUDA 11.8** (matching the server) in YOUR environment.

### Step 3: Load Model

Run **Cells 2-3** to:
- Import libraries
- Load SF3D model onto GPU
- Create output directory

Expected output:
```
Using device: cuda
Loading SF3D model...
✅ SF3D model loaded in 5.2s
GPU: NVIDIA RTX 6000 Ada Generation
```

### Step 4: Start Server

Run **Cell 5** to start the FastAPI server:
```python
PORT = 8765  # Change if needed
uvicorn.run(app, host="0.0.0.0", port=PORT)
```

**IMPORTANT**:
- This cell will run indefinitely
- Keep the notebook open
- You'll see server logs here

Expected output:
```
🚀 Starting SF3D API Server
======================================================================
Server URL: http://itp-ml.itp.tsoa.nyu.edu:8765/
Device: cuda

Available endpoints:
  GET  /           - API info
  GET  /health     - Health check
  POST /generate   - Generate 3D mesh

⚠️  KEEP THIS CELL RUNNING - Press ■ to stop server
======================================================================

INFO:     Started server process
INFO:     Uvicorn running on http://0.0.0.0:8765
```

### Step 5: Test from Your Mac

Open a terminal on your Mac and test:

```bash
# Quick health check
curl http://itp-ml.itp.tsoa.nyu.edu:8765/health

# Generate mesh from image
python tests/sf3d_api_client.py comfyui_generated_images/1762650549/generated_clothing.png
```

Expected output:
```
======================================================================
SF3D API Client - Image to 3D Mesh
======================================================================
Image: comfyui_generated_images/1762650549/generated_clothing.png
Server: http://itp-ml.itp.tsoa.nyu.edu:8765
Output: sf3d_api_output.glb

Settings:
  Texture Resolution: 1024
  Remesh Option: none
  Foreground Ratio: 0.85
======================================================================

🌐 Testing server connection...
   ✓ Server is healthy
   Device: cuda
   CUDA available: True

📤 Uploading image...

🚀 Generating 3D mesh...
   (This may take 5-30 seconds on GPU)

✅ Generation completed!
   Server processing time: 1.8s
   Total time (with upload): 2.3s

📥 Mesh saved!
   Path: /Users/you/Projects/spoken_wardrobe_2/sf3d_api_output.glb
   Size: 1523.4 KB

🎯 Next Steps:
   1. View in Blender
   2. View online: https://gltf-viewer.donmccurdy.com/
   3. Use with rigged clothing: python tests/test_a_rigged_clothing.py
======================================================================
```

## Client Script Options

```bash
# Basic usage
python tests/sf3d_api_client.py <image_path>

# Custom server URL (if you changed the port)
python tests/sf3d_api_client.py image.png --server http://itp-ml.itp.tsoa.nyu.edu:9000

# High resolution texture
python tests/sf3d_api_client.py image.png --texture-resolution 2048

# With remeshing for animation
python tests/sf3d_api_client.py image.png --remesh quad

# Custom foreground ratio
python tests/sf3d_api_client.py image.png --foreground-ratio 0.75

# Custom output filename
python tests/sf3d_api_client.py image.png --output my_clothing.glb
```

## Troubleshooting

### "Cannot connect to server"

**Possible causes**:
1. Not on NYU network/VPN
2. Notebook cell stopped running
3. Wrong port number

**Fix**:
```bash
# Test from Mac terminal:
curl http://itp-ml.itp.tsoa.nyu.edu:8765/health

# If this fails, check:
1. Are you on NYU network/VPN?
2. Is the JupyterHub notebook cell still running?
3. Check the port number in the notebook output
```

### "Port already in use"

**Fix**: Change the port in Cell 5 of the notebook:
```python
PORT = 8766  # Try different port
```

### "SF3D model not found"

**Fix**: The notebook has a fallback to use transformers pipeline. If both fail:
```python
# In notebook cell 1, try:
!pip install git+https://github.com/Stability-AI/stable-fast-3d.git
```

### "CUDA out of memory"

**Fix**: Reduce texture resolution:
```bash
python tests/sf3d_api_client.py image.png --texture-resolution 512
```

### Server is slow (>10 seconds per mesh)

**Possible causes**:
1. First run (model is loading)
2. GPU is being used by someone else
3. Image is very large

**Fix**:
- First generation is always slower (model loading)
- Subsequent generations should be 0.5-2 seconds
- Try smaller texture resolution: `--texture-resolution 512`

## Comparison: ComfyUI vs JupyterHub API

| Feature | ComfyUI-3D-Pack | JupyterHub API |
|---------|----------------|----------------|
| **Installation** | ❌ Failed (CUDA mismatch) | ✅ Full control |
| **Permissions** | ❌ Need `itp` user access | ✅ Your environment |
| **Speed** | 🚀 0.5-2s (if it worked) | 🚀 0.5-2s |
| **GPU** | ✅ Same RTX 6000 Ada | ✅ Same RTX 6000 Ada |
| **Maintenance** | ❌ Can't update | ✅ You control |
| **Dependencies** | ❌ Fixed by admin | ✅ Install what you need |
| **API** | ComfyUI workflow JSON | FastAPI (simpler) |

## Next Steps

### Integration into `speech_to_clothing.py`

Once this works, you can integrate into your main pipeline:

```python
import requests

def generate_3d_mesh_remote(image_path, output_path, server_url="http://itp-ml.itp.tsoa.nyu.edu:8765"):
    """Generate 3D mesh using remote SF3D API."""
    with open(image_path, 'rb') as f:
        files = {'file': (image_path.name, f, 'image/png')}
        data = {
            'texture_resolution': 1024,
            'remesh_option': 'none',
            'foreground_ratio': 0.85
        }

        response = requests.post(
            f"{server_url}/generate",
            files=files,
            data=data,
            timeout=120
        )

        if response.status_code == 200:
            with open(output_path, 'wb') as out_f:
                out_f.write(response.content)
            return True
    return False

# Use in pipeline:
if result_image:
    print("\n🎲 Generating 3D mesh...")
    mesh_path = output_dir / "clothing_mesh.glb"

    if generate_3d_mesh_remote(result_image, mesh_path):
        print(f"✅ 3D mesh: {mesh_path}")
```

### Keep Server Running

**Option 1: During active development**
- Keep JupyterHub browser tab open
- Server stays running as long as notebook is running

**Option 2: Background process (advanced)**
- Use `nohup` or `screen` in JupyterHub terminal
- Requires starting uvicorn from command line instead of notebook

**Option 3: Systemd service (requires IT)**
- Ask IT to set up as a permanent service
- Only needed if you want 24/7 availability

## Files Created

```
sf3d_api_server.ipynb          # JupyterHub notebook (upload this)
tests/sf3d_api_client.py       # Client script (run from Mac)
SF3D_JUPYTERHUB_QUICKSTART.md  # This guide
```

## Summary

**Total pipeline time** (Speech → 3D Mesh):
- Speech recognition: ~10s
- 2D generation (ComfyUI): 3-5s
- 3D generation (SF3D API): 0.5-2s
- **Total: ~15 seconds** 🚀

**vs Local (Mac CPU)**:
- Speech recognition: ~10s
- 2D generation (ComfyUI): 3-5s
- 3D generation (Local SF3D): **50s**
- **Total: ~65 seconds**

**4x faster with remote GPU!**

---

**Status**: ✅ Ready to test
**Created**: November 10, 2025
**Next**: Upload notebook to JupyterHub and run test
