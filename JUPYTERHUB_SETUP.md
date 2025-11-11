# JupyterHub 3D API Setup - Step by Step

## ⚠️ IMPORTANT: Use Version 3

**Upload `sf3d_api_server_v3.ipynb`** - This uses **real SF3D** from Stability AI, not TripoSR!

## The Problem

ComfyUI-3D-Pack won't install because of CUDA version mismatch. You need to run your own server.

## The Solution

Run an **SF3D API server** in JupyterHub (where you have full control).

**Why SF3D?**
- ✅ UV-mapped textures (better than TripoSR)
- ✅ Higher quality meshes
- ✅ Proper texture baking
- ✅ Same speed as TripoSR on GPU (0.5-2s)

---

## Step-by-Step Instructions

### 1. Upload Notebook to JupyterHub

1. Download `sf3d_api_server_v3.ipynb` from this project
2. Go to your JupyterHub: http://itp-ml.itp.tsoa.nyu.edu (or your school's URL)
3. Click "Upload" button (top right)
4. Select `sf3d_api_server_v3.ipynb`
5. Click "Upload" to confirm

### 2. Open the Notebook

- Click on `sf3d_api_server_v3.ipynb` in the file list
- It will open in a new tab

### 3. Run Installation (Cell 1)

**In the notebook:**
- Click on **Cell 1** (says "Install Dependencies")
- Click **Run** button (▶️) or press `Shift+Enter`
- Wait for it to finish (30-60 seconds)
- You'll see output like:
```
Successfully installed fastapi-0.104.1 uvicorn-0.24.0 ...
✅ Installation complete!
⚠️  IMPORTANT: Now go to Kernel → Restart Kernel
```

### 4. Restart Kernel

**CRITICAL STEP:**
- Go to menu: **Kernel → Restart Kernel**
- Click "Restart" when asked to confirm
- This reloads the newly installed packages

### 5. Run Setup Cells (2, 3, 4)

**Cell 2: Clone SF3D Repository**
- Click on Cell 2
- Press `Shift+Enter`
- Clones SF3D repository from GitHub (one-time, ~10 seconds)
- Should see: `✅ SF3D cloned to: /home/jovyan/stable-fast-3d`

**Cell 3: Load SF3D Model**
- Click on Cell 3
- Press `Shift+Enter`
- Shows GPU info
- **First time**: Downloads ~2GB SF3D model (30-60 seconds)
- Should see: `✅ SF3D model loaded in X.Xs`
- Should see: `Model: stabilityai/stable-fast-3d`
- Should see: `GPU: NVIDIA RTX 6000 Ada Generation`

**Cell 4: Define API Endpoints**
- Click on Cell 4
- Press `Shift+Enter`
- Should see: `✅ API endpoints defined`

### 6. Start Server (Cell 5)

**Cell 5: Start Server**
- Click on Cell 5
- Press `Shift+Enter`
- You'll see:
```
======================================================================
🚀 Starting Stable Fast 3D API Server
======================================================================
Server URL: http://itp-ml.itp.tsoa.nyu.edu:8765/
Device: cuda
Model: SF3D (stabilityai/stable-fast-3d)

Features:
  ✅ UV-mapped textures
  ✅ High-quality meshes
  ✅ Fast GPU inference (0.5-2s)

Endpoints:
  GET  http://itp-ml.itp.tsoa.nyu.edu:8765/health
  POST http://itp-ml.itp.tsoa.nyu.edu:8765/generate
======================================================================

⚠️  KEEP THIS CELL RUNNING

INFO:     Started server process
INFO:     Uvicorn running on http://0.0.0.0:8765
INFO:     Application startup complete.
```

**IMPORTANT**: Keep this notebook tab open! The server only runs while this cell is running.

### 7. Test from Your Mac

Open a terminal on your Mac:

```bash
# Make sure you're on NYU network/VPN first!

# Quick health check
curl http://itp-ml.itp.tsoa.nyu.edu:8765/health

# Should return:
# {"status":"healthy","device":"cuda","cuda_available":true,"model_loaded":true}
```

If health check works, try generating a mesh:

```bash
# Use one of your generated clothing images
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

🌐 Testing server connection...
   ✓ Server is healthy
   Device: cuda
   CUDA available: True

📤 Uploading image...

🚀 Generating 3D mesh...
   (This may take 5-30 seconds on GPU)

✅ Generation completed!
   Server processing time: 2.1s
   Total time (with upload): 2.5s

📥 Mesh saved!
   Path: /Users/you/Projects/spoken_wardrobe_2/sf3d_api_output.glb
   Size: 1823.4 KB
======================================================================
```

---

## Common Issues

### Issue: "Module not found" in Cell 2

**Cause**: Didn't restart kernel after Cell 1

**Fix**:
1. Go to: Kernel → Restart Kernel
2. Run Cell 2 again

### Issue: "Port already in use" in Cell 5

**Cause**: Another notebook is using port 8765

**Fix**:
1. In Cell 5, change: `PORT = 8765` to `PORT = 8766`
2. Re-run Cell 5 only
3. Update your client command: `--server http://itp-ml.itp.tsoa.nyu.edu:8766`

### Issue: "Cannot connect to server" from Mac

**Causes**:
1. Not on NYU network/VPN
2. Wrong server URL
3. Notebook cell stopped running

**Fix**:
1. Connect to NYU VPN
2. Check JupyterHub - is Cell 5 still running?
3. Try curl test: `curl http://itp-ml.itp.tsoa.nyu.edu:8765/health`

### Issue: "CUDA out of memory"

**Cause**: Someone else is using the GPU

**Fix**: Try again later, or ask in your class Slack who's using the GPU

---

## Stopping the Server

When you're done:

1. Go to JupyterHub notebook
2. Click the **■ Stop** button (top toolbar)
3. Or: **Kernel → Interrupt Kernel**

The server will stop and the cell will finish.

---

## Files

**Use this version (Real SF3D):**
- ✅ `sf3d_api_server_v3.ipynb` - **USE THIS** - Real SF3D from Stability AI
- ✅ `tests/sf3d_api_client.py` - Client script (run from your Mac)
- ✅ `JUPYTERHUB_SETUP.md` - This guide

**Old files (don't use):**
- ❌ `sf3d_api_server.ipynb` - First version (had issues)
- ❌ `sf3d_api_server_v2.ipynb` - Used TripoSR instead of SF3D
- ❌ `tests/comfyui_sf3d_test0.py` - ComfyUI version (won't work due to CUDA)

---

## Next Steps

Once this works:

1. **Test with different images** - Try various clothing items
2. **Integrate into pipeline** - Add to `speech_to_clothing.py`
3. **Keep server running** - During active development, keep notebook open

**Total pipeline time**: Speech (10s) + 2D image (3-5s) + 3D mesh (2s) = **~15 seconds** 🚀

---

**Quick Reference:**

| Step | Command |
|------|---------|
| Upload | Use JupyterHub upload button |
| Install | Run Cell 1, then Kernel → Restart |
| Setup | Run Cells 2, 3, 4 in order |
| Start | Run Cell 5 (keep running) |
| Test | `python tests/sf3d_api_client.py <image>` |
| Stop | Click ■ Stop button |
