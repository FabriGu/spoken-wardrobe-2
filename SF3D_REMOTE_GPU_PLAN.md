# SF3D Remote GPU Integration Plan

## 🚨 UPDATE: ComfyUI-3D-Pack Installation Failed

**Problem**: ComfyUI-3D-Pack cannot be installed due to **CUDA version mismatch**:
- School GPU has: CUDA 11.8
- ComfyUI PyTorch compiled with: CUDA 12.8
- You don't have permissions to rebuild the environment

**Solution**: Use **JupyterHub FastAPI Server** instead (Option E below)

## ✅ RECOMMENDED: JupyterHub FastAPI Server

Run your own SF3D API server in JupyterHub where you have full control!

**Quick Start** (3 steps):
1. Upload `sf3d_api_server.ipynb` to JupyterHub
2. Run all cells to start server
3. Test from Mac: `python tests/sf3d_api_client.py <image_path>`

**Complete guide**: See `SF3D_JUPYTERHUB_QUICKSTART.md`

**Why this works**:
- ✅ You control the environment (no permission issues)
- ✅ Install PyTorch with CUDA 11.8 (matches server)
- ✅ Same GPU performance (RTX 6000 Ada)
- ✅ Simple FastAPI endpoint (like ComfyUI)

**Files created**:
- ✅ `sf3d_api_server.ipynb` - JupyterHub notebook
- ✅ `tests/sf3d_api_client.py` - Client script
- ✅ `SF3D_JUPYTERHUB_QUICKSTART.md` - Setup guide

---

## ❌ ComfyUI-3D-Pack Attempt (Failed)

We tried using ComfyUI-3D-Pack (pre-installed on school GPU) but installation failed:

**Error**:
```
RuntimeError: The detected CUDA version (11.8) mismatches the version
that was used to compile PyTorch (12.8)
```

**Files created** (for reference, not usable):
- `workflows/sf3d_generation_api_correct.json` - SF3D workflow template
- `tests/comfyui_sf3d_test0.py` - Test script (won't work until CUDA fixed)

**To use these**: Would need IT to rebuild PyTorch environment with CUDA 11.8

---

## Current Situation

### What You Have Working
1. ✅ **Speech-to-Clothing Pipeline** (src/modules/speech_to_clothing.py)
   - Whisper speech recognition
   - OAK-D Pro body detection
   - BodyPix segmentation
   - ComfyUI remote GPU generation (3-10 seconds)
   - Organized output saving

2. ✅ **SF3D Local** (tests/sf3d_test_1_basic.py)
   - Working on Mac CPU (~50 seconds)
   - Generates textured GLB meshes
   - UV-unwrapped textures (better than TripoSR)

### What You Need
- **Real-time 3D mesh generation** from clothing images
- Use school GPU (2x RTX 6000 Ada, 48GB VRAM each)
- Integrate into existing pipeline
- **Target: 0.5-2 seconds** generation time (vs 50s on Mac)

### School GPU Options Available
1. **JupyterHub** - Interactive notebooks in browser
2. **Energy Monitoring** - Dashboard (not relevant)
3. **Ollama** - LLM inference API (not relevant)

---

## Option Analysis

### Option F: ComfyUI-3D-Pack (Direct Integration) ⭐⭐ NEW BEST OPTION

**Description**: Use the SF3D nodes you already have installed in ComfyUI via ComfyUI-3D-Pack.

**Architecture**:
```
[Your Mac]                      [School GPU Machine]
┌─────────────────┐            ┌──────────────────────┐
│ speech_to_      │   HTTP     │ ComfyUI Server       │
│ clothing.py     │──────────> │ - Already running    │
│                 │            │ - Has SF3D installed │
│ comfyui_client  │            │ - Same endpoint!     │
│                 │<────────── │                      │
└─────────────────┘  GLB mesh  └──────────────────────┘
```

**Pros**:
- ✅ ✅ **ALREADY INSTALLED**: ComfyUI-3D-Pack is working on school GPU
- ✅ ✅ **NO DEPLOYMENT NEEDED**: Uses existing ComfyUI infrastructure
- ✅ ✅ **SAME CLIENT CODE**: Reuse comfyui_client.py completely
- ✅ Fast: 0.5-2 seconds on GPU
- ✅ Proven: ComfyUI is already working for 2D generation
- ✅ Unified: Single server for both 2D and 3D

**Cons**:
- ⚠️ Depends on ComfyUI-3D-Pack being maintained
- ⚠️ Slightly less control than dedicated server

**Implementation Complexity**: LOW (just add workflow JSON + test script)

**Best For**: ✅ YOUR USE CASE - Already installed and working!

---

### Option A: REST API Service ⭐ ORIGINAL RECOMMENDATION

**Description**: Deploy SF3D as a persistent REST API service on the GPU machine, similar to your current ComfyUI setup.

**Architecture**:
```
[Your Mac]                      [School GPU Machine]
┌─────────────────┐            ┌──────────────────────┐
│ speech_to_      │   HTTP     │ FastAPI Server       │
│ clothing.py     │──────────> │ - Receives image     │
│                 │            │ - Runs SF3D          │
│ sf3d_client.py  │            │ - Returns GLB mesh   │
│                 │<────────── │                      │
└─────────────────┘  GLB file  └──────────────────────┘
```

**Pros**:
- ✅ **Fast**: 0.5-2 seconds on GPU (vs 50s on CPU)
- ✅ **Always available**: Persistent service like ComfyUI
- ✅ **Easy integration**: Reuse ComfyUI client pattern
- ✅ **Multiple users**: Can handle concurrent requests
- ✅ **Familiar pattern**: You already did this with ComfyUI

**Cons**:
- ⚠️ Requires deploying server code to GPU machine
- ⚠️ Need to verify network access/firewall rules
- ⚠️ May need admin help for persistent service

**Implementation Complexity**: Medium (but you've done this before with ComfyUI)

**Best For**: Production pipeline, real-time generation, multiple users

---

### Option B: JupyterHub Notebook

**Description**: Run SF3D in a Jupyter notebook on the GPU machine, save outputs to shared folder.

**Architecture**:
```
[Your Mac]                      [School GPU Machine]
┌─────────────────┐            ┌──────────────────────┐
│ speech_to_      │   Upload   │ JupyterHub Notebook  │
│ clothing.py     │──────────> │ - Manual execution   │
│                 │            │ - Saves to folder    │
│ (poll folder)   │<────────── │                      │
└─────────────────┘  Download  └──────────────────────┘
```

**Pros**:
- ✅ No server deployment needed
- ✅ Already available (JupyterHub exists)
- ✅ Good for testing/prototyping

**Cons**:
- ❌ **Not real-time**: Manual execution required
- ❌ **Not automated**: Can't integrate into pipeline
- ❌ **Polling required**: Need to watch for file changes
- ❌ **Inefficient**: File transfer overhead

**Implementation Complexity**: Low (but limited functionality)

**Best For**: One-off tests, prototyping, development

---

### Option C: SSH + Direct Execution

**Description**: SSH into GPU machine, run SF3D script remotely, transfer files back.

**Architecture**:
```
[Your Mac]                      [School GPU Machine]
┌─────────────────┐   SSH      ┌──────────────────────┐
│ speech_to_      │──────────> │ python sf3d_test.py  │
│ clothing.py     │   scp/rsync│ - Runs on command    │
│                 │<────────── │                      │
└─────────────────┘  GLB file  └──────────────────────┘
```

**Pros**:
- ✅ Direct control
- ✅ Can automate with SSH commands
- ✅ No persistent server needed

**Cons**:
- ⚠️ Need SSH access (may not be available)
- ⚠️ File transfer overhead
- ⚠️ Less elegant than REST API
- ⚠️ Harder to handle errors

**Implementation Complexity**: Medium

**Best For**: If SSH is available and REST API not allowed

---

### Option D: Gradio/Streamlit Web Interface

**Description**: Deploy a web UI on GPU machine for interactive use.

**Pros**:
- ✅ User-friendly interface
- ✅ Good for demos

**Cons**:
- ❌ **Not suitable for pipeline automation**
- ❌ Harder to integrate programmatically
- ❌ Overkill for API use

**Best For**: Demos, manual testing, not for production pipeline

---

## Recommendation: Option A (REST API Service)

### Why This is Best for Your Use Case

1. **Mirrors ComfyUI Success**: You already have working ComfyUI remote GPU integration. This uses the exact same pattern.

2. **Real-Time Performance**:
   - Current: 50 seconds on Mac CPU
   - Target: 0.5-2 seconds on GPU (25-100x faster!)

3. **Pipeline Integration**:
   ```python
   # Current pipeline
   speech → ComfyUI (3s) → 2D clothing image

   # Extended pipeline
   speech → ComfyUI (3s) → 2D clothing image → SF3D (2s) → 3D mesh
   ```

4. **Scalability**: Can be used by multiple people/projects simultaneously

5. **Familiar Implementation**: Reuse patterns from `comfyui_client.py`

---

## Implementation Plan

### Phase 1: Server Setup (On GPU Machine)

**Files to Create**:

1. **`sf3d_server.py`** - FastAPI server that runs SF3D
   - Endpoint: `POST /generate` (accepts image, returns GLB)
   - Endpoint: `GET /health` (check if server is running)
   - Handles image upload, preprocessing, SF3D generation
   - Returns GLB file as binary response

2. **`requirements_sf3d_server.txt`** - Server dependencies
   - fastapi
   - uvicorn
   - sf3d dependencies (already installed from yesterday)

3. **`start_sf3d_server.sh`** - Startup script
   - Activates environment
   - Starts uvicorn server on specific port (e.g., 9299)

**Server Architecture**:
```python
# Simplified structure
@app.post("/generate")
async def generate_mesh(
    image: UploadFile,
    texture_resolution: int = 1024,
    remesh_option: str = "none"
):
    # 1. Save uploaded image
    # 2. Preprocess (remove background, resize)
    # 3. Run SF3D model
    # 4. Return GLB file
    return FileResponse(glb_path)
```

**Deployment Steps**:
1. Transfer server files to GPU machine (via JupyterHub upload or SCP)
2. SSH into machine (or use terminal from JupyterHub)
3. Install dependencies
4. Start server: `uvicorn sf3d_server:app --host 0.0.0.0 --port 9299`
5. Note the server URL (e.g., `http://itp-ml.itp.tsoa.nyu.edu:9299`)

---

### Phase 2: Client Integration (On Your Mac)

**Files to Create**:

1. **`src/modules/sf3d_client.py`** - Client for SF3D server
   - Similar structure to `comfyui_client.py`
   - Methods:
     - `test_connection()` - Check server availability
     - `generate_mesh(image)` - Send image, receive GLB
     - `generate_from_file(path)` - Convenience method

2. **`tests/test_sf3d_integration.py`** - Test script
   - Test connection
   - Test mesh generation
   - Verify GLB output

**Client Architecture**:
```python
class SF3DClient:
    def __init__(self, base_url):
        self.base_url = base_url

    def generate_mesh(self, image_array, texture_resolution=1024):
        # 1. Convert numpy array to PNG bytes
        # 2. POST to /generate endpoint
        # 3. Receive GLB binary data
        # 4. Save to file
        # 5. Return path to GLB
```

**Integration Points**:
```python
# In speech_to_clothing.py, after ComfyUI generation:

# Step 10: Generate 3D mesh (NEW)
if result_image:
    print("\n🎲 Generating 3D mesh with SF3D...")

    sf3d_client = SF3DClient("http://itp-ml.itp.tsoa.nyu.edu:9299")
    mesh_path = sf3d_client.generate_mesh(
        np.array(result_image),
        texture_resolution=1024
    )

    if mesh_path:
        print(f"✅ 3D mesh generated: {mesh_path}")
        # Save mesh_path to metadata.json
```

---

### Phase 3: Testing & Validation

**Test Sequence**:

1. **Server Health Check**:
   ```bash
   curl http://itp-ml.itp.tsoa.nyu.edu:9299/health
   # Expected: {"status": "ok", "gpu": "cuda", "model_loaded": true}
   ```

2. **Simple Generation Test**:
   ```python
   python tests/test_sf3d_integration.py \
       --image generated_images/clothing_123.png \
       --server http://itp-ml.itp.tsoa.nyu.edu:9299
   ```

3. **Full Pipeline Test**:
   ```bash
   python src/modules/speech_to_clothing.py
   # Should generate: 2D image + 3D mesh
   ```

**Expected Output Structure**:
```
comfyui_generated_images/1762650549/
├── original_frame.png
├── mask.png
├── generated_clothing.png
├── mesh.glb              # NEW: 3D mesh
└── metadata.json         # Updated with mesh info
```

**Performance Targets**:
- ComfyUI generation: 3-10 seconds (already working)
- SF3D generation: 0.5-2 seconds (new)
- **Total pipeline: 13-15 seconds** (speech → 3D mesh)

---

## Alternative: Hybrid Approach

If REST API deployment is complex, consider this compromise:

**Option E: JupyterHub + REST API**

1. Run FastAPI server **inside** a JupyterHub notebook
2. Use `!uvicorn` magic command to start server
3. Keep notebook running (don't close browser tab)
4. Access from your Mac via network

**Pros**:
- No permanent deployment needed
- Still get REST API benefits
- Easy to start/stop

**Cons**:
- Need to keep notebook tab open
- Restarts when browser closes
- Less robust than dedicated service

**Implementation**:
```python
# In JupyterHub notebook cell:
!pip install fastapi uvicorn
!uvicorn sf3d_server:app --host 0.0.0.0 --port 9299 &
```

---

## Questions to Answer Before Implementation

### School IT/Permissions:

1. **Network Access**:
   - Can you access the GPU machine from outside the school network?
   - Is there a firewall? What ports are open?
   - Is VPN required?

2. **Deployment Permissions**:
   - Can you run persistent services (e.g., keep a server running)?
   - Can you install Python packages (pip install)?
   - Do you have SSH access, or only JupyterHub?

3. **Resource Limits**:
   - Are there time limits for running processes?
   - GPU usage quotas or restrictions?
   - Disk space for storing models and outputs?

### Technical Details:

4. **Server URL Pattern**:
   - What's the pattern? (e.g., `http://itp-ml.itp.tsoa.nyu.edu:PORT`)
   - Can you choose port numbers?
   - Is there a proxy/load balancer?

5. **ComfyUI Setup** (for reference):
   - How is ComfyUI deployed on the same machine?
   - Can we mirror that setup for SF3D?

---

## Recommended Next Steps

### Step 1: Information Gathering (Do This First)

Contact school IT or check documentation:
- [ ] Verify network access (can you reach GPU machine from Mac?)
- [ ] Check if you can run persistent services
- [ ] Confirm available ports
- [ ] Ask about ComfyUI deployment (how was it set up?)

### Step 2: Choose Deployment Method

Based on answers:
- **Full permissions** → Option A (REST API service)
- **Limited permissions** → Option E (JupyterHub + temporary API)
- **Very limited** → Option B (JupyterHub notebook only)

### Step 3: Implementation

Once you choose approach:
1. I'll create all necessary code files
2. You deploy server to GPU machine
3. We test connection and generation
4. Integrate into speech_to_clothing.py pipeline

---

## Success Criteria

### Minimum Viable Product (MVP):
- ✅ SF3D runs on GPU machine (0.5-2s generation)
- ✅ Can send image from Mac, receive GLB mesh
- ✅ Basic error handling

### Full Integration:
- ✅ MVP +
- ✅ Integrated into speech_to_clothing.py
- ✅ Organized output (mesh saved with 2D images)
- ✅ Metadata tracking (settings, timing)
- ✅ Comprehensive error handling

### Stretch Goals:
- ⭐ Multiple texture resolutions (fast preview vs high quality)
- ⭐ Remeshing options for animation-ready meshes
- ⭐ Batch processing (multiple images → multiple meshes)
- ⭐ Progress updates via WebSocket

---

## Timeline Estimate

**Assuming REST API approach with good permissions**:

- **Phase 1 (Server)**: 2-3 hours
  - Write server code: 1 hour
  - Deploy to GPU machine: 1 hour
  - Test and debug: 1 hour

- **Phase 2 (Client)**: 1-2 hours
  - Write client code: 30 min
  - Write tests: 30 min
  - Integration: 30 min

- **Phase 3 (Testing)**: 1 hour
  - End-to-end testing
  - Performance validation
  - Bug fixes

**Total**: 4-6 hours (assuming no permission roadblocks)

**With limited permissions** (JupyterHub only): Add 1-2 hours for workarounds

---

## Cost-Benefit Analysis

### Current (Local SF3D):
- Generation time: 50 seconds
- Quality: Good (same model)
- Cost: Free
- Availability: Always

### Proposed (Remote GPU):
- Generation time: 0.5-2 seconds (25-100x faster!)
- Quality: Good (same model)
- Cost: Free (school resource)
- Availability: When connected to school network/VPN

### Impact on Pipeline:
```
Current: Speech (10s) → ComfyUI (5s) → Total: 15s
With SF3D Local: +50s → Total: 65s (too slow!)
With SF3D Remote: +2s → Total: 17s (acceptable!)
```

**Conclusion**: Remote GPU is essential for real-time experience.

---

## Comparison with ComfyUI Setup

Since you already have ComfyUI working on the same machine, let's mirror that:

### ComfyUI (Working):
```
Server: http://itp-ml.itp.tsoa.nyu.edu:9199
Client: src/modules/comfyui_client.py
Protocol: REST API (POST /prompt, GET /history)
```

### SF3D (Proposed):
```
Server: http://itp-ml.itp.tsoa.nyu.edu:9299
Client: src/modules/sf3d_client.py
Protocol: REST API (POST /generate, GET /health)
```

### Reuse Patterns:
- Upload handling (image → bytes)
- Error handling (connection, timeout)
- Result downloading (bytes → file)
- Progress tracking

---

## Final Recommendation

**🎯 PRIMARY: Option F (ComfyUI-3D-Pack) - NEWLY DISCOVERED!**

This is now the BEST approach because:

1. ✅ ✅ **ALREADY INSTALLED**: You have ComfyUI-3D-Pack working on the GPU server
2. ✅ ✅ **ZERO DEPLOYMENT**: No new server setup needed
3. ✅ ✅ **REUSE EXISTING CODE**: Same comfyui_client.py you already have
4. ✅ **Real-time**: 0.5-2 seconds on GPU (same performance as Option A)
5. ✅ **Unified pipeline**: Single server for 2D clothing + 3D mesh
6. ✅ **Complete pipeline**: Speech → 2D → 3D in ~15 seconds total

**Implementation Status**:
- ✅ Workflow JSON created: `workflows/sf3d_generation_api.json`
- ✅ Test script created: `tests/comfyui_sf3d_test0.py`
- 🔄 Ready to test now!

**Fallback: Option A (REST API Service)**

Only if ComfyUI-3D-Pack doesn't work for some reason:

1. ✅ More control over SF3D
2. ✅ Independent from ComfyUI
3. ⚠️ Requires deployment
4. ⚠️ Separate server to maintain

**Not Recommended**: Options B, C, D, E
- Unnecessary now that you have ComfyUI-3D-Pack

---

## What to Do Next

1. **Review this plan** - Check if it makes sense for your needs

2. **Answer permission questions**:
   - Can you run persistent services on the GPU machine?
   - Do you have network access to the machine?
   - How is ComfyUI deployed? Can you mirror that?

3. **Choose deployment approach**:
   - Option A (REST API) - if you have permissions
   - Option E (JupyterHub temp API) - if limited permissions

4. **Let me know** and I'll:
   - Create all server code
   - Create all client code
   - Write deployment instructions
   - Integrate into your pipeline

---

**Created**: November 8, 2025
**Status**: 📋 PLAN - Awaiting review and permission verification
**Next**: Gather school IT information, choose deployment method
