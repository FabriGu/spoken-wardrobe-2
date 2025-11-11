# Speech-to-Clothing with Direct Rodin API Integration

## What's Different

`speech_to_clothing_with_rodin_api.py` uses the **Rodin API directly** instead of going through ComfyUI.

**Benefits**:
- ✅ No ComfyUI server configuration needed
- ✅ Use your own Rodin API key
- ✅ No IT admin required
- ✅ Same pipeline, just different 3D generation method

**Trade-offs**:
- You need a Rodin subscription (same as ComfyUI version)
- API key must be set on your local machine

---

## Quick Start

### 1. Get Rodin API Key

1. Subscribe to Rodin at: https://hyperhuman.deemos.com/
2. Go to **Account → API Keys**
3. Click **"+Create new API Keys"**
4. Copy your key (shown only once!)

### 2. Set API Key

**Option A: Environment Variable (Recommended)**
```bash
export RODIN_API_KEY="your_api_key_here"
```

To make it permanent, add to your `~/.zshrc` or `~/.bash_profile`:
```bash
echo 'export RODIN_API_KEY="your_api_key_here"' >> ~/.zshrc
source ~/.zshrc
```

**Option B: Command Line Argument**
```bash
python src/modules/speech_to_clothing_with_rodin_api.py --api-key "your_api_key_here"
```

### 3. Run Pipeline

**With 3D generation**:
```bash
python src/modules/speech_to_clothing_with_rodin_api.py
```

**Skip 3D (2D only)**:
```bash
python src/modules/speech_to_clothing_with_rodin_api.py --skip-3d
```

**With viewer (debug mode)**:
```bash
python src/modules/speech_to_clothing_with_rodin_api.py --viewer
```

---

## Output Structure

All outputs save to: `comfyui_generated_mesh/<timestamp>/`

```
comfyui_generated_mesh/
└── 1731305678/
    ├── original_frame.png          # Camera capture
    ├── mask.png                     # BodyPix segmentation
    ├── generated_clothing.png       # 2D result from SDXL
    ├── clothing_mesh.glb            # 3D mesh from Rodin API
    └── metadata.json                # Settings and prompt
```

---

## How It Works

### 2D Generation (SDXL)
- Server: `http://itp-ml.itp.tsoa.nyu.edu:9199` (ComfyUI)
- Workflow: `workflows/sdxl_inpainting_api.json`
- Time: ~3-5 seconds

### 3D Generation (Rodin API Direct)
- **NEW**: Direct API calls to `https://api.hyper3d.com/api/v2`
- Tier: **Regular** (good quality, saves credits)
- Time: ~60-90 seconds
- Process:
  1. Submit task with 2D image → get `task_uuid` + `subscription_key`
  2. Poll `/status` endpoint every 5 seconds
  3. When status = "Done", download GLB from `/download` endpoint

---

## Rodin API Settings

The pipeline uses **Rodin Regular** tier with **low-poly** optimized settings:

```python
{
  'tier': 'Regular',               # Not Gen-2 (saves credits)
  'quality_override': '5000',      # ~5000 faces (low-poly, fast rendering)
  'material': 'PBR',               # Physically Based Rendering (includes textures)
  'mesh_mode': 'Raw',              # Triangular faces (simpler geometry)
  'mesh_simplify': 'true',         # Simplify mesh for better performance
  'geometry_file_format': 'glb'   # GLB format with embedded textures
}
```

**Why Low-Poly (~5000 vertices)?**
- Fast real-time rendering
- Smaller file sizes (~2-5 MB)
- Easier rigging/animation
- Still includes PBR textures for visual quality

### Available Tiers

| Tier | Speed | Quality | Credits | Use Case |
|------|-------|---------|---------|----------|
| **Regular** | 70s | Good | Lower | **Default** - Production use |
| Gen-2 | 90s | Best | Higher | Premium quality only |
| Sketch | 20s | Basic | Lowest | Quick tests |

**Default is Regular** - best balance of quality and cost.

---

## Comparison: ComfyUI vs Direct API

| Feature | ComfyUI Version | Direct API Version |
|---------|----------------|-------------------|
| **3D Generation** | Via ComfyUI Rodin node | Direct Rodin API |
| **API Key Config** | IT admin on server | You set locally |
| **Setup Difficulty** | ⚠️ Needs IT help | ✅ Easy |
| **2D Generation** | ✅ ComfyUI | ✅ ComfyUI (same) |
| **Output** | ✅ Same | ✅ Same |
| **Speed** | ~60-90s | ~60-90s |
| **Quality** | Same | Same |

**Both versions produce identical results**, just different paths to Rodin API.

---

## Troubleshooting

### "RODIN_API_KEY not set"

**Cause**: API key not configured

**Fix**:
```bash
# Check if set
echo $RODIN_API_KEY

# If empty, set it
export RODIN_API_KEY="your_key_here"

# Or use --api-key flag
python src/modules/speech_to_clothing_with_rodin_api.py --api-key "your_key"
```

### "API error: 401 Unauthorized"

**Causes**:
1. Wrong API key
2. API key expired
3. Rodin subscription not active

**Fix**:
1. Verify key at https://hyperhuman.deemos.com/
2. Check subscription is active
3. Generate new API key if needed

### "Generation timed out after 300s"

**Causes**:
1. Rodin API is busy/slow
2. Network issues
3. Complex image taking longer

**Fix**:
- Try again (Rodin queue might be busy)
- Check internet connection
- Use simpler prompt for faster generation

### "Failed to download mesh"

**Causes**:
1. Generation completed but download failed
2. Network timeout
3. Invalid task UUID

**Fix**:
- Check terminal for task UUID
- Retry manually via Rodin web interface
- Check if files are in `/tmp/rodin_output_*` directory

---

## Example Session

```bash
# Set API key
export RODIN_API_KEY="sk_123456789abcdef"

# Run pipeline
python src/modules/speech_to_clothing_with_rodin_api.py

# Output:
# ======================================================================
# Speech-to-Clothing Pipeline with Rodin API (Direct)
# ======================================================================
# ✓ Pipeline initialized
# ✓ 3D mesh generation enabled (Rodin API Direct)
# ✓ API key configured: sk_12345...cdef
# ======================================================================
#
# 🎤 Calibrating microphone (3s)...
# ✓ Calibration complete!
#
# 📷 Initializing OAK-D Pro camera...
# ✓ OAK-D Pro initialized
#
# 👤 Waiting for body detection...
# ✓ Body detected!
#
# 🎤 Listening for speech...
# ✓ Speech detected! (volume: 823)
#
# 🎙️  Recording for 10 seconds...
# ✓ Recording complete
#
# 📝 Transcribing with Whisper...
# ✅ Transcription: 'flowing blue dress with golden embroidery'
#
# 🤸 Please stand in A-pose...
# 📸 Frame captured!
#
# 🎭 Running BodyPix segmentation...
# ✓ BodyPix complete in 1.23s
#
# 🎨 Generating 2D clothing with ComfyUI...
# ✓ Generation complete!
#
# 🎲 Generating 3D mesh with Rodin API (Direct)...
#    Input: /tmp/clothing_2d_1731305678.png
#    Tier: Regular
#    Submitting to Rodin API...
#    ✓ Task submitted: abc123def456
#    ⏳ Waiting for generation...
#    Job abc123de: Processing
#    Job abc123de: Processing
#    Job abc123de: Done
#    ✓ Generation complete in 73.2s
#    📥 Downloading mesh...
#    Downloading: model.glb
#    ✓ Saved: /tmp/rodin_output_1731305751/model.glb (2345.6 KB)
# ✓ Mesh downloaded: /tmp/rodin_output_1731305751/model.glb
#
# 💾 Saving to: comfyui_generated_mesh/1731305678
#   ✓ original_frame.png
#   ✓ mask.png
#   ✓ generated_clothing.png
#   ✓ clothing_mesh.glb (2345.6 KB)
#   ✓ metadata.json
#
# ✅ All files saved to: comfyui_generated_mesh/1731305678
# ✅ Pipeline complete! Check output folder for results.
```

---

## Next Steps

### View 3D Mesh

**Option 1: Custom Rodin Viewer (Recommended)**
```bash
open tests/rodin_mesh_viewer.html
```

Features:
- Optimized for low-poly Rodin meshes
- Shows PBR textures (Base Color, Normal, Roughness, Metalness)
- Quick load from pipeline output (enter timestamp)
- Stats: FPS, vertices, faces, texture breakdown
- Controls: Rotate, pan, zoom, wireframe, lighting toggle

**Option 2: Online Viewer**
https://gltf-viewer.donmccurdy.com/

Drag and drop `clothing_mesh.glb`

**Option 3: Blender**
1. File → Import → glTF 2.0
2. Select `clothing_mesh.glb`

### Use with Rigging

```bash
python tests/test_a_rigged_clothing.py comfyui_generated_mesh/1731305678/clothing_mesh.glb
```

---

## Files

```
src/modules/
└── speech_to_clothing_with_rodin_api.py    # NEW - Direct API version

workflows/
└── sdxl_inpainting_api.json                 # 2D generation (unchanged)

comfyui_generated_mesh/                      # Output folder (auto-created)
└── <timestamp>/
    ├── original_frame.png
    ├── mask.png
    ├── generated_clothing.png
    ├── clothing_mesh.glb                    # 3D mesh!
    └── metadata.json
```

---

## Security Notes

**⚠️ Keep your API key secret!**

- Don't commit API key to git
- Don't share API key publicly
- Revoke old keys when no longer needed
- Use environment variables (not hardcoded)

**Safe practices**:
```bash
# ✅ Good - environment variable
export RODIN_API_KEY="sk_..."

# ✅ Good - command line (for testing)
python script.py --api-key "sk_..."

# ❌ Bad - hardcoded in script
api_key = "sk_123456789..."  # Never do this!
```

---

**Status**: ✅ Ready to use (requires Rodin subscription)
**Created**: November 11, 2025
**Base**: speech_to_clothing_with_3d.py + Direct Rodin API integration
**See**: Rodin API docs at https://hyperhuman.deemos.com/
