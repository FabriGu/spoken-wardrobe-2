# Speech-to-Clothing with Rodin 3D Mesh Generation

## What's New

`speech_to_clothing_with_3d.py` extends the original pipeline to generate **3D meshes** using **Rodin Regular** via ComfyUI.

**Complete Pipeline**:
1. Speech recognition (Whisper)
2. 2D clothing generation (SDXL inpainting)
3. **3D mesh generation (Rodin Regular)** ← NEW!
4. Save all outputs to organized folder

**Why Rodin Regular?** Good quality, faster, and saves credits vs Gen-2.

---

## Output Structure

All outputs save to: `comfyui_generated_mesh/<timestamp>/`

```
comfyui_generated_mesh/
└── 1731305678/
    ├── original_frame.png          # Camera capture
    ├── mask.png                     # BodyPix segmentation
    ├── generated_clothing.png       # 2D result from SDXL
    ├── clothing_mesh.glb            # 3D mesh from Rodin Gen-2
    └── metadata.json                # Settings and prompt
```

---

## Usage

### Basic (with 3D generation)
```bash
python src/modules/speech_to_clothing_with_3d.py
```

### Skip 3D generation (2D only)
```bash
python src/modules/speech_to_clothing_with_3d.py --skip-3d
```

### With viewer (debug mode)
```bash
python src/modules/speech_to_clothing_with_3d.py --viewer
```

---

## How It Works

### 2D Generation (SDXL)
- Server: `http://itp-ml.itp.tsoa.nyu.edu:9199`
- Workflow: `workflows/sdxl_inpainting_api.json`
- Time: ~3-5 seconds

### 3D Generation (Rodin Regular)
- Server: Same ComfyUI instance
- Workflow: `workflows/rodin_regular_api.json`
- Time: ~60-90 seconds
- Node: `Rodin3D_Regular`

### Rodin Regular Settings (in workflow)

```json
{
  "Seed": 0,
  "Material_Type": "PBR",
  "Polygon_count": "200K-Triangle"
}
```

- **Material_Type**: `"PBR"` (Physically Based Rendering - realistic textures)
  - Alternative: `"Shaded"` (simpler, faster)
- **Polygon_count**: `"200K-Triangle"` (good quality)
  - Options: `"50K"`, `"100K"`, `"200K"`, `"400K"`
  - Higher = more detail, larger file
- **Seed**: `0` (random), or set specific number for reproducibility

---

## Requirements

### ComfyUI Server

Must have **Rodin3D_Regular node** installed:
- Part of Rodin extension for ComfyUI
- Check: Load `workflows/rodin_regular_api.json` in ComfyUI web interface
- If missing: Ask IT to install via ComfyUI Manager

### Rodin API Key

**REQUIRED**: The Rodin node needs an API key to work.

**See complete setup guide**: `RODIN_API_KEY_SETUP.md`

**Quick summary**:
1. Get API key from https://hyperhuman.deemos.com/ (Rodin subscription required)
2. Ask IT to configure it on the ComfyUI server
3. Test with: `python src/modules/speech_to_clothing_with_3d.py --skip-3d` first

---

## Workflow Customization

Edit `workflows/rodin_regular_api.json` to change settings:

```json
{
  "2": {
    "inputs": {
      "Seed": 0,
      "Material_Type": "PBR",           // Change to "Shaded" for simpler
      "Polygon_count": "200K-Triangle", // Change to "50K", "100K", "400K"
      "Images": ["1", 0]
    },
    "class_type": "Rodin3D_Regular"
  }
}
```

**Quality vs Speed**:
- `"50K"` = Faster, lower poly count, smaller file
- `"100K"` = Balanced
- `"200K"` = **Recommended** - Good quality
- `"400K"` = Highest quality, slower, large file

**Material Type**:
- `"PBR"` = **Recommended** - Realistic textures with metalness/roughness
- `"Shaded"` = Simpler, faster rendering

---

## Troubleshooting

### "Rodin3D_Regular node does not exist"

**Cause**: Rodin extension not installed in ComfyUI

**Fix**:
1. Ask IT to install Rodin extension via ComfyUI Manager
2. Verify in web interface: http://itp-ml.itp.tsoa.nyu.edu:9199
3. Try loading `workflows/rodin_regular_api.json` manually

### "Failed to download mesh"

**Possible causes**:
1. Generation timed out (increase timeout in code)
2. Rodin API key not configured
3. Rodin quota exceeded

**Fix**:
- Check ComfyUI logs on server
- Verify API key is set
- Try `tier: "sketch"` for faster generation

### "3D generation takes too long"

**Normal time for Rodin Regular**: 60-90 seconds

**If longer**:
- Server may be busy (check ComfyUI queue)
- Try lower polygon count: `"100K"` or `"50K"`
- Test 2D first: `--skip-3d` flag

---

## Comparison: Original vs 3D Version

| Feature | Original | With 3D |
|---------|----------|---------|
| **Output Folder** | `comfyui_generated_images/` | `comfyui_generated_mesh/` |
| **2D Image** | ✅ PNG | ✅ PNG |
| **3D Mesh** | ❌ | ✅ GLB |
| **Mask** | ✅ | ✅ |
| **Frame** | ✅ | ✅ |
| **Metadata** | ✅ | ✅ (extended) |
| **Total Time** | ~15s | ~75-105s |

---

## Next Steps

### View 3D Mesh

**Option 1: Online Viewer**
https://gltf-viewer.donmccurdy.com/

Drag and drop `clothing_mesh.glb`

**Option 2: Blender**
1. File → Import → glTF 2.0
2. Select `clothing_mesh.glb`

**Option 3: Three.js**
Load the GLB in your existing Three.js viewer

### Use with Rigging

```bash
python tests/test_a_rigged_clothing.py comfyui_generated_mesh/1731305678/clothing_mesh.glb
```

---

## Files Created

```
src/modules/
└── speech_to_clothing_with_3d.py       # New pipeline with 3D

workflows/
├── rodin_regular_api.json              # Rodin Regular (recommended)
└── rodin_gen2_api.json                 # Rodin Gen-2 (premium, optional)

comfyui_generated_mesh/                 # Output folder (auto-created)
└── <timestamp>/
    ├── original_frame.png
    ├── mask.png
    ├── generated_clothing.png
    ├── clothing_mesh.glb               # 3D mesh!
    └── metadata.json
```

---

**Status**: ✅ Ready to use (requires Rodin API key configuration)
**Created**: November 10, 2025
**Base**: speech_to_clothing.py + Rodin Regular
**See**: RODIN_API_KEY_SETUP.md for configuration
