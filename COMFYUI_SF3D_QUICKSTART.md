# ComfyUI SF3D Quick Start Guide

## What This Is

You discovered that **ComfyUI-3D-Pack** is already installed on your school GPU server! This means you can generate 3D meshes using the same ComfyUI endpoint you're using for 2D clothing generation. No separate server needed!

## How It Works

```
Your Mac                        School GPU (ComfyUI)
┌────────────────┐             ┌──────────────────┐
│ Image          │   Upload    │ ComfyUI Server   │
│ ↓              │──────────>  │ ↓                │
│ SF3D Workflow  │   Queue     │ SF3D Model       │
│ ↓              │<────────    │ ↓                │
│ 3D Mesh (GLB)  │   Download  │ Output Folder    │
└────────────────┘             └──────────────────┘
```

**Same server** as your 2D clothing generation!
- 2D: `http://itp-ml.itp.tsoa.nyu.edu:9199` + SDXL workflow
- 3D: `http://itp-ml.itp.tsoa.nyu.edu:9199` + SF3D workflow

## Quick Test

### Step 1: Find a test image

Use one of your generated clothing images:
```bash
ls comfyui_generated_images/*/generated_clothing.png
```

Example: `comfyui_generated_images/1762650549/generated_clothing.png`

### Step 2: Run the test script

```bash
python tests/comfyui_sf3d_test0.py comfyui_generated_images/1762650549/generated_clothing.png
```

### Step 3: Check the output

If successful, you'll get:
- ✅ `sf3d_test_output.glb` - 3D mesh file
- Generation time: **0.5-2 seconds** on GPU (vs 50s on your Mac!)

## Test Script Options

```bash
# Basic usage
python tests/comfyui_sf3d_test0.py <image_path>

# With custom output name
python tests/comfyui_sf3d_test0.py image.png --output my_mesh.glb

# High resolution texture
python tests/comfyui_sf3d_test0.py image.png --texture-resolution 2048

# With remeshing for animation
python tests/comfyui_sf3d_test0.py image.png --remesh quad

# Custom foreground ratio (more/less padding)
python tests/comfyui_sf3d_test0.py image.png --foreground-ratio 0.75
```

## Expected Output

```
======================================================================
ComfyUI SF3D Test - Image to 3D Mesh
======================================================================
Image: comfyui_generated_images/1762650549/generated_clothing.png
Server: http://itp-ml.itp.tsoa.nyu.edu:9199
Output: sf3d_test_output.glb

Settings:
  Texture Resolution: 1024
  Remesh Option: none
  Foreground Ratio: 0.85
======================================================================

🌐 Connecting to ComfyUI server...
✓ ComfyUI server is reachable

📷 Loading image...
   Image shape: (648, 1152, 3)

📤 Uploading image to server...
✓ Uploaded: sf3d_input_12345.png

📋 Loading workflow template...
✓ Loaded workflow template: workflows/sf3d_generation_api.json

⚙️  Preparing SF3D workflow...
   Input: sf3d_input_12345.png
   Texture: 1024x1024
   Remesh: none
   Foreground Ratio: 0.85
   Output: sf3d_test_output
   ✓ Workflow prepared

🚀 Queueing SF3D generation...
✓ Queued prompt: abc123-def456-...

⏳ Generating 3D mesh (this may take 5-30 seconds)...
✓ Generation complete in 2.3s

✅ Generation completed in 2.3 seconds!

📥 Downloading mesh file...
   Found mesh: sf3d_test_output_00001_.glb

✅ Success!
   Mesh saved to: /path/to/sf3d_test_output.glb
   File size: 1523.4 KB

🎯 Next Steps:
   1. View in Blender
   2. View online: https://gltf-viewer.donmccurdy.com/
   3. Use with rigged clothing: python tests/test_a_rigged_clothing.py
======================================================================
```

## Viewing the Mesh

### Option 1: Online Viewer (Easiest)
1. Go to https://gltf-viewer.donmccurdy.com/
2. Drag and drop your `.glb` file
3. Rotate, zoom, inspect textures

### Option 2: Blender (Best for editing)
1. Open Blender
2. File → Import → glTF 2.0 (.glb/.gltf)
3. Select your `.glb` file
4. Mesh will appear with textures

### Option 3: Your existing SF3D viewer
```bash
python tests/sf3d_test_2_viewer.py sf3d_test_output.glb
```

## Troubleshooting

### "Cannot connect to ComfyUI server"
**Fix**:
- Verify you're on school network or VPN
- Check server is running: `curl http://itp-ml.itp.tsoa.nyu.edu:9199/system_stats`

### "No mesh file found in output"
**Possible causes**:
1. ComfyUI-3D-Pack not installed correctly
2. SF3D model not loaded
3. Workflow node names don't match

**Fix**:
- Check ComfyUI web interface: http://itp-ml.itp.tsoa.nyu.edu:9199
- Load the workflow manually: `workflows/sf3d_generation_api.json`
- Check if "Load SF3D Model" and "StableFast3D" nodes exist

### "Generation timed out"
**Fix**:
- First run may take longer (downloading model)
- Increase timeout: edit `comfyui_client.py` line 42: `timeout=600`

### "Image upload failed"
**Fix**:
- Check image is valid: `file <image_path>`
- Try smaller image: `--texture-resolution 512`

## Workflow Explanation

The workflow JSON (`workflows/sf3d_generation_api.json`) defines these steps:

```
1. LoadSF3DModel
   └─> Loads "stabilityai/stable-fast-3d" model

2. LoadImage
   └─> Loads your uploaded image

3. StableFast3D
   ├─> Input: sf3d_model (from step 1)
   ├─> Input: foreground_image (from step 2)
   ├─> Settings: texture_resolution, remesh_option, foreground_ratio
   └─> Output: 3D mesh

4. SwitchMeshAxis (optional)
   └─> Rotates mesh if needed

5. Save3DMesh
   └─> Saves as GLB file to ComfyUI output folder
```

## Parameters Explained

### Texture Resolution
- `512` - Fast, lower quality (good for testing)
- `1024` - **Recommended** - Good balance
- `2048` - High quality, slower, larger files

### Remesh Option
- `none` - **Default** - Original mesh from SF3D
- `triangle` - Remeshed to triangles (better for game engines)
- `quad` - Remeshed to quads (**best for animation/rigging**)

### Foreground Ratio
- `0.5` - Lots of padding (object is small)
- `0.75` - Medium padding
- `0.85` - **Default** - Less padding (object fills frame)
- `1.0` - No padding (tight crop)

**For clothing**: Use `0.75-0.85` to ensure mesh isn't cut off

## Next: Integration into Pipeline

Once this test works, you can integrate into `speech_to_clothing.py`:

```python
# After ComfyUI 2D generation:
if result_image:
    print("\n🎲 Generating 3D mesh...")

    # Reuse same ComfyUI client!
    mesh_glb = generate_3d_mesh(
        comfyui_client,
        result_image,
        output_dir
    )

    if mesh_glb:
        print(f"✅ 3D mesh: {mesh_glb}")
```

**Total pipeline time**:
- Speech recognition: 10s
- 2D generation (ComfyUI): 3-5s
- 3D generation (SF3D): 0.5-2s
- **Total: ~15 seconds** speech → 3D mesh! 🚀

## Files Created

```
workflows/
└── sf3d_generation_api.json      # SF3D workflow template

tests/
└── comfyui_sf3d_test0.py         # Test script (reuses comfyui_client.py)
```

## Comparison with Local SF3D

### Local (Your Mac CPU):
```bash
python tests/sf3d_test_1_basic.py image.png
# Time: 50 seconds
# Output: generated_meshes/sf3d_test_1/mesh.glb
```

### Remote (School GPU via ComfyUI):
```bash
python tests/comfyui_sf3d_test0.py image.png
# Time: 0.5-2 seconds (25-100x faster!)
# Output: sf3d_test_output.glb
```

**Same quality, 25-100x faster!**

---

**Created**: November 8, 2025
**Status**: ✅ Ready to test
**Next**: Run test with your generated clothing image
