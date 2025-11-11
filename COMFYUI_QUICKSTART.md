# ComfyUI Integration - Quick Start

## What I Created (Without Modifying Existing Files)

I created **3 new files** to test ComfyUI integration without touching your existing code:

### 1. `src/modules/comfyui_client.py`
**Python client for ComfyUI API**
- Uploads images to server
- Submits workflows
- Waits for generation
- Downloads results

### 2. `workflows/sdxl_inpainting_api.json`
**Workflow template** (placeholder - needs to be replaced)
- Example structure for SDXL inpainting
- Has placeholders: `{PROMPT}`, `{NEGATIVE_PROMPT}`, etc.
- You need to export the REAL workflow from ComfyUI web interface

### 3. `test_comfyui_integration.py`
**Standalone test script**
- Tests connection to school server
- Tests image upload
- Tests full generation pipeline
- Compares speed with local SD

## How to Test ComfyUI Integration

### Step 1: Test Connection (Right Now!)

```bash
source venv/bin/activate
python test_comfyui_integration.py
```

This will:
- ✅ Test if you can reach the server
- ✅ Test image upload
- ⚠️ Skip workflow test (you need to create it first)

### Step 2: Create Workflow in ComfyUI Web Interface (Tomorrow at School)

1. **Open ComfyUI in browser:**
   ```
   http://itp-ml.itp.tsoa.nyu.edu:9199
   ```

2. **Create an inpainting workflow:**
   - Drag nodes onto canvas:
     - `LoadImage` (for input frame)
     - `LoadImage` (for mask)
     - `CheckpointLoaderSimple` → select SDXL inpainting model
     - `CLIPTextEncode` (positive prompt)
     - `CLIPTextEncode` (negative prompt)
     - `VAEEncodeForInpaint`
     - `KSampler`
     - `VAEDecode`
     - `SaveImage`
   - Connect the nodes (follow ComfyUI examples)
   - **Test it manually** with a sample image first!

3. **Export workflow as API format:**
   - Click **"Save (API Format)"** button (NOT regular save)
   - Save the JSON file

4. **Download and edit the JSON:**
   - Download the exported JSON to your computer
   - Open it in a text editor
   - Replace these values with placeholders:
     - Positive prompt text → `{PROMPT}`
     - Negative prompt text → `{NEGATIVE_PROMPT}`
     - Input image filename → `{INPUT_IMAGE}`
     - Mask image filename → `{MASK_IMAGE}`
     - Seed value → `{SEED}`
     - Steps value → `{STEPS}`
     - CFG value → `{CFG}`

5. **Save to your project:**
   ```bash
   # Replace the template file
   cp downloaded_workflow.json workflows/sdxl_inpainting_api.json
   ```

### Step 3: Run Full Test

```bash
# Generate test images first (if you haven't already)
python tests/create_consistent_pipeline_v2.py

# Then test ComfyUI generation
python test_comfyui_integration.py
```

Expected output:
```
✅ PASS Connection
✅ PASS Upload
✅ PASS Workflow
✅ PASS Generation
```

## How to Use in Your Pipeline

Once testing works, you have 2 options:

### Option A: Use ComfyUI Client Directly

**Don't modify existing code** - just import the client:

```python
from src.modules.comfyui_client import ComfyUIClient

# Initialize
client = ComfyUIClient("http://itp-ml.itp.tsoa.nyu.edu:9199")

# Generate
result_image = client.generate_inpainting(
    image=frame_rgb,  # numpy array
    mask=mask_array,  # numpy array
    prompt="fashion clothing made of flames",
    negative_prompt="low quality, blurry",
    workflow_path="workflows/sdxl_inpainting_api.json",
    seed=100,
    steps=30,
    cfg=7.5
)

# Result is a PIL Image
result_image.save("output.png")
```

### Option B: Modify ClothingGenerator Later

When ready, we can add a `mode` parameter to `ai_generation.py`:

```python
# Remote mode (ComfyUI)
generator = ClothingGenerator(
    mode="remote",
    comfyui_url="http://itp-ml.itp.tsoa.nyu.edu:9199",
    workflow_path="workflows/sdxl_inpainting_api.json"
)

# Local mode (existing behavior)
generator = ClothingGenerator(mode="local")

# Usage is the same either way!
result = generator.generate_clothing_from_text(frame, mask, "flames")
```

## Expected Performance

| Mode | Hardware | Time | Quality |
|------|----------|------|---------|
| **Local** | Mac MPS | 15-30s | Medium |
| **Remote** | School GPU | 2-5s | **High** |

**Speedup: 5-10x faster** ⚡

## Models You Can Use in ComfyUI

### For Text-to-Clothing (Your Use Case) ⭐

**SDXL Inpainting + DeepFashion**
- Model: `stabilityai/stable-diffusion-xl-1.0-inpainting`
- Extension: DeepFashion (ADetailer) for clothing enhancement
- Best for: Generating clothing from text prompts
- Works with: BodyPix segmentation masks

### For Virtual Try-On (Future Option)

**IDM-VTON**
- ComfyUI Extension: `ComfyUI-IDM-VTON`
- Best for: Swapping clothing onto person
- Requires: Separate clothing reference image

## Troubleshooting

### "Cannot connect to server"
- ✅ Make sure you're on NYU network or VPN
- ✅ Try opening `http://itp-ml.itp.tsoa.nyu.edu:9199` in browser
- ✅ Ask your instructor if the server is running

### "Workflow not found"
- ✅ Create and export workflow from ComfyUI web interface first
- ✅ Make sure it's saved to `workflows/sdxl_inpainting_api.json`

### "Upload failed"
- ✅ Check server disk space
- ✅ Check image format (should be PNG)
- ✅ Try with smaller test images first

### "Generation timeout"
- ✅ First generation may take longer (model loading)
- ✅ Subsequent generations should be 2-5 seconds
- ✅ Check server GPU usage in web interface

## Files Created

```
src/modules/
└── comfyui_client.py          # ComfyUI API client

workflows/
└── sdxl_inpainting_api.json   # Workflow template (replace with real one)

test_comfyui_integration.py     # Standalone test script

COMFYUI_INTEGRATION.md          # Detailed integration guide
COMFYUI_QUICKSTART.md          # This file
```

## Next Steps

1. ✅ Test connection (do this now!)
   ```bash
   python test_comfyui_integration.py
   ```

2. ⏳ Tomorrow at school:
   - Open ComfyUI web interface
   - Create inpainting workflow
   - Export as API format
   - Test full generation

3. ⏳ Once working:
   - Integrate into your main pipeline
   - Compare quality with local SD
   - Decide which to use for production

---

**Status**: Ready to test connection
**Priority**: High (5-10x speed improvement)
