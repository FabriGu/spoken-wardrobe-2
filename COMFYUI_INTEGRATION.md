# ComfyUI Integration Guide

## Overview

This guide explains how to integrate your school's ComfyUI GPU server (`itp-ml.itp.tsoa.nyu.edu:9199`) with the Spoken Wardrobe pipeline to get faster, higher-quality clothing generation.

## Current vs Proposed Architecture

### Current (Local Stable Diffusion)
```
Speech Input (Whisper)
  → Local Stable Diffusion Inpainting (Mac MPS, ~15-30s, low quality)
  → Clothing Image
  → TripoSR/SF3D (3D mesh)
  → Rigging Pipeline
```

### Proposed (Remote ComfyUI)
```
Speech Input (Whisper - Local)
  → ComfyUI API (Remote GPU, ~2-5s, high quality)
  → Clothing Image
  → SF3D (3D mesh)
  → Rigging Pipeline
```

## Benefits

1. **Speed**: ~2-5 seconds vs 15-30 seconds (5-15x faster)
2. **Quality**: Higher resolution textures, better detail
3. **GPU Power**: School server has more VRAM for higher settings
4. **Specialized Models**: Can use clothing-specific models not possible locally
5. **No Local GPU**: Frees up Mac resources for other tasks

## Recommended Models for Clothing

### Option 1: IDM-VTON (Best for Virtual Try-On) ⭐

**Use Case**: When you have separate person and clothing images
- **Model**: `yisol/IDM-VTON` (ECCV 2024)
- **Strengths**:
  - State-of-the-art virtual try-on quality
  - Preserves clothing details perfectly
  - Handles complex garments (wrinkles, patterns, folds)
- **Weaknesses**:
  - Requires pre-existing clothing image (can't generate from text alone)
  - Best with upper-body clothing

**ComfyUI Node**: `ComfyUI-IDM-VTON`
- GitHub: https://github.com/TemryL/ComfyUI-IDM-VTON
- Auto-downloads from HuggingFace

### Option 2: SDXL Inpainting + DeepFashion (Best for Text-to-Clothing) ⭐⭐

**Use Case**: Generate clothing from text prompts (your current workflow)
- **Model**: `stabilityai/stable-diffusion-xl-1.0-inpainting`
- **Enhancement**: DeepFashion ADetailer (800K+ garment dataset)
- **Strengths**:
  - Generates clothing from text prompts
  - High quality (SDXL)
  - Works with your existing BodyPix segmentation
  - 13 clothing categories understood
- **Weaknesses**:
  - Not as realistic as virtual try-on models

**Recommended for your project** since you're generating clothing from speech, not swapping existing garments.

### Option 3: CatVTON-Flux (Cutting Edge)

**Use Case**: Highest quality virtual try-on
- **Model**: CatVTON + Flux fill inpainting
- **Strengths**: State-of-the-art quality (2024)
- **Weaknesses**: Requires clothing reference image

## ComfyUI API Integration

### API Endpoints

Your school's ComfyUI server should expose:

```
Base URL: http://itp-ml.itp.tsoa.nyu.edu:9199

POST /prompt                  # Submit workflow for generation
GET  /history/{prompt_id}     # Check status and get results
GET  /view                    # Retrieve generated images
POST /upload/image            # Upload input images (frame, mask)
WS   /ws                      # WebSocket for real-time updates
```

### Workflow JSON Format

ComfyUI workflows are defined in JSON. You need to export your workflow as "API format" JSON from ComfyUI.

**Basic Inpainting Workflow Structure**:
```json
{
  "1": {
    "class_type": "LoadImage",
    "inputs": {
      "image": "input_frame.png"
    }
  },
  "2": {
    "class_type": "LoadImage",
    "inputs": {
      "image": "mask.png"
    }
  },
  "3": {
    "class_type": "CLIPTextEncode",
    "inputs": {
      "text": "fashion clothing made of flames",
      "clip": ["4", 1]
    }
  },
  "4": {
    "class_type": "CheckpointLoaderSimple",
    "inputs": {
      "ckpt_name": "sdxl_inpainting.safetensors"
    }
  },
  "5": {
    "class_type": "KSampler",
    "inputs": {
      "seed": 100,
      "steps": 30,
      "cfg": 7.5,
      "sampler_name": "euler",
      "scheduler": "normal",
      "denoise": 1.0,
      "model": ["4", 0],
      "positive": ["3", 0],
      "negative": ["6", 0],
      "latent_image": ["7", 0]
    }
  },
  "6": {
    "class_type": "CLIPTextEncode",
    "inputs": {
      "text": "low quality, blurry, distorted",
      "clip": ["4", 1]
    }
  },
  "7": {
    "class_type": "VAEEncodeForInpaint",
    "inputs": {
      "pixels": ["1", 0],
      "mask": ["2", 0],
      "vae": ["4", 2]
    }
  },
  "8": {
    "class_type": "VAEDecode",
    "inputs": {
      "samples": ["5", 0],
      "vae": ["4", 2]
    }
  },
  "9": {
    "class_type": "SaveImage",
    "inputs": {
      "filename_prefix": "clothing_output",
      "images": ["8", 0]
    }
  }
}
```

### Python API Client

You'll need to:
1. Upload input images (frame + mask)
2. Submit workflow JSON with your prompt
3. Poll for completion or use WebSocket
4. Download result image

## Implementation Plan

### Phase 1: Create ComfyUI Client Module

Create `src/modules/comfyui_client.py`:
- Upload images to server
- Submit workflow with prompt
- Wait for completion
- Download results
- Handle errors/timeouts

### Phase 2: Modify ClothingGenerator

Update `src/modules/ai_generation.py`:
- Add `mode` parameter: `"local"` or `"remote"`
- Keep local SD code for fallback
- Add ComfyUI path for remote generation
- Same interface for both modes

### Phase 3: Create Workflow Templates

Create workflow JSON templates in `workflows/`:
- `sdxl_inpainting.json` - Text-to-clothing inpainting
- `idm_vton.json` - Virtual try-on (if needed later)
- Template variables: `{PROMPT}`, `{NEGATIVE_PROMPT}`, `{SEED}`

### Phase 4: Test Integration

1. Test connection to school server
2. Upload test frame + mask
3. Generate clothing from speech prompt
4. Compare quality with local SD
5. Measure speed improvement

## Setup Instructions

### 1. Install Dependencies

```bash
source venv/bin/activate
pip install requests websocket-client
```

### 2. Test Server Connection

```bash
curl http://itp-ml.itp.tsoa.nyu.edu:9199/system_stats
```

Should return server stats JSON if accessible.

### 3. Export Workflow from ComfyUI

**On the school's ComfyUI interface** (http://itp-ml.itp.tsoa.nyu.edu:9199):

1. Create your inpainting workflow:
   - Load SDXL Inpainting checkpoint
   - Add nodes for: Image input, Mask input, Prompt, KSampler, VAE Decode, Save
2. Test it manually with a sample image
3. Click **"Save (API Format)"**
4. Save as `workflows/sdxl_inpainting_api.json`
5. Download to your local machine

### 4. Configure in Your Code

```python
from src.modules.ai_generation import ClothingGenerator

# Remote mode (ComfyUI)
generator = ClothingGenerator(
    mode="remote",
    comfyui_url="http://itp-ml.itp.tsoa.nyu.edu:9199",
    workflow_path="workflows/sdxl_inpainting_api.json"
)

# Usage stays the same!
inpainted_full, clothing_png = generator.generate_clothing_from_text(
    frame, mask, "flames"
)
```

## Workflow Template Variables

Your workflow JSON will have placeholders:

```json
{
  "3": {
    "class_type": "CLIPTextEncode",
    "inputs": {
      "text": "{PROMPT}",
      "clip": ["4", 1]
    }
  },
  "5": {
    "class_type": "KSampler",
    "inputs": {
      "seed": {SEED},
      "steps": 30,
      ...
    }
  }
}
```

Python client replaces them:
```python
workflow_json = workflow_template.replace("{PROMPT}", user_prompt)
workflow_json = workflow_json.replace("{SEED}", str(seed))
```

## Expected Performance

### Local SD (Mac MPS)
- Generation time: 15-30 seconds
- Quality: Medium (limited by MPS memory)
- Texture resolution: 512-1024px
- Inference steps: 10-20 (limited by speed)

### Remote ComfyUI (School GPU)
- Generation time: 2-5 seconds
- Quality: High (full GPU power)
- Texture resolution: 1024-2048px
- Inference steps: 30-50 (no performance penalty)

**Expected improvement**: 5-10x faster, 2x better quality

## Security Considerations

1. **Network**: Ensure you're on NYU network or VPN
2. **Authentication**: Check if server requires API key/token
3. **Rate Limiting**: Ask about usage limits
4. **Data Privacy**: Images are sent to school server (check policies)

## Fallback Strategy

Keep local SD as fallback:

```python
try:
    # Try remote first
    result = generator.generate_remote(frame, mask, prompt)
except (ConnectionError, TimeoutError) as e:
    print(f"Remote failed: {e}, falling back to local")
    result = generator.generate_local(frame, mask, prompt)
```

## Next Steps

1. ✅ Research ComfyUI API and models (Done)
2. ⏳ Test connectivity to school server
3. ⏳ Export workflow JSON from ComfyUI interface
4. ⏳ Implement `comfyui_client.py`
5. ⏳ Modify `ai_generation.py` to support remote mode
6. ⏳ Test end-to-end with speech → ComfyUI → SF3D pipeline
7. ⏳ Compare quality and speed with local SD

## Recommended Workflow for Your Use Case

For the Spoken Wardrobe project, I recommend:

**SDXL Inpainting + DeepFashion** because:
- ✅ Generates clothing from text prompts (matches your speech input)
- ✅ Works with BodyPix segmentation masks
- ✅ High quality results
- ✅ Widely supported in ComfyUI
- ✅ Can handle various clothing types

**Workflow steps**:
1. User speaks: "flames"
2. Whisper transcribes (local)
3. BodyPix creates mask (local)
4. Send frame + mask + prompt to ComfyUI
5. SDXL generates clothing (remote GPU)
6. Receive high-quality clothing image
7. SF3D converts to 3D mesh (local or PC)
8. Rigging pipeline (local)

---

**Created**: November 7, 2024
**Status**: ⏳ Ready to implement
**Priority**: High (significant quality and speed improvement)
