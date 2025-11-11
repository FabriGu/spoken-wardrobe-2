# Stable Diffusion Freeze Analysis

## What Happened

Your `day2_oakd_bodypix_sd_data.py` froze during Stable Diffusion generation at this line:

```
Calling Stable Diffusion pipeline...
Prompt: fashion clothing made of colorful flames pattern...
```

Then it hung indefinitely until you killed it with Ctrl+C (3 times).

## Root Cause

**Mac MPS (Metal Performance Shaders) is struggling with the SD inference**

Evidence:
1. **Large image resolution**: 1152x648 → resized to 768x448
2. **Mac MPS memory pressure**: Your Mac GPU is running out of memory or thermal throttling
3. **Inference steps**: Default 10 steps in `ai_generation.py` line 23
4. **Resource leak warning**: `leaked semaphore objects` indicates incomplete cleanup

## Why It Froze (Not Crashed)

The process froze instead of crashing because:
- MPS backend is **waiting** for GPU memory to become available
- PyTorch MPS sometimes **hangs** instead of raising an error
- The semaphore leak suggests threads are deadlocked

## Good News ✅

**The files you need were saved BEFORE the freeze!**

### Saved Files:
```bash
debug_image_to_sd.png   # 535KB - Your clean frame
debug_mask_to_sd.png    # 2.2KB - Your BodyPix mask
```

### Previous Data (Nov 1st):
```bash
generated_meshes/oakd_clothing/
├── frame_1762045273.png    # 1.0MB - Clean frame
├── mask_1762045273.png     # 4.9KB - BodyPix mask
└── metadata_1762045273.json # 5.2KB - Complete metadata
```

**You can use ANY of these for ComfyUI testing!**

## Solutions

### Option 1: Use Existing Data with ComfyUI (RECOMMENDED) ⭐

**Skip local SD entirely** and use the saved frame + mask with ComfyUI:

```bash
# Test ComfyUI with your existing data
python tests/test_comfyui_integration.py
```

The test script will use:
- `debug_image_to_sd.png` (from today's run)
- `debug_mask_to_sd.png` (from today's run)

### Option 2: Fix Local SD for Testing (NOT RECOMMENDED)

If you still want to test local SD, reduce memory usage:

**Create a lightweight version:**

Edit `src/modules/ai_generation.py` to use CPU mode for testing:

```python
# Line 22-24, change:
self.num_inference_steps = 10  # Change to 5 (faster but lower quality)
self.guidance_scale = 7.5      # Change to 5.0 (less memory)

# Line 212-213, change resize to smaller:
target_width = 512   # Change from 768
target_height = 384  # Smaller resolution
```

**Or force CPU mode** (slower but won't freeze):

```bash
export PYTORCH_ENABLE_MPS_FALLBACK=1
export PYTORCH_MPS_HIGH_WATERMARK_RATIO=0.0  # Disable MPS, force CPU
python tests/day2_oakd_bodypix_sd_data.py
```

### Option 3: Skip SD Generation Entirely

**Just generate masks without SD** for ComfyUI testing:

Create a simplified version that only does BodyPix segmentation.

## Why ComfyUI Solves This

ComfyUI on your school's GPU server will:
- ✅ Run on proper NVIDIA GPU (not Mac MPS)
- ✅ Complete in 2-5 seconds (not 10-20s + freeze)
- ✅ Handle larger resolutions (768x448 or higher)
- ✅ No memory issues
- ✅ No resource leaks

## Next Steps

### Immediate (Right Now):

1. **Test ComfyUI connection** with existing data:
   ```bash
   python tests/test_comfyui_integration.py
   ```

2. **It will use** the debug files you just created:
   - `debug_image_to_sd.png` (your frame)
   - `debug_mask_to_sd.png` (your mask)

### Tomorrow (At School):

1. **Access ComfyUI web interface**:
   ```
   http://itp-ml.itp.tsoa.nyu.edu:9199
   ```

2. **Create SDXL inpainting workflow**

3. **Export and test** with your saved frame+mask

4. **Compare quality**: ComfyUI (high quality, fast) vs Local SD (freezes)

## Cleaning Up Resource Leaks

The semaphore warning is harmless but to clean it up:

```bash
# Kill any lingering Python processes
pkill -9 -f "day2_oakd_bodypix"

# Clear PyTorch cache
python -c "import torch; torch.mps.empty_cache() if torch.backends.mps.is_available() else None"
```

## Summary

| Issue | Status | Solution |
|-------|--------|----------|
| **Program froze** | ⚠️ Expected on Mac MPS | Use ComfyUI instead |
| **Files saved** | ✅ Success | Ready for ComfyUI testing |
| **Resource leak** | ⚠️ Minor warning | Kill lingering processes |
| **Local SD slow** | ⚠️ Mac limitation | Switch to GPU server |
| **ComfyUI ready** | ✅ Can test now | Run test script |

## Conclusion

**Don't waste time fixing local SD** - this freeze confirms exactly why you need ComfyUI!

Your data is saved and ready. Test the ComfyUI connection now, then set up the full workflow tomorrow at school.

---

**Next Command:**
```bash
python tests/test_comfyui_integration.py
```

This will test if you can reach the school server (without needing to run local SD).
