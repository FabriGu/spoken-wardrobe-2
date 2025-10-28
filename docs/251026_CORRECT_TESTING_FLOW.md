# CORRECT Testing Flow for V2 Pipeline

**Date**: October 26, 2025  
**Status**: Updated - Now uses consistent data throughout

---

## 🎯 The Problem You Identified

You were **100% correct**! The previous testing guide had a fundamental flaw:

**❌ OLD (WRONG) Flow**:

1. Create mock reference data
2. Use a mesh generated BEFORE (not from the same frame)
3. Test V2 system

**Result**: Not testing the actual benefit of V2 (consistency throughout pipeline)

**✅ NEW (CORRECT) Flow**:

1. Capture frame from camera
2. Run BodyPix + MediaPipe on THAT frame
3. Generate clothing with Stable Diffusion using THAT frame
4. Generate 3D mesh with TripoSR from generated clothing
5. Save ALL reference data from THAT same frame
6. Test V2 system with this consistently-generated data

**Result**: True end-to-end test of V2's consistent pipeline

---

## 📦 New Files Created

### 1. `tests/triposr_pipeline.py`

**Purpose**: Extracts TripoSR logic into callable functions

**What it does**:

- Takes the working logic from `clothing_to_3d_triposr_2.py`
- Makes it importable/callable without changing the logic
- Preserves all the orientation correction and Z-scaling that works

**Key function**:

```python
generate_mesh_from_image(
    image_path,
    output_dir,
    z_scale=0.8,
    auto_orient=True,
    apply_flip=True,
    ...
)
```

### 2. `tests/create_consistent_pipeline_v2.py`

**Purpose**: Complete interactive pipeline from scratch

**What it does**:

1. Loads BodyPix model
2. Loads MediaPipe Pose
3. Loads Stable Diffusion model
4. Captures frame from camera (user in T-pose)
5. Runs BodyPix on that frame → saves all 24 body part masks
6. Runs MediaPipe on that frame → saves reference keypoints
7. Lets user select body parts (torso, arms, etc.)
8. Generates clothing with Stable Diffusion
9. Generates 3D mesh with TripoSR
10. Saves ALL reference data consistently

**Output**:

- `generated_meshes/{timestamp}/mesh.obj`
- `generated_meshes/{timestamp}/reference_data.pkl`
- `generated_meshes/{timestamp}/reference_frame.png`
- `generated_meshes/{timestamp}/generated_full.png`
- `generated_meshes/{timestamp}/generated_clothing.png`

---

## 🚀 Updated Testing Steps

### Step 1: Run Complete Pipeline

```bash
python tests/create_consistent_pipeline_v2.py
```

**What happens**:

1. Loads all models (may take a few minutes first time)
2. Opens camera
3. Asks you to stand in T-pose
4. Press SPACE to capture
5. Runs BodyPix and MediaPipe on captured frame
6. Asks you to select clothing type:
   ```
   1. Torso
   2. Left Upper Arm
   3. Right Upper Arm
   ...
   7. T-Shirt (torso + upper arms)  ← DEFAULT
   8. Long Sleeve Shirt
   ```
7. Asks for clothing prompt (e.g., "flames", "roses")
8. Generates clothing (takes ~30 seconds)
9. Generates 3D mesh with TripoSR (takes ~1 minute)
10. Saves everything

**Expected output at end**:

```
======================================================================
PIPELINE COMPLETE!
======================================================================

Generated files:
  Mesh: generated_meshes/1234567890/mesh.obj
  Reference data: generated_meshes/1234567890/reference_data.pkl

Next step: Run test_integration_v2.py

  python tests/test_integration_v2.py \
      --mesh generated_meshes/1234567890/mesh.obj \
      --reference generated_meshes/1234567890/reference_data.pkl
======================================================================
```

---

### Step 2: Run V2 Integration System

Copy the command from Step 1 output and run it:

```bash
python tests/test_integration_v2.py \
    --mesh generated_meshes/1234567890/mesh.obj \
    --reference generated_meshes/1234567890/reference_data.pkl
```

**What happens**:

1. Loads mesh
2. Loads reference data (from SAME frame)
3. Generates cage from reference BodyPix masks
4. Computes MVC weights
5. Starts real-time deformation
6. Opens camera and web viewer

---

### Step 3: Open Web Viewer

Open `tests/enhanced_mesh_viewer_v2.html` in your browser.

**What you should see**:

- Blue mesh (your generated clothing)
- Magenta cage around it
- Debug info showing positions, keypoints, etc.

---

### Step 4: Test Movement

**Move your body and watch the mesh**:

- Move left → mesh warps left
- Move right → mesh warps right
- Move up → mesh warps up
- Move down → mesh warps down

---

## 🔑 Why This Is Better

### Consistency Throughout

**Same Frame Used For**:

- BodyPix segmentation → cage structure
- MediaPipe keypoints → reference pose
- Stable Diffusion input → clothing generation
- TripoSR input → 3D mesh generation

**Result**: Everything is aligned and consistent!

### Real Test of V2

**Now we're actually testing**:

- Does cage match the mesh? (It should - same BodyPix masks)
- Does reference pose match the mesh? (It should - same frame)
- Does real-time deformation work? (This is what we want to validate)

---

## 📋 Files Used (Latest Versions)

### From Your Existing Code:

1. **`tests_backup/bodypix_tf_051025_2.py`**

   - Latest BodyPix script
   - Used for BodyPix model loading pattern

2. **`src/modules/ai_generation.py`**

   - Stable Diffusion inpainting
   - `generate_clothing_from_text()` method

3. **`tests/clothing_to_3d_triposr_2.py`**

   - Latest TripoSR script with working orientation correction
   - Logic extracted to `triposr_pipeline.py`

4. **`tests_backup/s0_consistent_skeleton_2D_3D_1.py`**
   - Latest calibration script (for future depth calibration)

### New V2 Files:

5. **`tests/triposr_pipeline.py`**

   - TripoSR as callable module

6. **`tests/create_consistent_pipeline_v2.py`**

   - Complete interactive pipeline

7. **`tests/test_integration_v2.py`**

   - Real-time deformation system

8. **`tests/enhanced_mesh_viewer_v2.html`**
   - Web-based 3D viewer

---

## ⚙️ Configuration Options

### In `create_consistent_pipeline_v2.py`:

**Z-Scale** (line 362):

```python
z_scale=0.8  # < 1.0 reduces "fatness", > 1.0 increases
```

**Body Part Selection** (interactive):

- Choose what parts of the body to include in mesh
- Default: T-shirt (torso + upper arms)

**Clothing Prompt** (interactive):

- Describe the pattern/design
- Examples: "flames", "roses", "starry night"

### In `test_integration_v2.py`:

**2D vs 3D Warping**:

```bash
# 2D only (default)
python tests/test_integration_v2.py --mesh ... --reference ...

# 3D (with Z-axis)
python tests/test_integration_v2.py --mesh ... --reference ... --enable-z-warp
```

**Headless Mode**:

```bash
python tests/test_integration_v2.py --mesh ... --reference ... --headless
```

---

## 🐛 Troubleshooting

### Pipeline Step 1 Issues

**"Could not open camera"**:

- Check camera permissions
- Close other apps using camera
- Try different camera index

**"BodyPix model failed to load"**:

- Check internet connection (downloads model)
- Ensure `tf-bodypix` installed: `pip install tf-bodypix`

**"Stable Diffusion failed"**:

- Check if you have enough disk space (~5GB for model)
- Check VRAM (GPU) or RAM (CPU)

**"TripoSR failed"**:

- Check if `tsr` package installed
- Check GPU/VRAM availability
- Try reducing `mc_resolution` (default 110)

### Pipeline Step 2 Issues

**"Mesh not visible in web viewer"**:

- Check debug logs for mesh position
- Press `R` to reset camera
- Use WASD to navigate

**"Mesh doesn't warp"**:

- Check "Keypoints detected" in debug logs
- Ensure you're in frame and well-lit
- Try moving closer to camera

---

## 📊 What Success Looks Like

After completing both steps, you should have:

✅ **Generated Data**:

- 3D mesh file
- Reference data (same frame as mesh)
- Cage generated from reference data
- Everything aligned and consistent

✅ **Real-Time Deformation**:

- Mesh visible in web viewer
- Mesh warps when you move
- Direction is correct (left → left, up → up)
- Cage visible and follows mesh

✅ **Debug Logs Show**:

- Keypoints detected: 13/13
- Delta magnitude changes with movement
- Mesh stays visible (not offscreen)
- Cage deformation occurs

---

## 🎯 Next Steps After Validation

If V2 works correctly with consistent data:

1. ✅ **Core concept is valid**

   - Consistent data matters
   - Cage-based deformation works
   - Section-wise deformation works

2. **Phase 2 Enhancements**:

   - Add depth calibration (use `s0_consistent_skeleton_2D_3D_1.py` approach)
   - Add mesh scaling to match user size
   - Add rotation handling (not just translation)
   - Add temporal smoothing for jitter reduction

3. **Integration**:
   - Integrate with full app pipeline
   - Add UI for clothing selection
   - Real-time overlay in camera feed

---

## 📝 Summary

**Old way (mock data)**:

```
Existing mesh (from who knows when)
  ↓
Mock reference data (synthetic)
  ↓
Test V2
```

❌ Not testing consistency

**New way (V2 pipeline)**:

```
Capture frame
  ↓
BodyPix + MediaPipe on THAT frame
  ↓
Generate clothing from THAT frame
  ↓
Generate mesh from THAT clothing
  ↓
Save ALL reference data from THAT frame
  ↓
Test V2 with consistent data
```

✅ Actually testing what V2 is designed for!

---

**You were right to call this out!** The updated pipeline now properly tests the V2 system with truly consistent data throughout.

---

**Ready to test!** 🚀

Run: `python tests/create_consistent_pipeline_v2.py`
