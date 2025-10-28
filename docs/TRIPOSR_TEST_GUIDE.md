# TripoSR Settings Test Guide

## 🎯 Quick Start

Run all 5 tests at once:

```bash
python tests/run_all_triposr_tests.py
```

Or run individual tests:

```bash
python tests/triposr_test_1_default.py
python tests/triposr_test_2_high_res.py
python tests/triposr_test_3_balanced.py
python tests/triposr_test_4_tight_frame.py
python tests/triposr_test_5_performance.py
```

---

## 📋 Test Configurations

All tests use:

- **Input**: `generated_meshes/1761618888/generated_clothing.png`
- **Background removal**: OFF (`--no-remove-bg`)
- **Texture baking**: ON (`--bake-texture`)
- **Format**: OBJ with texture

### Test 1: Default Settings (Baseline)

```
mc_resolution: 256 (official default)
foreground_ratio: 0.85 (official default)
z_scale: 1.0 (no scaling)
texture_resolution: 2048
```

**Purpose**: Baseline comparison - official TripoSR defaults  
**Output**: `generated_meshes/triposr_test_1_default/0/`

---

### Test 2: High Resolution (Maximum Quality)

```
mc_resolution: 320 (HIGHER)
foreground_ratio: 0.85
z_scale: 0.85 (slight compression)
texture_resolution: 2048
```

**Purpose**: Maximum detail capture - best for fixing holes  
**Output**: `generated_meshes/triposr_test_2_high_res/0/`

**Expected improvements**:

- ✅ Fewer holes in mesh
- ✅ Better capture of thin geometry (sleeves, lace, edges)
- ✅ Smoother surfaces
- ⚠️ Slower processing (~1.5-2x longer)

---

### Test 3: Balanced Settings (⭐ Recommended)

```
mc_resolution: 196 (balanced)
foreground_ratio: 0.75 (more padding)
z_scale: 0.80 (your current)
texture_resolution: 2048
```

**Purpose**: Best quality/performance trade-off  
**Output**: `generated_meshes/triposr_test_3_balanced/0/`

**Expected improvements**:

- ✅ Better than mc_res=110 (fewer holes)
- ✅ Faster than mc_res=256 or 320
- ✅ More context around clothing (better framing)
- ✅ Good for real-time prototype

---

### Test 4: Tight Framing (More Padding)

```
mc_resolution: 256
foreground_ratio: 0.70 (MUCH more padding)
z_scale: 0.75 (flatter)
texture_resolution: 2048
```

**Purpose**: Test if extra padding helps edge reconstruction  
**Output**: `generated_meshes/triposr_test_4_tight_frame/0/`

**Expected improvements**:

- ✅ Better edge reconstruction (no clipping)
- ✅ More spatial context for TripoSR
- ⚠️ Clothing appears smaller in frame

---

### Test 5: Performance Optimized (Fastest)

```
mc_resolution: 160 (lower but better than 110)
foreground_ratio: 0.80
z_scale: 0.80
texture_resolution: 1024 (lower for speed)
```

**Purpose**: Minimum viable quality for rapid iteration  
**Output**: `generated_meshes/triposr_test_5_performance/0/`

**Expected characteristics**:

- ⚡ Fastest generation time
- ⚠️ Some detail loss vs. higher resolutions
- ✅ Still better than mc_res=110
- ✅ Good for rapid prototyping

---

## 🔍 What to Compare

### 1. Holes & Missing Geometry

- Open each mesh in Blender or your 3D viewer
- Look for holes in fabric (especially sleeves, edges)
- Compare Test 1 (baseline) vs Test 2 (high res)

### 2. Thin Detail Capture

- Check if sleeves, collars, edges are properly captured
- Look for "paper-thin" collapsed areas
- Test 2 should capture thin details best

### 3. Edge Quality

- Check if clothing edges are sharp or blurry
- Compare Test 1 vs Test 4 (tight framing)
- Test 4 should have cleaner edges if your clothing was touching frame edges

### 4. Surface Smoothness

- Look for "blocky" or "jagged" surfaces
- Higher mc_resolution = smoother surfaces
- Test 2 should be smoothest

### 5. Processing Time

- The master script `run_all_triposr_tests.py` will report timing
- Compare quality/time trade-offs
- Test 5 should be fastest

### 6. Frame Padding

- Check the `input.png` in each output directory
- Test 4 should have the most padding around clothing
- Compare if extra padding helps or hurts

---

## 🎯 Expected Results for Your Mesh Issues

### If you have HOLES:

**Try**: Test 2 (High Resolution) - `mc_resolution=320`  
**Why**: Higher resolution grid captures thin geometry better

### If mesh is FLAT/COLLAPSED:

**Try**: Test 1 (Default) with `z_scale=1.0` first  
**Why**: See the "raw" depth before scaling  
**Then**: Adjust z_scale in post-processing

### If edges are WARPED/DISTORTED:

**Try**: Test 4 (Tight Framing) - `foreground_ratio=0.70`  
**Why**: More padding gives TripoSR better context

### If you need SPEED:

**Try**: Test 5 (Performance) - `mc_resolution=160`  
**Why**: Fast iteration for prototyping  
**Then**: Use Test 3 (Balanced) for final quality

---

## 📊 Recommended Decision Tree

```
START
  │
  ├─ Are there HOLES in the mesh?
  │   YES → Use Test 2 (mc_resolution=320)
  │   NO  → Continue
  │
  ├─ Is processing time critical?
  │   YES → Use Test 5 (mc_resolution=160)
  │   NO  → Continue
  │
  ├─ Are edges warped/distorted?
  │   YES → Use Test 4 (foreground_ratio=0.70)
  │   NO  → Continue
  │
  └─ Default choice
      → Use Test 3 (Balanced) ⭐
```

---

## 🔧 How to Apply Chosen Settings

Once you've identified the best settings, update your pipeline:

### In `tests/triposr_pipeline.py`:

```python
# Update default parameters in generate_mesh_from_image()
def generate_mesh_from_image(
    image_path: str,
    output_dir: str,
    model=None,
    device="cuda:0",
    auto_orient=True,
    z_scale=0.8,           # ← Adjust based on tests
    apply_flip=True,
    no_remove_bg=False,
    foreground_ratio=0.75,  # ← Adjust based on tests
    mc_resolution=196,      # ← Adjust based on tests
    chunk_size=8192,
    model_save_format="obj"
):
```

### In `tests/create_consistent_pipeline_v2.py`:

```python
# Update the call to triposr_pipeline.generate_mesh_from_image()
mesh_path, mesh = triposr_pipeline.generate_mesh_from_image(
    image_path=str(clothing_path),
    output_dir=mesh_output_dir,
    device=device,
    auto_orient=True,
    z_scale=0.8,           # ← Adjust based on tests
    no_remove_bg=True,
    foreground_ratio=0.75,  # ← Adjust based on tests
    mc_resolution=196,      # ← Adjust based on tests
    model_save_format="obj"
)
```

---

## 📈 Performance vs Quality Matrix

| Test   | mc_res | Quality    | Speed      | Use Case        |
| ------ | ------ | ---------- | ---------- | --------------- |
| Test 1 | 256    | ⭐⭐⭐⭐   | ⭐⭐⭐     | Baseline        |
| Test 2 | 320    | ⭐⭐⭐⭐⭐ | ⭐⭐       | Maximum quality |
| Test 3 | 196    | ⭐⭐⭐⭐   | ⭐⭐⭐⭐   | **Recommended** |
| Test 4 | 256    | ⭐⭐⭐⭐   | ⭐⭐⭐     | Edge issues     |
| Test 5 | 160    | ⭐⭐⭐     | ⭐⭐⭐⭐⭐ | Rapid iteration |

---

## ⚠️ Important Notes

### About `z_scale`:

- TripoSR itself doesn't have a `z_scale` parameter
- This is applied in **post-processing** (in your pipeline)
- The test scripts note the intended z_scale but don't apply it
- To see actual z_scale effects, use your full pipeline

### About `--no-remove-bg`:

- All tests use `--no-remove-bg` as requested
- Make sure your input image (`generated_clothing.png`) is clean
- TripoSR expects images with proper background (not transparent)

### About texture baking:

- All tests enable `--bake-texture` as requested
- This generates `texture.png` alongside `mesh.obj`
- Slower than vertex colors but better visual quality

---

## 🚀 Next Steps

1. **Run all tests**:

   ```bash
   python tests/run_all_triposr_tests.py
   ```

2. **Compare meshes** in Blender or your 3D viewer

3. **Identify best settings** based on your priorities:

   - Quality? → Test 2
   - Balance? → Test 3 ⭐
   - Speed? → Test 5

4. **Update your pipeline** with chosen settings

5. **Re-test** your full pipeline (`create_consistent_pipeline_v2.py`)

---

## 🐛 Troubleshooting

### Test fails with "TripoSR not found":

```bash
# Make sure TripoSR is cloned:
git clone https://github.com/VAST-AI-Research/TripoSR.git external/TripoSR
```

### Test fails with "Input image not found":

```bash
# Check if the image exists:
ls -la generated_meshes/1761618888/generated_clothing.png

# If not, update the INPUT_IMAGE variable in test scripts
```

### CUDA out of memory:

- Try Test 5 (Performance) with lower mc_resolution
- Or add `--device cpu` (slower but works)

### Mesh looks wrong orientation:

- The tests use raw TripoSR output
- Your pipeline's `detect_and_correct_orientation()` handles this
- Compare with your full pipeline output

---

## 📚 References

- TripoSR official defaults: `external/TripoSR/run.py`
- Your custom pipeline: `tests/triposr_pipeline.py`
- Full pipeline: `tests/create_consistent_pipeline_v2.py`
- Original recommendations: `docs/BODYPIX_MODELS_AVAILABLE.md`
