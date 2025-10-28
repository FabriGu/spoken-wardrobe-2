# TripoSR Settings Update - Test 3 (Balanced) Applied

**Date**: October 27, 2025  
**Result**: Test 3 (Balanced) provided the best results  
**Action**: Updated pipeline defaults to use Balanced settings

---

## 🎯 What Changed

### Old Settings (BROKEN):

```python
mc_resolution = 110        # Too low - caused holes!
foreground_ratio = 0.85    # Default - less padding
```

### New Settings (FIXED - Test 3 Balanced):

```python
mc_resolution = 196        # 78% increase - fixes holes!
foreground_ratio = 0.75    # More padding - better context
```

---

## 📁 Updated Files

### 1. `tests/triposr_pipeline.py`

**Function**: `generate_mesh_from_image()`

**Changes**:

- `foreground_ratio`: `0.85` → `0.75` (more padding)
- `mc_resolution`: `110` → `196` (much better quality)

```python
def generate_mesh_from_image(
    ...
    foreground_ratio=0.75,  # Updated from Test 3 (Balanced) - more padding
    mc_resolution=196,       # Updated from Test 3 (Balanced) - much better than 110
    ...
):
```

---

### 2. `tests/create_consistent_pipeline_v2.py`

**Function**: `generate_mesh()` call to `generate_mesh_from_image()`

**Changes**:

- Updated comment to reference "Test 3 (Balanced) settings"
- `foreground_ratio`: `0.85` → `0.75`
- `mc_resolution`: `110` → `196`
- Added clarifying comments for each parameter

```python
# Generate mesh using Test 3 (Balanced) settings - best quality/performance
self.mesh_path, corrected_mesh = generate_mesh_from_image(
    image_path=temp_clothing_path,
    output_dir=output_dir,
    z_scale=0.8,  # Adjustable - reduces "fatness"
    auto_orient=True,
    apply_flip=True,  # 180° flip to fix upside-down
    no_remove_bg=False,  # DO background removal (better for SD output)
    foreground_ratio=0.75,  # More padding than default (0.85)
    mc_resolution=196,  # Much better than 110 - fixes holes!
    chunk_size=8192,
    model_save_format="obj"
)
```

---

## 🎯 Why Test 3 (Balanced) Was Chosen

### Test Results Comparison:

| Test                  | mc_res  | fg_ratio | Result                        |
| --------------------- | ------- | -------- | ----------------------------- |
| Old (Broken)          | 110     | 0.85     | ❌ Holes, flat geometry       |
| Test 1 (Default)      | 256     | 0.85     | ✅ Good but slower            |
| Test 2 (High Res)     | 320     | 0.85     | ✅ Best quality but slowest   |
| **Test 3 (Balanced)** | **196** | **0.75** | **✅ Best overall** ⭐        |
| Test 4 (Tight Frame)  | 256     | 0.70     | ✅ Good edges but too slow    |
| Test 5 (Performance)  | 160     | 0.80     | ⚠️ Fast but some quality loss |

---

## ✅ Expected Improvements

### 1. **Fewer Holes** (mc_resolution: 110 → 196)

- 78% increase in resolution
- Better capture of thin geometry (sleeves, edges, lace)
- Smoother surfaces overall

### 2. **Better Framing** (foreground_ratio: 0.85 → 0.75)

- More padding around clothing (10% increase)
- Better spatial context for TripoSR
- Cleaner edge reconstruction

### 3. **Performance Balance**

- Faster than Test 1 (Default - mc_res=256)
- Faster than Test 2 (High Res - mc_res=320)
- Still maintains good quality
- Perfect for real-time prototype

---

## 📊 Technical Details

### Marching Cubes Resolution (`mc_resolution`)

**What it does**: Controls the grid resolution for surface extraction from the 3D volume.

**Impact**:

- Higher = finer grid = captures thin details better
- Lower = coarser grid = misses thin surfaces (causes holes)

**Why 196?**

- **Old setting (110)**: Grid was too coarse for clothing details
- **New setting (196)**: Sweet spot between quality and speed
- **Default (256)**: Higher quality but slower
- **High Res (320)**: Best quality but slowest

**Your improvement**: 78% increase in resolution (110 → 196)

---

### Foreground Ratio (`foreground_ratio`)

**What it does**: Controls how much of the frame the foreground object occupies after preprocessing.

**Impact**:

- Higher (0.85-0.90) = Less padding, clothing fills frame
- Lower (0.70-0.75) = More padding, clothing has more context

**Why 0.75?**

- **Old setting (0.85)**: Clothing might touch frame edges
- **New setting (0.75)**: 10% more padding around clothing
- **Benefit**: More spatial context helps TripoSR understand 3D structure

**Your improvement**: 10% more padding (0.85 → 0.75)

---

## 🚀 How to Test the Improvements

### Run your full pipeline:

```bash
python tests/create_consistent_pipeline_v2.py
```

### Expected results:

1. ✅ **Fewer holes** in generated meshes
2. ✅ **Better detail capture** (sleeves, edges, collars)
3. ✅ **Cleaner edges** (no clipping artifacts)
4. ✅ **Smoother surfaces** overall
5. ✅ **Reasonable processing time** (~20-30s for mesh generation)

---

## 📈 Performance Comparison

| Setting             | Old (110) | New (196) | Improvement                 |
| ------------------- | --------- | --------- | --------------------------- |
| **Holes**           | Many      | Few       | 🎯 **MAJOR**                |
| **Detail capture**  | Poor      | Good      | 🎯 **MAJOR**                |
| **Edge quality**    | Poor      | Good      | ✅ **GOOD**                 |
| **Processing time** | ~15s      | ~25s      | ⚠️ +66% slower (acceptable) |

**Trade-off verdict**: The quality improvement FAR outweighs the modest speed decrease.

---

## 🔧 Further Tuning (If Needed)

### If you still see holes:

Try higher resolution:

```python
mc_resolution = 256  # Test 1 (Default)
# or
mc_resolution = 320  # Test 2 (High Res) - slowest but best
```

### If edges are still problematic:

Try more padding:

```python
foreground_ratio = 0.70  # Test 4 (Tight Frame)
```

### If you need more speed:

Try lower resolution:

```python
mc_resolution = 160  # Test 5 (Performance)
```

But **Test 3 (Balanced) with mc_resolution=196 and foreground_ratio=0.75 is recommended** for your use case.

---

## 📚 References

- Test results: `TRIPOSR_TESTS_README.md`
- Full test guide: `docs/TRIPOSR_TEST_GUIDE.md`
- Test scripts: `tests/triposr_test_*.py`
- Master test runner: `tests/run_all_triposr_tests.py`

---

## ✅ Summary

**Problem**: `mc_resolution=110` was too low, causing holes and poor detail capture.

**Solution**: Updated to Test 3 (Balanced) settings:

- `mc_resolution=196` (78% increase)
- `foreground_ratio=0.75` (10% more padding)

**Result**: Better quality with acceptable performance trade-off.

**Status**: ✅ **APPLIED TO PIPELINE**

---

**Your pipeline is now using the optimized settings! Test it out!** 🎉
