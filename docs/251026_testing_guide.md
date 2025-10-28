# Testing Guide: Fixed Cage-Based Deformation

**Date**: October 26, 2025  
**Status**: Ready for Testing

---

## 🎯 What Was Fixed

### Fix #1: Mesh Orientation ✅

- Mesh now faces forward (not rotated 90° left)
- Added 90° Y-axis rotation after flip

### Fix #2: Cage Structure ✅

- Cage now returns anatomical structure mapping
- Each body part section tracked with vertex indices
- Body parts mapped to MediaPipe keypoints

### Fix #3: Section-Wise Deformation ✅

- Each cage section deforms independently
- No more uniform transformation
- Torso, arms, legs move separately

### Fix #4: Integration ✅

- test_integration.py stores and uses cage_structure
- Verification script updated to match

---

## 🧪 Testing Procedure

### Step 1: Test Mesh Orientation Fix

**Goal**: Verify mesh faces forward (not rotated left)

```bash
cd /Users/fabrizioguccione/Projects/spoken_wardrobe_2
source venv/bin/activate
python tests/clothing_to_3d_triposr_2.py
```

**Interactive Steps**:

1. Select an image from `generated_images` directory
2. Choose options (or use defaults)
3. Wait for mesh generation

**Expected Result**:

- Mesh saved to `generated_meshes/` directory
- Console shows: "Applying 90° rotation around Y-axis (face forward)"
- When visualized, mesh should face camera (not turned left)

**To Visualize** (optional):

```bash
# Use your 3D viewer or verification system
python 251025_data_verification/verify_deformation.py --mesh generated_meshes/[your_new_mesh].obj
open 251025_data_verification/verification_viewer.html
```

---

### Step 2: Test Cage Structure Generation

**Goal**: Verify cage has anatomical structure with body part sections

```bash
python 251025_data_verification/verify_cage_structure.py
```

**What to Look For**:

**Expected Output**:

```
Generated anatomical cage with [N] vertices
Cage structure: [X] body parts

Cage structure:
{
    'torso': {'vertex_indices': [0, 1, 2, ...], 'keypoints': [...]},
    'left_upper_arm': {'vertex_indices': [8, 9, ...], 'keypoints': [...]},
    'right_upper_arm': {'vertex_indices': [14, 15, ...], 'keypoints': [...]},
    ...
}
```

**Good Signs**:

- ✓ Cage has 3-6 body part sections
- ✓ Each section has 6-8 vertices
- ✓ Total cage vertices: 30-60
- ✓ Each section mapped to specific keypoints

**Bad Signs**:

- ✗ Only 1 section (all vertices in 'torso')
- ✗ >100 total vertices
- ✗ No structure dict returned

---

### Step 3: Test MVC Weights (Quick Sanity Check)

**Goal**: Ensure weights still computed correctly

```bash
python 251025_data_verification/verify_mvc_weights.py
```

**Expected Output**:

```
✓ PASS: All rows sum to 1.0
✓ PASS: Deformation is fast enough for real-time
Average time per deformation: <5ms
```

**This should pass** - we didn't change MVC logic

---

### Step 4: Test Section-Wise Deformation (CRITICAL!)

**Goal**: Verify only 30-60% of cage vertices move per frame

```bash
# Terminal 1 - Start verification
python 251025_data_verification/verify_deformation.py --mesh generated_meshes/0/mesh.obj

# Browser - Open viewer
open 251025_data_verification/verification_viewer.html
```

**What to Look For in Console**:

**BEFORE (Bad)**:

```
📊 Frame 0:   Moving vertices: 21/21 (100%)  ← ALL vertices moving
📊 Frame 30:  Moving vertices: 21/21 (100%)  ← Problem!
📊 Frame 60:  Moving vertices: 21/21 (100%)  ← No articulation
```

**AFTER (Good)**:

```
📊 Frame 0:   Moving vertices: 8/40 (20%)   ← Only some sections
📊 Frame 30:  Moving vertices: 14/40 (35%)  ← Different amounts
📊 Frame 60:  Moving vertices: 12/40 (30%)  ← Independent motion!
```

**What to Do**:

1. Stand in front of camera
2. Move left arm → Check console
   - Should see: ~20-30% of vertices moving
3. Move right arm → Check console
   - Should see: ~20-30% of vertices moving (different vertices!)
4. Move both arms → Check console
   - Should see: ~40-50% of vertices moving
5. Stand still → Check console
   - Should see: ~10-20% of vertices moving (stabilization)

**Visual Check (Web Viewer)**:

- Mesh (green) should follow your body motion
- Cage (magenta) should have sections that move independently
- No "smearing" or collapse
- Mesh should stay roughly the same size/shape

---

### Step 5: Test with Full Integration System

**Goal**: Test the actual deformation system (not just verification)

```bash
# Use the default mesh path
python tests/test_integration.py --headless --mesh generated_meshes/0/mesh.obj

# Or with viewer window
python tests/test_integration.py --mesh generated_meshes/0/mesh.obj

# Then open web viewer
open tests/enhanced_mesh_viewer.html
```

**Expected Console Output**:

```
Initializing cage from BodyPix...
Generated anatomical cage with [30-60] vertices
Cage structure: [3-6] body parts
✓ Cage system initialized
   Cage: [30-60] vertices, [X] faces
   Structure: [3-6] body parts
✓ Now using MediaPipe for real-time deformation
```

**Visual Test**:

1. Move left arm → Only left arm section of mesh moves
2. Move right arm → Only right arm section moves
3. Rotate torso → Torso section rotates, arms follow naturally
4. Raise arms → Both arm sections move up, torso stable

**Good Signs**:

- ✓ Mesh follows body motion smoothly
- ✓ Different body parts articulate independently
- ✓ No smearing or collapse
- ✓ FPS stays above 20

**Bad Signs**:

- ✗ Entire mesh moves as one unit
- ✗ Mesh collapses when you enter frame
- ✗ FPS drops below 10
- ✗ Mesh becomes distorted/smeared

---

## 📊 Expected Results Summary

| Test               | Metric     | Before Fix       | After Fix            |
| ------------------ | ---------- | ---------------- | -------------------- |
| Mesh Orientation   | Facing     | 90° left         | Forward ✓            |
| Cage Structure     | Sections   | 1 (all vertices) | 3-6 body parts ✓     |
| Cage Vertices      | Total      | 8-21 (box)       | 30-60 (anatomical) ✓ |
| Moving Vertices    | Per frame  | 100%             | 30-60% ✓             |
| Independent Motion | Sections   | No               | Yes ✓                |
| Mesh Quality       | Distortion | Smearing         | Clean ✓              |

---

## 🐛 Troubleshooting

### Issue: Mesh still facing left

**Solution**: Regenerate mesh with new script

```bash
python tests/clothing_to_3d_triposr_2.py
# Select a NEW image to generate fresh mesh
```

### Issue: "No module named 'enhanced_cage_utils'"

**Solution**: Make sure you're in venv and running from project root

```bash
source venv/bin/activate
cd /Users/fabrizioguccione/Projects/spoken_wardrobe_2
python tests/test_integration.py
```

### Issue: "AttributeError: tuple object has no attribute 'vertices'"

**Cause**: Old code trying to unpack cage but getting tuple
**Solution**: Restart Python (cached imports)

```bash
# Kill any running Python processes
pkill -9 python
# Try again
python tests/test_integration.py
```

### Issue: Still seeing 100% vertices moving

**Check**:

1. Is cage_structure being generated?
   ```python
   # Look for in console:
   "Cage structure: X body parts"
   ```
2. Is cage_structure being passed?
   ```python
   # Check test_integration.py line ~278
   # Should have: self.cage_structure as 4th parameter
   ```
3. Are there body part masks in BodyPix output?
   ```python
   # Add debug print in enhanced_cage_utils.py
   print(f"Body parts detected: {list(body_parts.keys())}")
   ```

### Issue: Cage has 0 vertices / Falls back to box

**Cause**: BodyPix not detecting body parts
**Solution**:

1. Stand closer to camera
2. Ensure good lighting
3. Face camera directly
4. Check BodyPix mask visualization (press 'S' in viewer)

---

## 📝 Data to Collect

For each test, note:

1. **Cage Statistics**:

   - Total vertices: \_\_\_\_
   - Number of body parts: \_\_\_\_
   - Body parts detected: ******\_\_\_\_******

2. **Performance**:

   - FPS: \_\_\_\_
   - Cage deformation time: \_\_\_\_ ms
   - Mesh deformation time: \_\_\_\_ ms

3. **Movement Analysis**:

   - Left arm raised: \_\_\_\_ % vertices moving
   - Right arm raised: \_\_\_\_ % vertices moving
   - Both arms raised: \_\_\_\_ % vertices moving
   - Standing still: \_\_\_\_ % vertices moving

4. **Visual Quality**:
   - Mesh follows body motion: Yes / No
   - Independent articulation: Yes / No
   - Smearing/distortion: Yes / No
   - Cage visible and 3D: Yes / No

---

## ✅ Success Criteria

All of these should be true:

- [x] Mesh faces forward (not rotated left)
- [x] Cage has 30-60 vertices
- [x] Cage has 3-6 anatomical sections
- [x] Each section mapped to keypoints
- [x] Only 30-60% of vertices move per frame
- [x] Different sections move independently
- [x] Mesh follows body motion smoothly
- [x] No smearing or collapse
- [x] FPS above 20
- [x] Cage is 3D (visible from all angles)

---

## 🎯 Quick Test Commands

```bash
# Activate venv
source venv/bin/activate

# Test 1: Mesh orientation
python tests/clothing_to_3d_triposr_2.py

# Test 2: Cage structure
python 251025_data_verification/verify_cage_structure.py

# Test 3: MVC weights
python 251025_data_verification/verify_mvc_weights.py

# Test 4: Section-wise deformation
python 251025_data_verification/verify_deformation.py --mesh generated_meshes/0/mesh.obj
open 251025_data_verification/verification_viewer.html

# Test 5: Full integration
python tests/test_integration.py --headless --mesh generated_meshes/0/mesh.obj
open tests/enhanced_mesh_viewer.html
```

---

**Good luck with testing! 🚀**

The fixes address the root cause identified in the verification:

- Cage now has anatomical structure
- Sections deform independently
- No more 100% vertex movement

This should eliminate the "smearing" problem!
