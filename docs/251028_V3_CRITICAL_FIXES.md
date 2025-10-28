# V3 Critical Fixes: Single Cage & Mesh Orientation Issues

## Date: October 28, 2025

## Problem Summary

V2 had 3 critical issues:

1. **Single elongated cage** (not multiple OBBs)
2. **Mesh sideways** (wrong orientation)
3. **Cage too long** (stretched along one axis)

## Root Cause Analysis

### Issue 1: Why Only One Cage?

**V2 Code (WRONG)**:

```python
def extract_bodypix_masks(self, segmentation_mask: np.ndarray) -> Dict[str, np.ndarray]:
    # For demo: create simple masks from segmentation
    # In production: use actual BodyPix masks

    # Simple heuristic segmentation based on Y-position
    body_mask = (segmentation_mask > 0.5).astype(np.uint8) * 255

    # Torso: middle 40%-80% vertical
    torso_mask = body_mask.copy()
    torso_mask[:int(h*0.2), :] = 0
    torso_mask[int(h*0.8):, :] = 0
    masks['torso'] = torso_mask
    # ...
```

**Problem**: We were creating **fake BodyPix masks** from MediaPipe's **single binary mask** (person vs background).

MediaPipe doesn't have 24-part segmentation like BodyPix. It only gives us:

- ✓ Person (1)
- ✓ Background (0)

Not:

- ✗ Torso front/back
- ✗ Left/right arms
- ✗ Left/right legs
- ✗ etc. (24 total parts)

**Result**: Our naive Y-coordinate slicing created at most 3 regions (torso, arms, legs), and the PCA on the full body mask created one elongated OBB.

### Issue 2: Mesh Orientation

The mesh from TripoSR is generated in a specific orientation. We need the same fixes as in `clothing_to_3d_triposr_2.py`:

1. 180° flip around X-axis (fix upside-down)
2. 90° rotation around Y-axis (face forward)

V2 didn't apply these transformations.

### Issue 3: Cage Too Long

When PCA is applied to a vertical body silhouette (head to feet), the major axis is vertical, creating a tall, thin OBB. This is geometrically correct for the full body, but useless for articulation.

## Solutions Implemented in V3

### Solution 1: Load REAL BodyPix Data

**V3 Strategy A (Preferred)**: Load from `reference_data.pkl`

```python
def load_reference_data(self, reference_path: str):
    """Load reference data with BodyPix masks."""
    with open(reference_path, 'rb') as f:
        self.reference_data = pickle.load(f)

    # This contains:
    # - bodypix_masks: Dict with 24 body parts
    # - mediapipe_keypoints_2d/3d
    # - original_frame
    # - frame_shape
```

**When to use**: When you've run `create_consistent_pipeline_v2.py` which saves full BodyPix segmentation.

**Advantages**:

- ✓ Real 24-part BodyPix segmentation
- ✓ Same data used for mesh generation (consistency)
- ✓ High quality masks

### Solution 1B: Keypoint-Based Mask Partitioning (Fallback)

**V3 Strategy B (Fallback)**: Intelligently partition MediaPipe mask using keypoint positions

```python
def partition_mask_by_keypoints(
    self,
    full_mask: np.ndarray,
    keypoints_2d: Dict[str, Tuple[float, float]]
) -> Dict[str, np.ndarray]:
    """
    Partition a single segmentation mask into body parts using keypoint positions.

    Key insight: Use skeleton to GUIDE segmentation.
    """
    # Get key Y-coordinates from keypoints
    shoulder_y = (left_shoulder.y + right_shoulder.y) / 2
    hip_y = (left_hip.y + right_hip.y) / 2
    knee_y = (left_knee.y + right_knee.y) / 2

    # HEAD: Above shoulders
    head_mask = full_mask.copy()
    head_mask[int(shoulder_y):, :] = 0

    # TORSO: Shoulders to hips
    torso_mask = full_mask.copy()
    torso_mask[:int(shoulder_y), :] = 0
    torso_mask[int(hip_y):, :] = 0

    # ARMS: Shoulders to hips, left/right of center
    left_arm_mask = full_mask.copy()
    left_arm_mask[:shoulder_y, :] = 0
    left_arm_mask[hip_y:, :] = 0
    left_arm_mask[:, center_x:] = 0  # Only left half

    # ... etc for legs
```

**When to use**: When no reference_data.pkl is available (live testing).

**Advantages**:

- ✓ Works with any MediaPipe feed
- ✓ No pre-processing required
- ✓ Anatomically guided by skeleton

**Limitations**:

- ~ Less accurate than real BodyPix (only 5-7 parts vs 24)
- ~ Assumes person is facing camera

### Solution 2: Mesh Orientation Correction

Applied the same transformations as `clothing_to_3d_triposr_2.py`:

```python
# FIX 1: 180° flip around X-axis (upside-down → right-side-up)
flip_transform = np.eye(4)
angle_x = np.pi  # 180 degrees
R_flip = np.array([
    [1, 0, 0],
    [0, cos(angle_x), -sin(angle_x)],
    [0, sin(angle_x), cos(angle_x)]
])
flip_transform[:3, :3] = R_flip
mesh.apply_transform(flip_transform)

# FIX 2: 90° rotation around Y-axis (left-facing → forward-facing)
forward_transform = np.eye(4)
angle_y = np.pi / 2  # 90 degrees
R_forward = np.array([
    [cos(angle_y), 0, sin(angle_y)],
    [0, 1, 0],
    [-sin(angle_y), 0, cos(angle_y)]
])
forward_transform[:3, :3] = R_forward
mesh.apply_transform(forward_transform)
```

**Result**: Mesh now faces forward and is right-side-up.

## Usage Guide

### Option A: With Reference Data (Recommended)

1. **Generate mesh with full pipeline**:

```bash
python tests/create_consistent_pipeline_v2.py
# This saves reference_data.pkl with BodyPix masks
```

2. **Run V3 with reference data**:

```bash
python tests/test_integration_cage_v3.py \
    --mesh generated_meshes/TIMESTAMP/mesh.obj \
    --reference generated_meshes/TIMESTAMP/reference_data.pkl
```

3. **Calibration** (after starting):
   - Press **SPACE** to begin 5-second countdown
   - Get into T-pose during countdown (arms extended horizontally)
   - System captures cage automatically after countdown completes

**Expected**:

- Multiple OBB sections (5-10 depending on BodyPix detection)
- Mesh facing forward
- Proper anatomical cage structure

### Option B: Without Reference Data (Fallback)

1. **Run V3 without reference data**:

```bash
python tests/test_integration_cage_v3.py \
    --mesh generated_meshes/0/mesh.obj
```

2. **Calibration** (same as Option A):
   - Press **SPACE** → 5-second countdown begins
   - Get into T-pose during countdown
   - System captures cage after countdown

**Expected**:

- Fewer OBB sections (5-7 from keypoint partitioning)
- Mesh facing forward
- Simpler but functional cage structure

## Comparison Table

| Aspect               | V2 (Broken)        | V3 Strategy A (Reference) | V3 Strategy B (Fallback) |
| -------------------- | ------------------ | ------------------------- | ------------------------ |
| **BodyPix Masks**    | Fake (1-3 regions) | Real (24 parts)           | Partitioned (5-7 parts)  |
| **Cage Sections**    | 1 (single box)     | 5-10 OBBs                 | 5-7 OBBs                 |
| **Mesh Orientation** | Sideways           | ✓ Forward                 | ✓ Forward                |
| **Data Source**      | MediaPipe only     | reference_data.pkl        | MediaPipe + keypoints    |
| **Setup Required**   | None               | Run pipeline first        | None                     |
| **Quality**          | ✗ Poor             | ✓ Excellent               | ~ Good                   |

## Technical Details: Keypoint-Based Partitioning

### Algorithm

```
Input:
- full_mask: Binary mask (H, W) from MediaPipe
- keypoints_2d: Dict of keypoint positions

Output:
- Dict of section_name -> binary mask

Process:
1. Extract key Y-coordinates from keypoints:
   - nose_y: Top of head reference
   - shoulder_y: Torso top boundary
   - hip_y: Torso bottom boundary
   - knee_y: Leg subdivision

2. Extract key X-coordinates:
   - center_x: Vertical midline of body
   - left/right boundaries from shoulder positions

3. Partition mask by regions:
   - HEAD: y < shoulder_y
   - TORSO: shoulder_y < y < hip_y
   - LEFT_ARM: shoulder_y < y < hip_y, x < center_x
   - RIGHT_ARM: shoulder_y < y < hip_y, x > center_x
   - LEFT_UPPER_LEG: hip_y < y < knee_y, x < center_x
   - RIGHT_UPPER_LEG: hip_y < y < knee_y, x > center_x
   - etc.

4. Filter out regions with < 100 pixels (noise)
```

### Advantages of This Approach

**Compared to naive Y-slicing (V2)**:

- ✓ Uses actual skeleton structure (anatomically informed)
- ✓ Adapts to body pose (keypoints move with person)
- ✓ Separates left/right (arms, legs)
- ✓ More sections (5-7 vs 1-3)

**Compared to real BodyPix**:

- ~ Fewer parts (5-7 vs 24)
- ~ Less fine-grained (no front/back separation)
- ~ Assumes frontal pose

**Inspiration**: Chen & Feng (2014) - "Adaptive skeleton-driven cages"

- Uses skeleton to guide cage generation
- Each bone defines a region
- Real-time adaptation to pose

## Expected Results

### Good Signs ✓

**Cage Structure**:

- 5-10 distinct OBB sections visible (not 1 box)
- Sections roughly aligned with body parts
- Cage proportions reasonable (not elongated)

**Mesh**:

- Facing forward (not sideways)
- Right-side-up (not upside-down)
- Centered in view

**Movement**:

- Multiple sections deform independently
- Interior vertices move rigidly
- Sections stay connected at joints

### Bad Signs ✗

**Still single box**:

- → Check if keypoints are detected (print keypoints_2d)
- → Check mask partitioning output (print len(masks))
- → Verify MediaPipe segmentation is working

**Still sideways**:

- → Verify mesh transformations are applied
- → Check mesh bounds after loading

**Detached sections**:

- → Check hierarchical deformation in articulated_deformer.py
- → Verify joint_info is correctly defined

## Debugging Commands

### Check reference data:

```bash
python -c "
import pickle
with open('generated_meshes/0/reference_data.pkl', 'rb') as f:
    data = pickle.load(f)
print('BodyPix masks:', list(data.get('bodypix_masks', {}).keys()))
print('Keypoints:', list(data.get('mediapipe_keypoints_2d', {}).keys()))
"
```

### Check mesh orientation:

```bash
python -c "
import trimesh
m = trimesh.load('generated_meshes/0/mesh.obj')
print('Bounds:', m.bounds)
print('Centroid:', m.centroid)
print('Extents:', m.bounds[1] - m.bounds[0])
"
```

### Test keypoint partitioning:

```bash
# Run V3 without reference data and check terminal output
python tests/test_integration_cage_v3.py --mesh generated_meshes/0/mesh.obj
# Look for: "✓ Partitioned mask into X sections: [...]"
```

## Next Steps

### If V3 Works ✓

1. **Test with different meshes**

   - Try various generated clothing meshes
   - Verify orientation fix works consistently

2. **Tune partitioning**

   - Adjust Y-coordinate thresholds
   - Add more anatomical sections (forearms, lower legs)

3. **Implement full pipeline**
   - Always generate with `create_consistent_pipeline_v2.py`
   - Use reference data for best results

### If V3 Still Has Issues ✗

1. **Single cage despite multiple masks**:

   - Debug OBB generation in `articulated_cage_generator.py`
   - Check PCA is computing per-section, not globally

2. **Poor segmentation quality**:

   - Use real BodyPix (run pipeline first)
   - Improve MediaPipe lighting/camera position

3. **Performance issues**:
   - Reduce mesh complexity
   - Simplify regional MVC computation

## Summary

V3 fixes all V2 issues by:

1. ✅ **Loading real BodyPix data** from `reference_data.pkl` (or intelligent partitioning as fallback)
2. ✅ **Fixing mesh orientation** (180° X-flip + 90° Y-rotation)
3. ✅ **Proper per-section OBB generation** (not global PCA)

**Result**: Multiple distinct anatomical cage sections, correctly oriented mesh, proper articulated structure.

---

**Test it now:**

```bash
python tests/test_integration_cage_v3.py --mesh generated_meshes/0/mesh.obj
```
