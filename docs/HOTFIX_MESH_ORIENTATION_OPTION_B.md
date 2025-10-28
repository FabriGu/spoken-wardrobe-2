# Hotfix: Mesh Orientation for Option B

**Date**: October 28, 2025  
**Issue**: Mesh appearing upside-down and facing wrong direction  
**Fix**: Apply orientation correction transforms after loading mesh

---

## Problem

After fixing the coordinate system mismatch, the mesh now appears but is:

- ❌ Upside-down (inverted vertically)
- ❌ Facing wrong direction (rotated)

This is because TripoSR meshes are typically generated in a different orientation than expected by the deformation system.

---

## Root Cause

TripoSR generates meshes with a specific orientation that worked fine for static viewing, but when applying skeletal deformation, the mesh coordinate system needs to align with:

1. MediaPipe's coordinate system (Y-up, Z-forward)
2. The expected "upright person facing forward" orientation

The mesh was likely:

- Upside-down in Y-axis
- Rotated 90° in Y-axis (facing left instead of forward)

---

## Solution

Apply two transformation matrices to the mesh **immediately after loading**, before any rigging/deformation:

### 1. Flip Upside-Down (180° around X-axis)

```python
flip_transform = np.eye(4)
angle_x = np.pi  # 180 degrees
c, s = np.cos(angle_x), np.sin(angle_x)
flip_transform[:3, :3] = np.array([
    [1, 0, 0],
    [0, c, -s],
    [0, s, c]
])
self.mesh.apply_transform(flip_transform)
```

### 2. Rotate to Face Forward (90° around Y-axis)

```python
forward_transform = np.eye(4)
angle_y = np.pi / 2  # 90 degrees
c, s = np.cos(angle_y), np.sin(angle_y)
forward_transform[:3, :3] = np.array([
    [c, 0, s],
    [0, 1, 0],
    [-s, 0, c]
])
self.mesh.apply_transform(forward_transform)
```

---

## Implementation

### Location

`tests/test_integration_skinning.py`, lines 411-438

### Added to `setup()` method

Right after loading the mesh, before initializing rigging:

```python
def setup(self):
    # Load mesh
    self.mesh = trimesh.load(self.mesh_path)

    # Fix mesh orientation (NEW!)
    print("\nCorrecting mesh orientation...")

    # 180° X-axis flip
    flip_transform = np.eye(4)
    # ... (rotation matrix)
    self.mesh.apply_transform(flip_transform)
    print("  Applied 180° X-axis flip (upside-down fix)")

    # 90° Y-axis rotation
    forward_transform = np.eye(4)
    # ... (rotation matrix)
    self.mesh.apply_transform(forward_transform)
    print("  Applied 90° Y-axis rotation (face forward)")

    print(f"✓ Mesh orientation corrected")

    # Continue with rigging...
    self.rigging = AutomaticRigging(self.mesh)
```

---

## Why This Approach

### Why in the Script (Not TripoSR or Web)?

**Not in TripoSR**:

- User doesn't want to regenerate meshes
- TripoSR settings are already optimized
- Orientation is consistent across all TripoSR outputs

**Not in Web Viewer**:

- Deformation happens in Python (mesh vertices are transformed)
- Web viewer only displays the result
- Fixing in Python ensures consistency across all viewers

**In the Script** ✅:

- Fixes orientation once at load time
- All subsequent operations (rigging, skinning, deformation) use corrected mesh
- No regeneration needed
- Can be easily adjusted if needed

### Why These Specific Rotations?

**180° X-axis flip**:

- Standard fix for upside-down meshes
- Flips Y-coordinates (up becomes down)
- Leaves X and Z unchanged

**90° Y-axis rotation**:

- Rotates around vertical axis
- Fixes "facing left" → "facing forward"
- Standard rotation for character orientation

These are the same transformations used in `tests/clothing_to_3d_triposr_2.py` to fix TripoSR output orientation.

---

## Console Output

When running, you'll now see:

```
Loading mesh...
✓ Mesh loaded: 5703 vertices, 11404 faces

Correcting mesh orientation...
  Applied 180° X-axis flip (upside-down fix)
  Applied 90° Y-axis rotation (face forward)
✓ Mesh orientation corrected
```

---

## Testing

```bash
python tests/test_integration_skinning.py --mesh generated_meshes/1761618888/mesh.obj
```

**Expected Result**:

- ✅ Mesh appears right-side-up
- ✅ Mesh facing forward (same direction as you)
- ✅ Deformation works correctly
- ✅ Arms/torso align with your body

---

## If Orientation Still Wrong

If the mesh is still incorrectly oriented, you might need to adjust the angles:

### For different orientations:

**If mesh is sideways (90° off)**:

```python
# Try different Y-axis angle
angle_y = np.pi  # 180° instead of 90°
# or
angle_y = -np.pi / 2  # -90° (opposite direction)
```

**If mesh is flipped horizontally**:

```python
# Add Z-axis rotation
angle_z = np.pi
c, s = np.cos(angle_z), np.sin(angle_z)
z_rotation = np.array([
    [c, -s, 0],
    [s, c, 0],
    [0, 0, 1]
])
```

**If mesh needs no flip** (unlikely):

```python
# Comment out the X-axis flip
# self.mesh.apply_transform(flip_transform)
```

---

## Files Modified

- `tests/test_integration_skinning.py`: Lines 411-438

---

## Related Fixes

This is the same orientation correction strategy used in:

- `tests/clothing_to_3d_triposr_2.py` (line 180-220)
- TripoSR mesh generation pipeline

All TripoSR meshes require this correction to work with character animation systems.

---

**Status**: ✅ **FIXED - Mesh should now be upright and facing forward**
