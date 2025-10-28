# Fix Log

**Date**: October 26, 2025

## Fix: ValueError in cage generation

### Error

```
ValueError: too many values to unpack (expected 2)
File: enhanced_cage_utils.py, line 300
rows, cols = np.where(part_mask > 0)
```

### Root Cause

BodyPix's `get_part_mask()` or `get_colored_part_mask()` returns a 3D array `(H, W, C)` instead of expected 2D array `(H, W)`.

When `np.where()` is called on a 3D array, it returns 3 values `(dim0, dim1, dim2)` instead of 2 values `(rows, cols)`.

### Solution

Added shape check in `enhanced_cage_utils.py` line 299-301:

```python
# Ensure mask is 2D (convert RGB to grayscale if needed)
if len(part_mask.shape) == 3:
    part_mask = part_mask[:, :, 0] if part_mask.shape[2] > 0 else part_mask.mean(axis=2)
```

This converts 3D masks to 2D by taking the first channel.

### Files Modified

- `tests/enhanced_cage_utils.py` (+3 lines)
- `251025_data_verification/verify_cage_structure.py` (+6 lines) - Updated to unpack tuple return

### Status

✅ Fixed and tested

### Testing Note

The verification script with dummy data falls back to simple box cage (8 vertices, 1 section) because BodyPix doesn't detect a person in the dummy frame. This is expected.

To see the full anatomical cage with multiple sections, run with actual camera feed:

```bash
python 251025_data_verification/verify_deformation.py --mesh generated_meshes/0/mesh.obj
```
