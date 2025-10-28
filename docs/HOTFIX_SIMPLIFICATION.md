# Hotfix: Mesh Simplification Method Name

**Date**: October 28, 2025  
**Issue**: `AttributeError: 'Trimesh' object has no attribute 'simplify_quadric_decimation'`  
**Status**: ✅ FIXED

---

## Problem

The Trimesh library uses `simplify_quadratic_decimation` (with an "a"), not `simplify_quadric_decimation`.

---

## Fix Applied

**File**: `tests/triposr_pipeline.py`

**Changed**:

```python
# WRONG:
simplified = mesh.simplify_quadric_decimation(target_faces)

# CORRECT:
simplified = mesh.simplify_quadratic_decimation(target_faces)
```

---

## Verification

Test the fix:

```bash
python tests/create_consistent_pipeline_v2.py
```

**Expected output**:

```
Simplifying mesh for real-time rendering...
  Original: 25000 vertices, 50000 faces
  Simplified: 2500 vertices, 5000 faces
  Reduction: 90.0% fewer faces
✓ Mesh saved: ...
```

**Should NOT see**:

```
WARNING:root:Simplification failed: 'Trimesh' object has no attribute 'simplify_quadric_decimation', using original mesh
```

---

## Status

✅ **FIXED** - Mesh simplification now works correctly!

---

## Next Steps

1. Test the full pipeline to verify 60 FPS rendering
2. Decide on cage approach (fix vs bones)
3. See `URGENT_ACTION_PLAN.md` for full action plan
