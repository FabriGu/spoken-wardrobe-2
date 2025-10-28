# Hotfix: SimplifyModifier Dependencies

**Date**: October 28, 2025  
**Issues**: Missing BufferGeometryUtils dependency, wrong variable name  
**Status**: ✅ FIXED

---

## Issues Found

### Issue 1: Missing BufferGeometryUtils

```
⚠️ Simplification failed: THREE.SimplifyModifier relies on THREE.BufferGeometryUtils
```

**Cause**: SimplifyModifier depends on BufferGeometryUtils, which wasn't imported.

**Fix**: Added BufferGeometryUtils import before SimplifyModifier:

```html
<script src="https://cdn.jsdelivr.net/npm/three@0.128.0/examples/js/utils/BufferGeometryUtils.js"></script>
<script src="https://cdn.jsdelivr.net/npm/three@0.128.0/examples/js/modifiers/SimplifyModifier.js"></script>
```

---

### Issue 2: Wrong Variable Name

```
Error parsing WebSocket message: ReferenceError: clothingMesh is not defined
```

**Cause**: Used `clothingMesh` but the actual variable is `mesh`.

**Fix**: Changed line 466:

```javascript
// WRONG:
clothingMesh.geometry = meshGeometry;

// CORRECT:
mesh.geometry = meshGeometry;
```

---

## Files Changed

**File**: `tests/enhanced_mesh_viewer_v2.html`

**Changes**:

1. Line 166: Added BufferGeometryUtils import
2. Line 466: Fixed variable name (`clothingMesh` → `mesh`)

---

## Test Now

```bash
python tests/test_integration_v2.py
```

### Expected console output:

```
✓ Three.js initialized
Connecting to WebSocket server...
✓ WebSocket connected
🔄 Mesh too complex (15015 vertices), simplifying...
  Target: Remove 9009 vertices (60%)
  ✅ Simplified: 15015 → 6006 vertices (60.0% reduction)
  ✅ Faces: 30000 → 12000
```

### Should NOT see:

```
⚠️ Simplification failed: THREE.SimplifyModifier relies on THREE.BufferGeometryUtils
Error parsing WebSocket message: ReferenceError: clothingMesh is not defined
```

---

## Summary

✅ **BufferGeometryUtils imported** - SimplifyModifier now has required dependency  
✅ **Variable name fixed** - Using correct `mesh` variable  
✅ **Ready to test** - Mesh simplification should work now!

**Test it and let me know if you see good mesh quality with simplified vertices!** 🚀
