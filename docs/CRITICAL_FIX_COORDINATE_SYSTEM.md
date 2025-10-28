# CRITICAL FIX: Coordinate System Mismatch

**Date**: October 28, 2025  
**Issue**: Mesh collapsed to (0, 0, 0) - all vertices at origin  
**Severity**: CRITICAL - Complete system failure

---

## The Problem

### Symptoms

- WebSocket connected ✅
- Data flowing ✅
- But mesh bounds: `X[0.000, 0.000]` - completely collapsed
- Mesh invisible in web viewer

### Root Cause: Coordinate System Mismatch

**During Calibration (OLD CODE)**:

```python
# calibrate_from_frame() - Line 478
keypoints_px[i] = [lm.x * w, lm.y * h, lm.z * w]
# Example: keypoints in PIXEL SPACE
# left_shoulder = (450, 200, 100)
# right_shoulder = (650, 200, 100)

self.rigging.calibrate(keypoints_px)
# Computes bone transforms in PIXEL COORDINATES
# bind_bone_transforms['torso'][:3, 3] = (550, 200, 100)
```

**During Runtime (OLD CODE)**:

```python
# MediaPipeSkeletonDriver.extract_keypoints() - Line 349
x_norm = (x_px - w/2) / (body_scale / 2)
y_norm = (y_px - h/2) / (body_scale / 2)
z_norm = z_px / (body_scale / 2)
# Example: keypoints in NORMALIZED SPACE [-1, 1]
# left_shoulder = (-0.3, 0.5, 0.2)
# right_shoulder = (0.3, 0.5, 0.2)

# Computes bone transforms in NORMALIZED SPACE
# current_bone_transforms['torso'][:3, 3] = (0.0, 0.5, 0.2)
```

**The Fatal Multiplication**:

```python
# In deform_mesh() - Line 298
skinning_matrix = current_transform @ bind_inv_transform
#                 [normalized: 0.0-0.5]  @  [pixel: 450-650]^-1
#                 = INCOMPATIBLE SCALES!
#                 = Near-zero transformation
#                 = All vertices collapse to origin
```

### Why This Caused Complete Failure

Linear Blend Skinning formula:

```
v' = Σ wᵢ * Mᵢ * Mᵢ_bind_inv * v
```

Where:

- `Mᵢ` = current bone transform (was in **normalized space**: ~0.5 units)
- `Mᵢ_bind_inv` = inverse bind transform (was in **pixel space**: ~500 pixels)
- When multiplied: `0.5 @ inv(500)` ≈ `0.001` (essentially zero)
- Result: All vertices transform to origin `(0, 0, 0)`

---

## The Solution: Option 2 - Normalize Everything

### Strategy

Use **normalized space throughout** for all transformations:

- Calibration: Normalize keypoints BEFORE computing bone transforms
- Runtime: Use same normalization
- Both use identical coordinate system: **[-1, 1] centered at origin**

### Implementation

#### Step 1: Compute Body Scale First (Still in Pixel Space)

```python
# calibrate_from_frame() - NEW
# Extract keypoints in pixel space
keypoints_px = np.zeros((33, 3))
for i, lm in enumerate(results.pose_landmarks.landmark):
    keypoints_px[i] = [lm.x * w, lm.y * h, lm.z * w]

# Compute body scale from pixel coordinates (for normalization)
left_shoulder = keypoints_px[11]
right_shoulder = keypoints_px[12]
shoulder_dist = np.linalg.norm(right_shoulder - left_shoulder)
body_scale = shoulder_dist * 2.5  # e.g., 457.3 pixels
```

#### Step 2: Normalize Keypoints for Calibration

```python
# NOW normalize keypoints to model space
keypoints_normalized = np.zeros((33, 3))

for i, lm in enumerate(results.pose_landmarks.landmark):
    x_px = lm.x * w
    y_px = lm.y * h
    z_px = lm.z * w

    # Normalize to [-1, 1] space centered at origin
    keypoints_normalized[i, 0] = (x_px - w/2) / (body_scale / 2)
    keypoints_normalized[i, 1] = (y_px - h/2) / (body_scale / 2)
    keypoints_normalized[i, 2] = z_px / (body_scale / 2)
```

**Example normalized values**:

```
left_shoulder:  (-0.87, 0.45, 0.22)
right_shoulder: ( 0.87, 0.45, 0.22)
left_hip:       (-0.43, -0.65, 0.18)
```

#### Step 3: Calibrate with Normalized Keypoints

```python
# Calibrate rigging with NORMALIZED keypoints
self.rigging.calibrate(keypoints_normalized, body_scale)

# Now bind_bone_transforms are in NORMALIZED SPACE
# bind_bone_transforms['torso'][:3, 3] = (0.0, -0.1, 0.2)  ✅
```

#### Step 4: Runtime Uses Same Normalization

```python
# MediaPipeSkeletonDriver.extract_keypoints() - UNCHANGED
# Already normalizing the same way:
x_norm = (x_px - w/2) / (body_scale / 2)
y_norm = (y_px - h/2) / (body_scale / 2)
z_norm = z_px / (body_scale / 2)

# Now current_bone_transforms are in NORMALIZED SPACE
# current_bone_transforms['torso'][:3, 3] = (0.1, -0.2, 0.3)  ✅
```

#### Step 5: Consistent LBS

```python
# In deform_mesh() - UNCHANGED, but now works correctly!
skinning_matrix = current_transform @ bind_inv_transform
#                 [normalized: 0.0-0.5]  @  [normalized: 0.0-0.5]^-1
#                 = COMPATIBLE SCALES!  ✅
#                 = Proper transformation
#                 = Vertices deform correctly
```

---

## Files Modified

### `tests/test_integration_skinning.py`

**Lines 469-518**: `calibrate_from_frame()`

- Added body scale computation from pixel coordinates
- Added normalization step before calibration
- Pass normalized keypoints to `rigging.calibrate()`

**Lines 79-98**: `AutomaticRigging.calibrate()`

- Changed signature: `calibrate(initial_keypoints, body_scale)`
- Removed internal body scale computation
- Added documentation that keypoints are already normalized

---

## Verification

### Before Fix:

```
Mesh:
  Position: (0.000, 0.000, 0.000)
  Size: (0.000, 0.000, 0.000)
  Bounds: X[0.000, 0.000]  ← COLLAPSED!
```

### After Fix (Expected):

```
Mesh:
  Position: (0.123, -0.456, 0.089)
  Size: (0.445, 0.934, 0.272)
  Bounds: X[-0.226, 0.222]  ← PROPER RANGE!
```

---

## Why This Works Now

### Consistent Coordinate System

| Stage                  | Coordinate System       | Example Values        |
| ---------------------- | ----------------------- | --------------------- |
| **Pixel Extraction**   | Pixels (0-1280)         | `(450, 200, 100)`     |
| **Body Scale**         | Pixels                  | `457.3`               |
| **Normalization**      | Normalized [-1, 1]      | `(-0.87, 0.45, 0.22)` |
| **Calibration**        | Normalized [-1, 1]      | Same                  |
| **Bind Transforms**    | Normalized [-1, 1]      | `(0.0, -0.1, 0.2)`    |
| **Runtime Keypoints**  | Normalized [-1, 1]      | Same                  |
| **Current Transforms** | Normalized [-1, 1]      | Same                  |
| **LBS**                | Normalized @ Normalized | ✅ Compatible!        |
| **Deformed Mesh**      | Normalized [-1, 1]      | Proper range          |

### Mathematical Consistency

**Before** (BROKEN):

```
Mᵢ = 0.5      (normalized)
Mᵢ_bind = 500 (pixels)
Mᵢ_bind_inv = 0.002

Skinning: 0.5 @ 0.002 = 0.001  ← Near zero!
```

**After** (FIXED):

```
Mᵢ = 0.5           (normalized)
Mᵢ_bind = 0.48     (normalized)
Mᵢ_bind_inv = 2.08

Skinning: 0.5 @ 2.08 = 1.04  ← Proper magnitude!
```

---

## Testing

```bash
python tests/test_integration_skinning.py --mesh generated_meshes/1761618888/mesh.obj
```

1. Press SPACE → 5-second countdown
2. Get into T-pose
3. After calibration, check console:
   - Should show: "Body scale: 457.3 pixels"
   - Should show: "Keypoints space: Normalized (centered at origin)"
4. Open `tests/enhanced_mesh_viewer_v2.html`
5. **Expected**: Mesh visible, proper bounds, follows movement

### Success Criteria

✅ Mesh bounds NOT at origin  
✅ Mesh visible in web viewer  
✅ Mesh size reasonable (not 0.0)  
✅ Mesh follows body movement  
✅ No collapse to origin

---

## Key Insight

The fundamental principle of Linear Blend Skinning:

> **All transformations in the LBS formula MUST use the same coordinate system.**

If bind pose is in pixels, current pose MUST be in pixels.  
If bind pose is normalized, current pose MUST be normalized.

We chose normalized space because:

- ✅ Independent of frame resolution
- ✅ Centered at origin (cleaner math)
- ✅ Consistent [-1, 1] range
- ✅ Matches typical 3D graphics conventions

---

## Lessons Learned

1. **Coordinate systems are critical** - Mixing spaces causes catastrophic failure
2. **Document coordinate assumptions** - Every function should specify its coordinate system
3. **Test with actual data** - Console logging revealed the mismatch
4. **Understand the math** - LBS formula requires compatible coordinate systems

This was a **textbook coordinate system mismatch** - one of the most common bugs in 3D graphics programming!

---

**Status**: ✅ **FIXED - Ready for testing**
