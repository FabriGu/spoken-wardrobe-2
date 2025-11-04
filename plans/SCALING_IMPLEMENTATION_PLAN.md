# Scaling Implementation Plan - Per-Bone Scale Calibration

## Overview

This plan implements **Solution 1: Per-Bone Scale Calibration** from `docs/SCALING_IMPROVEMENTS.md` to fix minor scaling/alignment mismatches between MediaPipe keypoints and rigged mesh bone joints.

**Goal**: Compute individual scale factors for each bone segment to match MediaPipe bone lengths to GLB mesh bone lengths.

**Expected Results**:
- 30-50% reduction in mesh warping
- Better alignment between keypoints and mesh joints
- RMSE: 0.05-0.10m → 0.02-0.05m (2-5cm error per joint)

**Estimated Time**: 1-2 hours

---

## Current State Analysis

### Root Cause
The scaling issue stems from **coordinate system mismatch**:

1. **MediaPipe World Coordinates**:
   - Origin at mid-hip
   - Arbitrary scale (varies per frame based on distance from camera)
   - Relative depth (Z) is not metric

2. **Blender Mesh Coordinates**:
   - Origin at mesh center or base
   - Absolute scale in meters (e.g., 1.7m tall)
   - Specific bone positions baked into mesh

3. **Current Calibration** (lines 339-364 in `test_a_rigged_clothing.py`):
   - Computes single global scale factor from torso height ratio
   - Applies Y-offset to align hip positions
   - **Problem**: Doesn't account for individual bone lengths

---

## Phase 1: Diagnostic Script (30 minutes)

Before implementing the fix, we need to measure the current alignment error to confirm the issue is primarily scale-related.

### Create `tests/diagnose_bone_alignment.py`

```python
"""
Diagnostic script to measure bone length mismatches and alignment errors
"""

import numpy as np
import trimesh
from pathlib import Path
import pickle
import sys

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.modules.rigged_mesh_loader import RiggedMesh
from src.modules.mediapipe_to_bones import MEDIAPIPE_LANDMARKS, BONE_MAPPING

# Simulated T-pose MediaPipe keypoints (from guide)
REFERENCE_KEYPOINTS = {
    'left_shoulder': np.array([-0.3, 0, 1.4]),
    'right_shoulder': np.array([0.3, 0, 1.4]),
    'left_hip': np.array([-0.15, 0, 0.9]),
    'right_hip': np.array([0.15, 0, 0.9]),
    'left_elbow': np.array([-0.6, 0, 1.4]),
    'left_wrist': np.array([-0.9, 0, 1.4]),
    'right_elbow': np.array([0.6, 0, 1.4]),
    'right_wrist': np.array([0.9, 0, 1.4]),
    'left_knee': np.array([-0.15, 0, 0.5]),
    'left_ankle': np.array([-0.15, 0, 0.0]),
    'right_knee': np.array([0.15, 0, 0.5]),
    'right_ankle': np.array([0.15, 0, 0.0]),
}

def compute_derived_points(keypoints):
    """Add mid_hip and mid_shoulder"""
    result = keypoints.copy()
    result['mid_hip'] = (keypoints['left_hip'] + keypoints['right_hip']) / 2
    result['mid_shoulder'] = (keypoints['left_shoulder'] + keypoints['right_shoulder']) / 2
    return result

def extract_mesh_bone_positions(glb_path):
    """
    Extract bone joint positions from GLB mesh
    Returns dict of keypoint_name → 3D position
    """
    mesh_obj = RiggedMesh(glb_path)

    bone_positions = {}

    # For each bone, extract head (start) and tail (end) positions
    for bone in mesh_obj.bones:
        # Get bone transform matrix
        transform = bone.local_transform
        head_pos = transform[:3, 3]  # Translation component

        # Store position by bone name
        # Map bone name to keypoint name
        bone_name_lower = bone.name.lower()

        if 'spine' in bone_name_lower:
            # Spine starts at mid_hip, ends at mid_shoulder
            bone_positions['mid_hip'] = head_pos
            # Approximate: spine end is head + length*direction
            # For simplicity, we'll compute from other bones
        elif 'left' in bone_name_lower:
            if 'upper_arm' in bone_name_lower:
                bone_positions['left_shoulder'] = head_pos
            elif 'lower_arm' in bone_name_lower:
                bone_positions['left_elbow'] = head_pos
                # Wrist is tail of this bone
            elif 'upper_leg' in bone_name_lower:
                bone_positions['left_hip'] = head_pos
            elif 'lower_leg' in bone_name_lower:
                bone_positions['left_knee'] = head_pos
        elif 'right' in bone_name_lower:
            if 'upper_arm' in bone_name_lower:
                bone_positions['right_shoulder'] = head_pos
            elif 'lower_arm' in bone_name_lower:
                bone_positions['right_elbow'] = head_pos
            elif 'upper_leg' in bone_name_lower:
                bone_positions['right_hip'] = head_pos
            elif 'lower_leg' in bone_name_lower:
                bone_positions['right_knee'] = head_pos

    return bone_positions

def diagnose_alignment(glb_path):
    """Run comprehensive alignment diagnostics"""

    print("=" * 60)
    print("BONE ALIGNMENT DIAGNOSTIC REPORT")
    print("=" * 60)
    print(f"\nMesh: {glb_path}\n")

    # Get mesh bone positions
    mesh_positions = extract_mesh_bone_positions(glb_path)

    # Compute derived points for reference keypoints
    ref_keypoints = compute_derived_points(REFERENCE_KEYPOINTS)

    # Compare bone lengths
    print("\n--- BONE LENGTH COMPARISON ---\n")

    bone_errors = []

    for bone_name, (start_key, end_key, parent) in BONE_MAPPING.items():
        if start_key not in ref_keypoints or end_key not in ref_keypoints:
            continue

        # MediaPipe bone length
        mp_start = ref_keypoints[start_key]
        mp_end = ref_keypoints[end_key]
        mp_length = np.linalg.norm(mp_end - mp_start)

        # Mesh bone length (if we have positions)
        if start_key in mesh_positions and end_key in mesh_positions:
            mesh_start = mesh_positions[start_key]
            mesh_end = mesh_positions[end_key]
            mesh_length = np.linalg.norm(mesh_end - mesh_start)

            # Compare
            ratio = mesh_length / (mp_length + 1e-6)
            error_pct = abs(1.0 - ratio) * 100

            bone_errors.append({
                'bone': bone_name,
                'mp_length': mp_length,
                'mesh_length': mesh_length,
                'ratio': ratio,
                'error_pct': error_pct
            })

            print(f"{bone_name:20s}  MP: {mp_length:.3f}m  Mesh: {mesh_length:.3f}m  "
                  f"Ratio: {ratio:.3f}  Error: {error_pct:.1f}%")

    # Summary statistics
    if bone_errors:
        print("\n--- SUMMARY ---\n")
        errors = [e['error_pct'] for e in bone_errors]
        print(f"Average bone length error: {np.mean(errors):.1f}%")
        print(f"Max bone length error: {np.max(errors):.1f}%")
        print(f"Min bone length error: {np.min(errors):.1f}%")

        # Identify problem bones
        high_error = [e for e in bone_errors if e['error_pct'] > 10]
        if high_error:
            print(f"\nBones with >10% error:")
            for e in high_error:
                print(f"  - {e['bone']}: {e['error_pct']:.1f}%")

    # Position alignment check
    print("\n--- JOINT POSITION ALIGNMENT ---\n")

    position_errors = []
    for key in ['left_shoulder', 'right_shoulder', 'left_elbow', 'right_elbow',
                'left_hip', 'right_hip', 'left_knee', 'right_knee']:
        if key in ref_keypoints and key in mesh_positions:
            mp_pos = ref_keypoints[key]
            mesh_pos = mesh_positions[key]
            error = np.linalg.norm(mp_pos - mesh_pos)
            position_errors.append(error)
            print(f"{key:20s}  Error: {error*100:.2f}cm")

    if position_errors:
        rmse = np.sqrt(np.mean(np.array(position_errors) ** 2))
        print(f"\nRMSE: {rmse*100:.2f}cm")
        print(f"Max position error: {np.max(position_errors)*100:.2f}cm")

    print("\n" + "=" * 60)
    print("RECOMMENDATION:")

    if bone_errors and np.mean([e['error_pct'] for e in bone_errors]) > 5:
        print("→ Significant bone length mismatch detected")
        print("→ Proceed with per-bone scale calibration (Solution 1)")
    else:
        print("→ Bone lengths are well-matched")
        print("→ Issue may be rotation or alignment (consider Solution 2: Procrustes)")

    print("=" * 60 + "\n")

if __name__ == "__main__":
    glb_path = "rigged_mesh/meshRigged_0.glb"
    diagnose_alignment(glb_path)
```

### Run Diagnostic

```bash
python tests/diagnose_bone_alignment.py
```

**Expected Output**: Report showing per-bone length mismatches and position errors.

**Decision Point**:
- If average bone error > 5%, proceed with per-bone scale calibration
- If average bone error < 5% but position errors high, consider rotation/alignment (Solution 2)

---

## Phase 2: Extract Mesh Bone Positions (30 minutes)

### Update `src/modules/rigged_mesh_loader.py`

Add method to extract bone joint positions:

```python
def get_bone_joint_positions(self) -> Dict[str, np.ndarray]:
    """
    Extract bone joint positions from rigged mesh

    Returns:
        Dict mapping keypoint names to 3D positions [x, y, z]
    """
    from src.modules.mediapipe_to_bones import GLB_BONE_NAME_MAPPING

    bone_positions = {}

    # Build reverse mapping: GLB name → MediaPipe name
    glb_to_mp = {}
    for mp_name, glb_names in GLB_BONE_NAME_MAPPING.items():
        for glb_name in glb_names:
            if glb_name in [b.name for b in self.bones]:
                glb_to_mp[glb_name] = mp_name
                break

    # Extract positions
    for bone in self.bones:
        if bone.name in glb_to_mp:
            mp_name = glb_to_mp[bone.name]

            # Bone head (start) position
            head_pos = bone.local_transform[:3, 3]

            # Store based on bone type
            if mp_name == 'spine':
                bone_positions['mid_hip'] = head_pos
                # mid_shoulder is the tail (approximate)
            elif 'upper_arm' in mp_name:
                bone_positions[mp_name.replace('upper_arm', 'shoulder')] = head_pos
            elif 'lower_arm' in mp_name:
                bone_positions[mp_name.replace('lower_arm', 'elbow')] = head_pos
            elif 'upper_leg' in mp_name:
                bone_positions[mp_name.replace('upper_leg', 'hip')] = head_pos
            elif 'lower_leg' in mp_name:
                bone_positions[mp_name.replace('lower_leg', 'knee')] = head_pos

    # Compute derived points
    if 'left_shoulder' in bone_positions and 'right_shoulder' in bone_positions:
        bone_positions['mid_shoulder'] = (bone_positions['left_shoulder'] +
                                          bone_positions['right_shoulder']) / 2

    return bone_positions
```

**Testing**:

```python
mesh = RiggedMesh("rigged_mesh/meshRigged_0.glb")
positions = mesh.get_bone_joint_positions()
print("Extracted positions:", positions.keys())
```

---

## Phase 3: Compute Per-Bone Scale Factors (30 minutes)

### Update `tests/test_a_rigged_clothing.py`

Modify `check_calibration()` method to compute per-bone scales:

```python
def check_calibration(self):
    """Check if user is in calibration pose (T-pose or A-pose)"""

    # ... existing detection code ...

    if is_calibrated:
        print("\n🎯 Calibration detected! Computing scale factors...")

        # Extract MediaPipe keypoints
        keypoints = {}
        for name, idx in MEDIAPIPE_LANDMARKS.items():
            lm = body.landmarks_world[idx]
            keypoints[name] = np.array([lm.x, -lm.y, lm.z], dtype=np.float32)

        # Add derived points
        keypoints['mid_hip'] = (keypoints['left_hip'] + keypoints['right_hip']) / 2
        keypoints['mid_shoulder'] = (keypoints['left_shoulder'] + keypoints['right_shoulder']) / 2

        # --- NEW: COMPUTE PER-BONE SCALE FACTORS ---

        print("\n=== Computing Per-Bone Scale Factors ===")

        # Get mesh bone positions
        mesh_positions = self.human_mesh.get_bone_joint_positions()

        self.bone_scale_factors = {}

        for bone_name, (start_key, end_key, parent) in BONE_MAPPING.items():
            if start_key not in keypoints or end_key not in keypoints:
                continue

            # MediaPipe bone vector
            mp_vec = keypoints[end_key] - keypoints[start_key]
            mp_length = np.linalg.norm(mp_vec)

            # Mesh bone vector (if available)
            if start_key in mesh_positions and end_key in mesh_positions:
                mesh_vec = mesh_positions[end_key] - mesh_positions[start_key]
                mesh_length = np.linalg.norm(mesh_vec)

                # Per-bone scale factor
                scale = mesh_length / (mp_length + 1e-6)
                self.bone_scale_factors[bone_name] = scale

                print(f"  {bone_name:20s}  MP: {mp_length:.3f}m  Mesh: {mesh_length:.3f}m  Scale: {scale:.3f}")
            else:
                # Fallback to global scale
                self.bone_scale_factors[bone_name] = self.scale_factor
                print(f"  {bone_name:20s}  Using global scale: {self.scale_factor:.3f}")

        # Compute global scale as average (for joints not covered)
        avg_scale = np.mean(list(self.bone_scale_factors.values()))
        print(f"\n  Average scale factor: {avg_scale:.3f}")

        # ... existing calibration code (hip offset, inverse bind matrices) ...
```

---

## Phase 4: Apply Per-Bone Scales During Animation (20 minutes)

### Update `animate_clothing()` in `test_a_rigged_clothing.py`

Apply per-bone scales when extracting keypoints:

```python
def animate_clothing(self, frame, body):
    """Deform clothing mesh based on body pose"""

    # Extract keypoints from MediaPipe
    current_keypoints = {}

    for name, idx in MEDIAPIPE_LANDMARKS.items():
        lm = body.landmarks_world[idx]

        # Flip Y and apply per-bone scale
        pos = np.array([lm.x, -lm.y, lm.z], dtype=np.float32)

        # --- NEW: APPLY PER-BONE SCALE ---

        # Determine which bone this keypoint belongs to
        bone_for_keypoint = self._get_bone_for_keypoint(name)

        if bone_for_keypoint and bone_for_keypoint in self.bone_scale_factors:
            scale = self.bone_scale_factors[bone_for_keypoint]
        else:
            scale = self.scale_factor  # Fallback to global

        pos *= scale
        pos[1] += self.hip_y_offset

        current_keypoints[name] = pos

    # Add derived points
    current_keypoints['mid_hip'] = (current_keypoints['left_hip'] +
                                    current_keypoints['right_hip']) / 2
    current_keypoints['mid_shoulder'] = (current_keypoints['left_shoulder'] +
                                         current_keypoints['right_shoulder']) / 2

    # ... rest of animation code ...

def _get_bone_for_keypoint(self, keypoint_name: str) -> str:
    """Map keypoint name to bone name"""
    from src.modules.mediapipe_to_bones import BONE_MAPPING

    for bone_name, (start_key, end_key, parent) in BONE_MAPPING.items():
        if keypoint_name == start_key or keypoint_name == end_key:
            return bone_name

    return None  # Not directly on a bone (use global scale)
```

---

## Phase 5: Testing and Validation (10-20 minutes)

### Test Procedure

1. **Run diagnostic before changes**:
   ```bash
   python tests/diagnose_bone_alignment.py > before_fix.txt
   ```

2. **Implement per-bone scale calibration** (follow Phases 2-4)

3. **Run test with per-bone scales**:
   ```bash
   python tests/test_a_rigged_clothing.py
   ```

4. **Observe improvements**:
   - Check terminal output for per-bone scale factors
   - Visually verify reduced warping in viewer
   - Specific areas to watch:
     - Elbows (often misaligned)
     - Knees (common scaling issue)
     - Shoulders (connection to spine)

5. **Measure improvement**:
   ```bash
   python tests/diagnose_bone_alignment.py > after_fix.txt
   diff before_fix.txt after_fix.txt
   ```

### Expected Results

**Before Fix**:
- Average bone length error: 8-15%
- RMSE: 5-10cm
- Visible warping at joints

**After Fix**:
- Average bone length error: 2-5%
- RMSE: 2-5cm
- Minimal warping, smooth deformation

---

## Phase 6: Commit and Document

### Commit Message Template

```
feat: implement per-bone scale calibration for improved alignment

- Add get_bone_joint_positions() to RiggedMesh
- Compute per-bone scale factors during T-pose calibration
- Apply bone-specific scales in animate_clothing()
- Add diagnostic script for alignment analysis

Results:
- Reduced average bone length error from X% to Y%
- RMSE improved from Xcm to Ycm
- Eliminated warping at elbows and knees

Closes #[issue-number]
```

### Update Documentation

Add to `docs/SCALING_IMPROVEMENTS.md`:

```markdown
## Implementation Status

✅ **Solution 1: Per-Bone Scale Calibration** - COMPLETED
- Implemented in commit [hash]
- Results: X% improvement in alignment
- RMSE: X.XXcm → Y.YYcm
```

---

## Troubleshooting

### Issue: Some bones still misaligned

**Cause**: Keypoint might belong to multiple bones (e.g., elbow is end of upper arm AND start of lower arm)

**Fix**: Use weighted average of parent and child bone scales:

```python
if keypoint_name in ['left_elbow', 'right_elbow', 'left_knee', 'right_knee']:
    # Average parent and child bone scales
    parent_bone = self._get_parent_bone(keypoint_name)
    child_bone = self._get_child_bone(keypoint_name)

    parent_scale = self.bone_scale_factors.get(parent_bone, self.scale_factor)
    child_scale = self.bone_scale_factors.get(child_bone, self.scale_factor)

    scale = (parent_scale + child_scale) / 2
```

### Issue: Scales varying too much per frame

**Cause**: MediaPipe keypoint jitter

**Fix**: Apply temporal smoothing to scale factors:

```python
# During calibration, average over multiple frames
self.scale_buffer = []

if is_calibrated:
    self.scale_buffer.append(computed_scales)

    if len(self.scale_buffer) >= 30:  # 1 second at 30fps
        # Average scales over buffer
        self.bone_scale_factors = {
            bone: np.mean([scales[bone] for scales in self.scale_buffer])
            for bone in computed_scales.keys()
        }
```

### Issue: Mesh now too small or too large

**Cause**: Applied per-bone scale on top of global scale

**Fix**: Remove global scale factor from per-bone computation:

```python
# In animate_clothing(), don't apply global scale if using per-bone:
if bone_for_keypoint and bone_for_keypoint in self.bone_scale_factors:
    pos *= self.bone_scale_factors[bone_for_keypoint]  # Only per-bone
    # Do NOT also multiply by self.scale_factor
```

---

## Next Steps (If Solution 1 Not Sufficient)

If per-bone scale calibration provides < 30% improvement:

1. **Run advanced diagnostics**: Check for rotation errors (not just scale)
2. **Consider Solution 2: Procrustes Alignment** (3-4 hours)
   - Finds optimal rigid transformation (scale + rotation + translation)
   - Better for cases where issue is both scale AND rotation
3. **See `docs/SCALING_IMPROVEMENTS.md`** for implementation details

---

## Success Criteria

✅ Per-bone scales computed and logged during calibration
✅ Different scale factors for arms vs legs vs torso
✅ Visual reduction in joint warping
✅ RMSE < 5cm
✅ Mesh maintains shape during all movements
✅ No regression in existing functionality

---

## File Checklist

Files to modify:
- [ ] `src/modules/rigged_mesh_loader.py` - Add `get_bone_joint_positions()`
- [ ] `tests/test_a_rigged_clothing.py` - Compute and apply per-bone scales
- [ ] Create `tests/diagnose_bone_alignment.py` - Diagnostic script

Files to update:
- [ ] `docs/SCALING_IMPROVEMENTS.md` - Mark Solution 1 as implemented
- [ ] `README.md` or `CLAUDE.md` - Update with new diagnostic tool

---

**END OF PLAN**
