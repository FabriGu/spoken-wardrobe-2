# Scaling & Alignment Improvements for LBS Animation

## Current State ✅

**What Works:**
- 9-bone manual rigging with Blender
- LBS deformation with proper bone transforms
- All limbs move correctly with body tracking
- Mesh maintains recognizable shape
- No warping or explosion

**Current Issue:**
- Minor scaling/alignment mismatch between MediaPipe keypoints and mesh bones
- Causes slight warping in some areas
- Some points are slightly off, affecting deformation quality

---

## Root Cause Analysis

The scaling issue stems from **coordinate system mismatch** between:

1. **MediaPipe World Coordinates**:
   - Origin at mid-hip
   - Arbitrary scale (varies per frame based on distance from camera)
   - Y-axis points UP in world space (after flip)
   - Relative depth (Z) is not metric

2. **Blender Mesh Coordinates**:
   - Origin at mesh center or base
   - Absolute scale in meters (e.g., 1.7m tall)
   - Specific bone positions baked into mesh

3. **Current Calibration** (lines 339-364 in `test_a_rigged_clothing.py`):
   - Computes scale factor from torso height ratio
   - Applies Y-offset to align hip positions
   - **Problem**: Uses approximate values (55%, 85%) for hip/shoulder heights
   - **Issue**: Doesn't account for individual bone lengths

---

## Proposed Solutions (Ranked by Complexity)

### 🟢 Solution 1: Per-Bone Scale Calibration (LOW Complexity)

**Concept**: Instead of one global scale factor, compute individual scale factors for each bone segment.

**Implementation**:
```python
# During calibration (T-pose):
for bone_name in ['left_upper_arm', 'left_lower_arm', ...]:
    start_key, end_key = BONE_CONNECTIONS[bone_name]

    # MediaPipe bone length
    mp_length = np.linalg.norm(curr_kp[end_key] - curr_kp[start_key])

    # Mesh bone length (from Blender)
    mesh_start = bone_positions_in_mesh[start_key]  # From rigging
    mesh_end = bone_positions_in_mesh[end_key]
    mesh_length = np.linalg.norm(mesh_end - mesh_start)

    # Per-bone scale
    bone_scale_factors[bone_name] = mesh_length / mp_length
```

**Pros**:
- ✅ Accounts for individual limb proportions
- ✅ Better accuracy for arms vs legs vs torso
- ✅ Simple to implement (30-45 min)
- ✅ No mesh changes needed

**Cons**:
- ❌ Still doesn't handle rotation/twist issues
- ❌ Assumes bones are straight lines

**Complexity**: ⭐ LOW (1-2 hours)
**Expected Improvement**: 30-50% better alignment

---

### 🟡 Solution 2: Procrustes Alignment (MEDIUM Complexity)

**Concept**: Use Procrustes analysis to find optimal rigid transformation (scale, rotation, translation) between MediaPipe keypoints and mesh bone positions.

**Implementation**:
```python
from scipy.spatial import procrustes

# During calibration:
# Source: MediaPipe keypoints (9 bone endpoints)
mp_points = np.array([
    curr_kp['left_shoulder'], curr_kp['left_elbow'], curr_kp['left_wrist'],
    curr_kp['right_shoulder'], curr_kp['right_elbow'], curr_kp['right_wrist'],
    # ... etc for all keypoints
])

# Target: Mesh bone positions (from Blender rigging)
mesh_points = np.array([
    mesh_bone_positions['left_shoulder'],
    mesh_bone_positions['left_elbow'],
    # ... etc
])

# Compute optimal transformation
mtx1, mtx2, disparity = procrustes(mesh_points, mp_points)

# mtx2 contains the aligned MediaPipe points
# Extract scale, rotation from transformation
```

**Pros**:
- ✅ Mathematically optimal alignment
- ✅ Handles scale + rotation + translation
- ✅ Proven algorithm (used in shape matching)
- ✅ One transformation for all points

**Cons**:
- ❌ Requires knowing mesh bone positions (need to extract from GLB)
- ❌ More complex implementation
- ❌ May overfit to T-pose

**Complexity**: ⭐⭐ MEDIUM (3-4 hours)
**Expected Improvement**: 60-80% better alignment

---

### 🟡 Solution 3: Iterative Closest Point (ICP) (MEDIUM-HIGH Complexity)

**Concept**: Iteratively refine alignment by minimizing distance between MediaPipe keypoints and mesh bone endpoints.

**Implementation**:
```python
from scipy.optimize import minimize

def alignment_error(params, mp_keypoints, mesh_bone_positions):
    scale, tx, ty, tz, rx, ry, rz = params

    # Apply transformation to MediaPipe keypoints
    transformed = scale * rotate(mp_keypoints, rx, ry, rz) + [tx, ty, tz]

    # Compute distance to mesh bones
    error = np.sum((transformed - mesh_bone_positions) ** 2)
    return error

# Optimize
initial_guess = [scale_factor, 0, hip_y_offset, 0, 0, 0, 0]
result = minimize(alignment_error, initial_guess, method='Powell')

# Use optimized transformation
optimal_scale, tx, ty, tz, rx, ry, rz = result.x
```

**Pros**:
- ✅ Highly accurate alignment
- ✅ Automatically finds best transformation
- ✅ Can handle non-rigid differences

**Cons**:
- ❌ Computationally expensive (may slow calibration)
- ❌ Requires good initial guess
- ❌ May be overkill for this problem

**Complexity**: ⭐⭐⭐ MEDIUM-HIGH (5-6 hours)
**Expected Improvement**: 80-90% better alignment

---

### 🟠 Solution 4: Multi-Point Correspondence (HIGH Complexity)

**Concept**: Instead of just keypoints, use additional anatomical landmarks (e.g., elbow inside/outside, knee cap) for more accurate correspondence.

**Implementation**:
- Extract more landmarks from MediaPipe (hand landmarks, face landmarks)
- Use these to constrain bone orientations
- Build denser correspondence map

**Pros**:
- ✅ Very accurate
- ✅ Handles twist/rotation better

**Cons**:
- ❌ Requires MediaPipe hand/face tracking
- ❌ Complex implementation
- ❌ More points to track = more noise

**Complexity**: ⭐⭐⭐⭐ HIGH (8-10 hours)
**Expected Improvement**: 90-95% better alignment

---

### 🔴 Solution 5: Learning-Based Alignment (VERY HIGH Complexity)

**Concept**: Train a small neural network to learn the mapping from MediaPipe keypoints to mesh bone positions.

**Implementation**:
- Collect training data (MediaPipe poses + corresponding mesh poses)
- Train regressor: `mesh_bones = NN(mediapipe_keypoints)`
- Use learned mapping during runtime

**Pros**:
- ✅ Can handle complex non-linear mappings
- ✅ Adapts to user's specific body proportions
- ✅ Best possible accuracy

**Cons**:
- ❌ Requires training data collection
- ❌ Need ML framework (PyTorch/TensorFlow)
- ❌ Overfitting risk with small dataset
- ❌ Overkill for this application

**Complexity**: ⭐⭐⭐⭐⭐ VERY HIGH (20+ hours)
**Expected Improvement**: 95-99% better alignment

---

## Recommended Approach

### Phase 1: Quick Win (Implement Now)
**Solution 1: Per-Bone Scale Calibration**

- ⏱️ Time: 1-2 hours
- 📈 Impact: 30-50% improvement
- 🎯 Lowest risk, immediate results

**Why**: Simple to implement, no dependencies, reversible if doesn't work well.

### Phase 2: If More Accuracy Needed
**Solution 2: Procrustes Alignment**

- ⏱️ Time: 3-4 hours
- 📈 Impact: 60-80% improvement
- 🎯 Mathematically sound, proven technique

**Why**: Optimal rigid transformation, handles global alignment issues.

### Phase 3: Polish (Optional)
**Fine-tuning per-limb**:
- Different scale factors for arms vs legs
- Account for camera distance variations
- Smooth temporal changes (Kalman filter)

---

## Implementation Plan for Solution 1

### Step 1: Extract Bone Lengths from Mesh (30 min)

Add to `RiggedMesh` class:
```python
def get_bone_positions(self):
    """Extract bone endpoint positions from mesh"""
    bone_positions = {}
    for bone in self.bones:
        # Get bone head/tail positions from local_transform
        # Store in dict keyed by bone name
    return bone_positions
```

### Step 2: Compute Per-Bone Scales (30 min)

Update `check_calibration()` in `test_a_rigged_clothing.py`:
```python
# After computing keypoints...
self.bone_scale_factors = {}

for bone_name in MEDIAPIPE_BONE_ORDER:
    start_key, end_key = MEDIAPIPE_BONE_CONNECTIONS[bone_name]

    # MediaPipe length
    mp_vec = keypoints[end_key] - keypoints[start_key]
    mp_length = np.linalg.norm(mp_vec)

    # Mesh length (from rigging)
    mesh_positions = self.human_mesh.get_bone_positions()
    mesh_start = mesh_positions[start_key]
    mesh_end = mesh_positions[end_key]
    mesh_length = np.linalg.norm(mesh_end - mesh_start)

    # Per-bone scale
    self.bone_scale_factors[bone_name] = mesh_length / (mp_length + 1e-6)
```

### Step 3: Apply Per-Bone Scales (30 min)

Update `animate_clothing()`:
```python
# For each keypoint:
for name, idx in MEDIAPIPE_LANDMARKS.items():
    lm = body.landmarks_world[idx]
    lm[1] = -lm[1]

    # Apply per-bone scale (if keypoint belongs to a bone)
    bone_name = get_bone_for_keypoint(name)
    if bone_name in self.bone_scale_factors:
        lm *= self.bone_scale_factors[bone_name]
    else:
        lm *= self.scale_factor  # Fallback to global scale

    current_keypoints[name] = lm
```

---

## Measuring Improvement

Add metrics to track alignment quality:

```python
def compute_alignment_error(mp_keypoints, mesh_bone_positions):
    """Compute RMSE between keypoints and mesh bones"""
    errors = []
    for key in mp_keypoints:
        mp_pos = mp_keypoints[key]
        mesh_pos = mesh_bone_positions[key]
        error = np.linalg.norm(mp_pos - mesh_pos)
        errors.append(error)

    rmse = np.sqrt(np.mean(np.array(errors) ** 2))
    return rmse

# After calibration:
print(f"Alignment RMSE: {rmse:.4f}m")
```

**Target RMSE**:
- Current (estimated): 0.05-0.10m (5-10cm error per joint)
- After Solution 1: 0.02-0.05m (2-5cm)
- After Solution 2: 0.01-0.02m (1-2cm)

---

## Alternative: Quick Hacks (If Time-Constrained)

### Hack 1: Manual Offset Tuning (15 min)
Add manual per-bone offsets in config:
```python
BONE_OFFSETS = {
    'left_upper_arm': [0.02, 0, 0],   # 2cm offset in X
    'right_lower_leg': [0, -0.01, 0], # 1cm down in Y
    # ... etc
}
```

**Pros**: Very fast
**Cons**: Not generalizable, brittle

### Hack 2: Smooth Temporal Filtering (30 min)
Apply exponential moving average to keypoint positions:
```python
# Smooth jitter
alpha = 0.7  # Smoothing factor
current_keypoints[name] = alpha * prev_keypoints[name] + (1-alpha) * new_keypoint
```

**Pros**: Reduces jitter, looks smoother
**Cons**: Doesn't fix alignment, adds lag

---

## Conclusion

**Recommended Path**:
1. ✅ **Now**: Implement Solution 1 (Per-Bone Scale) - 1-2 hours
2. ⏸️ **Test**: Evaluate improvement, measure RMSE
3. 🤔 **Decision**: If still not good enough, implement Solution 2 (Procrustes)

**Expected Timeline**:
- Solution 1: 1-2 hours → 30-50% better
- Solution 2 (if needed): +3-4 hours → 60-80% better

**Total**: 1-6 hours depending on quality requirements

---

## Questions to Consider

Before implementing:

1. **How good is "good enough"?**
   - Are current results acceptable for demo/prototype?
   - Do you need production-quality alignment?

2. **What's the use case?**
   - Real-time preview (can tolerate some error)
   - Video recording (needs better quality)
   - Production AR app (needs highest quality)

3. **Camera setup**:
   - Fixed camera position?
   - User moving around?
   - Different distances from camera?

**My Recommendation**: Start with Solution 1 (per-bone scale). It's low-risk, fast to implement, and will likely give you the improvement you need. You can always add Solution 2 later if needed.
