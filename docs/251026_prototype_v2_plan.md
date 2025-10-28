# Prototype V2: Corrected Cage-Based Mesh Deformation System

**Date**: October 26, 2025  
**Status**: Implementation Plan

---

## 🎯 Core Goal

Create a **working prototype** that demonstrates cage-based mesh deformation using:

- BodyPix segmentation from the **same frame** used for mesh generation
- MediaPipe keypoints for real-time deformation
- Proper coordinate system handling
- Focus on **warping validation** over perfect alignment

---

## 📋 Key Principles

### 1. Consistency Throughout Pipeline

- **Use same image** for: BodyPix segmentation → AI generation → TripoSR → cage generation
- **Use same keypoints** for: reference pose and real-time comparison
- **Keep coordinate systems aligned** at every step

### 2. Cage Matches Mesh Area

- ❌ **OLD**: Cage covers entire body (even if mesh is just a shirt)
- ✅ **NEW**: Cage only covers body parts where mesh exists
- Example: For a t-shirt mesh → cage only covers torso + upper arms

### 3. Prioritize Warping Logic Over Visuals

- Cage deformation logic must be **mathematically correct**
- Visual alignment (scaling, positioning) is secondary
- Use debug logging to verify deformation occurs correctly

---

## 🔄 Corrected Pipeline

### PHASE 1: Generation (One-Time Per Clothing Item)

**Input**: Camera frame with person

**Steps**:

1. Run BodyPix segmentation → get 24 body part masks
2. Select body parts for clothing generation (e.g., torso + arms for shirt)
3. Run Stable Diffusion inpainting → generate clothing
4. Crop clothing using same mask → save `{name}_clothing.png`
5. Run TripoSR → generate 3D mesh → save `{name}_mesh.obj`

**NEW: Save Reference Data**:

```python
reference_data = {
    'original_frame': camera_frame,           # Original camera image
    'bodypix_masks': all_body_part_masks,     # All 24 part masks
    'selected_parts': ['torso', 'left_arm'],  # Parts used for mesh
    'mediapipe_keypoints_2d': keypoints_2d,   # Reference pose (2D)
    'mediapipe_keypoints_3d': keypoints_3d,   # Reference pose (3D)
    'frame_shape': (height, width),           # Image dimensions
    'mesh_path': 'path/to/mesh.obj',
    'timestamp': time.time()
}
```

**Saved Files**:

- `generated_images/{name}_full.png`
- `generated_images/{name}_clothing.png`
- `generated_meshes/{name}/mesh.obj`
- **NEW**: `generated_meshes/{name}/reference_data.pkl`

---

### PHASE 2: Cage Generation (One-Time Per Mesh)

**Input**: Saved reference data + 3D mesh

**Steps**:

1. Load mesh and reference data
2. For each selected body part (e.g., torso, left_arm):
   - Get BodyPix mask for that part
   - Calculate 2D bounding box from mask
   - Estimate 3D depth using mesh bounds and heuristics
   - Create 3D cage section (8 vertices per part)
3. Combine all sections into full cage
4. Store cage structure with anatomical labels

**Cage Structure**:

```python
cage_structure = {
    'torso': {
        'vertex_indices': [0, 1, 2, 3, 4, 5, 6, 7],  # Indices in cage.vertices
        'keypoints': ['left_shoulder', 'right_shoulder', 'left_hip', 'right_hip'],
        'bounds_2d': (x_min, y_min, x_max, y_max),
        'depth_range': (z_min, z_max)
    },
    'left_upper_arm': {
        'vertex_indices': [8, 9, 10, 11, 12, 13, 14, 15],
        'keypoints': ['left_shoulder', 'left_elbow'],
        ...
    },
    ...
}
```

**Key Improvement**: Cage covers **only** the body parts that have mesh, not the entire body.

---

### PHASE 3: Real-Time Deformation

**Input**: Current camera frame

**Steps**:

#### 3.1 Coordinate System Normalization

```python
# All coordinates converted to MESH-CENTERED NORMALIZED space
# Origin at mesh center, scale normalized to mesh bounds

def normalize_to_mesh_space(keypoints_2d, frame_shape, mesh_bounds):
    """
    Convert 2D keypoints (pixels) to mesh-centered normalized coordinates

    Args:
        keypoints_2d: {name: (x_px, y_px)}
        frame_shape: (height, width)
        mesh_bounds: mesh.bounds (2x3 array)

    Returns:
        {name: (x_norm, y_norm)}
    """
    h, w = frame_shape
    mesh_center = mesh_bounds.mean(axis=0)
    mesh_size = mesh_bounds[1] - mesh_bounds[0]

    normalized = {}
    for name, (x_px, y_px) in keypoints_2d.items():
        # Convert pixels to [-1, 1]
        x_norm = (x_px / w) * 2 - 1
        y_norm = 1 - (y_px / h) * 2  # Flip Y

        # Scale to mesh dimensions (X, Y only)
        x_mesh = x_norm * mesh_size[0] / 2
        y_mesh = y_norm * mesh_size[1] / 2

        normalized[name] = (x_mesh, y_mesh)

    return normalized
```

#### 3.2 Keypoint Delta Computation

```python
def compute_keypoint_delta(reference_kpts, current_kpts):
    """
    Compute translation vector for each keypoint

    Args:
        reference_kpts: {name: (x, y)} from generation phase
        current_kpts: {name: (x, y)} from current frame

    Returns:
        {name: (dx, dy)}
    """
    deltas = {}
    for name in reference_kpts.keys():
        if name in current_kpts:
            ref = np.array(reference_kpts[name])
            cur = np.array(current_kpts[name])
            deltas[name] = cur - ref
    return deltas
```

#### 3.3 Cage Section Deformation

```python
def deform_cage_section(cage_structure, section_name, keypoint_deltas, enable_z=False):
    """
    Apply translation to a cage section based on relevant keypoints

    Args:
        cage_structure: Dict with section info
        section_name: e.g., 'torso', 'left_upper_arm'
        keypoint_deltas: {name: (dx, dy, dz)}
        enable_z: Whether to use Z-axis deformation

    Returns:
        deformed_vertices: np.ndarray (N, 3)
    """
    section = cage_structure[section_name]
    vertex_indices = section['vertex_indices']
    relevant_keypoints = section['keypoints']

    # Compute mean translation from relevant keypoints
    translations = []
    for kpt_name in relevant_keypoints:
        if kpt_name in keypoint_deltas:
            delta = keypoint_deltas[kpt_name]
            if not enable_z:
                delta = np.array([delta[0], delta[1], 0])  # Zero out Z
            translations.append(delta)

    if len(translations) == 0:
        return original_cage_vertices[vertex_indices]

    mean_translation = np.mean(translations, axis=0)

    # Apply translation to all vertices in section
    deformed = original_cage_vertices[vertex_indices] + mean_translation

    return deformed
```

#### 3.4 MVC-Based Mesh Deformation

```python
# MVC weights computed ONCE during cage generation
mvc_weights = compute_mvc_weights(mesh.vertices, cage.vertices)  # (M, N)

# Each frame: apply to deformed cage
deformed_mesh_vertices = mvc_weights @ deformed_cage_vertices  # (M, 3)
```

---

## 🔧 Prototype V2 Features

### Toggle: 2D vs 3D Warping

**Command-line argument**:

```bash
python tests/test_integration_v2.py --enable-z-warp
```

**Default: 2D-only warping**

- Uses only X, Y from keypoints
- Z remains fixed at mesh depth
- Validates that warping logic works correctly

**With --enable-z-warp**

- Uses X, Y, Z from MediaPipe
- Z scaled by 1000 (MediaPipe units → mesh units)
- May have incorrect depth due to uncalibrated Z

### Debug Logging (Every 2 Seconds)

**Logged Information**:

```
[DEBUG] Frame: 120
  Keypoints detected: 13/13
  Mesh position: (x, y, z)
  Mesh scale: (width, height, depth)
  Cage deformation: mean delta = (dx, dy, dz)
  Vertex range: X[min, max], Y[min, max], Z[min, max]
  Mesh visible: (on-screen check based on bounds)
```

**Web Viewer Debug Display**:

- Real-time mesh bounds
- Camera position
- Mesh visibility status
- Deformation magnitude

### No Automatic Scaling (For Now)

**Problem**: Scaling can make mesh too large/small to see

**Solution**:

- Keep mesh at original size
- Log mesh bounds and camera position
- User manually adjusts camera with OrbitControls + WASD
- If mesh is offscreen, debug log will show coordinates

---

## 📊 Validation Checklist

### ✅ Success Criteria

1. **Cage Matches Mesh**:

   - [ ] Cage only covers body parts where mesh exists
   - [ ] No cage vertices for missing body parts (e.g., head, legs)
   - [ ] Cage structure dict has correct anatomical sections

2. **Coordinate System Consistency**:

   - [ ] All keypoints in same coordinate system
   - [ ] Mesh stays centered at origin
   - [ ] Cage vertices in same space as mesh vertices

3. **Deformation Occurs**:

   - [ ] Moving left → mesh warps left (X-axis)
   - [ ] Moving up → mesh warps up (Y-axis)
   - [ ] With `--enable-z-warp`: Moving forward → mesh changes (Z-axis)

4. **Mesh Visibility**:
   - [ ] Mesh visible in web viewer
   - [ ] Does not disappear when keypoints detected
   - [ ] Debug log shows reasonable position/scale

---

## 🚀 Future Scaling (Not in V2)

### Phase 2A: Proper Camera Calibration

**Problem**: MediaPipe Z-axis is unreliable for absolute depth

**Solution**: Use depth estimation (Depth Anything or MiDaS)

**Approach** (from `s0_consistent_skeleton_2D_3D.py`):

1. Capture background depth (empty scene)
2. Capture user depth (with person)
3. Compute relative depth: `depth_relative = depth_user - depth_background`
4. Use relative depth for Z-coordinate calibration

**Implementation**:

```python
# One-time calibration per camera setup
background_depth = get_depth_map(empty_frame)

# During generation
user_depth = get_depth_map(frame_with_person)
relative_depth = user_depth - background_depth

# Apply to keypoints
for name, (x, y) in keypoints_2d.items():
    z_depth = relative_depth[y, x]
    keypoints_3d[name] = (x, y, z_depth)
```

**Files to Reference**:

- `tests_backup/s0_consistent_skeleton_2D_3D_1.py` (latest version with updated calibration)
- `calibration_data/*.pkl` files

### Phase 2B: Dynamic Mesh Scaling

**Problem**: Mesh size doesn't match user's body size in camera

**Solution**: Scale mesh based on keypoint spread

```python
# Compute reference spread
ref_shoulder_width = distance(ref_left_shoulder, ref_right_shoulder)

# Compute current spread
cur_shoulder_width = distance(cur_left_shoulder, cur_right_shoulder)

# Scale factor
scale_factor = cur_shoulder_width / ref_shoulder_width

# Apply to mesh
mesh.vertices *= scale_factor
cage.vertices *= scale_factor
```

### Phase 2C: Rigid Transformation (Not Just Translation)

**Problem**: Bodies rotate and scale, not just translate

**Solution**: Estimate rigid transformation (rotation + scale + translation)

**Approach**:

```python
from scipy.spatial.transform import Rotation

# Get corresponding keypoint sets
ref_points = np.array([reference_kpts[k] for k in keypoint_names])
cur_points = np.array([current_kpts[k] for k in keypoint_names])

# Estimate transformation using SVD (Kabsch algorithm)
rotation, translation, scale = estimate_rigid_transform(ref_points, cur_points)

# Apply to cage
cage_vertices_deformed = scale * (rotation @ cage_vertices.T).T + translation
```

### Phase 2D: Handle Partial Clothing

**Problem**: Not all meshes cover the same body parts

**Solution**: Dynamic cage generation based on mesh coverage

**Approach**:

1. Analyze mesh vertices → determine which body parts are covered
2. Only create cage sections for covered parts
3. Ignore keypoints for uncovered parts

**Example**:

- T-shirt mesh: Only torso + upper arms → ignore leg/foot keypoints
- Pants mesh: Only pelvis + legs → ignore arm keypoints

---

## 🧪 Testing Strategy

### Test 1: Cage Structure Validation

```bash
python 251025_data_verification/verify_cage_structure.py \
    --mesh generated_meshes/0/mesh.obj \
    --reference generated_meshes/0/reference_data.pkl
```

**Expected Output**:

- Cage has sections only for body parts in mesh
- Each section has 8 vertices
- Anatomical labels match selected parts

### Test 2: Coordinate System Verification

```bash
python 251025_data_verification/verify_coordinate_system.py \
    --mesh generated_meshes/0/mesh.obj
```

**Expected Output**:

- Mesh centered at origin
- Keypoints in mesh-normalized space
- All coordinates in same scale

### Test 3: 2D Warping (Default)

```bash
python tests/test_integration_v2.py \
    --mesh generated_meshes/0/mesh.obj \
    --reference generated_meshes/0/reference_data.pkl
```

**Expected Behavior**:

- Move left/right → mesh warps in X
- Move up/down → mesh warps in Y
- Z remains constant

### Test 4: 3D Warping (With Z-axis)

```bash
python tests/test_integration_v2.py \
    --mesh generated_meshes/0/mesh.obj \
    --reference generated_meshes/0/reference_data.pkl \
    --enable-z-warp
```

**Expected Behavior**:

- Move forward/back → mesh changes (may be incorrect due to uncalibrated Z)

---

## 📝 Key Differences from V1

| Aspect                | V1 (Current)                         | V2 (Corrected)                  |
| --------------------- | ------------------------------------ | ------------------------------- |
| **Cage Source**       | User body in camera                  | Same image as mesh generation   |
| **Cage Coverage**     | Entire body                          | Only mesh-covered parts         |
| **Coordinate System** | Mixed (pixels, mesh, normalized)     | Single normalized mesh-centered |
| **Keypoint Usage**    | Raw MediaPipe (pixel coords)         | Normalized to mesh space        |
| **Deformation**       | Regenerate cage each frame           | Deform existing cage with delta |
| **Z-Axis**            | Always used (incorrect scaling)      | Toggle 2D/3D mode               |
| **Scaling**           | Automatic (causes visibility issues) | None (manual camera adjustment) |
| **Debug Info**        | Minimal                              | Extensive logging + web display |

---

## 🎓 Learnings & Principles

### 1. Consistency is Critical

- Using different images for different pipeline steps introduces errors
- Every transformation must preserve coordinate system

### 2. Start Simple, Add Complexity

- 2D warping first → validate logic works
- Then add Z-axis → handle depth complexity
- Then add scaling → handle size matching
- Then add rotation → handle full rigid transform

### 3. Prioritize Correctness Over Features

- Correct cage structure > automatic scaling
- Correct deformation logic > pretty visuals
- Debug-ability > convenience

### 4. Depth is Hard

- MediaPipe Z is relative, not absolute
- Need separate depth estimation (Depth Anything, MiDaS)
- Calibration required per camera setup

### 5. Cage = Anatomical Structure

- Not just a bounding box
- Each section corresponds to body part
- Deformation respects anatomy

---

## 📚 References

### Papers & Methods

1. **Mean Value Coordinates (MVC)**:

   - Ju et al., "Mean value coordinates for closed triangular meshes" (2005)
   - Used for: Binding mesh vertices to cage

2. **Depth Estimation**:

   - MiDaS (Intel): Relative depth estimation
   - Depth Anything: Improved depth estimation
   - Used for: Z-axis calibration

3. **MediaPipe Pose**:

   - Google's pose estimation
   - Provides 33 3D keypoints
   - Used for: Real-time body tracking

4. **BodyPix**:
   - TensorFlow model for body segmentation
   - Provides 24 body part masks
   - Used for: Anatomical cage structure

### Existing Codebase

- `tests/s0_consistent_skeleton_2D_3D.py`: Z-axis calibration approach
- `calibration_data/*.pkl`: Saved calibration data structure
- `src/modules/ai_generation.py`: Generation pipeline
- `tests/enhanced_cage_utils.py`: Current cage generation (to be improved)

---

## ⚠️ Known Limitations of V2

1. **No Camera Calibration**: Z-axis will be incorrect without depth estimation
2. **No Automatic Scaling**: Mesh may appear too large/small
3. **No Rotation**: Only handles translation (user must face forward)
4. **Fixed Cage Topology**: Can't handle dynamic clothing types yet
5. **No Temporal Smoothing**: May jitter between frames

**These are acceptable for V2 prototype** → Focus is on validating core warping logic.

---

## ✅ V2 Implementation Checklist

- [ ] Update `ai_generation.py` to save reference data
- [ ] Create `enhanced_cage_utils_v2.py` with corrected cage generation
- [ ] Create `keypoint_mapper_v2.py` with proper coordinate handling
- [ ] Create `test_integration_v2.py` with 2D/3D toggle
- [ ] Update WebSocket server for better error handling
- [ ] Update HTML viewer with debug display
- [ ] Test with existing mesh (`generated_meshes/0/mesh.obj`)
- [ ] Validate warping in X, Y axes
- [ ] Document observed behavior for future improvements

---

**End of Plan** - Ready for implementation! 🚀
