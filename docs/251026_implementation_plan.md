# Implementation Plan: Fixing Cage-Based Deformation

**Date**: October 26, 2025  
**Status**: Ready for Implementation

---

## 🎯 Problems Identified

### Problem 1: Mesh Orientation (90° Left Rotation)

**File**: `tests/clothing_to_3d_triposr_2.py`  
**Issue**: Mesh is upright but facing left instead of forward  
**Fix**: Add 90° rotation around Y-axis after upright correction

### Problem 2: Cage Has No Anatomical Structure

**Files**: `tests/enhanced_cage_utils.py`  
**Issue**: Cage generates vertices but doesn't return structure mapping  
**Current**: Returns only `cage_mesh`  
**Needed**: Return `(cage_mesh, cage_structure)` where structure maps vertices to body parts

### Problem 3: All Cage Vertices Move Together

**File**: `tests/keypoint_mapper.py`  
**Issue**: `simple_anatomical_mapping()` applies uniform transformation to all vertices  
**Current**: Translates entire cage based on centroid  
**Needed**: Transform each body part section independently

### Problem 4: Cage Not 3D

**Root Cause**: BodyPix gives 2D masks, cage generation needs to extrapolate to 3D  
**Current**: Only using 2D bounding boxes  
**Needed**: Estimate depth based on mesh bounds and BodyPix area ratios

---

## 🔍 Key Insights from User

1. **MediaPipe provides "bones"** - keypoint connections that define body structure
2. **BodyPix provides 2D area** - where each body part is located in the frame
3. **Need 3D extrapolation** - Both are 2D, need to infer Z-axis depth
4. **TripoSR depth info** - Can analyze mesh to understand depth distribution
5. **Stationary camera setup** - Can use depth calibration (Depth Anything) once at setup
6. **MediaPipe Z is inaccurate** - Need calibration for better depth estimates

---

## 📊 Data Available

### From MediaPipe:

- 2D keypoints: `(x, y)` in pixels
- 3D keypoints: `(x, y, z)` where z is relative depth (not metric)
- Keypoint connections: Define skeleton structure
- Available every frame (fast)

### From BodyPix:

- 24 body part masks (2D binary images)
- Person mask (full silhouette)
- Available at initialization (slow ~1400ms)

### From TripoSR Mesh:

- 3D clothing mesh vertices
- Bounding box in X, Y, Z
- Can analyze depth distribution (Z-axis extent)

---

## ✅ Implementation Strategy

### Phase 1: Fix Mesh Orientation (Quick Win)

Add Y-axis rotation to face forward

### Phase 2: Enhanced Cage Generation with Structure

Modify `generate_anatomical_cage()` to:

1. Create vertices for each body part
2. Track which vertices belong to which part
3. Return both mesh AND structure dict
4. Estimate 3D depth from mesh bounds

### Phase 3: Section-Wise Deformation

Modify `simple_anatomical_mapping()` to:

1. Accept cage_structure as parameter
2. For each body part section:
   - Get corresponding MediaPipe keypoints
   - Compute transformation (rotation + translation + scale)
   - Apply ONLY to vertices in that section
3. Keep sections that don't have keypoints stable

### Phase 4: Integration

Update `test_integration.py` to:

1. Store `cage_structure` from cage generation
2. Pass `cage_structure` to deformation method
3. Verify independent section movement

---

## 📝 Detailed Implementation

### Fix 1: Mesh Orientation

**File**: `tests/clothing_to_3d_triposr_2.py`  
**Location**: After the 180° flip (line ~371)

```python
# Step 1.6: Rotate to face forward (fix 90° left rotation)
# Add 90° rotation around Y-axis to face camera
logging.info("Applying 90° rotation around Y-axis (face forward)")
forward_transform = np.eye(4)
angle = np.pi / 2  # 90 degrees
c, s = np.cos(angle), np.sin(angle)
R_forward = np.array([
    [c, 0, s],
    [0, 1, 0],
    [-s, 0, c]
])
forward_transform[:3, :3] = R_forward
mesh_tri.apply_transform(forward_transform)
```

---

### Fix 2: Cage Structure Generation

**File**: `tests/enhanced_cage_utils.py`  
**Method**: `generate_anatomical_cage()`

**Key Changes**:

1. Build `cage_structure` dict as vertices are created:

```python
def generate_anatomical_cage(self, segmentation_data, frame_shape, subdivisions=3):
    height, width = frame_shape[:2]
    body_parts = segmentation_data['body_parts']

    cage_vertices = []
    cage_structure = {}  # NEW: Track structure

    # Get mesh bounds to estimate depth
    mesh_bounds = self.mesh.bounds
    mesh_depth = (mesh_bounds[1][2] - mesh_bounds[0][2])  # Z extent

    for part_name, part_group in self.body_part_groups.items():
        if part_name in body_parts:
            part_mask = body_parts[part_name]

            # Generate 3D cage vertices for this part
            part_vertices_3d = self.generate_part_cage_vertices_3d(
                part_mask,
                part_name,
                subdivisions,
                mesh_bounds,
                mesh_depth,
                frame_shape
            )

            # Store structure mapping
            start_idx = len(cage_vertices)
            cage_vertices.extend(part_vertices_3d)
            end_idx = len(cage_vertices)

            cage_structure[part_name] = {
                'vertex_indices': list(range(start_idx, end_idx)),
                'keypoints': self.get_keypoints_for_part(part_name)
            }

    # Create cage mesh
    cage_vertices = np.array(cage_vertices)
    hull = ConvexHull(cage_vertices)
    cage_mesh = trimesh.Trimesh(vertices=cage_vertices, faces=hull.simplices)

    return cage_mesh, cage_structure  # Return BOTH
```

2. Add method to generate 3D vertices from 2D mask:

```python
def generate_part_cage_vertices_3d(self, part_mask, part_name, subdivisions,
                                    mesh_bounds, mesh_depth, frame_shape):
    """
    Generate 3D cage vertices from 2D body part mask.
    Extrapolates depth based on mesh bounds and body part type.
    """
    height, width = frame_shape[:2]

    # Get 2D bounding box from mask
    rows, cols = np.where(part_mask > 0)
    if len(rows) == 0:
        return []

    # 2D bounds
    y_min, y_max = rows.min(), rows.max()
    x_min, x_max = cols.min(), cols.max()

    # Normalize to [-1, 1] range
    x_center = (x_min + x_max) / 2 / width - 0.5
    y_center = (y_min + y_max) / 2 / height - 0.5
    x_extent = (x_max - x_min) / width
    y_extent = (y_max - y_min) / height

    # Estimate depth based on body part and mesh
    # Torso and head are deepest, arms/legs are thinner
    depth_ratios = {
        'torso': 1.0,  # Full depth
        'left_upper_arm': 0.4,
        'right_upper_arm': 0.4,
        'left_lower_arm': 0.3,
        'right_lower_arm': 0.3,
    }
    depth_ratio = depth_ratios.get(part_name, 0.5)
    z_extent = mesh_depth * depth_ratio

    # Create 3D bounding box vertices (8 vertices)
    vertices_3d = []
    for dx in [-x_extent/2, x_extent/2]:
        for dy in [-y_extent/2, y_extent/2]:
            for dz in [-z_extent/2, z_extent/2]:
                vertices_3d.append([
                    x_center + dx,
                    y_center + dy,
                    dz
                ])

    return vertices_3d
```

3. Add method to map body parts to keypoints:

```python
def get_keypoints_for_part(self, part_name):
    """Map body part name to relevant MediaPipe keypoints."""
    keypoint_map = {
        'torso': ['left_shoulder', 'right_shoulder', 'left_hip', 'right_hip'],
        'left_upper_arm': ['left_shoulder', 'left_elbow'],
        'right_upper_arm': ['right_shoulder', 'right_elbow'],
        'left_lower_arm': ['left_elbow', 'left_wrist'],
        'right_lower_arm': ['right_elbow', 'right_wrist'],
    }
    return keypoint_map.get(part_name, [])
```

---

### Fix 3: Section-Wise Deformation

**File**: `tests/keypoint_mapper.py`  
**Method**: `simple_anatomical_mapping()`

**Complete Rewrite**:

```python
def simple_anatomical_mapping(self, mediapipe_landmarks, cage_mesh, frame_shape, cage_structure=None):
    """
    Map MediaPipe keypoints to cage vertices with section-wise deformation.

    Args:
        mediapipe_landmarks: MediaPipe pose landmarks
        cage_mesh: Original cage mesh
        frame_shape: (height, width) of video frame
        cage_structure: Dict mapping body parts to vertex indices and keypoints

    Returns:
        deformed_cage_vertices: New cage vertex positions
    """
    height, width = frame_shape[:2]
    cage_vertices = np.array(cage_mesh.vertices).copy()

    # If no structure, fall back to old behavior
    if cage_structure is None:
        return self.simple_anatomical_mapping_old(mediapipe_landmarks, cage_mesh, frame_shape)

    # Extract MediaPipe keypoints to dict
    keypoints_2d = {}
    for landmark_name, landmark_idx in self.MEDIAPIPE_LANDMARKS.items():
        if landmark_idx < len(mediapipe_landmarks.landmark):
            lm = mediapipe_landmarks.landmark[landmark_idx]
            # Convert to pixel coordinates centered at origin
            x = (lm.x - 0.5) * width
            y = (lm.y - 0.5) * height
            z = lm.z * 1000  # MediaPipe Z (relative depth)
            keypoints_2d[landmark_name] = np.array([x, y, z])

    # For each body part section, compute independent transformation
    for part_name, part_info in cage_structure.items():
        vertex_indices = part_info['vertex_indices']
        keypoint_names = part_info['keypoints']

        # Skip if no vertices in this section
        if len(vertex_indices) == 0:
            continue

        # Get keypoints for this section
        part_keypoints = [keypoints_2d.get(kp) for kp in keypoint_names]
        part_keypoints = [kp for kp in part_keypoints if kp is not None]

        if len(part_keypoints) < 2:
            # Not enough keypoints, keep section stable
            continue

        # Get original section vertices
        section_verts_original = cage_vertices[vertex_indices]
        section_center_original = section_verts_original.mean(axis=0)

        # Compute current section center from keypoints
        section_center_current = np.mean(part_keypoints, axis=0)

        # Compute translation
        translation = section_center_current - section_center_original

        # Apply transformation to this section only
        for idx in vertex_indices:
            cage_vertices[idx] += translation

    # Smooth temporally
    if self.previous_cage_positions is not None:
        alpha = 0.3
        cage_vertices = alpha * cage_vertices + (1 - alpha) * self.previous_cage_positions

    self.previous_cage_positions = cage_vertices.copy()

    return cage_vertices
```

---

### Fix 4: Integration

**File**: `tests/test_integration.py`

**Changes**:

1. Store cage_structure:

```python
def initialize_cage_from_segmentation(self, segmentation_data, frame_shape):
    if self.cage_initialized:
        return

    print("\nInitializing cage from BodyPix segmentation...")

    # Generate anatomical cage WITH structure
    self.cage, self.cage_structure = self.cage_generator.generate_anatomical_cage(
        segmentation_data, frame_shape, subdivisions=3
    )

    # Store original cage vertices
    self.original_cage_vertices = self.cage.vertices.copy()

    # Initialize MVC
    self.mvc = EnhancedMeanValueCoordinates(self.mesh.vertices, self.cage)
    self.mvc.compute_weights()

    self.cage_initialized = True
    print("✓ Cage system initialized")
    print(f"   Cage: {len(self.cage.vertices)} vertices, {len(self.cage.faces)} faces")
    print(f"   Structure: {len(self.cage_structure)} body parts")
```

2. Pass cage_structure to deformation:

```python
def deform_mesh_from_keypoints(self, keypoints, landmarks, frame_shape):
    if not self.cage_initialized:
        return self.mesh.vertices

    # Deform cage using keypoint mapping WITH structure
    deformed_cage_vertices = self.keypoint_mapper.simple_anatomical_mapping(
        landmarks,
        self.cage,
        frame_shape,
        self.cage_structure  # Pass structure
    )

    # Apply MVC deformation
    deformed_vertices = self.mvc.deform_mesh(deformed_cage_vertices)

    return deformed_vertices
```

3. Add cage_structure to **init**:

```python
def __init__(self, mesh_path=None):
    ...
    self.cage = None
    self.cage_structure = None  # NEW
    self.original_cage_vertices = None
    ...
```

---

## 🧪 Testing Plan

### Step 1: Test Mesh Orientation Fix

```bash
# Generate a new mesh
cd /Users/fabrizioguccione/Projects/spoken_wardrobe_2
source venv/bin/activate
python tests/clothing_to_3d_triposr_2.py

# Select an image from generated_images
# Check if mesh is facing forward (not left)
```

**Expected**: Mesh should face camera, not be rotated left

---

### Step 2: Test Cage Structure Generation

```bash
# Run cage structure verification
python 251025_data_verification/verify_cage_structure.py
```

**Expected Output**:

```
Cage structure: {
    'torso': {'vertex_indices': [0,1,2,3,4,5,6,7], 'keypoints': [...]},
    'left_upper_arm': {'vertex_indices': [8,9,10,11,12,13], ...},
    ...
}
```

---

### Step 3: Test Section-Wise Deformation

```bash
# Run real-time verification
python 251025_data_verification/verify_deformation.py --mesh generated_meshes/0/mesh.obj

# Open in browser
open 251025_data_verification/verification_viewer.html
```

**Expected Output**:

```
📊 Frame 0: Max cage displacement: 0.239, Moving vertices: 8/21 (38%)
📊 Frame 30: Max cage displacement: 0.252, Moving vertices: 12/21 (57%)
```

**NOT 21/21 (100%)!**

---

### Step 4: Visual Verification

1. Move left arm → Only left arm cage section moves
2. Move right arm → Only right arm cage section moves
3. Torso stays relatively stable
4. Mesh follows body motion smoothly

---

## 🎯 Success Criteria

- [ ] Mesh faces forward (not rotated left)
- [ ] Cage has anatomical structure with 4-6 body part sections
- [ ] Only 30-60% of cage vertices move per frame
- [ ] Different body parts articulate independently
- [ ] Mesh follows body motion without smearing
- [ ] Cage is 3D (visible from all angles)

---

## ⚠️ Notes on Depth Calibration

For now, we're using **estimated depth ratios** based on body part type:

- Torso: 100% of mesh depth
- Arms: 30-40% of mesh depth
- Legs: 40-50% of mesh depth

**Future Enhancement** (if needed):

- Use Depth Anything at setup to calibrate Z-axis
- Store depth map once when no user in frame
- Use depth map to refine MediaPipe Z values
- This can be added later without breaking current system

---

**Ready to implement!** 🚀
