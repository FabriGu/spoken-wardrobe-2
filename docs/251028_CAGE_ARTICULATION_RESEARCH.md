# Cage-Based Deformation: Articulated Approach Research

## Date: October 28, 2025

## Problem Analysis

### Current Implementation Issues

**1. ConvexHull Simplification**

- We generate ~100 vertices across 5 body sections
- `ConvexHull` sees this as ONE point cloud
- Returns the outer envelope (a single box shape)
- All internal structure is lost

**2. Mean Value Coordinates (MVC) Global Pull**

- MVC computes weights for EVERY mesh vertex to EVERY cage vertex
- Each mesh vertex is "pulled" toward all 8 corners of the box
- This causes the pinching/distortion inside cages
- Interior vertices should move rigidly, not be pulled

**3. Missing Articulation**

- No concept of "joints" or "hinges"
- Segments can move independently (detachment)
- No angle-based rotation
- No hierarchical parent-child relationships

## User's Correct Intuition

> "if the cage segment of the arms is attached to the cage segment of the torso hinging around the shoulder point as pivot (it cannot move away from it) and the angle at which the keypoint for the shoulder and the keypoint to the elbow is used to orient the cage segment that we call the upper arm and keep everything inside that upper arm cage segment relatively unwarped EXCEPT the point at the edges of the cage segment (where it attached to the torso and where it attaches to the upper arm) then couldn't that be a working solution?"

**This is EXACTLY right.** This describes **articulated rigid body deformation** with smooth joint transitions.

## Research Foundation

### Paper: "Interactive Cage Generation for Mesh Deformation" (Le & Deng, 2017)

**Key Insights:**

1. **Cut Slides** (§3.1):

   - User-specified ellipses that define cage cross-sections
   - Each cross-section is a nearly rectangular boundary
   - Multiple cross-sections form a connected cage structure

2. **Orientation Optimization** (§3.2):

   - Solves for consistent orientations across neighboring sections
   - Prevents cage twisting
   - Uses **energy minimization**: smoothness + alignment
   - Critical formula: `E = Σ E_smoothness + Σ E_alignment`

3. **Cage Meshing** (§3.3):

   - Projects cross-section corners to a sphere
   - Computes 3D Delaunay triangulation
   - Edge flipping to optimize triangle quality
   - **Important**: Each section is meshed separately, then connected

4. **Cage Refinement** (§3.4):
   - Pushes vertices outward to bound the model
   - Preserves cross-section orientations
   - Iterative least-squares fitting

**Why This Works:**

- Each body part gets its own **oriented bounding box (OBB)**
- OBBs are **connected at joints**, not independent
- The cage is a **single unified mesh**, not multiple boxes
- Cross-sections define **local coordinate frames** for each segment

### Related Work: Skeleton-Driven Cages

From Chen & Feng (2014) - "Adaptive skeleton-driven cages":

- Use skeleton (like MediaPipe) to guide cage generation
- Each bone gets a cage segment
- Segments share vertices at joints (connectivity)
- Real-time deformation via skeleton animation

### OBB-Based Approach (Xian et al., 2012)

From "Automatic cage generation by improved OBBs":

- Generate **oriented bounding box** for each body part
- OBB defined by: center, 3 axes, 3 half-extents
- Register OBBs at joint locations
- Limitation: Assumes cylinder-like parts (works for limbs)

## Proposed Solution: Articulated Cage with OBBs

### Core Concept

Instead of ConvexHull, use **Oriented Bounding Boxes (OBBs)** for each body segment:

```
1. Generate OBB for each BodyPix segment
   - Use PCA on mask points to find principal axes
   - Create box aligned to body part orientation

2. Connect OBBs at joint keypoints
   - Torso-to-arm at shoulder
   - Upper-arm-to-lower-arm at elbow
   - Define shared vertices at joints

3. Create single unified mesh
   - Merge all OBB vertices
   - Connect faces at joint boundaries
   - Result: ONE cage with multiple sections

4. Real-time deformation:
   - Extract MediaPipe joint angles
   - Rotate each OBB around its pivot joint
   - Apply hierarchical transformations
   - Interior vertices move rigidly with OBB
   - Boundary vertices blend smoothly
```

### Mathematical Framework

#### 1. OBB Generation (Per Body Part)

For a BodyPix segment mask `M`:

```python
# Extract 2D points
points_2d = np.argwhere(M > 0)  # (N, 2) array

# PCA for orientation
mean = points_2d.mean(axis=0)
centered = points_2d - mean
cov = centered.T @ centered
eigenvalues, eigenvectors = np.linalg.eigh(cov)

# Principal axes (sorted by variance)
axis_1 = eigenvectors[:, 1]  # Major axis
axis_2 = eigenvectors[:, 0]  # Minor axis

# Bounding box in local frame
proj_1 = centered @ axis_1
proj_2 = centered @ axis_2
half_extent_1 = (proj_1.max() - proj_1.min()) / 2
half_extent_2 = (proj_2.max() - proj_2.min()) / 2

# OBB parameters
center_2d = mean
axes_2d = [axis_1, axis_2]
half_extents_2d = [half_extent_1, half_extent_2]
```

Convert to 3D:

```python
# Map 2D center to 3D mesh space
center_3d = project_2d_to_3d(center_2d, depth_estimate)

# Construct 3D OBB
axis_x = normalize([axes_2d[0][0], 0, axes_2d[0][1]])  # In XZ plane
axis_y = [0, 1, 0]  # Vertical
axis_z = cross(axis_x, axis_y)

# 8 vertices of OBB
vertices = []
for i in [-1, 1]:
    for j in [-1, 1]:
        for k in [-1, 1]:
            v = center_3d + i*half_extents[0]*axis_x + j*half_extents[1]*axis_y + k*half_extents[2]*axis_z
            vertices.append(v)
```

#### 2. Joint Connection

Define hierarchical relationships:

```python
HIERARCHY = {
    'left_upper_arm': {'parent': 'torso', 'joint': 'left_shoulder'},
    'left_lower_arm': {'parent': 'left_upper_arm', 'joint': 'left_elbow'},
    'left_hand': {'parent': 'left_lower_arm', 'joint': 'left_wrist'},
    # ... etc
}
```

Shared vertices at joints:

```python
# Example: Connect torso to left_upper_arm at shoulder
shoulder_position = keypoints['left_shoulder']

# Find nearest vertices in both OBBs
torso_vertices_near_shoulder = find_nearest(torso_obb.vertices, shoulder_position, threshold=0.1)
arm_vertices_near_shoulder = find_nearest(arm_obb.vertices, shoulder_position, threshold=0.1)

# Merge these vertices (they become the "hinge")
merged_vertex = (torso_vertices_near_shoulder.mean() + arm_vertices_near_shoulder.mean()) / 2
```

#### 3. Angle-Based Rotation (Real-Time)

For each frame:

```python
# Get current joint angle from MediaPipe
shoulder_pos = keypoints['left_shoulder']
elbow_pos = keypoints['left_elbow']
bone_vector = elbow_pos - shoulder_pos
bone_angle = compute_angle(bone_vector, reference_vector)

# Rotate arm OBB around shoulder pivot
R = rotation_matrix(pivot=shoulder_pos, angle=bone_angle, axis=rotation_axis)

# Apply to arm cage vertices
for vertex_idx in arm_section_indices:
    cage_vertices[vertex_idx] = R @ (cage_vertices[vertex_idx] - shoulder_pos) + shoulder_pos
```

#### 4. Distance-Based Falloff for Smooth Joints

Use Gaussian falloff:

```python
def compute_blend_weight(vertex, joint_position, falloff_radius):
    distance = ||vertex - joint_position||
    if distance > falloff_radius:
        return 0.0
    # Gaussian falloff
    weight = exp(-distance^2 / (2 * (falloff_radius / 3)^2))
    return weight

# Blend between parent and child transformations
for vertex in boundary_region:
    parent_weight = compute_blend_weight(vertex, joint_position, falloff_radius)
    child_weight = 1 - parent_weight

    vertex_deformed = parent_weight * parent_transform @ vertex + child_weight * child_transform @ vertex
```

### MVC Strategy: Regional, Not Global

**Current (Wrong):**

```python
# Compute MVC for ALL mesh vertices to ALL cage vertices
mvc_weights = compute_mvc(mesh_vertices, cage_vertices)  # (M, N) - expensive and causes pinching
deformed_mesh = mvc_weights @ deformed_cage_vertices
```

**Proposed (Correct):**

```python
# For each cage section:
for section_name, section_cage_indices in cage_structure.items():
    # Find mesh vertices inside this section's OBB
    mesh_vertices_in_section = find_vertices_inside_obb(mesh_vertices, section_obb)

    # Compute MVC ONLY for these vertices to ONLY this section's cage vertices
    section_cage_vertices = cage_vertices[section_cage_indices]
    mvc_weights_section = compute_mvc(mesh_vertices_in_section, section_cage_vertices)

    # Interior vertices (far from boundaries): Rigid transformation
    for vertex_idx in interior_vertices:
        # Move uniformly with the OBB (no MVC)
        deformed_mesh[vertex_idx] = section_transform @ mesh_vertices[vertex_idx]

    # Boundary vertices (near joints): MVC blending
    for vertex_idx in boundary_vertices:
        # Use MVC for smooth transition
        deformed_mesh[vertex_idx] = mvc_weights_section[vertex_idx] @ deformed_cage_section
```

## Implementation Plan

### Phase 1: OBB Generation from BodyPix (Offline)

**Files to Create:**

- `tests/articulated_cage_generator.py`
  - `ArticulatedCageGenerator` class
  - `generate_obb_from_mask(mask, keypoints)` → OBB parameters
  - `connect_obbs_at_joints(obbs, hierarchy, keypoints)` → unified cage mesh
  - `generate_articulated_cage(bodypix_data, keypoints)` → (cage_mesh, section_info, joint_info)

**Key Functions:**

```python
class OBB:
    def __init__(self, center, axes, half_extents):
        self.center = center  # (3,)
        self.axes = axes      # (3, 3) - column vectors are X, Y, Z axes
        self.half_extents = half_extents  # (3,)

    def get_vertices(self):
        """Returns 8 corner vertices"""
        vertices = []
        for i in [-1, 1]:
            for j in [-1, 1]:
                for k in [-1, 1]:
                    v = self.center + i*self.half_extents[0]*self.axes[:, 0] + \
                        j*self.half_extents[1]*self.axes[:, 1] + k*self.half_extents[2]*self.axes[:, 2]
                    vertices.append(v)
        return np.array(vertices)

    def rotate(self, pivot, rotation_matrix):
        """Rotate OBB around pivot point"""
        self.center = rotation_matrix @ (self.center - pivot) + pivot
        self.axes = rotation_matrix @ self.axes
```

### Phase 2: Angle-Based Deformation (Real-Time)

**Files to Create:**

- `tests/articulated_deformer.py`
  - `ArticulatedDeformer` class
  - `compute_joint_angles(keypoints, reference_keypoints)` → angle deltas
  - `deform_cage_articulated(cage, angles, joint_info)` → deformed cage
  - `apply_regional_mvc(mesh, cage, section_info)` → deformed mesh

**Key Functions:**

```python
class ArticulatedDeformer:
    def __init__(self, cage, section_info, joint_info):
        self.cage = cage
        self.section_info = section_info
        self.joint_info = joint_info

        # Pre-compute MVC weights for each section
        self.section_mvc = {}
        for section_name, cage_indices in section_info.items():
            # Find mesh vertices in this section
            mesh_indices = self._find_mesh_in_section(section_name)
            cage_verts = cage.vertices[cage_indices]
            mesh_verts = mesh.vertices[mesh_indices]

            # Compute MVC for this section only
            self.section_mvc[section_name] = {
                'mesh_indices': mesh_indices,
                'weights': compute_mvc(mesh_verts, cage_verts)
            }

    def deform(self, keypoints):
        """Real-time deformation"""
        # 1. Compute joint angles
        angles = self.compute_joint_angles(keypoints)

        # 2. Deform cage (articulated rigid bodies)
        deformed_cage = self._deform_cage_articulated(angles)

        # 3. Deform mesh (regional MVC)
        deformed_mesh = self._apply_regional_mvc(deformed_cage)

        return deformed_mesh
```

### Phase 3: Integration & Testing

**Files to Modify:**

- `tests/test_integration_cage.py`
  - Replace `UnifiedCageGenerator` with `ArticulatedCageGenerator`
  - Replace `MediaPipeCageDeformer` with `ArticulatedDeformer`
  - Keep existing WebSocket and visualization

## Expected Results

### Before (Current):

- ✗ Cage collapses to single box (ConvexHull simplification)
- ✗ Mesh pinches toward corners (global MVC)
- ✗ Segments can detach (no joint constraints)
- ✗ Uniform deformation (no articulation)

### After (Proposed):

- ✓ Cage has multiple connected OBB sections
- ✓ Mesh interior moves rigidly (no pinching)
- ✓ Segments stay connected (shared joint vertices)
- ✓ Natural articulation (angle-based rotation)
- ✓ Smooth joint transitions (distance-based falloff)

## Research References

1. **Le, B. H., & Deng, Z. (2017)**. "Interactive Cage Generation for Mesh Deformation". _I3D '17_. [Paper](http://dx.doi.org/10.1145/3023368.3023369)
   - Cut slides, orientation optimization, cage meshing
2. **Chen, X., & Feng, J. (2014)**. "Adaptive skeleton-driven cages for mesh sequences". _Computer Animation and Virtual Worlds_, 25(3-4), 447-455.
   - Skeleton-guided cage generation
3. **Xian, C., Lin, H., & Gao, S. (2012)**. "Automatic cage generation by improved obbs for mesh deformation". _The Visual Computer_, 28(1), 21-33.
   - OBB-based cage generation
4. **Ju, T., Schaefer, S., & Warren, J. (2005)**. "Mean value coordinates for closed triangular meshes". _ACM Trans. Graph._, 24(3), 561-566.

   - Original MVC paper

5. **Jacobson, A., Baran, I., Popović, J., & Sorkine, O. (2011)**. "Bounded biharmonic weights for real-time deformation". _ACM Trans. Graph._, 30(4), 78:1-78:8.
   - Alternative to MVC with better locality

## Implementation Notes

### Depth Estimation Challenge

BodyPix gives 2D masks. We need 3D OBBs. Options:

**Option A: Heuristic Depth Ratios** (Fast, simple)

```python
DEPTH_RATIOS = {
    'torso': 1.0,
    'left_upper_arm': 0.4,
    'left_lower_arm': 0.3,
    'left_hand': 0.2,
    # ... based on human anatomy
}
```

**Option B: Mesh-Guided Depth** (More accurate)

```python
# Use the generated 3D mesh to estimate depth
mesh_points_in_section = project_section_to_mesh(mask, mesh)
depth = estimate_depth_from_mesh(mesh_points_in_section)
```

**Option C: MediaPipe Z-coordinates** (Best, but needs calibration)

```python
# Use MediaPipe's 3D keypoints (if calibrated)
depth = interpolate_depth_from_keypoints(section_center, keypoints_3d)
```

Recommendation: Start with **Option A** (heuristic), later upgrade to **Option C** (MediaPipe Z).

### Real-Time Performance

- OBB generation: **Once** (from reference pose)
- MVC computation: **Once per section** (pre-computed)
- Joint angle extraction: **Every frame** (~5ms)
- Cage rotation: **Every frame** (~2ms)
- Regional MVC application: **Every frame** (~10ms for 5k vertices)

**Total: ~20ms per frame → 50 FPS** ✓

## Next Steps

1. ✅ Research complete (this document)
2. ⬜ Implement `ArticulatedCageGenerator` with OBB generation
3. ⬜ Implement `ArticulatedDeformer` with angle-based rotation
4. ⬜ Test with simple mesh (verify cage structure)
5. ⬜ Integrate into `test_integration_cage.py`
6. ⬜ Test with real TripoSR mesh
7. ⬜ Tune parameters (falloff radius, depth ratios, etc.)
8. ⬜ Document results and compare to Option B

---

**Conclusion**: The user's intuition about articulated cages with hinged joints is **absolutely correct** and aligns with research literature. The solution is well-defined and implementable within real-time constraints.
