# Real-Time Cage-Based Mesh Warping: Steps Forward

**Date**: October 25, 2025  
**Status**: Analysis & Implementation Plan

---

## 🎯 Executive Summary

Current implementation shows mesh distortion ("flat smear") when user enters frame. Root cause: **improper cage structure and deformation logic**. The cage-based warping is not following anatomical structure as required by the research.

---

## 📚 Research Foundation

### Key Papers Referenced

1. **Interactive Cage Generation for Mesh Deformation** (Le & Deng, 2017)

   - Source: https://graphics.cs.uh.edu/wp-content/papers/2017/2017-I3D-CageGeneration.pdf
   - Key Insight: _"A high-quality cage needs to respect the topology of the enveloped model"_
   - Relevance: Our cage must follow anatomical structure (torso, arms, legs)

2. **Deforming Radiance Fields with Cages** (Xu & Harada, 2022)

   - Key Formula for cage-based deformation:
     ```
     x = Σ ωⱼ(x) vⱼ        (canonical position)
     x' = Σ ωⱼ(x) v'ⱼ     (deformed position)
     ```
   - Relevance: Shows how to compute deformation using pre-computed weights

3. **Mean Value Coordinates** (Ju et al., 2005)
   - Core algorithm for computing cage coordinates
   - Closed-form solution enables fast computation

---

## 🔴 Current Problems Identified

### Problem 1: Non-Anatomical Cage Structure

```python
# CURRENT (WRONG): Generic cage generation
cage = cage_generator.generate_anatomical_cage(segmentation_data, ...)
# → Produces ~400-500 vertices without clear anatomical boundaries
# → No separation between torso, arms, legs
```

**Impact**: Can't deform body parts independently → everything moves together → smearing

### Problem 2: Incorrect Keypoint-to-Cage Mapping

```python
# CURRENT (WRONG): From keypoint_mapper.py line 130-215
cage_vertices += translation  # Translates entire cage uniformly
cage_vertices = alpha * cage_vertices + (1-alpha) * previous  # Just smoothing
```

**Impact**: Cage doesn't follow body articulation → distortion

### Problem 3: MVC Weight Computation Unclear

From `test_integration.py`:

- Weights computed once in `initialize_cage_from_segmentation()` ✓
- But cage might be regenerated every frame ✗
- Dimension mismatches suggest cage size is changing

---

## ✅ Correct Implementation Pipeline

### Phase 1: ONE-TIME SETUP (Offline/Initialization)

#### Step 1.1: Load 3D Clothing Mesh

```python
mesh = trimesh.load('generated_meshes/clothing.obj')
# mesh.vertices: [N, 3] - N vertices
# mesh.faces: [M, 3] - M triangular faces
```

#### Step 1.2: Generate Anatomical Cage

**Requirements** (from Le & Deng 2017):

- Low resolution (30-60 vertices total)
- Tightly bounds the mesh
- Respects anatomical topology

**Structure**:

```python
cage_structure = {
    'torso': {
        'vertices': [0, 1, 2, 3, 4, 5, 6, 7],  # 8 vertices
        'keypoints': ['left_shoulder', 'right_shoulder', 'left_hip', 'right_hip']
    },
    'left_upper_arm': {
        'vertices': [8, 9, 10, 11, 12, 13],  # 6 vertices
        'keypoints': ['left_shoulder', 'left_elbow']
    },
    'right_upper_arm': {
        'vertices': [14, 15, 16, 17, 18, 19],
        'keypoints': ['right_shoulder', 'right_elbow']
    },
    'left_lower_arm': {
        'vertices': [20, 21, 22, 23, 24, 25],
        'keypoints': ['left_elbow', 'left_wrist']
    },
    'right_lower_arm': {
        'vertices': [26, 27, 28, 29, 30, 31],
        'keypoints': ['right_elbow', 'right_wrist']
    }
}
```

**Pseudocode**:

```python
def generate_anatomical_cage(mesh, bodypix_segmentation):
    """
    Generate cage with clear anatomical sections.

    Args:
        mesh: Trimesh object
        bodypix_segmentation: Dict with body part masks

    Returns:
        cage: Trimesh with ~30-60 vertices
        cage_structure: Dict mapping vertices to body parts
    """
    cage_vertices = []
    cage_structure = {}

    # For each body part (torso, arms, legs):
    for part_name, part_mask in bodypix_segmentation.items():
        # Find mesh vertices in this body part
        part_mesh_verts = get_vertices_in_mask(mesh, part_mask)

        # Create tight bounding cage section (6-8 vertices)
        part_cage_verts = create_cage_section(part_mesh_verts)

        # Store mapping
        start_idx = len(cage_vertices)
        cage_vertices.extend(part_cage_verts)
        end_idx = len(cage_vertices)

        cage_structure[part_name] = {
            'vertex_indices': list(range(start_idx, end_idx)),
            'keypoints': get_keypoints_for_part(part_name)
        }

    # Create cage mesh
    cage = create_mesh_from_sections(cage_vertices, cage_structure)

    return cage, cage_structure
```

#### Step 1.3: Compute Mean Value Coordinates (MVC Weights)

**ONE TIME COMPUTATION** - This is expensive (~10 seconds for 26k vertices)

**Mathematical Formula** (from Ju et al. 2005):

```
For point x and cage C with vertices {v₁, v₂, ..., vₙ}:

ωⱼ(x) = [tan(αⱼ₋₁/2) + tan(αⱼ/2)] / ||x - vⱼ||

where αⱼ is the angle at x formed by edges to adjacent cage vertices
```

**Pseudocode**:

```python
def compute_mvc_weights(mesh_vertices, cage):
    """
    Compute Mean Value Coordinates for all mesh vertices.

    THIS IS EXPENSIVE - ONLY COMPUTE ONCE!

    Args:
        mesh_vertices: [N, 3] array of mesh vertex positions
        cage: Cage mesh with M vertices

    Returns:
        weights: [N, M] array where weights[i,j] = influence of
                 cage vertex j on mesh vertex i
    """
    N = len(mesh_vertices)
    M = len(cage.vertices)
    weights = np.zeros((N, M))

    print(f"Computing MVC weights for {N} mesh vertices...")

    for i, v in enumerate(mesh_vertices):
        if i % 1000 == 0:
            print(f"  Processing vertex {i}/{N}...")

        # For each cage vertex
        for j in range(M):
            # Compute angle-based weight
            weights[i, j] = compute_mvc_single_weight(
                v, cage.vertices[j], cage, j
            )

        # Normalize weights to sum to 1
        weights[i] /= weights[i].sum()

    print("✓ MVC weights computed!")
    return weights
```

**Key Properties**:

- Weights are **constant** once computed
- Sum to 1 for each mesh vertex: `Σⱼ ωⱼ(x) = 1`
- Depend only on relative position of mesh vertex to cage

---

### Phase 2: REAL-TIME LOOP (Every Frame)

#### Step 2.1: Get MediaPipe Keypoints (Fast - Already Works)

```python
results = pose.process(rgb_frame)
keypoints_2d = {
    'left_shoulder': [x, y, z],
    'right_shoulder': [x, y, z],
    'left_elbow': [x, y, z],
    # ... etc
}
```

#### Step 2.2: Map Keypoints → Cage Vertex Deformation (CRITICAL!)

**This is where the magic happens - and where your code fails**

**Key Insight**: Each anatomical section of the cage deforms based on its corresponding keypoints

**Pseudocode**:

```python
def deform_cage_from_keypoints(
    original_cage_vertices,
    cage_structure,
    keypoints_2d,
    frame_shape
):
    """
    Deform cage by moving each anatomical section independently.

    Args:
        original_cage_vertices: [M, 3] original cage positions
        cage_structure: Dict mapping body parts to cage vertices
        keypoints_2d: Dict of MediaPipe keypoints
        frame_shape: (height, width) for normalization

    Returns:
        deformed_cage_vertices: [M, 3] new cage positions
    """
    deformed_cage = original_cage_vertices.copy()
    h, w = frame_shape[:2]

    # For each body part section:
    for part_name, part_info in cage_structure.items():
        cage_indices = part_info['vertex_indices']
        keypoint_names = part_info['keypoints']

        # Get keypoints for this part
        part_keypoints = [keypoints_2d[kp] for kp in keypoint_names]

        # Compute transformation for this section
        # (rotation + translation + scaling)
        transform = compute_section_transform(
            original_cage_vertices[cage_indices],
            part_keypoints,
            frame_shape
        )

        # Apply transformation to cage vertices in this section
        for idx in cage_indices:
            deformed_cage[idx] = apply_transform(
                original_cage_vertices[idx],
                transform
            )

    return deformed_cage

def compute_section_transform(section_cage_verts, keypoints, frame_shape):
    """
    Compute affine transform for a body part section.

    Example for arm:
    - keypoints = [shoulder, elbow]
    - Compute arm direction: elbow - shoulder
    - Compute arm length: ||elbow - shoulder||
    - Compute rotation to align cage with arm direction
    - Compute translation to move cage center to arm center
    """
    # Convert keypoints from pixels to normalized coordinates
    kp_normalized = normalize_keypoints(keypoints, frame_shape)

    # Compute center and orientation
    center = np.mean(kp_normalized, axis=0)
    direction = kp_normalized[1] - kp_normalized[0]  # e.g., elbow - shoulder
    length = np.linalg.norm(direction)

    # Compute cage section's current center and orientation
    cage_center = np.mean(section_cage_verts, axis=0)
    cage_direction = get_primary_axis(section_cage_verts)
    cage_length = get_section_length(section_cage_verts)

    # Build transformation matrix
    # 1. Rotation: align cage_direction with direction
    R = rotation_matrix_from_vectors(cage_direction, direction)

    # 2. Scale: match length
    s = length / cage_length

    # 3. Translation: move cage_center to center
    t = center - cage_center

    return {'rotation': R, 'scale': s, 'translation': t}
```

#### Step 2.3: Apply MVC Deformation (Fast!)

**This should be nearly instant** - just a matrix multiplication!

**Mathematical Formula**:

```
v'ᵢ = Σⱼ ωᵢⱼ · c'ⱼ

where:
  v'ᵢ = deformed position of mesh vertex i
  ωᵢⱼ = pre-computed MVC weight (from Step 1.3)
  c'ⱼ = deformed position of cage vertex j (from Step 2.2)
```

**Pseudocode**:

```python
def deform_mesh_with_mvc(
    mvc_weights,
    deformed_cage_vertices
):
    """
    Apply cage deformation to mesh using pre-computed weights.

    THIS IS FAST - Just matrix multiplication!

    Args:
        mvc_weights: [N, M] pre-computed weights
        deformed_cage_vertices: [M, 3] deformed cage positions

    Returns:
        deformed_mesh_vertices: [N, 3] deformed mesh positions
    """
    # Simple matrix multiplication: [N, M] @ [M, 3] = [N, 3]
    deformed_vertices = mvc_weights @ deformed_cage_vertices

    return deformed_vertices
```

**Performance**:

- Input: 26,633 mesh vertices, 40 cage vertices
- Operation: (26633, 40) @ (40, 3) = (26633, 3)
- Time: ~1-2 milliseconds with NumPy

#### Step 2.4: Send to Web Viewer

```python
send_mesh_to_web(
    deformed_mesh_vertices,
    mesh.faces,
    cage_vertices=deformed_cage_vertices,
    cage_faces=cage.faces
)
```

---

## 🔧 Implementation Checklist

### Phase 1: Setup (One-Time)

- [ ] **Fix cage generation** to produce anatomical structure
  - [ ] Implement `generate_anatomical_cage()` with body part sections
  - [ ] Store `cage_structure` dict mapping vertices to body parts
  - [ ] Verify cage has 30-60 vertices total, ~6-8 per body part
- [ ] **Verify MVC weights computation**
  - [ ] Ensure computed only ONCE during initialization
  - [ ] Store weights: shape should be `[num_mesh_verts, num_cage_verts]`
  - [ ] Add assertion: `np.allclose(weights.sum(axis=1), 1.0)`

### Phase 2: Real-Time Loop

- [ ] **Fix keypoint-to-cage mapping**
  - [ ] Implement `compute_section_transform()` for each body part
  - [ ] Apply transforms independently to each cage section
  - [ ] Remove uniform translation/smoothing of entire cage
- [ ] **Verify deformation application**
  - [ ] Check: `deformed_verts = mvc_weights @ deformed_cage`
  - [ ] Ensure NO cage regeneration per frame
  - [ ] Ensure NO weight recomputation per frame

### Phase 3: Verification (Critical!)

- [ ] **Cage structure verification**
  - [ ] Visualize cage with color-coded body part sections
  - [ ] Verify cage tightly bounds mesh
  - [ ] Verify ~6-8 vertices per anatomical section
- [ ] **Weight verification**
  - [ ] Print weight matrix shape
  - [ ] Verify weights sum to 1 per vertex
  - [ ] Check weights are NOT changing per frame
- [ ] **Deformation verification**
  - [ ] Visualize cage deformation in real-time
  - [ ] Verify different cage sections move independently
  - [ ] Verify mesh follows cage smoothly

---

## 📊 Expected Performance

| Operation              | Time (per frame) | Frequency             |
| ---------------------- | ---------------- | --------------------- |
| BodyPix segmentation   | ~1400ms          | Once (initialization) |
| Cage generation        | ~500ms           | Once (initialization) |
| MVC weight computation | ~10,000ms        | Once (initialization) |
| MediaPipe keypoints    | ~30ms            | Every frame           |
| Cage deformation       | ~2ms             | Every frame           |
| Mesh deformation (MVC) | ~2ms             | Every frame           |
| WebSocket send         | ~10ms            | Every frame           |
| **Total per frame**    | **~44ms**        | **~23 FPS**           |

---

## 🎓 Key Takeaways from Research

### From Le & Deng (2017):

> _"A high-quality cage for mesh deformation is expected to have the following desired qualities: (1) low resolution, (2) fully and tightly bound the model, (3) respect the topology of the enveloped model."_

**Application**: Our cage must have anatomical structure (torso, arms) not just be a bounding box.

### From Xu & Harada (2022):

> _"The core of cage-based deformation is cage coordinates, which represent the relative positions of spatial points w.r.t. the cage."_

**Application**: Pre-compute these coordinates once, then deformation is just matrix multiplication.

### From Ju et al. (2005) - Mean Value Coordinates:

> _"MVC provides a closed-form solution for cage coordinates with smooth interpolation properties."_

**Application**: We can compute weights in feedforward manner without iterative optimization.

---

## 🚀 Next Steps

1. **Verification Phase** (Current Priority)

   - Create visualization tools to inspect current cage structure
   - Verify MVC weights are computed and stored correctly
   - Check if cage sections move independently

2. **Implementation Phase**

   - Rewrite `generate_anatomical_cage()` with proper structure
   - Rewrite `deform_cage_from_keypoints()` with section-wise transforms
   - Ensure `deform_mesh_with_mvc()` is just matrix multiplication

3. **Testing Phase**
   - Test with simple gestures (raise arm, bend elbow)
   - Verify mesh follows body motion smoothly
   - Check performance meets ~20 FPS target

---

## 📝 Code Locations

**Files to Modify**:

- `tests/enhanced_cage_utils.py` - Cage generation logic
- `tests/keypoint_mapper.py` - Keypoint-to-cage mapping
- `tests/test_integration.py` - Main loop integration

**Files to Verify**:

- Check cage structure after initialization
- Check MVC weights matrix shape and properties
- Check cage deformation per frame

---

## 🔗 References

1. Le, B. H., & Deng, Z. (2017). Interactive Cage Generation for Mesh Deformation. _I3D'17_. https://graphics.cs.uh.edu/wp-content/papers/2017/2017-I3D-CageGeneration.pdf

2. Xu, T., & Harada, T. (2022). Deforming Radiance Fields with Cages. _arXiv:2207.12298_.

3. Ju, T., Schaefer, S., & Warren, J. (2005). Mean Value Coordinates for Closed Triangular Meshes. _ACM SIGGRAPH_.

4. Jacobson, A., et al. (2014). Skinning: Real-time Shape Deformation. _ACM SIGGRAPH Courses_.

---

**Document Version**: 1.0  
**Last Updated**: October 25, 2025  
**Status**: Ready for Verification Phase
