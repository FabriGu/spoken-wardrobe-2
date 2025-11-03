# Upgrade Paths: From Test A to Production Pipeline

This document outlines the upgrade paths from the current Test A prototype to a full production pipeline.

**Current State**: Test A (`tests/test_a_rigged_clothing.py`)
- ✅ Works with pre-generated clothing mesh (Rodin base.glb)
- ✅ Uses generic rigged human mesh (CAUCASIAN MAN.glb)
- ✅ Transfers skin weights (simple K-NN approach)
- ✅ Real-time animation with MediaPipe + OAK-D
- ✅ WebSocket streaming to Three.js viewer

**Limitations**:
- Generic human mesh may not fit user's proportions perfectly
- Clothing mesh is pre-generated (not from AI pipeline)
- Weight transfer uses simplified nearest-neighbor (not barycentric interpolation)

---

## Option B: Integrate TripoSR into Pipeline

### Goal
Generate clothing meshes directly from the Stable Diffusion output within the pipeline, then animate them.

### Pipeline Flow

```
User T-Pose
  ↓
OAK-D Capture
  ↓
BodyPix Segmentation (24 parts)
  ↓
Stable Diffusion Inpainting → Clothing Image
  ↓
TripoSR → 3D Clothing Mesh                    ← NEW STEP
  ↓
Load Generic Rigged Human
  ↓
Scale & Align (bounding box + skeleton)
  ↓
Transfer Weights (human → clothing)
  ↓
MediaPipe Real-Time Animation
  ↓
Three.js Viewer
```

### Implementation Steps

#### Step 1: Integrate TripoSR Module (1 day)

**Create**: `src/modules/triposr_generator.py`

```python
class TripoSRGenerator:
    def generate_mesh(self, clothing_image_path: str, output_path: str):
        """
        Generate 3D mesh from 2D clothing image using TripoSR

        Returns:
            mesh_path: Path to generated GLB file
        """
        # Load TripoSR model
        # Run inference
        # Apply orientation correction (90° Y-rotation + 180° flip)
        # Save as GLB
        pass
```

**Reference**: `tests/clothing_to_3d_triposr_2.py` (already exists)

#### Step 2: Update Pipeline Script (1 day)

**Modify**: `tests/day2_oakd_bodypix_sd.py`

Add TripoSR generation step after Stable Diffusion:

```python
# After SD generates clothing image
clothing_image_path = "generated_images/clothing.png"

# NEW: Generate 3D mesh
triposr = TripoSRGenerator()
mesh_path = triposr.generate_mesh(
    clothing_image_path,
    output_path="generated_meshes/clothing.glb"
)

# Save mesh path in reference data
reference_data['mesh_path'] = mesh_path
```

#### Step 3: Update Test A to Accept Pipeline Output (0.5 days)

**Modify**: `tests/test_a_rigged_clothing.py`

Add command-line argument to load pipeline-generated mesh instead of hardcoded `base.glb`:

```python
parser = argparse.ArgumentParser()
parser.add_argument('--clothing', type=str, default='pregenerated_mesh_clothing/base.glb')
parser.add_argument('--reference', type=str, help='Path to reference_data.pkl')

# Load clothing mesh from argument
clothing_mesh = RiggedMeshLoader.load(args.clothing)
```

### Timeline

- **Day 1**: Integrate TripoSR module
- **Day 2**: Update pipeline to generate meshes
- **Day 3**: Test end-to-end and fix issues

### Pros

- ✅ Uses your existing Stable Diffusion textures (fluid, BodyPix-based)
- ✅ Preserves 3D geometry quality from SD
- ✅ Minimal changes to existing codebase
- ✅ No new dependencies

### Cons

- ❌ Still uses generic rigged human (doesn't fit user perfectly)
- ❌ TripoSR meshes may have topology issues for animation
- ❌ Weight transfer quality limited by generic human template

---

## Option C: CharacterGen Pipeline (User-Specific Rigging)

### Goal
Generate a user-specific rigged body mesh, then use it as the weight template for clothing.

### Pipeline Flow

```
User T-Pose (FULL BODY IMAGE)
  ↓
CharacterGen → 3D User Mesh                    ← NEW
  ↓
UniRig → Rigged User Mesh                      ← NEW
  ↓
[Save as user_template.glb]                    ← REUSE FOR ALL CLOTHING

---

User T-Pose (CLOTHING REGION)
  ↓
BodyPix Crop → Clothing Area
  ↓
Stable Diffusion → Clothing Image
  ↓
CharacterGen / TripoSR → 3D Clothing Mesh      ← CHOICE
  ↓
Transfer Weights (user_template → clothing)
  ↓
MediaPipe Real-Time Animation
  ↓
Three.js Viewer
```

### Key Insight

CharacterGen creates a **user-specific** rigged body mesh. This mesh:
- Matches user's proportions (height, shoulder width, etc.)
- Acts as the perfect weight template for all future clothing items
- Only needs to be generated ONCE per user

### Implementation Steps

#### Step 1: Install CharacterGen + UniRig (1 day)

```bash
git clone https://github.com/zjp-shadow/CharacterGen external/CharacterGen
cd external/CharacterGen
pip install -r requirements.txt
# Download model weights (follow CharacterGen README)

# Optional: Install UniRig for auto-rigging
git clone https://github.com/UniRig/UniRig external/UniRig
```

#### Step 2: Generate User Template (One-Time Setup)

**Create**: `tests/setup_user_template.py`

```python
def generate_user_template():
    """
    One-time setup: Generate user-specific rigged body mesh

    1. Capture user in T-pose (full body, uncropped)
    2. Run CharacterGen → 3D user mesh
    3. Run UniRig → Add skeleton and weights
    4. Save as user_template.glb
    5. Store MediaPipe keypoints from T-pose for calibration
    """
    # Capture T-pose image
    # CharacterGen inference
    # UniRig auto-rigging
    # Save template
    pass
```

User runs this ONCE during initial setup. Result: `user_template.glb` (their personal rigged body).

#### Step 3: Update Clothing Pipeline

**Modify**: `tests/day2_oakd_bodypix_sd.py`

Same as Option B, but with option to use CharacterGen instead of TripoSR:

```python
# Generate clothing mesh
if USE_CHARACTERGEN:
    mesh = CharacterGen.generate(clothing_image)
else:
    mesh = TripoSR.generate(clothing_image)
```

#### Step 4: Update Test A to Use User Template

**Modify**: `tests/test_a_rigged_clothing.py`

```python
# Load user-specific rigged body (instead of generic CAUCASIAN MAN.glb)
human_mesh = RiggedMeshLoader.load("user_data/user_template.glb")

# Transfer weights: user_template → clothing
# This will be more accurate because template matches user's body
```

### CharacterGen Cropping Strategy (Advanced)

**Question**: Can we crop the CharacterGen-generated mesh to get just clothing?

**Answer**: Yes, but complex. Steps would be:

1. CharacterGen generates full 3D character
2. Project BodyPix 2D mask into 3D space using camera parameters
3. Delete vertices outside clothing region
4. Fill holes at boundaries (neck, wrists, hem)
5. Recompute normals

**Challenges**:
- Projecting 2D mask → 3D accurately
- Handling self-occlusion (back of body)
- Filling holes without artifacts
- May delete vertices needed for smooth deformation

**Recommendation**:
- For prototype: Use TripoSR on clothing image (simpler)
- For production: Use CharacterGen for user body template only
- Skip mesh cropping complexity

### Timeline

- **Week 1**: Install and test CharacterGen + UniRig
- **Week 2**: Build user template generation workflow
- **Week 3**: Integrate into pipeline and test
- **Week 4**: Refinement and optimization

### Pros

- ✅ User-specific rigging (perfect fit)
- ✅ Reusable template for all clothing items
- ✅ Better weight transfer quality
- ✅ More professional/polished result

### Cons

- ❌ Requires external dependencies (CharacterGen, UniRig)
- ❌ Longer setup time (one-time user calibration)
- ❌ More complex pipeline
- ❌ May require manual intervention if auto-rigging fails

---

## Option D: Hybrid Approach (RECOMMENDED)

### Best of Both Worlds

Combine Option B and Option C:

1. **Use CharacterGen** for user body template (one-time setup)
2. **Use TripoSR** for clothing mesh generation (from SD output)
3. Transfer weights: CharacterGen user body → TripoSR clothing
4. Animate with MediaPipe

### Why This Works Best

- **User-specific fit**: CharacterGen body matches user proportions
- **Fluid clothing**: TripoSR preserves SD texture quality
- **Simple pipeline**: TripoSR easier than CharacterGen for clothing
- **Best quality**: User template + fluid clothing geometry

### Implementation Order

1. **Immediate** (This Week):
   - Test A already works with generic human
   - Prove the concept works end-to-end

2. **Short-term** (Next Week):
   - Integrate TripoSR (Option B)
   - Replace hardcoded base.glb with pipeline-generated meshes

3. **Mid-term** (Week 3-4):
   - Install CharacterGen + UniRig
   - Generate user template (Option C)
   - Replace generic CAUCASIAN MAN.glb with user template

4. **Long-term** (Month 2):
   - Improve weight transfer (barycentric interpolation from IMPLEMENTATION_PLAN.md)
   - Add texture mapping (SD texture on mesh surface)
   - Optimize performance (reduce mesh complexity, GPU acceleration)

---

## Weight Transfer Improvements

**Current**: Simple K-NN (nearest neighbor)

**Production**: Barycentric interpolation (from IMPLEMENTATION_PLAN.md)

### Why Barycentric is Better

K-NN copies weights from a single closest vertex → sharp discontinuities
Barycentric blends weights from 3 triangle vertices → smooth deformation

### Implementation Reference

See `docs/IMPLEMENTATION_PLAN.md` Section 3.3:

1. Find closest point on source mesh surface
2. Determine which triangle contains that point
3. Compute barycentric coordinates
4. Interpolate weights from triangle's 3 vertices

**Estimated effort**: 2-3 days

**Priority**: Medium (K-NN works for prototype, upgrade for production)

---

## Comparison Matrix

| Feature                    | Test A (Current) | Option B (TripoSR) | Option C (CharacterGen) | Option D (Hybrid) |
|----------------------------|------------------|-------------------|------------------------|-------------------|
| User-specific fit          | ❌ Generic        | ❌ Generic         | ✅ User body            | ✅ User body       |
| Fluid clothing geometry    | ❌ Pre-generated  | ✅ From SD         | ⚠️ Depends on source   | ✅ From SD         |
| Implementation complexity  | ✅ Simple         | ✅ Moderate        | ❌ Complex              | ⚠️ Moderate-High  |
| Setup time (user)          | ✅ None           | ✅ None            | ❌ 5-10 min setup       | ⚠️ 5-10 min setup |
| Quality (overall)          | ⭐⭐⭐           | ⭐⭐⭐⭐          | ⭐⭐⭐⭐⭐             | ⭐⭐⭐⭐⭐        |

---

## Next Actions

### Immediate (This Weekend)

1. ✅ Test A is complete - verify it works
2. Run Test A with OAK-D and Three.js viewer
3. Document any issues or improvements needed

### Week 1

1. Implement Option B (TripoSR integration)
2. Test full pipeline: capture → SD → TripoSR → animate
3. Fix any mesh orientation or topology issues

### Week 2-3

1. Research CharacterGen setup requirements
2. Install CharacterGen + UniRig
3. Generate first user template
4. Test with existing clothing meshes

### Week 4

1. Full pipeline test with user template
2. Compare quality vs. generic rigged human
3. Optimize performance

---

## Technical Notes

### Bone Mapping: MediaPipe → Skeleton

**Current implementation** (`src/modules/mediapipe_to_bones.py`):
- Maps MediaPipe 33 landmarks → 9 main bones
- Simplified skeleton (spine, left/right arm, left/right leg)

**For CharacterGen/CAUCASIAN MAN**:
- 163 bones (includes fingers, toes, face, etc.)
- Need hierarchical bone mapping
- Most bones can stay at identity transform
- Only animate major bones (spine, shoulders, elbows, wrists, hips, knees, ankles)

**Improvement needed**:
Map MediaPipe landmarks to specific bone names in rigged mesh:
- `left_shoulder` MediaPipe → `upperarm01_L` bone
- `left_elbow` MediaPipe → `lowerarm01_L` bone
- etc.

### Mesh Alignment Strategy

**Current** (`test_a_rigged_clothing.py`):
- Scale by bounding box height
- No rotation or positional alignment

**Improved version needed**:
1. Compute MediaPipe skeleton in T-pose
2. Compute rigged mesh skeleton in bind pose
3. Align skeletons (match hip position, shoulder line, spine direction)
4. Scale based on torso height (hip to shoulder)
5. This ensures clothing mesh overlays correctly on user

### Performance Optimization

**Target**: 30 FPS real-time animation

**Current bottlenecks**:
1. Linear Blend Skinning (CPU-bound)
2. WebSocket JSON serialization
3. TripoSR inference (if run in real-time)

**Solutions**:
1. Move LBS to GPU (use PyTorch/CUDA)
2. Use binary format for WebSocket (msgpack instead of JSON)
3. Pre-generate meshes (don't run TripoSR in real-time loop)
4. Reduce mesh complexity (decimate/simplify)

---

## Conclusion

**Recommended Path**: Option D (Hybrid)

1. **Start with Test A** (works now) - prove the concept
2. **Upgrade to Option B** (integrate TripoSR) - real clothing from SD
3. **Upgrade to Option C** (CharacterGen user template) - perfect fit
4. **Result**: Production-quality AR clothing try-on system

Each step builds on the previous, allowing incremental development and testing.
