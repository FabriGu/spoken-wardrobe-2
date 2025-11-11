# SF3D (Stable Fast 3D) Integration Guide

## What is SF3D?

SF3D (Stable Fast 3D) is the successor to TripoSR from Stability AI (August 2024). It generates high-quality 3D meshes from single images with several key improvements:

### Advantages over TripoSR

| Feature | TripoSR | SF3D |
|---------|---------|------|
| **Generation Speed** | 5-15 seconds | **0.5 seconds** (10-30x faster) |
| **Mesh Topology** | Marching Cubes (artifacts) | **Marching Tetrahedra** (smooth) |
| **Textures** | Vertex colors only | **UV-unwrapped textures** |
| **Animation Ready** | Poor (needs post-processing) | **Quad remeshing option** |
| **Simplification** | Breaks vertex colors | **Preserves UV textures** |
| **Output Format** | OBJ/GLB | **GLB with baked textures** |

### Key Features

- **UV-Unwrapped Textures**: Real texture maps instead of vertex colors
- **Remeshing Options**: `none`, `triangle`, or `quad` (quad recommended for animation)
- **Material Parameters**: Predicts normals, roughness, metalness
- **Delighting**: Removes baked lighting for use in any lighting condition
- **Fast**: 0.5s on GPU, ~10s on CPU

---

## Installation

### 1. Install Dependencies

The SF3D dependencies have been installed in the `stable-fast-3d-testing` branch:

```bash
# Already installed:
- jaxtyping==0.2.31
- open_clip_torch==2.24.0
- pynanoinstantmeshes==0.0.3
- texture_baker (local package)
- uv_unwrapper (local package with OpenMP)
```

### 2. Hugging Face Authentication

SF3D is a **gated model** requiring Hugging Face access:

#### Step 1: Request Access
1. Visit https://huggingface.co/stabilityai/stable-fast-3d
2. Click **"Request Access"**
3. Wait for approval (usually instant)

#### Step 2: Create Access Token
1. Visit https://huggingface.co/settings/tokens
2. Click **"Create new token"**
3. Select **"Read"** permissions
4. Copy the token

#### Step 3: Login via CLI
```bash
source venv/bin/activate
huggingface-cli login
# Paste your token when prompted
```

### 3. Verify Installation

```bash
source venv/bin/activate
python -c "
import sys
sys.path.insert(0, 'external/stable-fast-3d')
from sf3d.system import SF3D
print('✓ SF3D installed successfully')
"
```

---

## Usage

### Test 1: Basic Mesh Generation

Generate a 3D mesh from a single image:

```bash
source venv/bin/activate

# IMPORTANT FOR MAC: Use CPU mode to avoid MPS memory errors
export SF3D_USE_CPU=1

# Basic usage (default settings)
python tests/sf3d_test_1_basic.py path/to/image.png

# With custom settings
python tests/sf3d_test_1_basic.py path/to/image.png \
    --output-dir generated_meshes/sf3d_custom \
    --texture-resolution 2048 \
    --remesh-option quad \
    --target-vertex-count 10000
```

#### Parameters

- `--texture-resolution`: Texture size in pixels (default: 1024)
  - Higher = better quality but slower
  - Recommended: 1024 for testing, 2048 for production

- `--remesh-option`: Mesh topology (default: none)
  - `none`: Fast, original topology
  - `triangle`: Triangle mesh (better for some cases)
  - `quad`: **Recommended for animation** - proper quad flow

- `--foreground-ratio`: Object size in frame (default: 0.75)
  - Lower = more padding around object
  - Recommended: 0.75 for clothing

- `--target-vertex-count`: Vertex count for remeshing (default: -1)
  - -1 = no reduction
  - Set to desired count (e.g., 10000 for performance)

#### Example Output

```
SF3D Test 1: Basic Mesh Generation
====================================
Input: generated_images/clothing.png
Output: generated_meshes/sf3d_test_1

Settings:
  - texture_resolution: 1024
  - remesh_option: quad
  - foreground_ratio: 0.75
  - target_vertex_count: -1
====================================

🖥️  Device: mps
📦 Loading SF3D model...
   ✓ Model loaded successfully
🖼️  Processing image...
   ✓ Saved preprocessed image
🚀 Generating 3D mesh...
   This should take ~0.5 seconds on GPU
   ✓ Generation completed in 0.52 seconds

✅ Success!
   Mesh saved to: generated_meshes/sf3d_test_1/mesh.glb

📊 Mesh Information:
   Vertices: 15,234
   Faces: 30,112
   Peak GPU Memory: 2,345.6 MB
```

### Test 2: View Textured Mesh

Open the Three.js viewer to inspect generated meshes:

```bash
# Open viewer with a specific mesh
python tests/sf3d_test_2_viewer.py generated_meshes/sf3d_test_1/mesh.glb

# Open viewer without a mesh (drag and drop later)
python tests/sf3d_test_2_viewer.py
```

#### Viewer Controls

- **Left Mouse**: Rotate
- **Right Mouse**: Pan
- **Scroll**: Zoom
- **R**: Reset camera view
- **T**: Toggle wireframe mode
- **L**: Toggle lighting

#### Features

- Drag and drop GLB files
- Displays UV-mapped textures
- Shows mesh statistics (vertices, faces, textures)
- Real-time FPS counter

---

## Integration with Existing Pipeline

### Option 1: Replace TripoSR in Pipeline

Update `tests/test_a_rigged_clothing.py` to use SF3D-generated meshes:

```python
# Line 117-118: Change clothing mesh path
# OLD:
clothing_path = "generated_meshes/triposr_glb/0/mesh.glb"

# NEW:
clothing_path = "generated_meshes/sf3d_test_1/mesh.glb"
```

### Option 2: Create SF3D-Specific Pipeline

Create a new test file `tests/test_b_sf3d_clothing.py` based on `test_a_rigged_clothing.py` but optimized for SF3D meshes:

```python
# Use SF3D quad-remeshed meshes
clothing_path = "generated_meshes/sf3d_quad/mesh.glb"

# SF3D meshes have better topology, may need different alignment
# Experiment with alignment settings in scale_and_align_meshes()
```

### Option 3: Batch Processing

Process multiple clothing images:

```bash
for img in generated_images/*.png; do
    python tests/sf3d_test_1_basic.py "$img" \
        --output-dir "generated_meshes/sf3d_batch/$(basename $img .png)" \
        --remesh-option quad \
        --texture-resolution 1024
done
```

---

## Comparison: TripoSR vs SF3D

### When to Use TripoSR

- You need to tweak marching cubes resolution
- You're okay with vertex colors (no textures)
- You have specific TripoSR workflows already working

### When to Use SF3D ⭐

- **Speed matters** (0.5s vs 5-15s)
- You need **textures** instead of vertex colors
- You want **animation-ready** meshes (quad topology)
- You need to **simplify meshes** without losing quality
- You want **better mesh quality** (no marching cubes artifacts)

### Recommendation

**Use SF3D** for the Spoken Wardrobe project because:
1. Faster generation = better user experience
2. UV textures = can apply different materials/patterns
3. Quad topology = better deformation for rigged animation
4. Can simplify meshes without losing texture data

---

## Troubleshooting

### Authentication Errors

**Error**: `401 Client Error: Unauthorized`

**Solution**:
```bash
# 1. Request access at https://huggingface.co/stabilityai/stable-fast-3d
# 2. Create token at https://huggingface.co/settings/tokens
# 3. Login:
huggingface-cli login
```

### OpenMP Library Not Found (Mac)

**Error**: `ld: library 'omp' not found`

**Solution**:
```bash
# Install OpenMP
brew install libomp

# Reinstall uv_unwrapper with library paths
source venv/bin/activate
export LDFLAGS="-L/opt/homebrew/opt/libomp/lib"
export CPPFLAGS="-I/opt/homebrew/opt/libomp/include"
pip install ./external/stable-fast-3d/uv_unwrapper/
```

### Slow Generation on Mac

**Issue**: Generation takes 10+ seconds instead of 0.5s

**Explanation**: SF3D uses MPS (Mac GPU) but it's not as fast as CUDA. This is expected.

**Solutions**:
- Use `--remesh-option none` to skip remeshing (faster)
- Lower `--texture-resolution` to 512 or 1024
- Accept that Mac will be slower than NVIDIA GPU

### Memory Errors

**Error**: `RuntimeError: MPS backend out of memory`

This is **common on Mac** with SF3D. The model uses ~5GB GPU memory which exceeds most Mac GPU limits.

**Solution (REQUIRED FOR MAC)**:
```bash
# Use CPU instead of MPS (Mac GPU)
export SF3D_USE_CPU=1
python tests/sf3d_test_1_basic.py image.png

# Lower texture resolution to reduce memory
python tests/sf3d_test_1_basic.py image.png --texture-resolution 512

# Process one image at a time (not batches)
```

**Performance on Mac CPU**:
- Generation time: ~50 seconds (acceptable for non-real-time use)
- GPU would be ~10 seconds, but doesn't work due to memory

**Alternative**: Keep using TripoSR for Mac GPU, which uses less memory

### Mesh Orientation Wrong

**Issue**: Mesh faces wrong direction in viewer

**Solution**: This was already fixed for TripoSR meshes in `test_a_rigged_clothing.py` (lines 175-184). SF3D meshes may need the same fix:

```python
# Apply -90° Y-axis rotation
theta_y = -np.pi / 2
rot_y = np.array([
    [np.cos(theta_y), 0, np.sin(theta_y)],
    [0, 1, 0],
    [-np.sin(theta_y), 0, np.cos(theta_y)]
])
mesh.vertices = mesh.vertices @ rot_y.T
```

---

## Advanced Usage

### Using Different Remeshing Options

```bash
# No remeshing (fastest)
python tests/sf3d_test_1_basic.py image.png --remesh-option none

# Triangle remeshing (balanced)
python tests/sf3d_test_1_basic.py image.png --remesh-option triangle

# Quad remeshing (best for animation)
python tests/sf3d_test_1_basic.py image.png \
    --remesh-option quad \
    --target-vertex-count 10000
```

### High-Quality Settings

For final production meshes:

```bash
python tests/sf3d_test_1_basic.py image.png \
    --texture-resolution 2048 \
    --remesh-option quad \
    --target-vertex-count 15000 \
    --foreground-ratio 0.85
```

### Performance Settings

For rapid prototyping:

```bash
python tests/sf3d_test_1_basic.py image.png \
    --texture-resolution 512 \
    --remesh-option none \
    --foreground-ratio 0.75
```

---

## Improving Mesh Quality

### Understanding Quality Trade-offs

SF3D mesh quality depends on several factors:

**Hardware Constraints**:
- **Mac CPU** (~50s): Lower quality due to limited computational resources
- **Mac GPU (MPS)**: Would be faster (~10s) but **out of memory** with SF3D (~5GB required)
- **PC GPU (CUDA)**: Highest quality, fastest speed (0.5-2s), no memory limits

**Quality Settings** (in order of impact):

#### 1. Texture Resolution (`--texture-resolution`)

Controls the size of the baked texture map:

```bash
# Low quality (faster, Mac CPU recommended)
--texture-resolution 512   # 512x512 texture

# Medium quality (default)
--texture-resolution 1024  # 1024x1024 texture

# High quality (requires GPU)
--texture-resolution 2048  # 2048x2048 texture

# Ultra quality (GPU only)
--texture-resolution 4096  # 4096x4096 texture
```

**Current limitation on Mac CPU**: Stick to 512-1024 for acceptable generation times.

#### 2. Remeshing Option (`--remesh-option`)

Controls mesh topology and structure:

```bash
# No remeshing (fastest but raw marching tetrahedra topology)
--remesh-option none

# Triangle remeshing (balanced topology)
--remesh-option triangle

# Quad remeshing (BEST for animation, proper edge flow)
--remesh-option quad
```

**Why quad is better**: Provides proper quad flow which deforms more naturally during animation. **Highly recommended** for the Spoken Wardrobe rigging system.

#### 3. Target Vertex Count (`--target-vertex-count`)

Controls mesh density:

```bash
# Automatic (model decides based on detail)
--target-vertex-count -1

# Low poly (performance, 5K vertices)
--target-vertex-count 5000

# Medium poly (balanced, 10K vertices)
--target-vertex-count 10000

# High poly (quality, 15K+ vertices)
--target-vertex-count 15000
```

**Trade-off**: Higher vertex count = more detail but slower real-time deformation.

#### 4. Foreground Ratio (`--foreground-ratio`)

Controls how much of the frame the object fills:

```bash
# More padding (object smaller in frame)
--foreground-ratio 0.65

# Default (balanced)
--foreground-ratio 0.75

# Tight crop (object fills more of frame)
--foreground-ratio 0.85
```

**Recommendation**: 0.75-0.85 for clothing to capture full detail.

### Recommended Settings by Use Case

#### Mac CPU (Current Setup)
```bash
# Balanced quality/speed for testing
python tests/sf3d_test_1_basic.py image.png \
    --texture-resolution 1024 \
    --remesh-option quad \
    --target-vertex-count 10000 \
    --foreground-ratio 0.75
```

**Expected**: ~50 seconds, decent quality for testing

#### PC GPU (Tomorrow)
```bash
# High quality production settings
python tests/sf3d_test_1_basic.py image.png \
    --texture-resolution 2048 \
    --remesh-option quad \
    --target-vertex-count 15000 \
    --foreground-ratio 0.85
```

**Expected**: 0.5-2 seconds, high quality suitable for final meshes

#### Rapid Prototyping (Mac CPU)
```bash
# Fast iteration for testing
python tests/sf3d_test_1_basic.py image.png \
    --texture-resolution 512 \
    --remesh-option none \
    --target-vertex-count 5000
```

**Expected**: ~30 seconds, lower quality but fast iteration

### Input Image Quality Matters

SF3D quality also depends heavily on input image:

**Good Input Images**:
- Clear, well-lit subject
- Subject fills 60-80% of frame
- Minimal background clutter
- Sharp focus (not blurry)
- Consistent lighting

**Poor Input Images**:
- Dark or underexposed
- Subject too small in frame
- Busy background
- Motion blur
- Extreme angles

**Tip**: The clothing images generated by your Stable Diffusion pipeline should work well since they're already segmented and well-framed.

### Why Mac CPU Quality is Lower

The quality difference you're seeing is NOT due to bugs - it's fundamental to CPU vs GPU:

1. **Memory bandwidth**: CPU has lower memory bandwidth than GPU
2. **Parallel processing**: GPU has thousands of cores vs CPU's handful
3. **Precision**: SF3D may use reduced precision on CPU to fit in memory
4. **Timeout constraints**: CPU may hit internal timeouts causing early termination

**Solution**: Use PC with NVIDIA GPU for production-quality meshes. Mac CPU is fine for testing and development.

---

## Understanding Textures and Materials (Why is it Shiny?)

### How SF3D Textures Work

SF3D generates **UV-unwrapped textures** with **PBR (Physically Based Rendering) materials**. This is very different from TripoSR's simple vertex colors.

#### What is UV Unwrapping?

Think of UV unwrapping like unfolding a 3D box into a flat pattern:

```
3D Mesh → Unfold → 2D Texture Map → Wrap back → Textured 3D Mesh
```

**Benefits**:
- Textures can be edited in 2D (Photoshop, GIMP)
- Higher resolution than vertex colors
- Can be swapped without changing geometry
- Standard format for game engines and 3D software

#### What is PBR?

PBR (Physically Based Rendering) materials simulate realistic light interaction using multiple texture maps:

1. **Base Color (Albedo)**: The actual color/pattern of the surface
2. **Normal Map**: Simulates surface details (wrinkles, bumps) without adding geometry
3. **Roughness Map**: Controls how shiny/matte the surface is
   - 0.0 = mirror-like (very shiny)
   - 1.0 = matte (no shine)
4. **Metalness Map**: Whether surface is metal or non-metal
   - 0.0 = non-metal (fabric, plastic, skin)
   - 1.0 = metal (steel, gold, aluminum)

**SF3D includes all of these maps** in the GLB file!

### Why Your Mesh Appears Shiny

The shininess you're seeing is likely due to:

#### 1. Default Material Properties in GLB

SF3D may be setting conservative material values:
- Roughness too low (too shiny)
- Metalness too high (looks like metal instead of fabric)

#### 2. Viewer Lighting

The Three.js viewer uses **Phong lighting** which can make materials appear more reflective than they should. Different lighting models show materials differently:

- **Phong lighting**: More specular highlights (shiny)
- **PBR lighting**: More physically accurate
- **Flat lighting**: No highlights at all

#### 3. Missing Environment Maps

PBR materials look best with **environment mapping** (reflections of surroundings). Without it, they can look overly shiny or unrealistic.

### How to Inspect Materials in Your GLB

You can check what SF3D actually generated by loading the GLB in Blender:

```bash
# Open Blender, then:
File → Import → glTF 2.0 (.glb/.gltf)
# Select: generated_meshes/sf3d_test_1/mesh.glb

# In Shading workspace, select the mesh and check:
# - Base Color texture
# - Roughness value (should be ~0.8 for fabric)
# - Metallic value (should be ~0.0 for fabric)
# - Normal map (adds surface detail)
```

### Solutions to Reduce Shininess

#### Option 1: Adjust Viewer Lighting (Quick Fix)

The `sf3d_viewer_textured.html` viewer can be modified to reduce shininess:

**Current lighting** (in viewer HTML):
```javascript
// Uses MeshPhongMaterial which can be shiny
```

**Better approach** (edit viewer):
```javascript
// Change to MeshStandardMaterial for better PBR
material = new THREE.MeshStandardMaterial({
    map: textureMap,
    roughness: 0.8,  // Reduce shininess
    metalness: 0.0   // Non-metallic (fabric)
});
```

#### Option 2: Modify GLB in Blender (Proper Fix)

1. Open GLB in Blender
2. Switch to **Shading** workspace
3. Select the mesh object
4. In the shader nodes, find **Principled BSDF**:
   - Set **Roughness** to 0.8-1.0 (matte fabric)
   - Set **Metallic** to 0.0 (non-metal)
   - Adjust **Specular** to 0.2-0.3 (reduce reflections)
5. Export as GLB: `File → Export → glTF 2.0`

#### Option 3: Edit Viewer Environment (Best for Realism)

Add environment lighting to viewer for more realistic PBR:

```javascript
// Add HDRI environment map
const pmremGenerator = new THREE.PMREMGenerator(renderer);
scene.environment = pmremGenerator.fromScene(
    new THREE.RoomEnvironment()
).texture;
```

This makes PBR materials render more realistically.

### What SF3D Actually Generates

When you look inside the GLB file, SF3D includes:

```
mesh.glb (binary file)
├── Geometry (vertices, faces, UVs)
├── Base Color Texture (PNG/JPG embedded)
├── Normal Map (for surface detail)
├── Roughness Map (controls shine)
├── Metalness Map (metal vs non-metal)
└── Material Properties (PBR parameters)
```

**You can extract these textures** using Blender or glTF tools if you want to edit them separately.

### Comparing to TripoSR

| Feature | TripoSR | SF3D |
|---------|---------|------|
| **Texture Type** | Vertex colors | UV-mapped textures |
| **Material System** | Simple RGB | PBR (roughness, metalness, normals) |
| **Shininess Control** | None (matte only) | Full control via roughness map |
| **Editability** | Hard (must recolor vertices) | Easy (edit textures in 2D) |
| **File Format** | OBJ/GLB (no materials) | GLB with embedded PBR materials |

**Why SF3D looks shiny**: Because it's trying to be more realistic! You just need to adjust the material parameters to match fabric instead of the default conservative values.

### Recommended Workflow

For the Spoken Wardrobe project:

1. **Generate mesh with SF3D** (as you're doing now)
2. **Import GLB into Blender** (one-time setup)
3. **Adjust material to look like fabric**:
   - Roughness: 0.8-0.9 (matte fabric)
   - Metalness: 0.0 (not metal)
   - Specular: 0.2-0.3 (subtle highlights)
4. **Export as GLB** with adjusted materials
5. **Use in your pipeline** for rigging and animation

**OR** modify the Three.js viewer to automatically apply fabric-like materials to loaded meshes.

---

## Next Steps

1. **Authenticate with Hugging Face** (required)
   ```bash
   huggingface-cli login
   ```

2. **Test with a sample image**
   ```bash
   python tests/sf3d_test_1_basic.py generated_images/your_image.png
   ```

3. **View the result**
   ```bash
   python tests/sf3d_test_2_viewer.py generated_meshes/sf3d_test_1/mesh.glb
   ```

4. **Integrate with rigged clothing**
   - Update `tests/test_a_rigged_clothing.py` to use SF3D mesh
   - Test alignment and deformation
   - Adjust orientation if needed

5. **Compare TripoSR vs SF3D**
   - Generate same clothing with both methods
   - Compare speed, quality, and animation results
   - Decide which to use for production

---

## Files Created

```
tests/
├── sf3d_test_1_basic.py          # Generate mesh from image
├── sf3d_test_2_viewer.py         # Launch textured mesh viewer
└── sf3d_viewer_textured.html     # Three.js GLB viewer with textures

external/
└── stable-fast-3d/               # SF3D repository (cloned)

SF3D_README.md                    # This guide
```

---

## Resources

- **SF3D Paper**: https://arxiv.org/abs/2408.00653
- **SF3D GitHub**: https://github.com/Stability-AI/stable-fast-3d
- **Hugging Face**: https://huggingface.co/stabilityai/stable-fast-3d
- **Online GLB Viewer**: https://gltf-viewer.donmccurdy.com/

---

## Notes

- SF3D is gated - you **must** request access on Hugging Face
- First run downloads ~2GB model (cached for future use)
- Mac MPS support is experimental but works
- UV-unwrapped textures mean you can:
  - Apply different materials in Blender
  - Simplify mesh without losing textures
  - Use standard texture editing tools
  - Export to game engines easily

---

**Created**: November 7, 2024
**Branch**: `stable-fast-3d-testing`
**Status**: ✅ Installed and Ready to Test
