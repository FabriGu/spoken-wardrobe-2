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

**Error**: `RuntimeError: Out of memory`

**Solutions**:
```bash
# Use CPU instead of GPU
SF3D_USE_CPU=1 python tests/sf3d_test_1_basic.py image.png

# Lower texture resolution
python tests/sf3d_test_1_basic.py image.png --texture-resolution 512

# Process one image at a time (not batches)
```

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
