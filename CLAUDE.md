# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

**Spoken Wardrobe 2** is an AR clothing visualization system that generates and animates 3D clothing meshes in real-time based on user pose. The pipeline consists of:

1. **AI Generation**: Stable Diffusion inpainting to generate clothing from text prompts on segmented body parts
2. **3D Reconstruction**: TripoSR to convert 2D clothing images into 3D meshes
3. **Pose Tracking**: MediaPipe for real-time skeletal tracking
4. **Mesh Deformation**: Cage-based or skeletal skinning systems to animate meshes with user movement

## Environment Setup

### Python Version
- Python 3.11.5 (specified in venv)

### Virtual Environment
```bash
source venv/bin/activate
```

### Dependencies
Install dependencies with:
```bash
pip install -r requirements.txt
pip install -r requirements-ai.txt
```

Core dependencies include:
- PyTorch with MPS support (Mac GPU) or CUDA (NVIDIA GPU)
- MediaPipe for pose detection
- Stable Diffusion (diffusers library)
- TripoSR (in `external/TripoSR/`)
- BodyPix for body part segmentation
- trimesh for mesh operations

## Key Commands

### Generate Clothing Image
Run Stable Diffusion inpainting to generate clothing on segmented body:
```bash
python tests/create_consistent_pipeline_v2.py
```
This captures a frame, runs BodyPix segmentation, generates clothing with AI, and saves reference data.

### Generate 3D Mesh from Image
Convert 2D clothing image to 3D mesh using TripoSR:
```bash
# Interactive mode (prompts for image selection)
python tests/clothing_to_3d_triposr_2.py

# Direct mode
python tests/clothing_to_3d_triposr_2.py path/to/image.png
```

Output goes to `generated_meshes/TIMESTAMP/` containing:
- `mesh.obj` - 3D mesh file
- `reference_data.pkl` - BodyPix masks and MediaPipe keypoints

### Real-Time Mesh Deformation Testing

The project has multiple deformation approaches:

**Option A: Unified Cage Deformation** (recommended for soft/flowing clothing)
```bash
python tests/test_integration_cage.py --mesh generated_meshes/0/mesh.obj
```

**Option B: Skeletal Skinning** (recommended for fitted clothing)
```bash
python tests/test_integration_skinning.py --mesh generated_meshes/0/mesh.obj
# Press SPACE to calibrate in T-pose
```

**Option V2: Articulated Cage System**
```bash
python tests/test_integration_cage_v2.py --mesh generated_meshes/0/mesh.obj
# Press SPACE to initialize cage in T-pose
```

**Option V3: Fixed Articulated Cage** (latest)
```bash
python tests/test_integration_cage_v3.py --mesh generated_meshes/0/mesh.obj
# Press SPACE for 5-second T-pose countdown
```

With reference data for better quality:
```bash
python tests/test_integration_cage_v3.py \
    --mesh generated_meshes/TIMESTAMP/mesh.obj \
    --reference generated_meshes/TIMESTAMP/reference_data.pkl
```

### Web Viewer
All deformation tests require opening the web viewer first:
```bash
open tests/enhanced_mesh_viewer_v2.html
```
The viewer connects via WebSocket on `ws://localhost:8765`

### Run Single Test
To test individual components:
```bash
# Test BodyPix segmentation
python tests/run_all_triposr_tests.py

# Test cage generation
python 251025_data_verification/verify_cage_structure.py

# Test MVC weights
python 251025_data_verification/verify_mvc_weights.py
```

## Architecture

### Pipeline Flow

```
User Input (speech/text)
  → BodyPix Segmentation (24-part body mask)
  → Stable Diffusion Inpainting (generate clothing on mask)
  → TripoSR (2D image → 3D mesh)
  → Reference Data (save BodyPix masks + MediaPipe keypoints)
  → Real-time Deformation (cage/skinning system)
  → WebSocket Stream (send to Three.js viewer)
```

### Core Modules

**`src/modules/`** - Main application modules
- `ai_generation.py`: Stable Diffusion pipeline wrapper
  - `ClothingGenerator` class handles model loading, prompt creation, inpainting
  - Uses runwayml/stable-diffusion-inpainting model
  - Caches models in `./models/` directory
  - Saves reference data including BodyPix masks for cage generation

- `body_tracking.py`: MediaPipe pose detection wrapper
- `keypoint_matching.py`: Maps MediaPipe keypoints to BodyPix segments
- `speech_recognition.py`: Voice input for clothing prompts

**`tests/`** - Implementation and testing (40+ test files)

Key test files:
- `clothing_to_3d_triposr_2.py`: Interactive TripoSR runner with automatic mesh orientation correction (90° Y-rotation + 180° flip)
- `create_consistent_pipeline_v2.py`: Complete pipeline from camera to mesh with consistent reference data
- `test_integration_cage.py`: Unified cage deformation system (Option A)
- `test_integration_skinning.py`: Skeletal skinning system (Option B)
- `test_integration_cage_v3.py`: Latest articulated cage with 5-second T-pose calibration
- `articulated_cage_generator.py`: OBB-based cage generation with anatomical structure
- `articulated_deformer.py`: Regional MVC deformation to prevent pinching
- `keypoint_mapper.py`: Maps MediaPipe keypoints to cage transformations
- `enhanced_cage_utils.py`: BodyPix-based cage generation with 3D depth estimation
- `enhanced_websocket_server_v2.py`: WebSocket server for Three.js viewer

**`external/`** - Third-party dependencies
- `TripoSR/`: Image-to-3D mesh generation (must be cloned separately)
- `VIBE/`: Video-based pose estimation (legacy)
- `EasyMocap/`: Multi-view mocap (legacy)

### Deformation Systems Explained

The project implements multiple 3D mesh deformation approaches:

**Cage-Based Deformation (MVC)**
- Generates an anatomical cage (30-60 vertices) around the mesh
- Uses Mean Value Coordinates to compute weights (once during setup)
- Real-time: transforms cage vertices based on MediaPipe keypoints, mesh follows smoothly
- Critical: cage must have anatomical structure (torso, arms, legs as separate sections)
- Prevents pinching by using regional MVC (vertices influenced only by nearby cage vertices)

**Skeletal Skinning (LBS)**
- Creates virtual bones from MediaPipe keypoints (spine, arms, legs)
- Computes skinning weights using Gaussian falloff
- Real-time: applies 4x4 transformation matrices to bones, vertices blend based on weights
- Requires T-pose calibration to compute inverse bind matrices
- More accurate for fitted clothing that should follow body articulation

**Key Difference from 2D Warping**
- These are true 3D systems (not Snapchat-style 2D face mesh)
- Meshes have actual volume and depth (Z-axis)
- Handle rotation and viewing from any angle
- Proper depth ordering and occlusion

### Coordinate Systems

**BodyPix**: 2D pixel coordinates (x, y) in image space

**MediaPipe**:
- 2D: (x, y) normalized [0, 1]
- 3D: (x, y, z) where z is relative depth (not metric, requires calibration)

**TripoSR Mesh**: 3D world coordinates (X, Y, Z)
- Output meshes require orientation correction: 90° Y-rotation + 180° flip to face forward
- Depth estimation is approximate (no metric scale)

**WebSocket Protocol**: JSON messages with mesh vertices, cage vertices, transformation matrices

### Critical Implementation Details

**Cage Generation Must Be Anatomical**
- ConvexHull collapses to single box → causes smearing (DO NOT USE)
- Use OBB (Oriented Bounding Box) per body part
- Track which cage vertices belong to which anatomical section
- Return `(cage_mesh, cage_structure)` where structure maps vertices to body parts

**MVC Weight Computation**
- Compute ONCE during initialization (expensive operation)
- Store weight matrix: (n_mesh_vertices × n_cage_vertices)
- Each row must sum to 1.0
- DO NOT recompute every frame
- If cage size changes, weights are invalid → must regenerate

**Section-Wise Deformation**
- Each body part section transforms independently
- Hierarchical: arms inherit 50% of torso movement (prevents detachment)
- Use MediaPipe keypoint connections to define parent-child relationships
- Apply rigid transformation to interior vertices, blend only at section boundaries

**Depth Estimation Strategy**
- BodyPix provides 2D masks → need to extrapolate to 3D
- Use mesh bounding box Z-extent as reference
- Different body parts have different depth ratios: torso=100%, arms=30-40%, head=50%
- With reference data: use actual mesh vertices within each BodyPix segment

## Common Issues and Solutions

### Mesh Orientation Wrong
**Symptom**: Mesh is sideways or upside down
**Fix**: `clothing_to_3d_triposr_2.py` applies 180° flip + 90° Y-rotation automatically (lines ~373-385)

### Mesh Smearing/Pinching
**Symptom**: Mesh collapses toward corners when user moves
**Cause**: Using global MVC (all vertices influenced by all cage vertices)
**Fix**: Use regional MVC - each vertex influenced only by nearest cage section (see `articulated_deformer.py`)

### Only Single Cage Box Generated
**Symptom**: Cage has no anatomical sections
**Cause**: MediaPipe segmentation doesn't separate limbs (it's a single person mask)
**Fix**: Use reference data from BodyPix which has 24-part segmentation
**Workaround**: Keypoint-based partitioning (groups keypoints by anatomical region)

### Arms Detach from Torso
**Symptom**: Arms float away during movement
**Cause**: Independent section transformations
**Fix**: Implement hierarchical transforms (arms inherit partial torso movement)

### Mesh Not Visible in Viewer
**Cause**: Coordinate mismatch or WebSocket not connected
**Fix**:
1. Check browser console (F12) for errors
2. Verify WebSocket connection to localhost:8765
3. Try zooming out in viewer (scroll wheel)
4. Check terminal for "✓ Loaded mesh: X vertices"

### Low FPS (< 20)
**Cause**: Mesh too complex or MVC computation in real-time loop
**Fix**:
1. Simplify mesh (reduce vertex count)
2. Verify MVC weights computed only once during setup
3. Check CPU usage (should be < 60%)

## Testing Workflow

### End-to-End Test
1. Open web viewer: `open tests/enhanced_mesh_viewer_v2.html`
2. Run V3 integration: `python tests/test_integration_cage_v3.py --mesh generated_meshes/0/mesh.obj`
3. Press SPACE when ready
4. Wait 5 seconds for T-pose calibration countdown
5. Move around and verify mesh follows smoothly

### Expected Good Results
- Terminal shows 5-10 OBB sections generated (not just 1)
- Mesh orientation: facing forward (not sideways)
- Cage visible as pink wireframe with multiple connected sections
- FPS > 30 in terminal
- No pinching in mesh interior
- Sections stay connected during movement

### With Reference Data (Higher Quality)
1. Generate complete pipeline: `python tests/create_consistent_pipeline_v2.py`
2. Follow prompts to capture frame, select body parts, generate clothing
3. Use resulting mesh with reference: `python tests/test_integration_cage_v3.py --mesh generated_meshes/TIMESTAMP/mesh.obj --reference generated_meshes/TIMESTAMP/reference_data.pkl`
4. Benefits: Real 24-part BodyPix segmentation → 5-10 OBB sections (vs 5-7 from keypoint partitioning)

## Directory Structure

```
spoken_wardrobe_2/
├── src/modules/          # Core application modules
├── tests/                # Implementation and testing (40+ files)
├── external/             # Third-party dependencies (TripoSR, VIBE, EasyMocap)
├── generated_images/     # Stable Diffusion outputs
├── generated_meshes/     # TripoSR mesh outputs + reference data
├── models/               # Cached AI models (Stable Diffusion, TripoSR)
├── docs/                 # Detailed documentation (45+ markdown files)
├── 251025_data_verification/  # Cage verification tools
├── calibration_data/     # Camera calibration data
└── venv/                 # Python virtual environment
```

## Important Notes

### TripoSR Path
The system expects TripoSR at `external/TripoSR/`. All test files add this to `sys.path`:
```python
triposr_path = Path(__file__).parent.parent / "external" / "TripoSR"
sys.path.insert(0, str(triposr_path))
```

### GPU Acceleration
The system auto-detects and uses:
- Mac GPU (MPS) for Apple Silicon
- NVIDIA GPU (CUDA) if available
- CPU as fallback (slow)

Check `src/modules/ai_generation.py` for device selection logic.

### Reference Data Format
`reference_data.pkl` (saved by pipeline) contains:
```python
{
    'bodypix_masks': {
        'left_upper_arm': binary_mask_array,
        'torso': binary_mask_array,
        # ... 24 body part masks
    },
    'keypoints_2d': {
        'left_shoulder': [x, y],
        'right_shoulder': [x, y],
        # ... MediaPipe keypoint positions
    },
    'frame_shape': (height, width, 3)
}
```

### WebSocket Protocol
Server sends JSON messages:
```json
{
  "type": "update",
  "mesh_vertices": [[x, y, z], ...],
  "cage_vertices": [[x, y, z], ...],
  "cage_edges": [[i, j], ...]
}
```

Viewer (Three.js) updates mesh geometry and cage wireframe in real-time.

## Research References

The cage-based deformation system is based on academic research:

1. **Mean Value Coordinates** (Ju et al. 2005) - Core MVC algorithm
2. **Interactive Cage Generation** (Le & Deng 2017) - OBB-based anatomical cages
3. **Skeleton-Driven Hierarchical Cages** (Chen & Feng 2014) - Joint-based deformation
4. **Linear Blend Skinning** (standard in game engines) - Skeletal animation

See documentation in `docs/` for detailed research notes and implementation comparisons.

## Documentation

The `docs/` directory contains 45+ detailed markdown files:

**Quick Starts**:
- `QUICK_START_V3.md`: Latest V3 articulated cage system
- `QUICK_START_ARTICULATED_CAGE.md`: Articulated cage concepts
- `TRIPOSR_TESTS_README.md`: TripoSR testing guide

**Implementation Summaries**:
- `IMPLEMENTATION_SUMMARY_OPTION_A_B.md`: Comparison of cage vs skinning approaches
- `docs/251026_IMPLEMENTATION_COMPLETE.md`: Complete cage implementation details
- `docs/251028_CAGE_DEFORMATION_FIXES.md`: Fixes for smearing/pinching issues

**Research and Planning**:
- `docs/251025_steps_forward.md`: Research foundation and correct pipeline
- `docs/251026_implementation_plan.md`: Detailed implementation strategy
- `docs/251028_ARTICULATED_CAGE_IMPLEMENTATION.md`: Technical deep-dive

Always check the relevant quick start guide before making changes to deformation systems.
