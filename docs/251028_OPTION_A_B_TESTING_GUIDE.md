# Testing Guide: Option A & B - Proper 3D Deformation

**Date**: October 28, 2025  
**Purpose**: Guide for testing unified cage deformation (Option A) and skeletal skinning (Option B)

---

## Overview

We've implemented TWO proper approaches to 3D mesh deformation that should handle rotation correctly:

### **Option A: Unified Cage Deformation**

- **File**: `tests/test_integration_cage.py`
- **Approach**: Single unified cage with MVC binding
- **Best for**: Smooth, organic deformations
- **Complexity**: Medium

### **Option B: Skeletal Skinning**

- **File**: `tests/test_integration_skinning.py`
- **Approach**: Automatic bone rigging with LBS
- **Best for**: Precise, articulated movement
- **Complexity**: Medium-High

Both use the **same web viewer** (`tests/enhanced_mesh_viewer_v2.html`) without mesh simplification.

---

## Prerequisites

1. **Mesh File**: You need a generated mesh (e.g., `generated_meshes/0/mesh.obj`)
2. **Web Browser**: For viewing the 3D deformation
3. **Webcam**: For MediaPipe pose tracking
4. **Python Environment**: With all dependencies installed

---

## Option A: Unified Cage Deformation

### What's Different from Previous Attempts

❌ **Previous (Broken)**:

- Multiple independent cage boxes
- Recomputed weights every frame
- Each section moved independently (caused pinching/smearing)

✅ **Now (Fixed)**:

- **ONE** unified cage around entire mesh
- Weights computed **ONCE** during setup
- Sections deform with hierarchical constraints
- Smooth, connected deformation

### How It Works

```
1. Create unified humanoid-shaped cage around mesh
2. Compute MVC weights binding mesh vertices to cage (ONCE)
3. Each frame:
   - MediaPipe detects pose
   - Cage sections translate based on keypoints
   - Mesh deforms using pre-computed weights (fast!)
```

### Running Option A

```bash
# Basic usage
python tests/test_integration_cage.py --mesh generated_meshes/0/mesh.obj

# Headless mode (no Python window)
python tests/test_integration_cage.py --mesh generated_meshes/0/mesh.obj --headless
```

### Testing Steps

1. **Start the script**:

   ```bash
   python tests/test_integration_cage.py --mesh generated_meshes/0/mesh.obj
   ```

2. **Observe Python window**:

   - Should see your webcam feed
   - MediaPipe skeleton overlay
   - FPS counter
   - "Option A: Unified Cage" label

3. **Open web viewer**:

   - Open `tests/enhanced_mesh_viewer_v2.html` in your browser
   - Or press `O` in the Python window

4. **Test rotation handling**:

   - Stand in front of camera
   - Turn your body left/right (45°, 90°)
   - **Expected**: Mesh should follow your rotation smoothly
   - **No** pinching, smearing, or detachment

5. **Test cage visualization**:

   - Press `C` to toggle cage on/off
   - Cage should appear as magenta wireframe
   - Cage should move with your body sections

6. **Verify smooth deformation**:
   - Raise arms
   - Move torso
   - **Expected**: Mesh stays connected, no tears

### Expected Output

**Console**:

```
======================================================================
UNIFIED CAGE DEFORMATION SYSTEM - OPTION A
======================================================================
Mesh: generated_meshes/0/mesh.obj
Headless: False
======================================================================

Loading mesh...
✓ Mesh loaded: 15234 vertices, 30468 faces

============================================================
GENERATING UNIFIED CAGE
============================================================
Mesh bounds: [-1.2, -0.8, -0.3] to [1.2, 0.8, 0.3]
Mesh center: [0.0, 0.0, 0.0]
...
  Section 'torso': 64 vertices
  Section 'left_upper_arm': 48 vertices
...

✓ Unified cage generated
  Total vertices: 240
  Total faces: 478
  Sections: ['torso', 'left_upper_arm', ...]

============================================================
COMPUTING MVC WEIGHTS
============================================================
Mesh vertices: 15234
Cage vertices: 240
  Processing vertices 0/15234...
  Processing vertices 5000/15234...
...

✓ MVC weights computed: shape (15234, 240)
  Weight sum per vertex: min=1.0000, max=1.0000

✓ SYSTEM READY
```

---

## Option B: Skeletal Skinning

### What's Different from Previous LBS Attempts

❌ **Previous (Failed)**:

- Displacement vectors only (no rotation)
- Poor weight computation
- No hierarchical constraints
- Coordinate system mismatches

✅ **Now (Fixed)**:

- **4x4 transformation matrices** (rotation + translation)
- Automatic weight computation with Gaussian falloff
- Proper bone hierarchy (arms inherit from torso)
- Bind pose with inverse bind matrices
- Normalized coordinate system

### How It Works

```
1. Calibrate bind pose from T-pose
   - Generate skeleton from MediaPipe
   - Compute bone transformations (4x4 matrices)
   - Automatically compute skinning weights
   - Store inverse bind matrices

2. Each frame:
   - MediaPipe detects current pose
   - Compute bone transformations with hierarchy
   - Apply LBS: v' = Σ wᵢ * Mᵢ * Mᵢ_bind_inv * v
   - Fast matrix operations!
```

### Running Option B

```bash
# Basic usage
python tests/test_integration_skinning.py --mesh generated_meshes/0/mesh.obj

# Headless mode
python tests/test_integration_skinning.py --mesh generated_meshes/0/mesh.obj --headless
```

### Testing Steps

1. **Start the script**:

   ```bash
   python tests/test_integration_skinning.py --mesh generated_meshes/0/mesh.obj
   ```

2. **Calibrate bind pose**:

   - Stand in a **T-pose** (arms out to sides)
   - Press **SPACE** to calibrate
   - **Critical**: This sets the reference pose for deformation

   Console should show:

   ```
   🔄 Calibrating bind pose...

   ============================================================
   CALIBRATING BIND POSE
   ============================================================
   Body scale: 457.3 pixels
   ...
   Computing skinning weights...

   ✓ Calibration complete
   ...

   🎉 Calibration successful! System is now tracking your movements.
   ```

3. **Open web viewer**:

   - Open `tests/enhanced_mesh_viewer_v2.html` in your browser
   - Or press `O` in the Python window

4. **Test deformation**:
   - **Raise arms**: Should deform smoothly
   - **Rotate body**: Mesh should rotate with you
   - **Move around**: Mesh tracks your position
5. **Test rotation handling** (KEY TEST):
   - Turn 45° left
   - Turn 45° right
   - Turn 90° (profile view)
   - **Expected**: Mesh maintains integrity, rotates naturally

### Expected Output

**Console**:

```
======================================================================
SKELETAL SKINNING SYSTEM - OPTION B
======================================================================
Mesh: generated_meshes/0/mesh.obj
Headless: False
======================================================================

Loading mesh...
✓ Mesh loaded: 15234 vertices, 30468 faces

============================================================
AUTOMATIC RIGGING SYSTEM
============================================================
Mesh: 15234 vertices, 30468 faces
Skeleton: 5 bones
  Bones: ['torso', 'left_upper_arm', 'right_upper_arm', ...]
============================================================

✓ SYSTEM READY - WAITING FOR CALIBRATION

Controls:
  SPACE - Calibrate bind pose (stand in T-pose)
  Q - Quit
  O - Open web viewer
```

After pressing SPACE:

```
🔄 Calibrating bind pose...

============================================================
CALIBRATING BIND POSE
============================================================
Body scale: 457.3 pixels

Computing skinning weights...

✓ Calibration complete
  Skinning weights: (15234, 5)
  Weight sum per vertex: min=1.0000, max=1.0000
============================================================

🎉 Calibration successful! System is now tracking your movements.
```

---

## Comparison: Which Option is Better?

| Aspect                  | Option A (Cage)                | Option B (Skinning)               |
| ----------------------- | ------------------------------ | --------------------------------- |
| **Setup**               | Automatic, instant             | Requires T-pose calibration       |
| **Deformation Quality** | Smooth, organic                | Precise, articulated              |
| **Rotation Handling**   | Good                           | Excellent                         |
| **Computational Cost**  | Medium                         | Medium-High                       |
| **Best for**            | Soft clothing (dresses, capes) | Fitted clothing (shirts, jackets) |
| **Pros**                | No calibration, smooth         | Accurate, bone-based              |
| **Cons**                | Less precise joint control     | Requires good T-pose              |

### Recommendations

**Try Option A first if**:

- Your mesh is flowing/soft (dress, skirt)
- You want instant results (no calibration)
- You prioritize smooth, organic deformation

**Try Option B if**:

- Your mesh has defined sleeves/limbs
- You need precise joint control
- You can provide a good T-pose for calibration
- You want the most accurate rotation handling

---

## Troubleshooting

### Option A Issues

**Problem**: Cage doesn't move with body

- **Check**: MediaPipe skeleton is visible in Python window
- **Fix**: Ensure good lighting, stand clearly in frame

**Problem**: Mesh pinches or tears

- **Check**: Cage sections in console output
- **Fix**: This shouldn't happen with unified cage. If it does, report as bug.

**Problem**: Deformation too stiff

- **Check**: MVC weights are computed correctly (sum = 1.0)
- **Adjust**: Can modify `smooth_alpha` in code for more/less smoothing

### Option B Issues

**Problem**: Calibration fails

- **Check**: "No pose detected" message
- **Fix**: Stand in clear T-pose with arms fully visible

**Problem**: Mesh deforms incorrectly after calibration

- **Cause**: Poor T-pose during calibration
- **Fix**: Restart script, calibrate again with better T-pose

**Problem**: Arms detach from torso

- **Check**: Bone hierarchy in console (should show parent-child)
- **Fix**: This shouldn't happen. If it does, report as bug.

**Problem**: Jittery movement

- **Adjust**: Increase `smooth_alpha` in code (higher = more smoothing)

### Common Issues (Both)

**Problem**: WebSocket connection fails

- **Check**: Browser console for errors
- **Fix**: Ensure port 8765 is not in use, restart script

**Problem**: Mesh too small/large in web viewer

- **Use**: OrbitControls to zoom (mouse wheel)
- **Use**: WASD to move camera

**Problem**: Mesh not visible in web viewer

- **Check**: Browser console debug info
- **Fix**: Use camera controls to find mesh, check vertex ranges

**Problem**: Low FPS

- **Note**: These are complex deformations, expect 20-30 FPS
- **Fix**: Lower mesh resolution or use `--headless` mode

---

## Success Criteria

### ✅ Option A is working if:

1. Mesh deforms smoothly with your body
2. No pinching or tearing at joints
3. Rotation 90° left/right: mesh stays intact
4. Cage (if shown) moves with body sections
5. FPS > 20

### ✅ Option B is working if:

1. Calibration completes successfully
2. Arms follow your arm movement precisely
3. Torso follows body rotation
4. Rotation 90° left/right: mesh rotates naturally
5. No joint detachment
6. FPS > 15

---

## Next Steps

After testing both options:

1. **Document your findings**:

   - Which option works better for your mesh?
   - What are the pros/cons you observed?
   - Any specific issues?

2. **If both work**:

   - Great! Choose based on your use case
   - Option A for soft/flowing clothing
   - Option B for fitted/structured clothing

3. **If neither works perfectly**:

   - Document specific failure modes
   - We may need to combine approaches
   - Or investigate hybrid solutions

4. **Future enhancements**:
   - Option A: More sophisticated cage shapes
   - Option B: More bones for finer control
   - Both: Better temporal smoothing
   - Both: Depth calibration for Z-axis

---

## Files Modified/Created

### New Files

- `tests/test_integration_cage.py` - Option A implementation
- `tests/test_integration_skinning.py` - Option B implementation
- `docs/251028_OPTION_A_B_TESTING_GUIDE.md` - This guide

### Modified Files

- `tests/enhanced_mesh_viewer_v2.html` - Reverted SimplifyModifier (back to working version)

### Unchanged (Used by both)

- `tests/enhanced_websocket_server_v2.py` - WebSocket server
- All other existing files remain untouched

---

## Technical Details

### Option A: Unified Cage

**Key Algorithm**: Mean Value Coordinates (MVC)

```python
# 1. Generate unified cage (ONE mesh)
cage = UnifiedCageGenerator(mesh).generate_unified_cage()

# 2. Compute weights (ONCE)
mvc_weights = compute_mvc_weights(mesh.vertices, cage.vertices)
# Shape: (n_mesh_verts, n_cage_verts)
# Each row sums to 1.0

# 3. Deform (real-time)
deformed_cage = deform_cage_from_keypoints(cage, keypoints)
deformed_mesh = mvc_weights @ deformed_cage  # Fast matrix multiply!
```

**Mathematical Foundation**:

- Simplified MVC using inverse distance weighting
- Full MVC formula from "Mean Value Coordinates for Closed Triangular Meshes" (Ju et al. 2005)
- For production, consider implementing full formula for better quality

### Option B: Skeletal Skinning

**Key Algorithm**: Linear Blend Skinning (LBS)

```python
# 1. Calibrate (one-time)
bind_transforms = compute_bone_transforms(t_pose_keypoints)
inv_bind_transforms = invert(bind_transforms)
skinning_weights = compute_weights(mesh.vertices, bone_positions)
# Shape: (n_mesh_verts, n_bones)

# 2. Deform (real-time)
current_transforms = compute_bone_transforms(current_keypoints)
for bone_i:
    skinning_matrix = current_transforms[i] @ inv_bind_transforms[i]
    deformed_mesh += skinning_weights[:, i] * (skinning_matrix @ mesh.vertices)
```

**Mathematical Foundation**:

- LBS formula: v' = Σ wᵢ _ Mᵢ _ Mᵢ_bind_inv \* v
- Rodrigues' rotation formula for bone orientation
- Gaussian falloff for automatic weight computation
- Hierarchical transformations (child = parent @ child)

---

## Questions?

If you encounter issues not covered in this guide:

1. Check console output for error messages
2. Verify mesh file exists and is valid
3. Ensure MediaPipe detects your pose (skeleton visible)
4. Check web browser console for WebSocket/rendering errors

**Report any bugs or unexpected behavior!**
