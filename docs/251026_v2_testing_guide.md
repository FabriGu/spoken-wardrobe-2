# Testing Guide for Prototype V2

**Date**: October 26, 2025

---

## Overview

This guide explains how to test the corrected cage-based deformation system (V2).

**Key Features Being Tested**:

1. Cage generation from reference data (not from user in camera)
2. Proper coordinate system handling
3. Section-wise cage deformation
4. 2D vs 3D warping modes
5. Debug logging and visualization

---

## Prerequisites

### Required Files

- **Mesh**: `generated_meshes/0/mesh.obj` (or any generated mesh)
- **Reference Data**: To be created in Step 1

### Required Packages

```bash
pip install trimesh mediapipe opencv-python numpy websockets
```

---

## Testing Steps

### Step 1: Create Reference Data

Since we don't have the full generation pipeline running, we'll create mock reference data.

**Option A: With Camera (Recommended)**

```bash
python tests/create_mock_reference_data.py \
    --mesh generated_meshes/0/mesh.obj \
    --output generated_images/0_reference.pkl
```

This will:

1. Open your camera
2. Ask you to stand in a T-pose
3. Capture your pose as reference
4. Save reference data

**Option B: Without Camera (Synthetic Data)**

```bash
python tests/create_mock_reference_data.py \
    --mesh generated_meshes/0/mesh.obj \
    --output generated_images/0_reference.pkl \
    --no-camera
```

**Expected Output**:

```
CREATING MOCK REFERENCE DATA
✓ Created 10 mock BodyPix masks
✓ Captured reference pose with 13 keypoints
✓ Reference data saved: generated_images/0_reference.pkl
✓ Reference frame saved: generated_images/0_reference_frame.png
```

---

### Step 2: Test 2D Warping (Default)

**Run the main script**:

```bash
python tests/test_integration_v2.py \
    --mesh generated_meshes/0/mesh.obj \
    --reference generated_images/0_reference.pkl
```

**What should happen**:

1. Terminal shows setup phase (loading mesh, generating cage, etc.)
2. Camera window opens showing your video feed with MediaPipe skeleton
3. WebSocket server starts on port 8765

**Expected Terminal Output**:

```
======================================================================
INTEGRATED DEFORMATION SYSTEM V2
======================================================================
Mesh: generated_meshes/0/mesh.obj
Reference: generated_images/0_reference.pkl
Z-axis warping: DISABLED (2D only)
Headless mode: False
======================================================================

SETUP PHASE
1. Loading mesh and reference data...
✓ Mesh loaded: 16 vertices
✓ Reference data loaded

2. Generating cage from reference data...
✓ Cage generated: 24 vertices

3. Computing MVC weights...
✓ MVC weights computed

4. Initializing MediaPipe Pose...
✓ MediaPipe initialized

5. Initializing keypoint mapper...
✓ Keypoint mapper initialized

6. Initializing camera...
✓ Camera initialized

7. Starting WebSocket server...
✓ WebSocket server started

SETUP COMPLETE
```

---

### Step 3: Open Web Viewer

1. Open `tests/enhanced_mesh_viewer_v2.html` in your browser
2. You should see:
   - **Top-left overlay**: Connection status, FPS, debug info
   - **Bottom-left controls**: Toggle cage, wireframe, reset camera
   - **Top-right help**: Keyboard controls
   - **3D scene**: Grid, axes, mesh (blue), cage (magenta wireframe)

**Expected Web Console Output**:

```
✓ Three.js initialized
Connecting to WebSocket server...
✓ WebSocket connected
✓ Viewer initialized and running
```

---

### Step 4: Validate Warping

**Test X-axis warping** (left/right):

1. Stand in front of camera
2. Move your body to the left
3. **Expected**: Mesh in web viewer warps/moves left
4. Move your body to the right
5. **Expected**: Mesh warps/moves right

**Test Y-axis warping** (up/down):

1. Crouch down
2. **Expected**: Mesh warps/moves down
3. Stand on tiptoes (or raise arms)
4. **Expected**: Mesh warps/moves up

**Z-axis** (forward/backward):

- Should NOT change in 2D mode
- Mesh stays at constant depth

---

### Step 5: Check Debug Logs

**In Terminal** (every 2 seconds):

```
======================================================================
DEBUG INFO - Frame 120 (29.8 FPS)
======================================================================

Mesh:
  Position: (0.012, -0.034, 0.000)
  Size: (0.450, 0.620, 0.180)
  X range: [-0.213, 0.237]
  Y range: [-0.344, 0.276]
  Z range: [-0.090, 0.090]

Cage Deformation:
  Mean delta: (0.023, -0.012, 0.000)
  Max delta: 0.145

Keypoints:
  Detected: 13/13
  Delta magnitude: 0.026
  Max delta: 0.145
  Top movers:
    left_wrist: 0.145
    right_wrist: 0.132
    left_elbow: 0.089

Mesh likely VISIBLE
======================================================================
```

**Key things to check**:

- **Keypoints detected**: Should be 13/13 when you're in frame
- **Delta magnitude**: Should change when you move
- **Mesh likely VISIBLE**: Should say VISIBLE (if OFFSCREEN, adjust camera)
- **Mean delta**: Should have non-zero X, Y when you move

**In Web Viewer** (real-time):

- **FPS**: Should be 20-30 FPS
- **Mesh Position**: Should update as you move
- **Visibility Status**: Should say "✓ VISIBLE"

---

### Step 6: Test 3D Warping (Z-Axis)

**Run with Z-axis enabled**:

```bash
python tests/test_integration_v2.py \
    --mesh generated_meshes/0/mesh.obj \
    --reference generated_images/0_reference.pkl \
    --enable-z-warp
```

**Test forward/backward movement**:

1. Move forward (toward camera)
2. **Expected**: Mesh changes (may be incorrect due to uncalibrated Z)
3. Move backward (away from camera)
4. **Expected**: Mesh changes

**Note**: Z-axis warping will likely be incorrect because MediaPipe's Z is uncalibrated. This is expected in V2 prototype.

---

### Step 7: Test Headless Mode

**Run without Python window**:

```bash
python tests/test_integration_v2.py \
    --mesh generated_meshes/0/mesh.obj \
    --reference generated_images/0_reference.pkl \
    --headless
```

- No camera window should appear
- Only terminal output
- Web viewer still works

---

## Troubleshooting

### Problem: Mesh Not Visible in Web Viewer

**Check debug logs**:

- If "Mesh likely OFFSCREEN": Mesh is rendered but outside view
- Check "Mesh Position" values - should be near (0, 0, 0)

**Solutions**:

1. Press **R** to reset camera
2. Use **WASD** + **Q/E** to navigate
3. Scroll mouse wheel to zoom

---

### Problem: Mesh Doesn't Warp When Moving

**Check terminal debug logs**:

- "Keypoints detected" should be > 0
- "Delta magnitude" should change when you move

**If keypoints = 0**:

- Ensure you're in frame
- Check lighting (MediaPipe needs good lighting)
- Try different pose/distance from camera

**If keypoints detected but mesh doesn't warp**:

- Check "Mean delta" in debug logs
- Should have non-zero values when you move
- If zero, there may be a coordinate system issue

---

### Problem: Cage Not Visible

**Check**:

- "Show Cage" button should be green/active
- Try toggling it off and on

**If still not visible**:

- Cage may be offscreen
- Use camera controls to look around
- Check if mesh is visible (cage should be near mesh)

---

### Problem: WebSocket Connection Fails

**Check**:

- Port 8765 not in use by another app
- WebSocket server started (terminal should say "✓ WebSocket server started")

**If connection fails**:

- Close other WebSocket connections
- Restart the Python script
- Check browser console for errors

---

### Problem: "ValueError: too many values to unpack"

**This means**:

- Reference data format is incorrect
- Regenerate reference data with `create_mock_reference_data.py`

---

## Success Criteria

### ✅ V2 Prototype is Working If:

1. **Mesh is visible** in web viewer
2. **Moving left/right → mesh warps left/right** (X-axis)
3. **Moving up/down → mesh warps up/down** (Y-axis)
4. **Cage is visible** and moves with mesh
5. **Debug logs show**:
   - Keypoints detected
   - Delta magnitude changes with movement
   - Mesh stays VISIBLE

---

## Known Limitations (Expected in V2)

1. **Z-axis is incorrect**: Expected without depth calibration
2. **Mesh may be wrong size**: No automatic scaling yet
3. **Rotation not handled**: Only translation works
4. **Jittering**: No temporal smoothing yet
5. **Mesh doesn't match user size**: Need scaling factor

**These are acceptable for V2** - focus is on validating that warping logic works.

---

## What to Report

When testing, please report:

### 1. Setup

- [ ] Reference data created successfully
- [ ] Mesh and cage loaded
- [ ] WebSocket connected

### 2. Warping Behavior

- [ ] Mesh visible in web viewer
- [ ] Mesh warps left/right when you move (X)
- [ ] Mesh warps up/down when you move (Y)
- [ ] Cage follows mesh

### 3. Debug Info

- Keypoints detected: \_\_/13
- FPS: \_\_ FPS
- Mesh position: (**, **, \_\_)
- Delta magnitude when moving: \_\_

### 4. Issues Encountered

- Describe any problems
- Include terminal/console errors
- Screenshots if helpful

---

## Next Steps After Validation

If V2 warping works correctly:

1. ✅ Core logic is sound
2. Implement proper camera calibration (depth estimation)
3. Add mesh scaling to match user size
4. Add rotation handling (not just translation)
5. Integrate with actual generation pipeline

---

## Quick Reference

**Camera Controls (Python window)**:

- `Q` - Quit
- `D` - Toggle debug logging
- `C` - Toggle cage visualization

**Web Viewer Controls**:

- Mouse drag - Orbit camera
- Scroll - Zoom
- `W`/`A`/`S`/`D` - Move camera
- `Q`/`E` - Move up/down
- `R` - Reset camera

**Command-line Arguments**:

```bash
--mesh PATH          # Path to .obj mesh
--reference PATH     # Path to .pkl reference data
--enable-z-warp      # Enable Z-axis warping
--headless           # No Python window
```

---

**End of Testing Guide**
