# Complete Testing Guide: Articulated Cage System

## October 28, 2025

## Overview

This guide will walk you through testing the new **Articulated Cage System** (Option A V2), which implements proper cage-based deformation using Oriented Bounding Boxes (OBBs) and regional Mean Value Coordinates (MVC).

## What's Different from Previous Version?

| Aspect             | V1 (ConvexHull)        | V2 (OBBs)                   |
| ------------------ | ---------------------- | --------------------------- |
| **Cage Structure** | Single box (collapsed) | Multiple connected segments |
| **MVC Strategy**   | Global (pinching)      | Regional (no pinching)      |
| **Joint Handling** | Independent sections   | Hierarchical parent-child   |
| **Deformation**    | Uniform translation    | Articulated rigid bodies    |

## Prerequisites

### 1. Ensure Dependencies are Installed

```bash
# Activate your virtual environment
source venv/bin/activate

# Verify key packages
python -c "import trimesh, numpy, cv2, mediapipe; print('✓ All dependencies OK')"
```

### 2. Have a Mesh Ready

You need a 3D mesh file (`.obj` format). Options:

- Use existing: `generated_meshes/0/mesh.obj`
- Generate new: Run `tests/create_consistent_pipeline_v2.py`

### 3. Open Web Viewer

Open `tests/enhanced_mesh_viewer_v2.html` in your browser before starting the Python script.

---

## Testing Stages

### Stage 1: Test Individual Components

#### Test 1.1: Cage Generator (OBB Generation)

**Purpose**: Verify that OBBs are created correctly from masks

```bash
cd /Users/fabrizioguccione/Projects/spoken_wardrobe_2
python tests/articulated_cage_generator.py
```

**Expected Output**:

```
Testing ArticulatedCageGenerator...
======================================================================
GENERATING ARTICULATED CAGE WITH OBBs
======================================================================
✓ Projected 4 keypoints to 3D
  ✓ Generated OBB for 'torso'

✓ Generated 1 OBBs

✓ Unified cage created:
   Vertices: 8
   Faces: 12
   Sections: ['torso']
======================================================================

✓ Test passed!
   Cage: 8 vertices, 12 faces
   Sections: ['torso']
   Joints: ['center']
```

**What to Check**:

- ✓ No errors or crashes
- ✓ Cage vertices > 0
- ✓ Sections list is populated
- ✓ Number of vertices = 8 × number of sections (each OBB has 8 corners)

#### Test 1.2: Articulated Deformer (Regional MVC)

**Purpose**: Verify regional MVC weights are computed correctly

```bash
python tests/articulated_deformer.py
```

**Expected Output**:

```
Testing ArticulatedDeformer...
======================================================================
ARTICULATED DEFORMER INITIALIZED
======================================================================
  Mesh vertices: 100
  Cage vertices: 20
  Sections: 2
  Joints: 2
  Regional MVC computed for 2 sections
======================================================================

Computing regional MVC weights...
  ✓ Section 'torso': 60 mesh vertices, 10 cage vertices
  ✓ Section 'left_arm': 40 mesh vertices, 10 cage vertices

✓ Test passed!
   Original mesh: (100, 3)
   Deformed mesh: (100, 3)
   Movement: 0.0150 avg distance
```

**What to Check**:

- ✓ Regional MVC computed successfully
- ✓ Each section has mesh vertices assigned
- ✓ Deformed mesh shows non-zero movement
- ✓ No "pinching" warnings

---

### Stage 2: Test Integrated System

#### Test 2.1: Run with Real Mesh

```bash
# Make sure your mesh exists
ls -lh generated_meshes/0/mesh.obj

# Run the integrated system
python tests/test_integration_cage_v2.py --mesh generated_meshes/0/mesh.obj
```

**Expected Startup Output**:

```
======================================================================
LOADING 3D MESH
======================================================================
✓ Loaded mesh: 5247 vertices, 10490 faces
  Bounds: [-0.23 -0.49 -0.04] to [0.23 0.49 1.04]
======================================================================

✓ WebSocket server started on ws://localhost:8765
======================================================================
STARTING ARTICULATED CAGE DEMO
======================================================================

Controls:
  Q - Quit
  SPACE - Capture T-pose and initialize cage
  R - Reset cage
  C - Toggle cage visualization
======================================================================
```

**What to Check**:

- ✓ Mesh loads without errors
- ✓ WebSocket server starts successfully
- ✓ Camera feed shows in OpenCV window (if not headless)
- ✓ MediaPipe skeleton is visible on your body

#### Test 2.2: Capture T-Pose and Initialize Cage

**Steps**:

1. Stand in front of camera in **T-pose** (arms extended horizontally)
2. Press **SPACE** key
3. Wait for cage initialization

**Expected Output**:

```
======================================================================
INITIALIZING ARTICULATED CAGE SYSTEM
======================================================================
✓ Extracted 13 keypoints (2D)
✓ Extracted 3 body part masks
======================================================================
GENERATING ARTICULATED CAGE WITH OBBs
======================================================================
✓ Projected 13 keypoints to 3D
  ✓ Generated OBB for 'torso'
  ✓ Generated OBB for 'left_upper_arm'
  ✓ Generated OBB for 'right_upper_arm'

✓ Generated 3 OBBs

✓ Unified cage created:
   Vertices: 24
   Faces: 36
   Sections: ['torso', 'left_upper_arm', 'right_upper_arm']
======================================================================

======================================================================
ARTICULATED DEFORMER INITIALIZED
======================================================================
  Mesh vertices: 5247
  Cage vertices: 24
  Sections: 3
  Joints: 3
  Regional MVC computed for 3 sections
======================================================================

✓ Reference T-pose captured

======================================================================
✓ ARTICULATED CAGE SYSTEM READY
======================================================================
```

**What to Check**:

- ✓ At least 3 OBBs generated (torso + 2 arms minimum)
- ✓ Cage vertices = 8 × number of sections
- ✓ Regional MVC computed for all sections
- ✓ No errors about missing keypoints

#### Test 2.3: Real-Time Deformation

**Steps**:

1. After cage initialization, **move your arms around**
2. Observe the mesh in the web viewer
3. Watch for FPS in terminal

**Expected Behavior**:

**In Web Viewer (`enhanced_mesh_viewer_v2.html`)**:

- ✓ Mesh appears centered and visible
- ✓ Mesh deforms when you move
- ✓ Cage (pink wireframe) visible around mesh
- ✓ Cage sections stay connected (no gaps between torso and arms)
- ✓ Mesh interior doesn't pinch or distort
- ✓ Smooth, natural movement

**In Terminal**:

```
FPS: 45.2
FPS: 47.8
FPS: 46.3
```

**What to Check**:

- ✓ FPS > 30 (real-time performance)
- ✓ No console errors in browser or terminal
- ✓ Mesh follows your body movement
- ✓ No extreme stretching or mesh collapse

---

### Stage 3: Visual Quality Checks

#### Check 3.1: Cage Structure (Press C to toggle)

**Enable cage visualization** and verify:

| Check                | What to Look For               | Good ✓                 | Bad ✗           |
| -------------------- | ------------------------------ | ---------------------- | --------------- |
| **Section Count**    | Multiple distinct boxes        | 3-5 OBBs               | Single box      |
| **Joint Connection** | Sections touch at joints       | Connected              | Floating apart  |
| **Size**             | Cage slightly larger than mesh | 10-20% padding         | Too tight/loose |
| **Orientation**      | Aligned with body parts        | Follows limb direction | Twisted/rotated |

#### Check 3.2: Mesh Deformation Quality

**Move your arms** and observe:

| Movement        | Expected Result        | Problem Signs             |
| --------------- | ---------------------- | ------------------------- |
| **Raise arms**  | Mesh follows smoothly  | Pinching at armpits       |
| **Lower arms**  | Returns to T-pose      | Mesh stays stretched      |
| **Move torso**  | Entire mesh translates | Arms detach from torso    |
| **Rotate body** | Mesh rotates as unit   | Individual sections twist |

#### Check 3.3: Performance Under Load

**Rapid movements** to stress-test:

1. **Wave arms quickly** (left-right)
2. **Jump** (vertical motion)
3. **Rotate body** (360°)

**Expected**:

- FPS stays > 30
- No lag or stuttering
- Mesh recovers quickly
- No crashes or freezes

---

### Stage 4: Compare to Option B (Skeletal Skinning)

Run both systems side-by-side to compare:

#### Run Option B:

```bash
python tests/test_integration_skinning.py --mesh generated_meshes/0/mesh.obj
```

#### Run Option A V2:

```bash
python tests/test_integration_cage_v2.py --mesh generated_meshes/0/mesh.obj
```

**Comparison Checklist**:

| Aspect                  | Option A (Cage)               | Option B (Skinning)           | Winner             |
| ----------------------- | ----------------------------- | ----------------------------- | ------------------ |
| **Setup Time**          | ~2 seconds                    | ~5 seconds (T-pose countdown) | A                  |
| **Deformation Quality** | Rigid sections, smooth joints | Smooth everywhere             | B (if rigged well) |
| **Mesh Compatibility**  | Any TripoSR mesh              | Needs proper topology         | A                  |
| **Pinching**            | None (regional MVC)           | Can occur at joints           | A                  |
| **Performance**         | 45-50 FPS                     | 40-45 FPS                     | A                  |
| **Detachment**          | None (hierarchical)           | None (bone constraints)       | Tie                |

---

## Troubleshooting

### Problem 1: "No segmentation mask available"

**Cause**: MediaPipe not detecting person

**Solution**:

- Ensure good lighting
- Stand fully in frame
- Move closer to camera
- Check camera permissions

### Problem 2: Only 1 OBB generated (torso)

**Cause**: MediaPipe segmentation doesn't separate limbs well

**Solution**:

- This is expected with MediaPipe's single-mask segmentation
- In production, use actual BodyPix with 24-part segmentation
- For now, test with torso-only deformation

### Problem 3: Mesh not visible in web viewer

**Cause**: Coordinate mismatch or mesh too small/large

**Solution**:

1. Check browser console for errors
2. Verify mesh bounds in terminal output
3. Try zooming out in web viewer (scroll wheel)
4. Check mesh is not at origin (0,0,0)

### Problem 4: Cage sections floating apart

**Cause**: Joint positions not correctly computed

**Solution**:

- Verify MediaPipe keypoints are detected (check terminal)
- Ensure T-pose is held properly when pressing SPACE
- Try resetting (press R) and re-initializing

### Problem 5: Low FPS (<20)

**Cause**: Mesh too complex or MVC computation too slow

**Solution**:

- Simplify mesh (reduce vertex count)
- Check CPU usage (should be <60%)
- Ensure no other heavy processes running
- Try reducing cage subdivisions in code

### Problem 6: Mesh pinching at corners

**Cause**: Regional MVC not working correctly

**Solution**:

- Check terminal for "Regional MVC computed" message
- Verify section assignments (check `regional_mvc` dict)
- Ensure mesh vertices are inside cage sections
- May need to adjust radius threshold in `_compute_regional_mvc()`

---

## Expected Results Summary

### ✓ Success Criteria

1. **Cage Structure**

   - Multiple OBB sections (3-5)
   - Sections connected at joints
   - No ConvexHull collapse

2. **Mesh Deformation**

   - No pinching inside sections
   - Smooth motion following body
   - Interior vertices move rigidly

3. **Joint Behavior**

   - Sections stay connected
   - No detachment when moving
   - Hierarchical parent-child motion

4. **Performance**
   - FPS > 30 consistently
   - Real-time response (<50ms latency)
   - Stable under rapid movement

### ✗ Failure Signs

1. **Cage collapses to single box** → OBB generation failed
2. **Mesh pinches toward corners** → Regional MVC not working
3. **Arms detach from torso** → Hierarchical transforms broken
4. **FPS < 20** → Performance optimization needed
5. **Mesh doesn't move** → Deformer not applying transformations

---

## Next Steps After Testing

### If Tests Pass ✓

1. **Enhance Angle Extraction**
   - Implement proper bone rotation (Rodrigues' formula)
   - Add joint angle computation from keypoint deltas
2. **Improve Joint Blending**

   - Add Gaussian falloff at joint boundaries
   - Smooth transitions between sections

3. **Integrate with Full Pipeline**
   - Use real BodyPix masks (24 parts)
   - Add depth calibration (Z-axis)
   - Connect to clothing generation workflow

### If Tests Fail ✗

1. **Debug Cage Generation**

   - Print OBB parameters (center, axes, extents)
   - Visualize masks used for PCA
   - Check keypoint-to-3D projection

2. **Debug Regional MVC**

   - Print mesh-to-section assignments
   - Verify weight sum = 1 for each vertex
   - Check cage vertex indices

3. **Debug Deformation**
   - Log joint transforms
   - Check hierarchical processing order
   - Verify matrix shapes match

---

## Performance Benchmarks

### Target Metrics (MacBook M2)

| Metric               | Target     | Acceptable | Poor    |
| -------------------- | ---------- | ---------- | ------- |
| **FPS**              | > 45       | 30-45      | < 30    |
| **Latency**          | < 30ms     | 30-50ms    | > 50ms  |
| **Mesh Vertices**    | 5000-10000 | 3000-5000  | < 3000  |
| **Cage Vertices**    | 40-80      | 24-40      | < 24    |
| **MVC Compute Time** | < 100ms    | 100-200ms  | > 200ms |

---

## Debugging Commands

### View mesh info:

```bash
python -c "import trimesh; m=trimesh.load('generated_meshes/0/mesh.obj'); print(f'Vertices: {len(m.vertices)}, Bounds: {m.bounds}')"
```

### Check WebSocket:

```bash
# In browser console (F12):
ws = new WebSocket('ws://localhost:8765');
ws.onopen = () => console.log('Connected!');
ws.onmessage = (e) => console.log('Received:', JSON.parse(e.data).type);
```

### Monitor FPS:

```bash
# Add to code for detailed FPS logging
import time
fps_log = []
start = time.time()
# ... (in loop) ...
fps_log.append(1 / (time.time() - start))
start = time.time()
if frame_count % 100 == 0:
    print(f"Avg FPS: {np.mean(fps_log[-100:]):.1f}")
```

---

## Documentation References

- **Research**: `docs/251028_CAGE_ARTICULATION_RESEARCH.md`
- **Implementation**: `docs/251028_ARTICULATED_CAGE_IMPLEMENTATION.md`
- **Code**:
  - `tests/articulated_cage_generator.py`
  - `tests/articulated_deformer.py`
  - `tests/test_integration_cage_v2.py`

---

**Good luck testing! 🚀**

If you encounter issues not covered here, check the terminal output carefully and look for specific error messages. Most problems have clear indicators in the logs.
