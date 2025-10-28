# Quick Start: V3 Articulated Cage (Fixed!)

## 5-Second T-Pose Calibration Included

## What's Fixed in V3

✅ **REAL BodyPix data** (or smart keypoint partitioning)
✅ **Mesh orientation corrected** (facing forward, right-side-up)
✅ **Multiple OBB sections** (not single box)
✅ **5-second countdown** for T-pose calibration

---

## Quick Test (2 Commands)

### Step 1: Open Web Viewer

```bash
open tests/enhanced_mesh_viewer_v2.html
```

### Step 2: Run V3

```bash
python tests/test_integration_cage_v3.py --mesh generated_meshes/0/mesh.obj
```

### Step 3: Calibration

1. Press **SPACE**
2. **5-second countdown** begins (displayed in terminal)
3. Get into **T-pose** during countdown
4. System captures cage automatically after countdown

### Step 4: Move Around

- Mesh should follow your movements
- Multiple cage sections visible
- No pinching or detachment

---

## Expected Results

### ✓ Good Signs

**Terminal Output**:

```
T-POSE CALIBRATION COUNTDOWN
======================================================================
Get into T-pose position (arms extended horizontally)...
  5...
  4...
  3...
  2...
  1...
✓ Capturing T-pose!

======================================================================
INITIALIZING ARTICULATED CAGE SYSTEM
======================================================================
✓ Extracted 13 keypoints (2D)
  Using keypoint-based mask partitioning (fallback)
  ✓ Partitioned mask into 5 sections: ['head', 'torso', 'left_upper_arm', 'right_upper_arm', 'left_upper_leg']
...
✓ Generated 5 OBBs
✓ Unified cage created:
   Vertices: 40
   Faces: 60
   Sections: ['head', 'torso', 'left_upper_arm', 'right_upper_arm', 'left_upper_leg']
```

**Web Viewer**:

- Multiple cage sections (pink wireframe)
- Mesh facing forward (not sideways)
- Mesh deforms smoothly
- Sections stay connected

### ✗ Bad Signs

**Still single box**:

- Check keypoint detection: `print(keypoints_2d)` in code
- Verify MediaPipe segmentation is working

**Still sideways**:

- Mesh orientation fixes should be automatic
- Check terminal for "✓ Orientation corrected" message

**Countdown doesn't give enough time**:

- Edit `test_integration_cage_v3.py` line 381-383
- Change range(5, 0, -1) to range(10, 0, -1) for 10 seconds

---

## With Reference Data (Better Quality)

If you have reference data from pipeline:

```bash
python tests/test_integration_cage_v3.py \
    --mesh generated_meshes/TIMESTAMP/mesh.obj \
    --reference generated_meshes/TIMESTAMP/reference_data.pkl
```

**Benefits**:

- Real 24-part BodyPix segmentation
- 5-10 OBB sections (vs 5-7 from keypoint partitioning)
- Higher quality cage structure

**How to generate reference data**:

```bash
python tests/create_consistent_pipeline_v2.py
# Follow prompts, saves reference_data.pkl automatically
```

---

## Troubleshooting

### "No segmentation mask available"

**Fix**: Improve lighting, stand fully in frame

### "Only 1 section generated"

**Fix**: Use reference data OR check MediaPipe detection

### "Countdown too fast"

**Fix**: Edit line 381 in `test_integration_cage_v3.py`:

```python
for i in range(10, 0, -1):  # 10 seconds instead of 5
```

### "Mesh not visible in web viewer"

**Fix**: Zoom out with scroll wheel, check browser console (F12)

---

## Controls

- **Q**: Quit
- **SPACE**: Start T-pose calibration (5s countdown)
- **R**: Reset cage
- **C**: Toggle cage visualization

---

## Comparison to V2

| Issue            | V2 (Broken) | V3 (Fixed)             |
| ---------------- | ----------- | ---------------------- |
| Cage sections    | 1 box       | 5-10 OBBs              |
| Mesh orientation | Sideways    | Forward ✓              |
| Calibration      | Instant     | 5s countdown ✓         |
| Data source      | Fake masks  | Real/smart partition ✓ |

---

## Documentation

- **This file**: Quick start
- **`docs/251028_V3_CRITICAL_FIXES.md`**: Detailed fixes explained
- **`tests/test_integration_cage_v3.py`**: Full implementation

---

**Test it now!**

```bash
python tests/test_integration_cage_v3.py --mesh generated_meshes/0/mesh.obj
```

Press SPACE, wait 5 seconds in T-pose, then move around!
