# Quick Start - Prototype V2

**3-Step Testing Guide**

---

## Step 1: Create Reference Data

```bash
python tests/create_mock_reference_data.py \
    --mesh generated_meshes/0/mesh.obj \
    --output generated_images/0_reference.pkl
```

**What happens**: Camera opens, stand in T-pose, press SPACE to capture.

---

## Step 2: Run V2 System

```bash
python tests/test_integration_v2.py \
    --mesh generated_meshes/0/mesh.obj \
    --reference generated_images/0_reference.pkl
```

**What happens**: System starts, camera window shows your video + skeleton.

---

## Step 3: Open Web Viewer

Open `tests/enhanced_mesh_viewer_v2.html` in your browser.

**What you should see**:

- Blue mesh in center
- Magenta cage around mesh
- Debug info in top-left

---

## Test Movement

- **Move left** → mesh warps left
- **Move right** → mesh warps right
- **Move up** → mesh warps up
- **Move down** → mesh warps down

---

## Controls

**Python Window**:

- `Q` - Quit
- `D` - Toggle debug
- `C` - Toggle cage

**Web Viewer**:

- Mouse drag - Orbit
- Scroll - Zoom
- `W/A/S/D` - Move camera
- `R` - Reset camera

---

## Troubleshooting

**Mesh not visible?**

- Press `R` to reset camera
- Use WASD to navigate
- Check debug logs for position

**Mesh doesn't warp?**

- Check terminal for "Keypoints detected"
- Try better lighting
- Move closer to camera

---

## Documentation

- **Full Testing Guide**: `docs/251026_v2_testing_guide.md`
- **Implementation Details**: `docs/251026_V2_IMPLEMENTATION_SUMMARY.md`
- **Pipeline Design**: `docs/251026_prototype_v2_plan.md`

---

**Need Help?** Check the full testing guide above.
