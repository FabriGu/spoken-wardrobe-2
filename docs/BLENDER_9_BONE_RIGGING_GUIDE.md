# Blender 9-Bone Rigging Guide for MediaPipe Integration

This guide will walk you through creating a 9-bone armature that matches MediaPipe's bone structure, allowing for clean 1:1 bone mapping without the warping issues.

## Prerequisites

- Blender 3.0+ (free download: https://www.blender.org/download/)
- Your mesh file (GLB or OBJ format)
- 1-3 hours for first-time rigging
- 30-60 minutes for subsequent meshes

## Overview

We'll create exactly 9 bones matching MediaPipe's structure:
1. Spine (mid_hip → mid_shoulder)
2. Left Upper Arm (left_shoulder → left_elbow)
3. Left Lower Arm (left_elbow → left_wrist)
4. Right Upper Arm (right_shoulder → right_elbow)
5. Right Lower Arm (right_elbow → right_wrist)
6. Left Upper Leg (left_hip → left_knee)
7. Left Lower Leg (left_knee → left_ankle)
8. Right Upper Leg (right_hip → right_knee)
9. Right Lower Leg (right_knee → right_ankle)

---

## Part 1: Setup (10 minutes)

### Step 1: Install Blender
1. Download from https://www.blender.org/download/
2. Install (no special options needed)
3. Open Blender - you'll see the default scene with a cube

### Step 2: Import Your Mesh
1. **Delete the default cube**:
   - Click the cube
   - Press `X` → Delete

2. **Import your mesh**:
   - File → Import → glTF 2.0 (.glb/.gltf)
   - Navigate to your mesh file (e.g., `pregenerated_mesh_clothing/base.glb`)
   - Click "Import glTF 2.0"

3. **Center the view**:
   - Press `Numpad .` (period) to frame the object
   - Or: View → Frame Selected

### Step 3: Put Mesh in T-Pose
**CRITICAL**: The mesh MUST be in T-pose (arms straight out to sides) for this to work!

If your mesh is NOT in T-pose:
1. Tab into Edit Mode (`Tab` key)
2. Select arm vertices
3. Rotate them to be horizontal (arms straight out)
4. Tab back to Object Mode (`Tab` key)

**Tip**: If you're not comfortable with this, find a T-pose mesh online or ask me to help you convert it.

---

## Part 2: Create the Armature (20 minutes)

### Step 4: Add Armature
1. **Add armature**:
   - `Shift + A` → Armature → Single Bone
   - You'll see one bone appear at the origin

2. **Switch to Edit Mode for the armature**:
   - Select the armature (click it)
   - Press `Tab` to enter Edit Mode
   - You'll see the bone with a "head" (bottom) and "tail" (top)

3. **Delete the default bone**:
   - Select all (`A`)
   - Press `X` → Delete Bones

### Step 5: Create Spine Bone

**Naming Convention**: Use exact names for MediaPipe mapping!

1. **Add spine bone**:
   - `Shift + A` → Single Bone

2. **Position the spine**:
   - **Head (bottom)**: Mid-hip position (center between left/right hip)
   - **Tail (top)**: Mid-shoulder position (center between left/right shoulder)

3. **How to position**:
   - Select the bone
   - Press `G` to grab/move
   - Press `X`, `Y`, or `Z` to constrain to axis
   - Type numbers for precision (e.g., `G Z 0.9` moves to Z=0.9)

4. **Rename**:
   - With bone selected, press `F2` (or in Properties panel)
   - Name it: `spine`

**Approximate spine positions for standard human**:
- Head: (0, 0, 0.9) - at hip height
- Tail: (0, 0, 1.4) - at shoulder height

### Step 6: Create Arm Bones

**Left Upper Arm**:
1. Add bone (`Shift + A`)
2. Position:
   - Head: Left shoulder (approx: -0.3, 0, 1.4)
   - Tail: Left elbow (approx: -0.6, 0, 1.4)
3. Rename to: `left_upper_arm`

**Left Lower Arm**:
1. Add bone (`Shift + A`)
2. Position:
   - Head: Left elbow (same as left_upper_arm tail: -0.6, 0, 1.4)
   - Tail: Left wrist (approx: -0.9, 0, 1.4)
3. Rename to: `left_lower_arm`

**Right Upper Arm**:
1. Add bone (`Shift + A`)
2. Position:
   - Head: Right shoulder (approx: 0.3, 0, 1.4)
   - Tail: Right elbow (approx: 0.6, 0, 1.4)
3. Rename to: `right_upper_arm`

**Right Lower Arm**:
1. Add bone (`Shift + A`)
2. Position:
   - Head: Right elbow (same as right_upper_arm tail: 0.6, 0, 1.4)
   - Tail: Right wrist (approx: 0.9, 0, 1.4)
3. Rename to: `right_lower_arm`

### Step 7: Create Leg Bones

**Left Upper Leg**:
1. Add bone (`Shift + A`)
2. Position:
   - Head: Left hip (approx: -0.15, 0, 0.9)
   - Tail: Left knee (approx: -0.15, 0, 0.5)
3. Rename to: `left_upper_leg`

**Left Lower Leg**:
1. Add bone (`Shift + A`)
2. Position:
   - Head: Left knee (same as left_upper_leg tail: -0.15, 0, 0.5)
   - Tail: Left ankle (approx: -0.15, 0, 0.0)
3. Rename to: `left_lower_leg`

**Right Upper Leg**:
1. Add bone (`Shift + A`)
2. Position:
   - Head: Right hip (approx: 0.15, 0, 0.9)
   - Tail: Right knee (approx: 0.15, 0, 0.5)
3. Rename to: `right_upper_leg`

**Right Lower Leg**:
1. Add bone (`Shift + A`)
2. Position:
   - Head: Right knee (same as right_upper_leg tail: 0.15, 0, 0.5)
   - Tail: Right ankle (approx: 0.15, 0, 0.0)
3. Rename to: `right_lower_leg`

### Step 8: Set Up Bone Hierarchy (Parent-Child)

**CRITICAL for proper deformation!**

1. Stay in Edit Mode
2. **Parent arm bones**:
   - Select `left_lower_arm` → Shift-select `left_upper_arm` → `Ctrl + P` → Keep Offset
   - Select `left_upper_arm` → Shift-select `spine` → `Ctrl + P` → Keep Offset
   - Select `right_lower_arm` → Shift-select `right_upper_arm` → `Ctrl + P` → Keep Offset
   - Select `right_upper_arm` → Shift-select `spine` → `Ctrl + P` → Keep Offset

3. **Parent leg bones**:
   - Select `left_lower_leg` → Shift-select `left_upper_leg` → `Ctrl + P` → Keep Offset
   - Select `left_upper_leg` → Shift-select `spine` → `Ctrl + P` → Keep Offset
   - Select `right_lower_leg` → Shift-select `right_upper_leg` → `Ctrl + P` → Keep Offset
   - Select `right_upper_leg` → Shift-select `spine` → `Ctrl + P` → Keep Offset

4. Press `Tab` to exit Edit Mode

---

## Part 3: Skin the Mesh (30-60 minutes)

### Step 9: Automatic Weights

1. **Select mesh, then armature**:
   - Click the mesh
   - Shift-click the armature
   - (Order matters! Mesh first, then armature)

2. **Parent with automatic weights**:
   - `Ctrl + P` → With Automatic Weights
   - Blender will calculate initial skin weights
   - Wait a few seconds

3. **Test the rigging**:
   - Select the armature
   - Press `Tab` to enter Pose Mode (NOT Edit Mode!)
   - Select a bone (click on it)
   - Press `R` to rotate
   - The mesh should deform!

### Step 10: Weight Painting (Optional but Recommended)

Automatic weights are often good enough, but you can refine them:

1. **Enter Weight Paint Mode**:
   - Select the mesh
   - Mode dropdown (top left) → Weight Paint

2. **Select bone**:
   - In the right panel → Object Data Properties → Vertex Groups
   - Click a bone name (e.g., `left_upper_arm`)

3. **Paint weights**:
   - Blue = 0 influence
   - Red = 1.0 full influence
   - Green = 0.5 medium influence

4. **Tools**:
   - Draw: Add weight
   - Subtract: Remove weight
   - Smooth: Blend weights
   - Weight: 1.0 (adjustable, top left)

**Tips**:
- Upper arm vertices should be red (1.0) for `left_upper_arm`
- Elbow region should blend (yellow/green) between upper and lower arm
- Torso should be fully red for `spine`

---

## Part 4: Export for Python (5 minutes)

### Step 11: Export as GLB

1. **Select mesh and armature**:
   - Click mesh
   - Shift-click armature
   - Both should be highlighted

2. **Export**:
   - File → Export → glTF 2.0 (.glb/.gltf)
   - **Format**: glTF Binary (.glb)
   - **Include**: Selected Objects (check this!)
   - **Transform**: +Y Up (important!)
   - Navigate to save location: `pregenerated_mesh_clothing/my_rigged_mesh.glb`
   - Click "Export glTF 2.0"

---

## Part 5: Test in Python (5 minutes)

### Step 12: Update Your Code

Edit `tests/test_a_rigged_clothing.py`:

```python
# Change line 116-119:
clothing_path = "pregenerated_mesh_clothing/my_rigged_mesh.glb"  # Your new mesh!
print(f"Loading clothing mesh: {clothing_path}")
self.clothing_mesh = RiggedMeshLoader.load(clothing_path)
print(f"✓ Clothing: {len(self.clothing_mesh.vertices)} verts")
```

### Step 13: Run the Test

```bash
open tests/clothing_viewer.html
python tests/test_a_rigged_clothing.py

# Press SPACE for T-pose calibration
# Move around - it should work!
```

---

## Bone Placement Reference Diagram

```
                    [mid_shoulder] (0, 0, 1.4)
                          |
      left_shoulder ------+------ right_shoulder
            |                           |
     (-0.3, 0, 1.4)               (0.3, 0, 1.4)
            |                           |
      left_elbow                  right_elbow
            |                           |
     (-0.6, 0, 1.4)               (0.6, 0, 1.4)
            |                           |
      left_wrist                  right_wrist
     (-0.9, 0, 1.4)               (0.9, 0, 1.4)


                      [mid_hip] (0, 0, 0.9)
                          |
        left_hip ---------+--------- right_hip
            |                           |
     (-0.15, 0, 0.9)              (0.15, 0, 0.9)
            |                           |
       left_knee                   right_knee
            |                           |
     (-0.15, 0, 0.5)              (0.15, 0, 0.5)
            |                           |
      left_ankle                  right_ankle
     (-0.15, 0, 0.0)              (0.15, 0, 0.0)
```

### Coordinate System
- **X-axis**: Left (-) to Right (+)
- **Y-axis**: Back (-) to Front (+)
- **Z-axis**: Bottom (0) to Top (+)

### Measurements for Standard Human (~1.7m tall)
- **Ankle to Hip**: 0.9m
- **Hip to Shoulder**: 0.5m (1.4 - 0.9)
- **Upper Leg**: 0.4m (0.9 - 0.5)
- **Lower Leg**: 0.5m (0.5 - 0.0)
- **Shoulder Width**: 0.6m (±0.3)
- **Hip Width**: 0.3m (±0.15)
- **Arm Length**: 0.6m per segment

---

## Troubleshooting

### "Mesh doesn't deform when I rotate bones"
- Did you parent mesh to armature with Automatic Weights?
- Are you in Pose Mode (not Edit Mode)?
- Check vertex groups exist (Properties → Object Data → Vertex Groups)

### "Mesh explodes when animated"
- Check bone names match EXACTLY: `spine`, `left_upper_arm`, etc.
- Make sure mesh is in T-pose before rigging
- Check bone hierarchy (parent-child relationships)

### "Automatic Weights failed"
- Mesh might have issues (non-manifold geometry, loose vertices)
- Try: Edit Mode → Select All → Mesh → Clean Up → Merge By Distance
- Try manual weight painting instead

### "Bones are too small/big"
- In armature properties: Display → Viewport Display → Size
- Adjust until comfortable (doesn't affect functionality)

### "I can't see bones in solid view"
- Armature → Viewport Display → In Front (check this)

---

## Quick Reference: Blender Shortcuts

| Shortcut | Action |
|----------|--------|
| `Tab` | Toggle Edit/Object Mode |
| `Shift + A` | Add (bone, object, etc.) |
| `X` | Delete |
| `G` | Grab/Move |
| `R` | Rotate |
| `S` | Scale |
| `Ctrl + P` | Parent |
| `F2` | Rename |
| `Numpad .` | Frame selected |
| `Ctrl + Z` | Undo |

---

## Expected Results

After completing this guide:

✅ You'll have a mesh with exactly 9 bones matching MediaPipe
✅ 1:1 bone mapping (no 163→9 remapping needed!)
✅ Proper inverse bind matrices from Blender
✅ Clean weight painting for smooth deformation
✅ No warping or coordinate system mismatches

The mesh should:
- Maintain its original shape ✓
- Deform smoothly with body movement ✓
- Have arms/legs rotate naturally ✓
- Look recognizable at all times ✓

---

## Next Steps After Rigging

1. **Test the rigged mesh**: Run your Python script
2. **Refine weights**: If any part looks weird, go back to weight painting
3. **Apply to more meshes**: Repeat process for different clothing items
4. **Consider automation**: Once you understand the process, you could script parts of it

---

## Resources

- **Blender Manual**: https://docs.blender.org/manual/en/latest/
- **Rigging Tutorial**: https://www.youtube.com/results?search_query=blender+rigging+tutorial
- **Weight Painting Guide**: https://docs.blender.org/manual/en/latest/sculpt_paint/weight_paint/
- **GLB Export**: https://docs.blender.org/manual/en/latest/addons/import_export/scene_gltf2.html

---

**Good luck! This should solve your warping issues permanently. If you get stuck at any step, let me know and I can provide more detailed guidance or screenshots.**
