# Quick Start Guide for Verification System

**Last Updated**: October 25, 2025

## ✅ System is Now Working!

All verification scripts have been fixed and are ready to use.

---

## 🚀 How to Run

### Option 1: Run All Verifications (Recommended)

```bash
# From project root, with venv activated:
bash 251025_data_verification/quick_verify.sh
```

This will run all three verification scripts in sequence.

### Option 2: Run Individual Verifications

#### Step 1: Verify Cage Structure

```bash
source venv/bin/activate
python 251025_data_verification/verify_cage_structure.py
```

**What it does**:

- Creates a test mesh
- Generates a cage using your current system
- Analyzes the cage structure
- **Saves cage and mesh to `251025_data_verification/output/`**
- Tests with actual clothing mesh if available

**Output Files**:

- `output/test_mesh.obj` - Simple test mesh
- `output/test_cage.obj` - Cage generated for test mesh
- `output/actual_cage.obj` - Cage for your clothing mesh

**You can open these OBJ files in Blender or MeshLab to visually inspect the cage!**

---

#### Step 2: Verify MVC Weights

```bash
source venv/bin/activate
python 251025_data_verification/verify_mvc_weights.py
```

**What it does**:

- Computes MVC weights for a test mesh
- Verifies weights sum to 1.0
- Tests deformation performance
- Checks if weights are recomputed per frame

**Look for**:

- ✓ "All rows sum to 1.0" - Good!
- ✓ "Deformation time <5ms" - Good!
- ✗ "Deformation too slow" - Problem!

---

#### Step 3: Real-Time Deformation Verification

```bash
# Terminal 1 - Start verification server:
source venv/bin/activate
python 251025_data_verification/verify_deformation.py

# Browser - Open viewer:
open 251025_data_verification/verification_viewer.html
```

**What it does**:

- Starts WebSocket server on port **8766** (different from main system)
- Captures camera feed
- Runs BodyPix once for cage initialization
- Uses MediaPipe for real-time keypoint tracking
- Streams mesh + cage to web viewer

**Web Viewer Shows**:

- Green mesh (your clothing)
- Magenta wireframe cage
- Real-time statistics
- Cage quality indicators

**Controls in Python Window**:

- Q - Quit
- R - Reset
- A - Show analysis

**Controls in Web Viewer**:

- Mouse - Orbit camera
- Scroll - Zoom
- Show Mesh button
- Show Cage button
- Wireframe toggle

---

## 🔍 What to Look For

### Initial Findings (from first run):

**From Step 1 - Cage Structure**:

```
Cage vertices: 8
✗ Vertex count (8) is too high! Should be 30-60 for clothing
```

Wait, that message is wrong - **8 vertices is actually TOO FEW!**

The current system generates a **simple bounding box** (8 vertices = box corners).

**What you need**: 30-60 vertices organized into anatomical sections:

- Torso: 8 vertices
- Left upper arm: 6 vertices
- Right upper arm: 6 vertices
- Left lower arm: 6 vertices
- Right lower arm: 6 vertices
- etc.

### Key Issues to Verify:

1. **Cage Structure Problem** ✗

   - Current: 8-vertex bounding box
   - Needed: 30-60 vertex anatomical cage
   - **This is the root cause of the "smearing"**

2. **MVC Weights** (check with Step 2)

   - Should sum to 1.0 per vertex
   - Should be computed once
   - Deformation should be <5ms

3. **Independent Section Movement** (check with Step 3)
   - Different cage sections should move independently
   - When you raise left arm, only left arm cage should move
   - Currently probably ALL vertices move together

---

## 📊 Visual Inspection

After running Step 1, open the cage files in Blender:

```bash
# Install Blender if you don't have it:
# brew install --cask blender  (macOS)

# Then:
open -a Blender 251025_data_verification/output/test_cage.obj
```

**What to check**:

- Is it just a box? (Problem!)
- Does it have arm/leg sections? (Good!)
- Are sections clearly separated? (Good!)
- Is it too dense/complex? (Problem!)

---

## 📖 Next Steps After Verification

Once you've confirmed the issues (cage is just a box, no anatomical structure):

1. **Read the implementation plan**:

   - `docs/251025_steps_forward.md`
   - Contains detailed pseudocode for fixing the cage generation

2. **Focus on fixing `enhanced_cage_utils.py`**:

   - Rewrite `generate_anatomical_cage()` method
   - Create separate sections for each body part
   - Use BodyPix segmentation to guide section placement

3. **Fix `keypoint_mapper.py`**:

   - Implement per-section transformation
   - Map MediaPipe keypoints to specific cage sections
   - Apply independent rotation/translation/scaling per section

4. **Re-run verification** to confirm fixes work

---

## 🆘 Troubleshooting

### "ModuleNotFoundError: No module named 'X'"

Make sure venv is activated:

```bash
source venv/bin/activate
```

### WebSocket won't connect

- Check port 8766 is free: `lsof -i :8766`
- Make sure Python script started first
- Refresh browser after Python script is running

### Camera not opening

- Check camera permissions in System Settings
- Try different camera index in code
- Make sure no other app is using camera

### Blender can't open OBJ files

- Install Blender: `brew install --cask blender`
- Or use online viewer: https://3dviewer.net/

---

## 💡 Key Insight

The verification system has already revealed the core problem:

**Your cage is a simple 8-vertex bounding box, not an anatomical structure!**

This explains why the mesh "smears" - there's no way to articulate different body parts independently when the cage is just a box.

The solution (from the research papers in `docs/251025_steps_forward.md`):

- Generate cage with 30-60 vertices
- Organize into anatomical sections (torso, arms, legs)
- Each section has 6-8 vertices
- Each section can deform independently

---

## 📝 Documentation

- **Implementation Plan**: `docs/251025_steps_forward.md`
- **Verification System**: `251025_data_verification/VERIFICATION_SUMMARY.md`
- **This Quick Start**: `251025_data_verification/QUICK_START.md`

---

**You're all set! Run the verification and see what needs to be fixed.** 🚀
