# Quick Test Commands

**Run these in order to test the fixes**

## Setup

```bash
cd /Users/fabrizioguccione/Projects/spoken_wardrobe_2
source venv/bin/activate
```

## Test 1: Mesh Orientation

```bash
python tests/clothing_to_3d_triposr_2.py
# Mesh should now face forward (not rotated left)
```

## Test 2: Cage Structure

```bash
python 251025_data_verification/verify_cage_structure.py
# Should show: "Cage structure: 3-6 body parts"
```

## Test 3: Real-Time Deformation

```bash
# Terminal:
python 251025_data_verification/verify_deformation.py --mesh generated_meshes/0/mesh.obj

# Browser:
open 251025_data_verification/verification_viewer.html

# Expected: 30-60% vertices moving (NOT 100%!)
```

## Test 4: Full System

```bash
python tests/test_integration.py --headless --mesh generated_meshes/0/mesh.obj
open tests/enhanced_mesh_viewer.html

# Move arms → should see independent articulation
```

## Success Criteria

- [ ] Mesh faces forward
- [ ] Cage has 3-6 body part sections
- [ ] Only 30-60% of vertices move per frame
- [ ] Mesh follows body motion smoothly
- [ ] No smearing or collapse

## Docs

- Implementation: `docs/251026_IMPLEMENTATION_COMPLETE.md`
- Testing Guide: `docs/251026_testing_guide.md`
- Verification: `251025_data_verification/FINDINGS.md`
