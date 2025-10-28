# Comprehensive Fix Plan for Option B

**Date**: October 28, 2025

## Current Issues

1. **Mesh Orientation**: Still wrong after multiple rotation attempts
2. **No Deformation**: Mesh moves as rigid body, no skeletal deformation visible
3. **No Debug Visualization**: Can't see what's happening with bones/keypoints

## Root Causes Analysis

### Issue 1: Mesh Orientation

**Problem**: Random rotations without understanding the starting state

**Correct Approach** (from `clothing_to_3d_triposr_2.py`):

```python
# Step 1: Make Y tallest (if needed)
# Step 2: 180° X-flip (upside-down fix)
# Step 3: 90° Y-rotation (face forward)
```

**Current mesh state**: After TripoSR with auto-orient, mesh is likely:

- Y is already tallest (upright axis)
- But inverted (upside-down)
- And rotated 180° (facing away)

**Correct fix**:

1. Don't rotate axes (Y is already vertical)
2. Apply 180° X-flip
3. NO additional Y-rotation (or maybe 180° not 90°)

### Issue 2: No Deformation

**Suspected causes**:

1. Bone transforms might be collapsing to identity
2. Skinning weights might all be equal (no variation)
3. Hierarchical transforms not propagating
4. Deformed vertices not being sent to viewer

**Need**: Debug logging to trace through LBS

### Issue 3: No Visualization

**Need**:

- Skeleton overlay in web viewer
- Bone positions as spheres
- Current vs bind pose comparison

## Implementation Plan

### Phase 1: Fix Orientation (Take 1 more try)

Remove ALL rotations, start fresh:

```python
# Only apply what TripoSR script uses:
# 1. 180° X-flip
# 2. Test if mesh faces forward
# 3. If not, try 180° Y-rotation (not 90°!)
```

### Phase 2: Add Comprehensive Debugging

**In Python**:

- Log bone positions (bind vs current)
- Log vertex movement (before vs after LBS)
- Log skinning weight distribution

**In Web Viewer**:

- Draw MediaPipe keypoints as spheres
- Draw bone positions as spheres
- Draw skeleton connections as lines
- Color-code by bone influence

### Phase 3: Verify LBS is Working

Check each step:

1. Bind pose transforms computed ✓
2. Current pose transforms computed ?
3. Skinning matrix computed ?
4. Vertices actually transformed ?
5. Transformed vertices sent to web ?

## Files to Modify

1. `tests/test_integration_skinning.py`:

   - Simplify orientation (remove random rotations)
   - Add debug logging in `deform_mesh()`
   - Send bone/keypoint data to web

2. `tests/enhanced_websocket_server_v2.py`:

   - Add keypoint/bone data to WebSocket payload

3. `tests/enhanced_mesh_viewer_v2.html`:
   - Add skeleton visualization
   - Add bone sphere rendering
   - Add debug overlay for LBS state

## Expected Outcome

- Mesh oriented correctly (upright, facing forward)
- Mesh deforms with body movement (arms, torso)
- Visual skeleton overlay shows what's controlling mesh
- Debug info shows LBS is actually working
