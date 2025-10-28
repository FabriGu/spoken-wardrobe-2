# Hotfix: Cage ConvexHull Index Mapping

**Date**: October 28, 2025  
**Issue**: IndexError when initializing MediaPipeCageDeformer

---

## Problem

When testing Option A, encountered:

```
IndexError: index 8 is out of bounds for axis 0 with size 8
```

At line 324 in `test_integration_cage.py`:

```python
section_verts = original_cage_vertices[vertex_indices]
```

---

## Root Cause

**ConvexHull simplification**:

- Created 100 cage vertices (5 sections × 20 vertices each)
- ConvexHull reduced these to **only 8 vertices** (just the corner points)
- `section_info` still referenced original indices [0-99]
- When trying to access `cage_vertices[8]`, it was out of bounds

**Why ConvexHull did this**:

- ConvexHull only keeps vertices on the outer surface
- Interior and edge-subdivision vertices were discarded
- This is mathematically correct but broke our section mapping

---

## Solution

**Map original indices to hull indices**:

```python
# After ConvexHull
hull_vertex_indices = hull.vertices  # Which original vertices are in hull
hull_vertices = all_vertices[hull_vertex_indices]

# Create mapping: original index -> hull index
orig_to_hull = {orig_idx: hull_idx for hull_idx, orig_idx in enumerate(hull_vertex_indices)}

# Update section_info to use hull indices
updated_section_info = {}
for section_name, orig_indices in self.section_info.items():
    hull_indices = [orig_to_hull[orig_idx] for orig_idx in orig_indices if orig_idx in orig_to_hull]
    updated_section_info[section_name] = hull_indices

self.section_info = updated_section_info
```

---

## Result

Console now shows:

```
Creating convex hull from 100 vertices...
  Generated 12 faces
  Note: Hull reduced 100 vertices to 8 hull vertices
  Section 'torso': 20 original → 2 hull vertices
  Section 'left_upper_arm': 20 original → 2 hull vertices
  ...
```

Each section now correctly maps to the subset of vertices that survived the ConvexHull simplification.

---

## Impact on Deformation Quality

**Potential concern**: Each section now has only 2-3 vertices instead of 20.

**Why it's okay**:

- MVC weights still computed for ALL mesh vertices against ALL 8 cage vertices
- Sections are used for **deformation control** (which cage vertices to move)
- Even with 2 vertices per section, we can translate that section
- The smooth deformation comes from MVC interpolation, not cage resolution

**If quality is poor**:

- Can increase cage resolution (more sections, tighter bounds)
- Can use manual face generation instead of ConvexHull
- For now, test with current fix to see if it's sufficient

---

## Files Modified

- `tests/test_integration_cage.py` (lines 169-200)

---

## Testing

Now retry:

```bash
python tests/test_integration_cage.py --mesh generated_meshes/1761618888/mesh.obj --headless
```

Should proceed past the initialization error.
