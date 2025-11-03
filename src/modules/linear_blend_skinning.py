"""
Linear Blend Skinning (LBS) - Apply bone transformations to mesh vertices

This is the standard skeletal animation technique used in games and 3D applications.
Each vertex is influenced by up to 4 bones, with weights summing to 1.0.

Reference: https://en.wikipedia.org/wiki/Skeletal_animation
"""

import numpy as np
from typing import List, Dict
from rigged_mesh_loader import SkeletonBone


def apply_skinning(
    base_vertices: np.ndarray,
    skin_weights: np.ndarray,
    skin_indices: np.ndarray,
    bones: List[SkeletonBone],
    bone_world_transforms: Dict[str, np.ndarray]
) -> np.ndarray:
    """
    Apply Linear Blend Skinning to deform mesh

    Args:
        base_vertices: (N, 3) rest pose vertex positions
        skin_weights: (N, 4) vertex weights for up to 4 bones [0-1], sum to 1.0
        skin_indices: (N, 4) bone indices for each weight
        bones: List of SkeletonBone objects with inverse bind matrices
        bone_world_transforms: Dict of bone_name → current 4x4 world transform matrix

    Returns:
        (N, 3) deformed vertex positions
    """
    n_vertices = len(base_vertices)
    deformed_vertices = np.zeros((n_vertices, 3), dtype=np.float32)

    # Convert base vertices to homogeneous coordinates (N, 4)
    vertices_homogeneous = np.ones((n_vertices, 4), dtype=np.float32)
    vertices_homogeneous[:, :3] = base_vertices

    # For each vertex
    for i in range(n_vertices):
        vertex_pos_4d = np.zeros(4, dtype=np.float32)

        # Blend contributions from all influencing bones
        for j in range(4):  # Up to 4 bones
            bone_idx = skin_indices[i, j]
            weight = skin_weights[i, j]

            if weight == 0:
                continue

            # Get bone transformation
            bone = bones[bone_idx]

            # Compute skinning matrix:
            # M = WorldTransform * InverseBindMatrix
            # This transforms from bind pose → bone space → world space
            if bone.name in bone_world_transforms:
                world_transform = bone_world_transforms[bone.name]
            else:
                # No current transform - use identity
                world_transform = np.eye(4)

            skinning_matrix = world_transform @ bone.inverse_bind_matrix

            # Apply weighted transformation
            vertex_pos_4d += weight * (skinning_matrix @ vertices_homogeneous[i])

        # Extract 3D position
        deformed_vertices[i] = vertex_pos_4d[:3]

    return deformed_vertices


def apply_skinning_fast(
    base_vertices: np.ndarray,
    skin_weights: np.ndarray,
    skin_indices: np.ndarray,
    bones: List[SkeletonBone],
    bone_world_transforms: Dict[str, np.ndarray]
) -> np.ndarray:
    """
    Optimized version using vectorized operations

    Same as apply_skinning but much faster for large meshes.
    """
    n_vertices = len(base_vertices)

    # Build array of skinning matrices (one per bone)
    n_bones = len(bones)
    skinning_matrices = np.zeros((n_bones, 4, 4), dtype=np.float32)

    for bone in bones:
        if bone.name in bone_world_transforms:
            world_transform = bone_world_transforms[bone.name]
        else:
            world_transform = np.eye(4)

        skinning_matrices[bone.index] = world_transform @ bone.inverse_bind_matrix

    # Convert vertices to homogeneous coordinates
    vertices_h = np.ones((n_vertices, 4), dtype=np.float32)
    vertices_h[:, :3] = base_vertices

    # Initialize result
    deformed = np.zeros((n_vertices, 4), dtype=np.float32)

    # For each of the 4 bone slots
    for slot in range(4):
        bone_indices = skin_indices[:, slot]  # (N,) which bone for this slot
        weights = skin_weights[:, slot:slot+1]  # (N, 1) weight for this slot

        # Get skinning matrix for each vertex's bone
        matrices = skinning_matrices[bone_indices]  # (N, 4, 4)

        # Apply transformation: matrix @ vertex
        # Using einsum for efficient batched matrix-vector multiply
        transformed = np.einsum('nij,nj->ni', matrices, vertices_h)  # (N, 4)

        # Add weighted contribution
        deformed += weights * transformed

    # Return 3D positions
    return deformed[:, :3]


# Test
if __name__ == "__main__":
    from rigged_mesh_loader import SkeletonBone

    # Create simple test: 2 bones, 3 vertices
    bone0 = SkeletonBone("bone0", 0)
    bone0.inverse_bind_matrix = np.eye(4)  # Identity

    bone1 = SkeletonBone("bone1", 1)
    bone1.inverse_bind_matrix = np.eye(4)

    bones = [bone0, bone1]

    # Base vertices along X-axis
    base_verts = np.array([
        [0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [2.0, 0.0, 0.0],
    ], dtype=np.float32)

    # Weights: linearly interpolated
    skin_weights = np.array([
        [1.0, 0.0, 0.0, 0.0],  # 100% bone0
        [0.5, 0.5, 0.0, 0.0],  # 50/50
        [0.0, 1.0, 0.0, 0.0],  # 100% bone1
    ], dtype=np.float32)

    skin_indices = np.array([
        [0, 1, 0, 0],
        [0, 1, 0, 0],
        [0, 1, 0, 0],
    ], dtype=np.int32)

    # Test 1: No transformation (should match base)
    print("=== Test 1: Identity Transform ===")
    bone_transforms = {
        "bone0": np.eye(4),
        "bone1": np.eye(4),
    }

    deformed = apply_skinning(base_verts, skin_weights, skin_indices, bones, bone_transforms)
    print("Base vertices:")
    print(base_verts)
    print("\nDeformed vertices (should be same):")
    print(deformed)
    print(f"✓ Max error: {np.abs(deformed - base_verts).max():.6f}")

    # Test 2: Translate bone1 by (0, 1, 0)
    print("\n=== Test 2: Translate Bone1 ===")
    bone1_transform = np.eye(4)
    bone1_transform[1, 3] = 1.0  # Translate Y +1

    bone_transforms = {
        "bone0": np.eye(4),
        "bone1": bone1_transform,
    }

    deformed = apply_skinning(base_verts, skin_weights, skin_indices, bones, bone_transforms)
    print("Deformed vertices:")
    print(deformed)
    print("Expected: vertex 0 stays at (0,0,0), vertex 2 moves to (2,1,0), vertex 1 halfway")

    # Test vectorized version
    print("\n=== Test 3: Vectorized Version ===")
    deformed_fast = apply_skinning_fast(base_verts, skin_weights, skin_indices, bones, bone_transforms)
    print("Deformed (fast):")
    print(deformed_fast)
    print(f"✓ Match slow version: {np.allclose(deformed, deformed_fast)}")
