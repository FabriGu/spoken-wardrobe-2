"""
Simple Weight Transfer - Transfer skin weights from rigged mesh to clothing mesh

PROTOTYPE VERSION: Uses nearest-vertex approach instead of full barycentric interpolation.
This is sufficient for MVP to prove the concept works.

For production, implement the full algorithm from IMPLEMENTATION_PLAN.md:
- Closest point search with BVH acceleration
- Barycentric interpolation
- Weight inpainting for unmapped vertices
"""

import numpy as np
from typing import Dict, Tuple
from scipy.spatial import cKDTree


def transfer_weights_nearest(
    source_vertices: np.ndarray,
    source_weights: np.ndarray,
    source_indices: np.ndarray,
    target_vertices: np.ndarray,
    max_distance: float = 0.1
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Transfer skin weights from source mesh to target mesh using nearest-neighbor

    Args:
        source_vertices: (N, 3) source mesh vertices
        source_weights: (N, 4) skin weights [0-1] for up to 4 bones
        source_indices: (N, 4) bone indices for each weight
        target_vertices: (M, 3) target mesh vertices
        max_distance: Maximum search distance for nearest neighbor

    Returns:
        (target_weights, target_indices) - Transferred weights and bone indices
    """
    print(f"Transferring weights: {len(source_vertices)} source → {len(target_vertices)} target vertices")

    # Build KD-tree for fast nearest-neighbor search
    tree = cKDTree(source_vertices)

    # For each target vertex, find nearest source vertex
    distances, nearest_indices = tree.query(target_vertices, k=1)

    # Initialize target weights
    target_weights = np.zeros((len(target_vertices), 4), dtype=np.float32)
    target_indices = np.zeros((len(target_vertices), 4), dtype=np.int32)

    # Copy weights from nearest source vertex
    for i in range(len(target_vertices)):
        nearest_idx = nearest_indices[i]
        dist = distances[i]

        if dist < max_distance:
            # Close enough - copy weights directly
            target_weights[i] = source_weights[nearest_idx]
            target_indices[i] = source_indices[nearest_idx]
        else:
            # Too far - use fallback (assign to root bone)
            print(f"Warning: Vertex {i} is {dist:.3f}m from nearest source vertex (max: {max_distance})")
            target_weights[i, 0] = 1.0  # Full weight to bone 0 (root)
            target_indices[i, 0] = 0

    # Normalize weights
    weight_sums = target_weights.sum(axis=1, keepdims=True)
    target_weights = np.divide(
        target_weights,
        weight_sums,
        out=np.zeros_like(target_weights),
        where=weight_sums > 0
    )

    # Count how many vertices got valid weights
    valid = (distances < max_distance).sum()
    print(f"✓ {valid}/{len(target_vertices)} vertices within {max_distance}m")
    print(f"✗ {len(target_vertices) - valid} vertices using fallback (root bone)")

    return target_weights, target_indices


def transfer_weights_smooth(
    source_vertices: np.ndarray,
    source_weights: np.ndarray,
    source_indices: np.ndarray,
    target_vertices: np.ndarray,
    k_neighbors: int = 5,
    max_distance: float = 0.15
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Transfer skin weights using K-nearest neighbors with distance-based blending

    Better than nearest-neighbor, but still simpler than full barycentric interpolation.

    Args:
        source_vertices: (N, 3) source mesh vertices
        source_weights: (N, 4) skin weights
        source_indices: (N, 4) bone indices
        target_vertices: (M, 3) target mesh vertices
        k_neighbors: Number of neighbors to blend
        max_distance: Maximum distance for neighbors

    Returns:
        (target_weights, target_indices) - Transferred weights
    """
    print(f"Transferring weights (smooth k={k_neighbors}): {len(source_vertices)} source → {len(target_vertices)} target")

    # Build KD-tree
    tree = cKDTree(source_vertices)

    # Find K nearest neighbors for each target vertex
    distances, neighbor_indices = tree.query(target_vertices, k=k_neighbors)

    # Initialize output
    max_bones = source_indices.max() + 1  # Total number of bones
    target_weights = np.zeros((len(target_vertices), 4), dtype=np.float32)
    target_indices = np.zeros((len(target_vertices), 4), dtype=np.int32)

    for i in range(len(target_vertices)):
        # Get neighbors and their distances
        neighbors = neighbor_indices[i]
        dists = distances[i]

        # Filter by max distance
        valid_mask = dists < max_distance

        if not valid_mask.any():
            # No valid neighbors - use root bone
            target_weights[i, 0] = 1.0
            target_indices[i, 0] = 0
            continue

        valid_neighbors = neighbors[valid_mask]
        valid_dists = dists[valid_mask]

        # Compute blending weights (inverse distance)
        # Add small epsilon to avoid division by zero
        blend_weights = 1.0 / (valid_dists + 1e-6)
        blend_weights /= blend_weights.sum()

        # Accumulate bone influences from all neighbors
        bone_accumulator = np.zeros(max_bones, dtype=np.float32)

        for j, neighbor_idx in enumerate(valid_neighbors):
            neighbor_weight = blend_weights[j]

            for k in range(4):
                bone_idx = source_indices[neighbor_idx, k]
                bone_weight = source_weights[neighbor_idx, k]

                if bone_weight > 0:
                    bone_accumulator[bone_idx] += neighbor_weight * bone_weight

        # Select top 4 bones
        top_4_bones = np.argsort(bone_accumulator)[-4:][::-1]

        for k, bone_idx in enumerate(top_4_bones):
            target_indices[i, k] = bone_idx
            target_weights[i, k] = bone_accumulator[bone_idx]

    # Normalize weights
    weight_sums = target_weights.sum(axis=1, keepdims=True)
    target_weights = np.divide(
        target_weights,
        weight_sums,
        out=np.zeros_like(target_weights),
        where=weight_sums > 0
    )

    valid = (distances[:, 0] < max_distance).sum()
    print(f"✓ {valid}/{len(target_vertices)} vertices with valid neighbors")

    return target_weights, target_indices


# Test
if __name__ == "__main__":
    # Create simple test case
    # Source mesh: 3 vertices along X-axis
    source_verts = np.array([
        [0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [2.0, 0.0, 0.0],
    ], dtype=np.float32)

    # Source weights: linearly interpolated between bone 0 and bone 1
    source_weights = np.array([
        [1.0, 0.0, 0.0, 0.0],  # 100% bone 0
        [0.5, 0.5, 0.0, 0.0],  # 50/50
        [0.0, 1.0, 0.0, 0.0],  # 100% bone 1
    ], dtype=np.float32)

    source_indices = np.array([
        [0, 1, 0, 0],
        [0, 1, 0, 0],
        [0, 1, 0, 0],
    ], dtype=np.int32)

    # Target mesh: 2 vertices between source vertices
    target_verts = np.array([
        [0.5, 0.0, 0.0],  # Should get ~75% bone 0, 25% bone 1
        [1.5, 0.0, 0.0],  # Should get ~25% bone 0, 75% bone 1
    ], dtype=np.float32)

    # Test nearest-neighbor
    print("=== Nearest Neighbor Transfer ===")
    weights_nn, indices_nn = transfer_weights_nearest(
        source_verts, source_weights, source_indices, target_verts
    )
    print("\nResult:")
    for i in range(len(target_verts)):
        print(f"  Vertex {i}: weights={weights_nn[i]}, bones={indices_nn[i]}")

    # Test smooth blending
    print("\n=== Smooth K-NN Transfer ===")
    weights_smooth, indices_smooth = transfer_weights_smooth(
        source_verts, source_weights, source_indices, target_verts, k_neighbors=3
    )
    print("\nResult:")
    for i in range(len(target_verts)):
        print(f"  Vertex {i}: weights={weights_smooth[i]}, bones={indices_smooth[i]}")
