"""
MediaPipe Linear Blend Skinning - Option C: Pre-computed Inverse Bind Matrices

This module implements proper skeletal deformation by:
1. Computing bone transformation matrices from MediaPipe keypoints
2. Remapping GLB skin weights from 163 bones to 9 MediaPipe bones
3. Pre-computing inverse bind matrices during T-pose calibration (Option C)
4. Applying LBS using stored inverse bind matrices

OPTION C IMPLEMENTATION (Nov 2025):
====================================
Problem: GLB has 163 bones with inverse bind matrices designed for those specific
bone coordinate systems. When we remap to 9 MediaPipe bones with DIFFERENT
coordinate systems, the original inverse bind matrices no longer match.

Solution: Compute OUR OWN inverse bind matrices for the 9 MediaPipe bones during
T-pose calibration, rather than trying to use the GLB's 163-bone matrices.

Implementation:
1. During T-pose calibration: call set_bind_pose(vertices, keypoints)
   - Computes bone transformation matrices M_bind from T-pose keypoints
   - Inverts to get M_bind_inv for each of the 9 MediaPipe bones
   - Stores these matrices for use during animation

2. During animation: deform(vertices, ref_kp, curr_kp)
   - Computes current bone matrices M_current from current keypoints
   - Uses STORED M_bind_inv (not recomputed!)
   - Applies: M_skin[i] = M_current[i] * M_bind_inv[i]
   - Deforms vertices: v' = sum_i( w_i * M_skin[i] * v )

Standard LBS Formula:
  v' = sum_i( weight[i] * M_current[i] * M_bind_inv[i] * v )

where:
  - M_bind_inv[i] = stored inverse bind matrix for MediaPipe bone i (from T-pose)
  - M_current[i] = current transformation matrix for bone i (from current pose)
  - weight[i] = remapped skinning weight for bone i

Critical: The inverse bind matrices are computed ONCE during T-pose calibration
and remain constant. Only M_current changes during animation.

Reference: https://cseweb.ucsd.edu/classes/sp16/cse169-a/readings/3-Skin.html
"""

import numpy as np
from typing import Dict, Tuple
from scipy.spatial.transform import Rotation


# MediaPipe bone definitions (start keypoint, end keypoint)
MEDIAPIPE_BONE_CONNECTIONS = {
    'spine': ('mid_hip', 'mid_shoulder'),
    'left_upper_arm': ('left_shoulder', 'left_elbow'),
    'left_lower_arm': ('left_elbow', 'left_wrist'),
    'right_upper_arm': ('right_shoulder', 'right_elbow'),
    'right_lower_arm': ('right_elbow', 'right_wrist'),
    'left_upper_leg': ('left_hip', 'left_knee'),
    'left_lower_leg': ('left_knee', 'left_ankle'),
    'right_upper_leg': ('right_hip', 'right_knee'),
    'right_lower_leg': ('right_knee', 'right_ankle'),
}

# Ordered list for consistent indexing
MEDIAPIPE_BONE_ORDER = [
    'spine',
    'left_upper_arm',
    'left_lower_arm',
    'right_upper_arm',
    'right_lower_arm',
    'left_upper_leg',
    'left_lower_leg',
    'right_upper_leg',
    'right_lower_leg',
]


class MediaPipeLBS:
    """Linear Blend Skinning using MediaPipe keypoints with proper bind pose"""

    def __init__(self, mesh, bone_name_mapping: Dict[str, str]):
        """
        Initialize LBS system for a rigged mesh

        Args:
            mesh: RiggedMesh with vertices, bones, skin_weights, skin_indices
            bone_name_mapping: Dict mapping MediaPipe bone names to GLB bone names
                              e.g., {'left_upper_arm': 'upperarm01_L'}
        """
        self.mesh = mesh
        self.bone_name_mapping = bone_name_mapping
        self.n_bones = len(MEDIAPIPE_BONE_ORDER)
        self.n_vertices = len(mesh.vertices)

        # Build reverse mapping: GLB bone name -> MediaPipe bone index
        self.glb_to_mediapipe_idx = {}
        for mp_idx, mp_bone_name in enumerate(MEDIAPIPE_BONE_ORDER):
            if mp_bone_name in bone_name_mapping:
                glb_bone_name = bone_name_mapping[mp_bone_name]
                # Find the GLB bone index
                for bone in mesh.bones:
                    if bone.name == glb_bone_name:
                        self.glb_to_mediapipe_idx[bone.index] = mp_idx
                        break

        # Remap weights from GLB's 163 bones to our 9 MediaPipe bones
        self.remapped_weights = self._remap_weights()

        # Inverse bind matrices (computed during T-pose calibration)
        # Shape: (n_bones, 4, 4)
        self.inverse_bind_matrices = None
        self.is_calibrated = False

        print(f"✓ MediaPipeLBS initialized:")
        print(f"  Bones: {self.n_bones}")
        print(f"  Vertices: {self.n_vertices}")
        print(f"  Mapped GLB bones: {len(self.glb_to_mediapipe_idx)}")
        print(f"  ⚠️  Call set_bind_pose() with T-pose keypoints before deforming!")

    def _remap_weights(self) -> np.ndarray:
        """
        Remap GLB skin weights (n_verts, 4) for 163 bones
        to MediaPipe weights (n_verts, 9) for our 9 bones

        Returns:
            (n_vertices, 9) array of weights
        """
        new_weights = np.zeros((self.n_vertices, self.n_bones), dtype=np.float32)

        # For each vertex
        for v_idx in range(self.n_vertices):
            # Check all 4 bone influences
            for slot in range(4):
                glb_bone_idx = self.mesh.skin_indices[v_idx, slot]
                weight = self.mesh.skin_weights[v_idx, slot]

                if weight == 0:
                    continue

                # Map to MediaPipe bone index
                if glb_bone_idx in self.glb_to_mediapipe_idx:
                    mp_bone_idx = self.glb_to_mediapipe_idx[glb_bone_idx]
                    new_weights[v_idx, mp_bone_idx] += weight

        # Renormalize weights (some vertices may have lost influence)
        weight_sums = new_weights.sum(axis=1, keepdims=True)
        # Where sum is 0 (no mapped bones), keep as 0 (vertex won't deform)
        # Where sum > 0, normalize to sum to 1
        new_weights = np.divide(
            new_weights,
            weight_sums,
            out=new_weights,
            where=weight_sums > 0
        )

        # Count vertices with no influence
        no_influence = (weight_sums == 0).sum()
        if no_influence > 0:
            print(f"  Warning: {no_influence} vertices have no mapped bone weights (won't deform)")

        return new_weights

    def set_bind_pose(self, base_vertices: np.ndarray, reference_keypoints: Dict[str, np.ndarray]):
        """
        Compute and store inverse bind matrices from T-pose calibration

        This is the KEY to Option C: we compute OUR OWN inverse bind matrices
        for the 9 MediaPipe bones based on the T-pose, rather than trying to
        use the GLB's 163-bone inverse bind matrices.

        Args:
            base_vertices: (n, 3) rest pose vertices (T-pose)
            reference_keypoints: T-pose keypoints (scaled and aligned)
        """
        # Compute bone transformation matrices in T-pose
        M_bind = self.compute_bone_matrices(reference_keypoints)

        # Invert to get inverse bind matrices
        self.inverse_bind_matrices = np.zeros((self.n_bones, 4, 4), dtype=np.float32)

        for bone_idx in range(self.n_bones):
            try:
                self.inverse_bind_matrices[bone_idx] = np.linalg.inv(M_bind[bone_idx])
            except np.linalg.LinAlgError:
                # Singular matrix, use identity
                print(f"⚠️  Warning: Bone {MEDIAPIPE_BONE_ORDER[bone_idx]} has singular matrix, using identity")
                self.inverse_bind_matrices[bone_idx] = np.eye(4)

        self.is_calibrated = True
        print(f"✓ Bind pose calibrated! Inverse bind matrices computed for {self.n_bones} bones")

    @staticmethod
    def compute_derived_keypoints(keypoints: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        """
        Compute derived keypoints like mid_hip and mid_shoulder

        Args:
            keypoints: Dict of keypoint_name -> (3,) position

        Returns:
            Dict with additional derived keypoints
        """
        result = keypoints.copy()

        if 'left_hip' in keypoints and 'right_hip' in keypoints:
            result['mid_hip'] = (keypoints['left_hip'] + keypoints['right_hip']) / 2

        if 'left_shoulder' in keypoints and 'right_shoulder' in keypoints:
            result['mid_shoulder'] = (keypoints['left_shoulder'] + keypoints['right_shoulder']) / 2

        return result

    @staticmethod
    def compute_bone_rotation(ref_start: np.ndarray, ref_end: np.ndarray,
                             curr_start: np.ndarray, curr_end: np.ndarray) -> np.ndarray:
        """
        Compute 3x3 rotation matrix from reference bone to current bone

        Uses scipy's Rotation.align_vectors for robust rotation computation

        Args:
            ref_start: (3,) start position of bone in reference pose
            ref_end: (3,) end position of bone in reference pose
            curr_start: (3,) start position of bone in current pose
            curr_end: (3,) end position of bone in current pose

        Returns:
            (3, 3) rotation matrix
        """
        # Compute bone vectors
        ref_vec = ref_end - ref_start
        ref_length = np.linalg.norm(ref_vec)
        if ref_length < 1e-6:
            return np.eye(3)  # Degenerate bone, no rotation
        ref_vec = ref_vec / ref_length

        curr_vec = curr_end - curr_start
        curr_length = np.linalg.norm(curr_vec)
        if curr_length < 1e-6:
            return np.eye(3)  # Degenerate bone, no rotation
        curr_vec = curr_vec / curr_length

        # Use scipy to compute optimal rotation
        # This handles edge cases (parallel vectors, etc.) robustly
        rot, _ = Rotation.align_vectors([curr_vec], [ref_vec])

        return rot.as_matrix()

    def compute_bone_matrices(self,
                             keypoints: Dict[str, np.ndarray]) -> np.ndarray:
        """
        Compute 4x4 transformation matrices for all bones from keypoints

        Args:
            keypoints: Keypoint positions (already scaled and aligned)

        Returns:
            (n_bones, 4, 4) transformation matrices
        """
        # Add derived keypoints
        kp = self.compute_derived_keypoints(keypoints)

        matrices = np.zeros((self.n_bones, 4, 4), dtype=np.float32)

        # Compute for each bone
        for bone_idx, bone_name in enumerate(MEDIAPIPE_BONE_ORDER):
            if bone_name not in MEDIAPIPE_BONE_CONNECTIONS:
                matrices[bone_idx] = np.eye(4)  # Identity
                continue

            start_key, end_key = MEDIAPIPE_BONE_CONNECTIONS[bone_name]

            # Check if keypoints exist
            if start_key not in kp or end_key not in kp:
                matrices[bone_idx] = np.eye(4)
                continue

            start_pos = kp[start_key]
            end_pos = kp[end_key]

            # Compute bone vector
            bone_vec = end_pos - start_pos
            bone_length = np.linalg.norm(bone_vec)

            if bone_length < 1e-6:
                matrices[bone_idx] = np.eye(4)
                continue

            bone_dir = bone_vec / bone_length

            # Build local coordinate system for the bone
            # Bone points along +Y in local space (standard for many systems)
            local_y = bone_dir

            # Choose perpendicular X axis
            if abs(bone_dir[1]) < 0.9:
                # Y component isn't too dominant, cross with global Y
                local_x = np.cross(np.array([0, 1, 0]), local_y)
            else:
                # Y component is dominant, cross with global X
                local_x = np.cross(np.array([1, 0, 0]), local_y)

            local_x = local_x / (np.linalg.norm(local_x) + 1e-8)

            # Z axis completes the right-handed system
            local_z = np.cross(local_x, local_y)
            local_z = local_z / (np.linalg.norm(local_z) + 1e-8)

            # Build 4x4 transformation matrix
            # [local_x, local_y, local_z] as columns = rotation part
            # start_pos = translation part
            matrices[bone_idx] = np.eye(4)
            matrices[bone_idx][:3, 0] = local_x
            matrices[bone_idx][:3, 1] = local_y
            matrices[bone_idx][:3, 2] = local_z
            matrices[bone_idx][:3, 3] = start_pos

        return matrices

    def compute_bone_transforms(self,
                                reference_keypoints: Dict[str, np.ndarray],
                                current_keypoints: Dict[str, np.ndarray]) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute rotation matrices and translations for LBS

        This computes the RELATIVE transformation: M_current * M_bind_inverse

        Args:
            reference_keypoints: T-pose keypoints (already scaled and aligned)
            current_keypoints: Current pose keypoints (already scaled and aligned)

        Returns:
            Rs: (n_bones, 3, 3) rotation matrices
            Ts: (n_bones, 3) translation vectors
        """
        # Compute bind pose matrices (from reference T-pose)
        M_bind = self.compute_bone_matrices(reference_keypoints)

        # Compute current pose matrices
        M_current = self.compute_bone_matrices(current_keypoints)

        Rs = np.zeros((self.n_bones, 3, 3), dtype=np.float32)
        Ts = np.zeros((self.n_bones, 3), dtype=np.float32)

        # For each bone, compute M_relative = M_current * M_bind_inverse
        for bone_idx in range(self.n_bones):
            # Compute inverse of bind pose
            try:
                M_bind_inv = np.linalg.inv(M_bind[bone_idx])
            except np.linalg.LinAlgError:
                # Singular matrix, use identity
                Rs[bone_idx] = np.eye(3)
                Ts[bone_idx] = np.zeros(3)
                continue

            # Compute relative transformation
            M_relative = M_current[bone_idx] @ M_bind_inv

            # Extract rotation and translation
            Rs[bone_idx] = M_relative[:3, :3]
            Ts[bone_idx] = M_relative[:3, 3]

        return Rs, Ts

    def deform(self,
               base_vertices: np.ndarray,
               reference_keypoints: Dict[str, np.ndarray],
               current_keypoints: Dict[str, np.ndarray]) -> np.ndarray:
        """
        Deform mesh vertices using LBS with pre-computed inverse bind matrices

        Option C Implementation:
        Uses inverse bind matrices computed during T-pose calibration,
        rather than recomputing them every frame.

        Formula: v' = sum_i( w_i * M_current_i * M_bind_inv_i * v )

        Args:
            base_vertices: (n, 3) rest pose vertices
            reference_keypoints: T-pose keypoints (not used if calibrated)
            current_keypoints: Current pose keypoints (scaled and aligned)

        Returns:
            (n, 3) deformed vertices
        """
        # Check if calibrated
        if not self.is_calibrated or self.inverse_bind_matrices is None:
            print("⚠️  WARNING: LBS not calibrated! Call set_bind_pose() first.")
            print("    Falling back to computing inverse bind matrices on-the-fly...")
            # Fallback: compute on-the-fly (old behavior)
            M_bind = self.compute_bone_matrices(reference_keypoints)
            M_bind_inv = np.zeros((self.n_bones, 4, 4), dtype=np.float32)
            for bone_idx in range(self.n_bones):
                try:
                    M_bind_inv[bone_idx] = np.linalg.inv(M_bind[bone_idx])
                except np.linalg.LinAlgError:
                    M_bind_inv[bone_idx] = np.eye(4)
        else:
            # Use pre-computed inverse bind matrices (Option C!)
            M_bind_inv = self.inverse_bind_matrices

        # Compute current pose matrices
        M_current = self.compute_bone_matrices(current_keypoints)

        # Compute skinning matrices: M_skin_i = M_current_i * M_bind_inv_i
        M_skin = np.zeros((self.n_bones, 4, 4), dtype=np.float32)

        for bone_idx in range(self.n_bones):
            M_skin[bone_idx] = M_current[bone_idx] @ M_bind_inv[bone_idx]

        # Apply LBS: for each vertex, blend transformed positions
        n_verts = len(base_vertices)
        deformed = np.zeros((n_verts, 3), dtype=np.float32)

        # Convert vertices to homogeneous coordinates (x, y, z, 1)
        v_homogeneous = np.ones((n_verts, 4), dtype=np.float32)
        v_homogeneous[:, :3] = base_vertices

        # Vectorized LBS computation
        # For each vertex, accumulate weighted bone transformations
        for bone_idx in range(self.n_bones):
            # Get weights for this bone (n_verts,)
            weights = self.remapped_weights[:, bone_idx]

            # Skip bones with no influence
            if np.sum(weights) < 1e-6:
                continue

            # Transform all vertices by this bone's skinning matrix
            # M_skin[bone_idx] @ v_homogeneous.T → (4, n_verts)
            v_transformed = (M_skin[bone_idx] @ v_homogeneous.T).T  # (n_verts, 4)

            # Add weighted contribution to deformed vertices
            # Broadcasting: (n_verts, 1) * (n_verts, 3)
            deformed += weights[:, np.newaxis] * v_transformed[:, :3]

        return deformed


# Test
if __name__ == "__main__":
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).parent))

    from rigged_mesh_loader import RiggedMeshLoader
    from mediapipe_to_bones import GLB_BONE_NAME_MAPPING

    print("=== Testing MediaPipe LBS ===\n")

    # Load mesh
    mesh = RiggedMeshLoader.load("rigged_mesh/CAUCASIAN MAN.glb")

    # Build bone mapping
    bone_names = [bone.name for bone in mesh.bones]
    bone_mapping = {}
    for mp_name, glb_candidates in GLB_BONE_NAME_MAPPING.items():
        for candidate in glb_candidates:
            if candidate in bone_names:
                bone_mapping[mp_name] = candidate
                break

    # Initialize LBS
    lbs = MediaPipeLBS(mesh, bone_mapping)

    # Create dummy keypoints
    reference_kp = {
        'left_shoulder': np.array([0.3, 1.4, 0.0]),
        'right_shoulder': np.array([-0.3, 1.4, 0.0]),
        'left_hip': np.array([0.15, 0.9, 0.0]),
        'right_hip': np.array([-0.15, 0.9, 0.0]),
        'left_elbow': np.array([0.6, 1.4, 0.0]),
        'right_elbow': np.array([-0.6, 1.4, 0.0]),
        'left_wrist': np.array([0.9, 1.4, 0.0]),
        'right_wrist': np.array([-0.9, 1.4, 0.0]),
        'left_knee': np.array([0.15, 0.5, 0.0]),
        'right_knee': np.array([-0.15, 0.5, 0.0]),
        'left_ankle': np.array([0.15, 0.0, 0.0]),
        'right_ankle': np.array([-0.15, 0.0, 0.0]),
    }

    # Current pose: arms down
    current_kp = reference_kp.copy()
    current_kp['left_wrist'] = np.array([0.3, 0.9, 0.0])
    current_kp['right_wrist'] = np.array([-0.3, 0.9, 0.0])

    # Test deformation
    print("\n=== Testing Deformation ===")
    deformed = lbs.deform(mesh.vertices, reference_kp, current_kp)

    # Check results
    delta = np.abs(deformed - mesh.vertices).max()
    print(f"Max vertex displacement: {delta:.6f} meters")

    if delta > 1e-6:
        print("✓ Vertices deformed (arms moved)")
    else:
        print("⚠️  No deformation detected")

    print("\n✓ Test complete")
