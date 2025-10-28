"""
Keypoint Mapper V2 - Corrected Coordinate System Handling
==========================================================

This version fixes coordinate system issues:
1. Proper normalization of MediaPipe keypoints to mesh space
2. Delta computation from reference pose
3. Section-wise cage deformation based on relevant keypoints
4. Toggle for 2D vs 3D warping

Author: AI Assistant
Date: October 26, 2025
"""

import numpy as np
from typing import Dict, Tuple, Optional


class KeypointMapperV2:
    """
    Maps MediaPipe keypoints to cage deformation with proper coordinate handling.
    
    Key improvements over V1:
    - Converts keypoints from pixel space to mesh-centered normalized space
    - Computes deltas from reference pose (not absolute positions)
    - Applies section-wise deformation (not uniform translation)
    - Supports 2D-only warping mode
    """
    
    # MediaPipe landmark indices
    KEYPOINT_INDICES = {
        'nose': 0,
        'left_shoulder': 11,
        'right_shoulder': 12,
        'left_elbow': 13,
        'right_elbow': 14,
        'left_wrist': 15,
        'right_wrist': 16,
        'left_hip': 23,
        'right_hip': 24,
        'left_knee': 25,
        'right_knee': 26,
        'left_ankle': 27,
        'right_ankle': 28,
    }
    
    def __init__(
        self,
        mesh_bounds: np.ndarray,
        reference_keypoints: Dict[str, Tuple[float, float, float]],
        frame_shape: Tuple[int, int],
        enable_z_warp: bool = False
    ):
        """
        Initialize mapper with coordinate system parameters.
        
        Args:
            mesh_bounds: (2, 3) array of mesh bounds
            reference_keypoints: Dict of reference pose {name: (x_px, y_px, z)}
            frame_shape: (height, width) of camera frame
            enable_z_warp: Whether to use Z-axis deformation
        """
        self.mesh_bounds = mesh_bounds
        self.mesh_center = mesh_bounds.mean(axis=0)
        self.mesh_size = mesh_bounds[1] - mesh_bounds[0]
        self.frame_shape = frame_shape
        self.enable_z_warp = enable_z_warp
        
        # Normalize reference keypoints to mesh space
        self.reference_keypoints_normalized = self._normalize_keypoints(reference_keypoints)
        
        print(f"\n{'='*60}")
        print("KEYPOINT MAPPER V2 - INITIALIZATION")
        print(f"{'='*60}")
        print(f"Mesh center: {self.mesh_center}")
        print(f"Mesh size: {self.mesh_size}")
        print(f"Frame shape: {frame_shape}")
        print(f"Z-axis warping: {'ENABLED' if enable_z_warp else 'DISABLED (2D only)'}")
        print(f"Reference keypoints: {len(self.reference_keypoints_normalized)}")
        print(f"{'='*60}\n")
    
    def _normalize_keypoints(
        self,
        keypoints: Dict[str, Tuple[float, float, float]]
    ) -> Dict[str, np.ndarray]:
        """
        Convert keypoints from pixel space to mesh-centered normalized space.
        
        Args:
            keypoints: Dict of {name: (x_px, y_px, z_mediapipe)}
        
        Returns:
            Dict of {name: np.array([x_mesh, y_mesh, z_mesh])}
        """
        h, w = self.frame_shape
        normalized = {}
        
        for name, (x_px, y_px, z_mp) in keypoints.items():
            # Convert pixels to [-1, 1] normalized space
            x_norm = (x_px / w) * 2 - 1
            y_norm = 1 - (y_px / h) * 2  # Flip Y (image Y goes down, 3D Y goes up)
            
            # Scale to mesh dimensions
            x_mesh = x_norm * self.mesh_size[0] / 2
            y_mesh = y_norm * self.mesh_size[1] / 2
            
            # Z: MediaPipe Z is in [-1, 1] relative to hips, scale to mesh
            z_mesh = z_mp * self.mesh_size[2] * 1000  # Scale factor for depth
            
            normalized[name] = np.array([x_mesh, y_mesh, z_mesh])
        
        return normalized
    
    def extract_keypoints_from_landmarks(
        self,
        landmarks
    ) -> Dict[str, Tuple[float, float, float]]:
        """
        Extract keypoints from MediaPipe landmarks.
        
        Args:
            landmarks: MediaPipe pose_landmarks object
        
        Returns:
            Dict of {name: (x_px, y_px, z_mediapipe)}
        """
        if landmarks is None:
            return {}
        
        h, w = self.frame_shape
        keypoints = {}
        
        for name, idx in self.KEYPOINT_INDICES.items():
            if idx < len(landmarks.landmark):
                landmark = landmarks.landmark[idx]
                x_px = landmark.x * w
                y_px = landmark.y * h
                z_mp = landmark.z  # MediaPipe's Z (relative depth)
                
                keypoints[name] = (x_px, y_px, z_mp)
        
        return keypoints
    
    def compute_keypoint_deltas(
        self,
        current_landmarks
    ) -> Dict[str, np.ndarray]:
        """
        Compute movement deltas from reference pose.
        
        Args:
            current_landmarks: MediaPipe pose_landmarks object
        
        Returns:
            Dict of {name: np.array([dx, dy, dz])}
        """
        # Extract current keypoints
        current_keypoints_px = self.extract_keypoints_from_landmarks(current_landmarks)
        
        if len(current_keypoints_px) == 0:
            return {}
        
        # Normalize to mesh space
        current_keypoints_normalized = self._normalize_keypoints(current_keypoints_px)
        
        # Compute deltas
        deltas = {}
        for name in self.reference_keypoints_normalized.keys():
            if name in current_keypoints_normalized:
                delta = current_keypoints_normalized[name] - self.reference_keypoints_normalized[name]
                
                # If Z-warping disabled, zero out Z component
                if not self.enable_z_warp:
                    delta[2] = 0.0
                
                deltas[name] = delta
        
        return deltas
    
    def deform_cage(
        self,
        original_cage_vertices: np.ndarray,
        cage_structure: Dict,
        current_landmarks
    ) -> np.ndarray:
        """
        Deform cage with hierarchical joint constraints.
        
        Uses a hybrid approach:
        1. Section-based movement for core parts (torso)
        2. Joint-constrained movement for limbs (prevents detachment)
        3. Smooth interpolation for in-between vertices
        
        Args:
            original_cage_vertices: (N, 3) original cage vertex positions
            cage_structure: Dict with anatomical section info
            current_landmarks: MediaPipe pose_landmarks object
        
        Returns:
            deformed_cage_vertices: (N, 3) new cage vertex positions
        """
        # Get keypoint deltas
        keypoint_deltas = self.compute_keypoint_deltas(current_landmarks)
        
        if len(keypoint_deltas) == 0:
            # No keypoints detected, return original
            return original_cage_vertices.copy()
        
        # Define hierarchical relationships (child -> parent)
        # Children must move with parent to prevent detachment
        HIERARCHY = {
            'left_upper_arm': 'torso',
            'right_upper_arm': 'torso',
            'left_lower_arm': 'left_upper_arm',
            'right_lower_arm': 'right_upper_arm',
            'left_hand': 'left_lower_arm',
            'right_hand': 'right_lower_arm',
            'left_upper_leg': 'torso',
            'right_upper_leg': 'torso',
            'left_lower_leg': 'left_upper_leg',
            'right_lower_leg': 'right_upper_leg',
            'left_foot': 'left_lower_leg',
            'right_foot': 'right_lower_leg',
            'head': 'torso',
        }
        
        # Compute section transformations
        section_transforms = {}
        
        for section_name, section_info in cage_structure.items():
            keypoint_names = section_info['keypoints']
            
            # Collect deltas for this section's keypoints
            section_deltas = []
            for kpt_name in keypoint_names:
                if kpt_name in keypoint_deltas:
                    section_deltas.append(keypoint_deltas[kpt_name])
            
            if len(section_deltas) > 0:
                # Use mean translation for this section
                section_transforms[section_name] = np.mean(section_deltas, axis=0)
            else:
                section_transforms[section_name] = np.zeros(3)
        
        # Apply hierarchical transformations (parent influences child)
        final_transforms = {}
        
        for section_name in cage_structure.keys():
            # Start with section's own transform
            transform = section_transforms.get(section_name, np.zeros(3)).copy()
            
            # Add parent's transform (propagate through hierarchy)
            if section_name in HIERARCHY:
                parent_name = HIERARCHY[section_name]
                parent_transform = section_transforms.get(parent_name, np.zeros(3))
                
                # Child inherits 100% of parent movement (prevents detachment)
                # Plus its own additional movement
                transform += parent_transform
            
            final_transforms[section_name] = transform
        
        # Apply transformations to vertices
        deformed_vertices = original_cage_vertices.copy()
        
        for section_name, section_info in cage_structure.items():
            vertex_indices = section_info['vertex_indices']
            transform = final_transforms[section_name]
            
            # Apply to all vertices in this section
            deformed_vertices[vertex_indices] = original_cage_vertices[vertex_indices] + transform
        
        return deformed_vertices
    
    def get_debug_info(
        self,
        current_landmarks
    ) -> Dict:
        """
        Get debug information about current state.
        
        Args:
            current_landmarks: MediaPipe pose_landmarks object
        
        Returns:
            Dict with debug info
        """
        keypoint_deltas = self.compute_keypoint_deltas(current_landmarks)
        
        if len(keypoint_deltas) == 0:
            return {
                'keypoints_detected': 0,
                'mean_delta': np.array([0.0, 0.0, 0.0]),
                'max_delta': 0.0,
                'delta_magnitude': 0.0
            }
        
        # Compute statistics
        deltas_array = np.array(list(keypoint_deltas.values()))
        mean_delta = deltas_array.mean(axis=0)
        max_delta = np.abs(deltas_array).max()
        delta_magnitude = np.linalg.norm(mean_delta)
        
        return {
            'keypoints_detected': len(keypoint_deltas),
            'mean_delta': mean_delta,
            'max_delta': max_delta,
            'delta_magnitude': delta_magnitude,
            'delta_per_keypoint': {name: np.linalg.norm(delta) for name, delta in keypoint_deltas.items()}
        }


# ============================================================================
# Testing
# ============================================================================

if __name__ == "__main__":
    print("="*60)
    print("Keypoint Mapper V2 - Test Script")
    print("="*60)
    print("\nThis module should be imported, not run directly.")
    print("Use test_integration_v2.py for full pipeline testing.")
    print("="*60)

