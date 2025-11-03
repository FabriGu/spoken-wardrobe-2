"""
MediaPipe to Bones Mapper - Maps MediaPipe landmarks to skeleton bone transformations

This module converts MediaPipe pose keypoints into bone rotations and positions
for animating a rigged 3D mesh.
"""

import numpy as np
from typing import Dict, List, Tuple
from scipy.spatial.transform import Rotation


# MediaPipe landmark indices (33 total)
MEDIAPIPE_LANDMARKS = {
    # Torso
    'left_shoulder': 11,
    'right_shoulder': 12,
    'left_hip': 23,
    'right_hip': 24,

    # Left arm
    'left_elbow': 13,
    'left_wrist': 15,

    # Right arm
    'right_elbow': 14,
    'right_wrist': 16,

    # Left leg
    'left_knee': 25,
    'left_ankle': 27,

    # Right leg
    'right_knee': 26,
    'right_ankle': 28,

    # Head
    'nose': 0,
    'left_ear': 7,
    'right_ear': 8,
}


# Mapping from MediaPipe keypoint pairs to skeleton bones
# Format: bone_name → (start_keypoint, end_keypoint, parent_bone)
BONE_MAPPING = {
    # Spine (root → shoulders)
    'spine': ('mid_hip', 'mid_shoulder', None),

    # Left arm chain
    'left_upper_arm': ('left_shoulder', 'left_elbow', 'spine'),
    'left_lower_arm': ('left_elbow', 'left_wrist', 'left_upper_arm'),

    # Right arm chain
    'right_upper_arm': ('right_shoulder', 'right_elbow', 'spine'),
    'right_lower_arm': ('right_elbow', 'right_wrist', 'right_upper_arm'),

    # Left leg chain
    'left_upper_leg': ('left_hip', 'left_knee', 'spine'),
    'left_lower_leg': ('left_knee', 'left_ankle', 'left_upper_leg'),

    # Right leg chain
    'right_upper_leg': ('right_hip', 'right_knee', 'spine'),
    'right_lower_leg': ('right_knee', 'right_ankle', 'right_upper_leg'),
}


# Mapping from our MediaPipe bone names to typical GLB skeleton bone names
# GLB mesh might have bones like "upperarm01_L", "lowerarm01_L", etc.
# This allows LBS to find the correct bones in the rigged mesh
GLB_BONE_NAME_MAPPING = {
    # Spine - try multiple common names
    'spine': ['Spine', 'spine', 'spine01', 'spine02', 'spine03', 'spine04', 'spine05'],

    # Left arm - IMPORTANT: exact names from guide first!
    'left_upper_arm': ['left_upper_arm', 'upperarm01_L', 'upperarm02_L', 'LeftUpperArm', 'upperArm_L'],
    'left_lower_arm': ['left_lower_arm', 'lowerarm01_L', 'lowerarm02_L', 'LeftLowerArm', 'lowerArm_L'],

    # Right arm - IMPORTANT: exact names from guide first!
    'right_upper_arm': ['right_upper_arm', 'upperarm01_R', 'upperarm02_R', 'RightUpperArm', 'upperArm_R'],
    'right_lower_arm': ['right_lower_arm', 'lowerarm01_R', 'lowerarm02_R', 'RightLowerArm', 'lowerArm_R'],

    # Left leg - IMPORTANT: exact names from guide first!
    'left_upper_leg': ['left_upper_leg', 'upperleg01_L', 'upperleg02_L', 'thigh01_L', 'thigh02_L', 'LeftUpperLeg', 'upperLeg_L'],
    'left_lower_leg': ['left_lower_leg', 'lowerleg01_L', 'lowerleg02_L', 'calf01_L', 'calf02_L', 'shin_L', 'LeftLowerLeg', 'lowerLeg_L'],

    # Right leg - IMPORTANT: exact names from guide first!
    'right_upper_leg': ['right_upper_leg', 'upperleg01_R', 'upperleg02_R', 'thigh01_R', 'thigh02_R', 'RightUpperLeg', 'upperLeg_R'],
    'right_lower_leg': ['right_lower_leg', 'lowerleg01_R', 'lowerleg02_R', 'calf01_R', 'calf02_R', 'shin_R', 'RightLowerLeg', 'lowerLeg_R'],
}


class MediaPipeToBones:
    """Converts MediaPipe pose to skeleton bone transformations"""

    def __init__(self, glb_bone_names=None):
        """
        Args:
            glb_bone_names: List of actual bone names from the GLB mesh
                           Used to create mapping from MediaPipe bones to GLB bones
        """
        self.reference_pose = None  # T-pose keypoints for calibration
        self.glb_bone_mapping = {}  # MediaPipe bone name → actual GLB bone name

        # Build mapping if GLB bone names provided
        if glb_bone_names:
            self._build_glb_mapping(glb_bone_names)

    def _build_glb_mapping(self, glb_bone_names: List[str]):
        """
        Build mapping from MediaPipe bone names to actual GLB bone names

        Args:
            glb_bone_names: List of bone names from the GLB skeleton
        """
        glb_set = set(glb_bone_names)

        for mediapipe_name, possible_names in GLB_BONE_NAME_MAPPING.items():
            # Try each possible name in order
            for candidate in possible_names:
                if candidate in glb_set:
                    self.glb_bone_mapping[mediapipe_name] = candidate
                    print(f"  Mapped {mediapipe_name} → {candidate}")
                    break

        if len(self.glb_bone_mapping) == 0:
            print("WARNING: No bones matched between MediaPipe and GLB!")
            print(f"GLB has {len(glb_bone_names)} bones:")
            for name in glb_bone_names[:20]:
                print(f"  - {name}")
            print("  ...")

    def set_reference_pose(self, keypoints_3d: Dict[str, np.ndarray]):
        """
        Set the reference T-pose from MediaPipe keypoints

        Args:
            keypoints_3d: Dict of keypoint names → 3D positions [x, y, z]
        """
        self.reference_pose = self._compute_derived_points(keypoints_3d)

    def update_bones(self, keypoints_3d: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        """
        Compute bone transformations from current MediaPipe keypoints

        Args:
            keypoints_3d: Current MediaPipe keypoints

        Returns:
            Dict of bone_name → 4x4 transformation matrix
            (bone names are the actual GLB bone names if mapping was built)
        """
        # Add derived points (mid_hip, mid_shoulder)
        keypoints = self._compute_derived_points(keypoints_3d)

        bone_transforms = {}

        for bone_name, (start_key, end_key, parent) in BONE_MAPPING.items():
            if start_key not in keypoints or end_key not in keypoints:
                continue

            start_pos = keypoints[start_key]
            end_pos = keypoints[end_key]

            # Compute bone direction vector
            bone_vec = end_pos - start_pos
            bone_length = np.linalg.norm(bone_vec)

            if bone_length < 1e-6:
                # Zero-length bone, use identity
                transform = np.eye(4)
            else:
                bone_dir = bone_vec / bone_length

                # Compute rotation to align with bone direction
                # Assume bones point along +Y axis in rest pose
                rest_dir = np.array([0, 1, 0])

                rotation = self._rotation_from_vectors(rest_dir, bone_dir)

                # Build 4x4 transformation matrix
                transform = np.eye(4)
                transform[:3, :3] = rotation
                transform[:3, 3] = start_pos

            # Use GLB bone name if we have a mapping, otherwise use MediaPipe name
            output_name = self.glb_bone_mapping.get(bone_name, bone_name)
            bone_transforms[output_name] = transform

        return bone_transforms

    def _compute_derived_points(self, keypoints: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        """Compute additional points like mid_hip, mid_shoulder"""
        result = keypoints.copy()

        # Mid hip
        if 'left_hip' in keypoints and 'right_hip' in keypoints:
            result['mid_hip'] = (keypoints['left_hip'] + keypoints['right_hip']) / 2

        # Mid shoulder
        if 'left_shoulder' in keypoints and 'right_shoulder' in keypoints:
            result['mid_shoulder'] = (keypoints['left_shoulder'] + keypoints['right_shoulder']) / 2

        return result

    def _rotation_from_vectors(self, vec1: np.ndarray, vec2: np.ndarray) -> np.ndarray:
        """
        Compute rotation matrix that rotates vec1 to align with vec2

        Args:
            vec1: Source vector (normalized)
            vec2: Target vector (normalized)

        Returns:
            3x3 rotation matrix
        """
        vec1 = vec1 / (np.linalg.norm(vec1) + 1e-8)
        vec2 = vec2 / (np.linalg.norm(vec2) + 1e-8)

        # Cross product gives rotation axis
        axis = np.cross(vec1, vec2)
        axis_length = np.linalg.norm(axis)

        # Check if vectors are parallel
        if axis_length < 1e-6:
            # Vectors are parallel (same or opposite direction)
            if np.dot(vec1, vec2) > 0:
                return np.eye(3)  # Same direction
            else:
                # Opposite direction - rotate 180° around perpendicular axis
                # Find a perpendicular vector
                if abs(vec1[0]) < 0.9:
                    perp = np.cross(vec1, [1, 0, 0])
                else:
                    perp = np.cross(vec1, [0, 1, 0])
                perp = perp / np.linalg.norm(perp)
                return Rotation.from_rotvec(np.pi * perp).as_matrix()

        axis = axis / axis_length

        # Angle between vectors
        angle = np.arccos(np.clip(np.dot(vec1, vec2), -1.0, 1.0))

        # Rodrigues rotation formula
        rot = Rotation.from_rotvec(angle * axis)
        return rot.as_matrix()

    @staticmethod
    def mediapipe_keypoints_to_dict(landmarks_3d: List) -> Dict[str, np.ndarray]:
        """
        Convert MediaPipe landmark list to named dictionary

        Args:
            landmarks_3d: MediaPipe pose landmarks (33 points)

        Returns:
            Dict of landmark_name → [x, y, z] position
        """
        result = {}

        for name, idx in MEDIAPIPE_LANDMARKS.items():
            if idx < len(landmarks_3d):
                lm = landmarks_3d[idx]
                result[name] = np.array([lm.x, lm.y, lm.z], dtype=np.float32)

        return result


# Test
if __name__ == "__main__":
    # Simulated T-pose keypoints
    t_pose = {
        'left_shoulder': np.array([-0.3, 1.4, 0.0]),
        'right_shoulder': np.array([0.3, 1.4, 0.0]),
        'left_hip': np.array([-0.15, 0.9, 0.0]),
        'right_hip': np.array([0.15, 0.9, 0.0]),
        'left_elbow': np.array([-0.6, 1.4, 0.0]),
        'left_wrist': np.array([-0.9, 1.4, 0.0]),
        'right_elbow': np.array([0.6, 1.4, 0.0]),
        'right_wrist': np.array([0.9, 1.4, 0.0]),
        'left_knee': np.array([-0.15, 0.5, 0.0]),
        'left_ankle': np.array([-0.15, 0.0, 0.0]),
        'right_knee': np.array([0.15, 0.5, 0.0]),
        'right_ankle': np.array([0.15, 0.0, 0.0]),
    }

    mapper = MediaPipeToBones()
    mapper.set_reference_pose(t_pose)

    # Simulate arms down pose
    current_pose = t_pose.copy()
    current_pose['left_wrist'] = np.array([-0.3, 0.9, 0.0])  # Left arm down
    current_pose['right_wrist'] = np.array([0.3, 0.9, 0.0])  # Right arm down

    bone_transforms = mapper.update_bones(current_pose)

    print("=== Bone Transformations ===")
    for bone_name, transform in bone_transforms.items():
        print(f"\n{bone_name}:")
        print(f"  Position: {transform[:3, 3]}")
        print(f"  Rotation (euler): {Rotation.from_matrix(transform[:3, :3]).as_euler('xyz', degrees=True)}")
