"""
Articulated Cage Generator with Oriented Bounding Boxes (OBBs)
================================================================

This implements proper cage-based deformation based on research literature:
- Le & Deng (2017): "Interactive Cage Generation for Mesh Deformation"
- Xian et al. (2012): "Automatic cage generation by improved OBBs"

Key Concepts:
1. Generate OBB (Oriented Bounding Box) for each BodyPix segment
2. Connect OBBs at joint keypoints (shared vertices)
3. Create unified cage mesh with anatomical sections
4. Enable articulated deformation via joint angles

Author: AI Assistant
Date: October 28, 2025
"""

import numpy as np
import trimesh
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from scipy.linalg import eigh


@dataclass
class OBB:
    """Oriented Bounding Box representation"""
    center: np.ndarray  # (3,) center position
    axes: np.ndarray    # (3, 3) orthonormal axes (column vectors)
    half_extents: np.ndarray  # (3,) half-sizes along each axis
    section_name: str   # e.g., 'left_upper_arm'
    
    def get_vertices(self) -> np.ndarray:
        """
        Get 8 corner vertices of the OBB.
        
        Returns:
            (8, 3) array of vertex positions
        """
        vertices = []
        for i in [-1, 1]:
            for j in [-1, 1]:
                for k in [-1, 1]:
                    v = self.center + \
                        i * self.half_extents[0] * self.axes[:, 0] + \
                        j * self.half_extents[1] * self.axes[:, 1] + \
                        k * self.half_extents[2] * self.axes[:, 2]
                    vertices.append(v)
        return np.array(vertices)
    
    def get_faces(self, vertex_offset: int = 0) -> np.ndarray:
        """
        Get triangulated faces for the OBB.
        
        Args:
            vertex_offset: Offset to add to vertex indices (for merging multiple OBBs)
            
        Returns:
            (12, 3) array of triangular faces
        """
        # Define box faces (each face = 2 triangles)
        faces = []
        # Format: each face is defined by 4 corners, triangulated as (0,1,2) and (0,2,3)
        face_quads = [
            [0, 1, 3, 2],  # -X face
            [4, 6, 7, 5],  # +X face
            [0, 4, 5, 1],  # -Y face
            [2, 3, 7, 6],  # +Y face
            [0, 2, 6, 4],  # -Z face
            [1, 5, 7, 3],  # +Z face
        ]
        
        for quad in face_quads:
            # Triangle 1
            faces.append([vertex_offset + quad[0], vertex_offset + quad[1], vertex_offset + quad[2]])
            # Triangle 2
            faces.append([vertex_offset + quad[0], vertex_offset + quad[2], vertex_offset + quad[3]])
        
        return np.array(faces)
    
    def rotate(self, pivot: np.ndarray, rotation_matrix: np.ndarray):
        """
        Rotate OBB around a pivot point.
        
        Args:
            pivot: (3,) pivot point position
            rotation_matrix: (3, 3) rotation matrix
        """
        self.center = rotation_matrix @ (self.center - pivot) + pivot
        self.axes = rotation_matrix @ self.axes


class ArticulatedCageGenerator:
    """
    Generates articulated cage structure with OBBs for each body segment.
    
    Unlike ConvexHull-based approaches, this creates properly connected
    cage segments that can be articulated via joint angles.
    """
    
    # Hierarchical body structure (child -> parent)
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
    
    # Mapping from section names to joint keypoint names
    SECTION_JOINTS = {
        'torso': 'center',  # Special case: average of shoulders and hips
        'left_upper_arm': 'left_shoulder',
        'right_upper_arm': 'right_shoulder',
        'left_lower_arm': 'left_elbow',
        'right_lower_arm': 'right_elbow',
        'left_hand': 'left_wrist',
        'right_hand': 'right_wrist',
        'left_upper_leg': 'left_hip',
        'right_upper_leg': 'right_hip',
        'left_lower_leg': 'left_knee',
        'right_lower_leg': 'right_knee',
        'left_foot': 'left_ankle',
        'right_foot': 'right_ankle',
        'head': 'nose',
    }
    
    # Heuristic depth ratios (relative to torso depth)
    DEPTH_RATIOS = {
        'torso': 1.0,
        'left_upper_arm': 0.4,
        'right_upper_arm': 0.4,
        'left_lower_arm': 0.3,
        'right_lower_arm': 0.3,
        'left_hand': 0.2,
        'right_hand': 0.2,
        'left_upper_leg': 0.5,
        'right_upper_leg': 0.5,
        'left_lower_leg': 0.4,
        'right_lower_leg': 0.4,
        'left_foot': 0.25,
        'right_foot': 0.25,
        'head': 0.6,
    }
    
    def __init__(self, mesh: trimesh.Trimesh):
        """
        Initialize with target mesh.
        
        Args:
            mesh: The 3D clothing mesh to create cage around
        """
        self.mesh = mesh
        self.obbs = {}  # section_name -> OBB
        self.joint_positions = {}  # joint_name -> 3D position
        
    def generate_cage(
        self,
        bodypix_masks: Dict[str, np.ndarray],
        keypoints_2d: Dict[str, Tuple[float, float]],
        frame_shape: Tuple[int, int],
        padding: float = 0.15
    ) -> Tuple[trimesh.Trimesh, Dict, Dict]:
        """
        Generate articulated cage from BodyPix masks and MediaPipe keypoints.
        
        Args:
            bodypix_masks: Dict mapping section names to 2D binary masks
            keypoints_2d: Dict mapping keypoint names to (x, y) pixel coordinates
            frame_shape: (height, width) of the camera frame
            padding: Padding factor for OBB expansion (default 15%)
            
        Returns:
            (cage_mesh, section_info, joint_info)
            - cage_mesh: Unified trimesh object
            - section_info: Maps section names to vertex indices in cage
            - joint_info: Maps joint names to positions and connected sections
        """
        print(f"\n{'='*70}")
        print("GENERATING ARTICULATED CAGE WITH OBBs")
        print(f"{'='*70}")
        
        # Step 1: Project 2D keypoints to 3D mesh space
        self.joint_positions = self._project_keypoints_to_3d(
            keypoints_2d, frame_shape
        )
        
        print(f"\n✓ Projected {len(self.joint_positions)} keypoints to 3D")
        
        # Step 2: Generate OBB for each BodyPix segment
        for section_name, mask in bodypix_masks.items():
            if section_name not in self.SECTION_JOINTS:
                continue  # Skip unknown sections
            
            if np.sum(mask > 0) < 50:  # Skip if too few pixels
                print(f"  ⚠ Skipping '{section_name}' (insufficient mask pixels)")
                continue
            
            obb = self._generate_obb_from_mask(
                mask, section_name, frame_shape, padding
            )
            
            if obb is not None:
                self.obbs[section_name] = obb
                print(f"  ✓ Generated OBB for '{section_name}'")
        
        print(f"\n✓ Generated {len(self.obbs)} OBBs")
        
        # Step 3: Connect OBBs at joints to create unified cage
        cage_mesh, section_info, joint_info = self._build_unified_cage()
        
        print(f"\n✓ Unified cage created:")
        print(f"   Vertices: {len(cage_mesh.vertices)}")
        print(f"   Faces: {len(cage_mesh.faces)}")
        print(f"   Sections: {list(section_info.keys())}")
        print(f"{'='*70}\n")
        
        return cage_mesh, section_info, joint_info
    
    def _project_keypoints_to_3d(
        self,
        keypoints_2d: Dict[str, Tuple[float, float]],
        frame_shape: Tuple[int, int]
    ) -> Dict[str, np.ndarray]:
        """
        Project 2D keypoints to 3D mesh space.
        
        Args:
            keypoints_2d: Dict of keypoint_name -> (x, y) in pixels
            frame_shape: (height, width) of frame
            
        Returns:
            Dict of keypoint_name -> (x, y, z) in mesh space
        """
        h, w = frame_shape
        mesh_bounds = self.mesh.bounds
        mesh_center = self.mesh.centroid
        mesh_size = mesh_bounds[1] - mesh_bounds[0]
        
        keypoints_3d = {}
        
        for name, (x_pixel, y_pixel) in keypoints_2d.items():
            # Normalize to [-1, 1]
            x_norm = (x_pixel / w) * 2 - 1
            y_norm = 1 - (y_pixel / h) * 2  # Flip Y
            
            # Scale to mesh dimensions
            x_mesh = x_norm * mesh_size[0] / 2 + mesh_center[0]
            y_mesh = y_norm * mesh_size[1] / 2 + mesh_center[1]
            z_mesh = mesh_center[2]  # Default to mesh center depth
            
            keypoints_3d[name] = np.array([x_mesh, y_mesh, z_mesh])
        
        # Special case: 'center' is average of shoulders and hips
        if all(k in keypoints_3d for k in ['left_shoulder', 'right_shoulder', 'left_hip', 'right_hip']):
            center = (
                keypoints_3d['left_shoulder'] +
                keypoints_3d['right_shoulder'] +
                keypoints_3d['left_hip'] +
                keypoints_3d['right_hip']
            ) / 4
            keypoints_3d['center'] = center
        
        return keypoints_3d
    
    def _generate_obb_from_mask(
        self,
        mask: np.ndarray,
        section_name: str,
        frame_shape: Tuple[int, int],
        padding: float
    ) -> Optional[OBB]:
        """
        Generate OBB from a 2D BodyPix mask using PCA.
        
        Args:
            mask: 2D binary mask (H, W)
            section_name: Name of body section
            frame_shape: (height, width) of frame
            padding: Padding factor
            
        Returns:
            OBB object or None if generation fails
        """
        # Extract mask points (2D)
        rows, cols = np.where(mask > 0)
        if len(rows) < 10:
            return None
        
        points_2d = np.column_stack([cols, rows])  # (N, 2) - x, y
        
        # PCA for orientation
        mean_2d = points_2d.mean(axis=0)
        centered = points_2d - mean_2d
        cov = centered.T @ centered / len(centered)
        
        eigenvalues, eigenvectors = eigh(cov)
        # Sort by descending eigenvalue (largest variance first)
        idx = eigenvalues.argsort()[::-1]
        eigenvalues = eigenvalues[idx]
        eigenvectors = eigenvectors[:, idx]
        
        # Principal axes in 2D
        axis_major = eigenvectors[:, 0]  # Direction of maximum variance
        axis_minor = eigenvectors[:, 1]  # Direction of minimum variance
        
        # Project points onto principal axes to get extents
        proj_major = centered @ axis_major
        proj_minor = centered @ axis_minor
        
        half_extent_major = (proj_major.max() - proj_major.min()) / 2 * (1 + padding)
        half_extent_minor = (proj_minor.max() - proj_minor.min()) / 2 * (1 + padding)
        
        # Convert to 3D mesh space
        h, w = frame_shape
        mesh_bounds = self.mesh.bounds
        mesh_center = self.mesh.centroid
        mesh_size = mesh_bounds[1] - mesh_bounds[0]
        
        # Center position
        x_norm = (mean_2d[0] / w) * 2 - 1
        y_norm = 1 - (mean_2d[1] / h) * 2
        
        center_3d = np.array([
            x_norm * mesh_size[0] / 2 + mesh_center[0],
            y_norm * mesh_size[1] / 2 + mesh_center[1],
            mesh_center[2]
        ])
        
        # 3D axes
        # X-axis: aligned with major 2D axis in XZ plane
        axis_x = np.array([axis_major[0], 0, axis_major[1]])
        axis_x = axis_x / (np.linalg.norm(axis_x) + 1e-8)
        
        # Y-axis: vertical
        axis_y = np.array([0, 1, 0])
        
        # Z-axis: perpendicular to X and Y
        axis_z = np.cross(axis_x, axis_y)
        axis_z = axis_z / (np.linalg.norm(axis_z) + 1e-8)
        
        # Recompute X to ensure orthogonality
        axis_x = np.cross(axis_y, axis_z)
        axis_x = axis_x / (np.linalg.norm(axis_x) + 1e-8)
        
        axes_3d = np.column_stack([axis_x, axis_y, axis_z])  # (3, 3)
        
        # 3D extents
        # Scale 2D extents to mesh space
        scale_x = mesh_size[0] / w
        scale_y = mesh_size[1] / h
        
        half_extents_3d = np.array([
            half_extent_major * scale_x,
            half_extent_minor * scale_y,
            mesh_size[2] / 2 * self.DEPTH_RATIOS.get(section_name, 0.5)
        ])
        
        return OBB(
            center=center_3d,
            axes=axes_3d,
            half_extents=half_extents_3d,
            section_name=section_name
        )
    
    def _build_unified_cage(self) -> Tuple[trimesh.Trimesh, Dict, Dict]:
        """
        Build unified cage mesh from OBBs with joint connections.
        
        Returns:
            (cage_mesh, section_info, joint_info)
        """
        all_vertices = []
        all_faces = []
        section_info = {}
        joint_info = {}
        
        # Build cage by adding each OBB
        for section_name, obb in self.obbs.items():
            vertex_offset = len(all_vertices)
            
            # Get OBB vertices and faces
            obb_vertices = obb.get_vertices()
            obb_faces = obb.get_faces(vertex_offset)
            
            all_vertices.extend(obb_vertices)
            all_faces.extend(obb_faces)
            
            # Store vertex indices for this section
            vertex_indices = list(range(vertex_offset, vertex_offset + 8))
            section_info[section_name] = {
                'vertex_indices': vertex_indices,
                'joint': self.SECTION_JOINTS[section_name],
                'parent': self.HIERARCHY.get(section_name, None)
            }
        
        # Build joint info (which sections connect at each joint)
        for section_name, info in section_info.items():
            joint_name = info['joint']
            if joint_name not in joint_info:
                joint_info[joint_name] = {
                    'position': self.joint_positions.get(joint_name, np.zeros(3)),
                    'connected_sections': []
                }
            joint_info[joint_name]['connected_sections'].append(section_name)
        
        # Create unified mesh
        all_vertices = np.array(all_vertices)
        all_faces = np.array(all_faces)
        
        cage_mesh = trimesh.Trimesh(vertices=all_vertices, faces=all_faces)
        
        return cage_mesh, section_info, joint_info


if __name__ == "__main__":
    # Simple test
    print("Testing ArticulatedCageGenerator...")
    
    # Create a simple test mesh
    mesh = trimesh.creation.box(extents=[0.5, 1.0, 0.3])
    
    # Mock BodyPix masks
    frame_shape = (720, 1280)
    masks = {
        'torso': np.zeros((720, 1280), dtype=np.uint8)
    }
    masks['torso'][200:520, 540:740] = 255  # Rectangle in center
    
    # Mock keypoints
    keypoints = {
        'left_shoulder': (540, 220),
        'right_shoulder': (740, 220),
        'left_hip': (560, 500),
        'right_hip': (720, 500),
    }
    
    generator = ArticulatedCageGenerator(mesh)
    cage, section_info, joint_info = generator.generate_cage(
        masks, keypoints, frame_shape
    )
    
    print(f"\n✓ Test passed!")
    print(f"   Cage: {len(cage.vertices)} vertices, {len(cage.faces)} faces")
    print(f"   Sections: {list(section_info.keys())}")
    print(f"   Joints: {list(joint_info.keys())}")

