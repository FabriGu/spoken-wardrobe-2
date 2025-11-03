"""
Texture Mapper - Apply 2D clothing texture to 3D body mesh

This module handles UV mapping of AI-generated clothing onto depth-based body meshes.

Two projection modes:
1. Planar - Simple front-facing projection (fast, good for standing poses)
2. Cylindrical - Wraps around torso (better for rotation)
"""

import numpy as np
from typing import Tuple, Optional
import cv2
from PIL import Image


class TextureMapper:
    """Maps 2D clothing textures to 3D body mesh using UV coordinates"""

    def __init__(self, projection_mode: str = 'planar'):
        """
        Initialize texture mapper

        Args:
            projection_mode: 'planar' or 'cylindrical'
        """
        self.projection_mode = projection_mode
        print(f"TextureMapper initialized (mode: {projection_mode})")

    def compute_planar_uv(self,
                          vertices: np.ndarray,
                          landmarks_world: np.ndarray,
                          landmark_indices: dict) -> np.ndarray:
        """
        Compute UV coordinates using planar projection

        Projects vertices onto frontal plane defined by shoulders.

        Args:
            vertices: Nx3 vertex positions (camera coords, meters)
            landmarks_world: 33x3 BlazePose world landmarks
            landmark_indices: Dict mapping names to indices

        Returns:
            uv: Nx2 UV coordinates in [0, 1]
        """
        # Define frontal plane from shoulders
        left_shoulder = landmarks_world[landmark_indices['LEFT_SHOULDER']]
        right_shoulder = landmarks_world[landmark_indices['RIGHT_SHOULDER']]
        left_hip = landmarks_world[landmark_indices['LEFT_HIP']]
        right_hip = landmarks_world[landmark_indices['RIGHT_HIP']]

        # Plane origin (center of torso)
        plane_origin = (left_shoulder + right_shoulder + left_hip + right_hip) / 4

        # Plane axes
        x_axis = right_shoulder - left_shoulder
        x_axis = x_axis / (np.linalg.norm(x_axis) + 1e-8)

        y_axis = left_hip - left_shoulder
        y_axis = y_axis / (np.linalg.norm(y_axis) + 1e-8)

        # Bounding box in plane coordinates
        x_min = np.dot(left_shoulder - plane_origin, x_axis)
        x_max = np.dot(right_shoulder - plane_origin, x_axis)
        y_min = np.dot(left_shoulder - plane_origin, y_axis)
        y_max = np.dot(left_hip - plane_origin, y_axis)

        # Add padding
        padding = 0.05  # 5cm
        x_min -= padding
        x_max += padding
        y_min -= padding
        y_max += padding

        # Project vertices onto plane
        uv = np.zeros((len(vertices), 2), dtype=np.float32)

        for i, vertex in enumerate(vertices):
            # Get local coordinates
            local = vertex - plane_origin
            x_local = np.dot(local, x_axis)
            y_local = np.dot(local, y_axis)

            # Map to UV [0, 1]
            u = (x_local - x_min) / (x_max - x_min + 1e-8)
            v = (y_local - y_min) / (y_max - y_min + 1e-8)

            # Clamp to [0, 1]
            u = np.clip(u, 0.0, 1.0)
            v = np.clip(v, 0.0, 1.0)

            uv[i] = [u, v]

        return uv

    def compute_cylindrical_uv(self,
                               vertices: np.ndarray,
                               landmarks_world: np.ndarray,
                               landmark_indices: dict) -> np.ndarray:
        """
        Compute UV coordinates using cylindrical projection

        Wraps texture around torso like a cylinder.

        Args:
            vertices: Nx3 vertex positions
            landmarks_world: 33x3 landmarks
            landmark_indices: Dict of indices

        Returns:
            uv: Nx2 UV coordinates
        """
        # Cylinder axis (spine direction)
        left_shoulder = landmarks_world[landmark_indices['LEFT_SHOULDER']]
        right_shoulder = landmarks_world[landmark_indices['RIGHT_SHOULDER']]
        left_hip = landmarks_world[landmark_indices['LEFT_HIP']]
        right_hip = landmarks_world[landmark_indices['RIGHT_HIP']]

        mid_shoulder = (left_shoulder + right_shoulder) / 2
        mid_hip = (left_hip + right_hip) / 2

        # Cylinder center line
        cylinder_axis = mid_hip - mid_shoulder
        cylinder_axis = cylinder_axis / (np.linalg.norm(cylinder_axis) + 1e-8)

        # Cylinder center
        cylinder_center = (mid_shoulder + mid_hip) / 2

        uv = np.zeros((len(vertices), 2), dtype=np.float32)

        for i, vertex in enumerate(vertices):
            # Get position relative to cylinder center
            relative = vertex - cylinder_center

            # Project onto cylinder axis to get height (V coordinate)
            height = np.dot(relative, cylinder_axis)

            # Get radial component (perpendicular to axis)
            radial = relative - height * cylinder_axis

            # Compute angle around axis (U coordinate)
            # Use atan2 for full 360 degree range
            angle = np.arctan2(radial[0], radial[2])  # X and Z for horizontal angle

            # Map angle to [0, 1]
            u = (angle + np.pi) / (2 * np.pi)

            # Map height to [0, 1]
            height_min = -0.5  # Approximate torso height range
            height_max = 0.5
            v = (height - height_min) / (height_max - height_min + 1e-8)
            v = np.clip(v, 0.0, 1.0)

            uv[i] = [u, v]

        return uv

    def apply_texture_to_mesh(self,
                              vertices: np.ndarray,
                              faces: np.ndarray,
                              uv: np.ndarray,
                              texture_image: np.ndarray) -> np.ndarray:
        """
        Sample texture colors at UV coordinates

        Args:
            vertices: Nx3 positions
            faces: Mx3 triangle indices
            uv: Nx2 UV coordinates
            texture_image: Texture image (H x W x 3 or 4, RGB or RGBA)

        Returns:
            vertex_colors: Nx3 RGB colors (0-255)
        """
        h, w = texture_image.shape[:2]

        # Sample texture at UV coordinates
        vertex_colors = np.zeros((len(vertices), 3), dtype=np.uint8)

        for i, (u, v) in enumerate(uv):
            # Convert UV to pixel coordinates
            x = int(u * (w - 1))
            y = int(v * (h - 1))

            # Clamp
            x = np.clip(x, 0, w - 1)
            y = np.clip(y, 0, h - 1)

            # Sample color
            color = texture_image[y, x]

            # Handle RGBA (ignore alpha for now)
            if len(color) == 4:
                color = color[:3]

            vertex_colors[i] = color

        return vertex_colors

    def load_texture_image(self, image_path: str) -> np.ndarray:
        """
        Load texture image from file

        Args:
            image_path: Path to PNG/JPG image

        Returns:
            texture: Numpy array (H x W x 3 or 4)
        """
        # Load with PIL to handle RGBA properly
        img = Image.open(image_path)

        # Convert to numpy
        texture = np.array(img)

        # If grayscale, convert to RGB
        if len(texture.shape) == 2:
            texture = cv2.cvtColor(texture, cv2.COLOR_GRAY2RGB)

        # If RGBA, keep all 4 channels
        # If RGB, keep 3 channels

        print(f"Loaded texture: {texture.shape} from {image_path}")

        return texture


def main():
    """Test texture mapping"""
    print("="*70)
    print("Testing TextureMapper")
    print("="*70)

    mapper = TextureMapper(projection_mode='planar')

    # Create fake data
    vertices = np.random.rand(100, 3).astype(np.float32)
    landmarks = np.random.rand(33, 3).astype(np.float32)

    landmark_indices = {
        'LEFT_SHOULDER': 11,
        'RIGHT_SHOULDER': 12,
        'LEFT_HIP': 23,
        'RIGHT_HIP': 24
    }

    print("\nComputing planar UV...")
    uv = mapper.compute_planar_uv(vertices, landmarks, landmark_indices)
    print(f"✓ Generated UV coordinates: {uv.shape}")
    print(f"  UV range: [{uv.min():.3f}, {uv.max():.3f}]")

    print("\n" + "="*70)
    print("Test complete!")
    print("="*70)


if __name__ == "__main__":
    main()
