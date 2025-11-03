"""
Billboard Texture Overlay - Paper-like clothing drape

This creates a textured plane that follows the torso contour
like paper stuck to the body (as shown in user's prototype photos).

Key features:
- Single textured mesh (not rigid quads)
- Follows body center line
- Smooth edge warping at shoulders/arms
- Depth layering for 3D effect
"""

import numpy as np
from typing import Tuple, Optional


class BillboardOverlay:
    """Creates a paper-like billboard mesh that drapes on the body"""

    def __init__(self, mesh_resolution: int = 20):
        """
        Initialize billboard overlay generator

        Args:
            mesh_resolution: Grid density (higher = smoother, slower)
        """
        self.mesh_resolution = mesh_resolution
        print(f"BillboardOverlay initialized (resolution: {mesh_resolution}x{mesh_resolution})")

    def create_body_billboard(self,
                              landmarks: np.ndarray,
                              landmark_indices: dict) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Create a textured mesh plane that follows the torso

        Args:
            landmarks: 33x3 MediaPipe landmarks (normalized x,y,z)
            landmark_indices: Dict mapping landmark names to indices

        Returns:
            vertices: Nx3 3D positions in camera space
            faces: Mx3 triangle indices
            uvs: Nx2 texture coordinates [0,1]
        """
        # Key body points
        l_shoulder = landmarks[landmark_indices['LEFT_SHOULDER']]
        r_shoulder = landmarks[landmark_indices['RIGHT_SHOULDER']]
        l_hip = landmarks[landmark_indices['LEFT_HIP']]
        r_hip = landmarks[landmark_indices['RIGHT_HIP']]
        l_elbow = landmarks[landmark_indices['LEFT_ELBOW']]
        r_elbow = landmarks[landmark_indices['RIGHT_ELBOW']]

        # Torso center line
        shoulder_center = (l_shoulder + r_shoulder) / 2
        hip_center = (l_hip + r_hip) / 2

        # Torso dimensions
        shoulder_width = np.linalg.norm(r_shoulder - l_shoulder)
        torso_height = np.linalg.norm(hip_center - shoulder_center)

        # Extend slightly beyond body for natural drape
        width_extension = 1.3  # 30% wider
        height_extension = 1.2  # 20% taller

        billboard_width = shoulder_width * width_extension
        billboard_height = torso_height * height_extension

        # Create grid mesh
        res = self.mesh_resolution
        vertices = []
        uvs = []

        for row in range(res):
            for col in range(res):
                # Normalized grid position [0, 1]
                u = col / (res - 1)
                v = row / (res - 1)

                # Map to billboard space
                # u: 0=left, 1=right
                # v: 0=top (shoulders), 1=bottom (hips)

                # Horizontal position (left to right)
                h_offset = (u - 0.5) * billboard_width

                # Vertical position (shoulder to hip)
                vertical_blend = v

                # Center line point (shoulder → hip)
                center_point = shoulder_center * (1 - vertical_blend) + hip_center * vertical_blend

                # Create 3D position
                # Use torso orientation for natural alignment
                torso_axis = hip_center - shoulder_center
                torso_axis = torso_axis / (np.linalg.norm(torso_axis) + 1e-8)

                # Perpendicular axis (left-right)
                right_axis = r_shoulder - l_shoulder
                right_axis = right_axis / (np.linalg.norm(right_axis) + 1e-8)

                # Position = center + horizontal offset
                vertex_2d = center_point + right_axis * h_offset

                # Add depth (z) from MediaPipe landmarks
                # Use average depth of nearby landmarks for smooth surface
                if v < 0.3:  # Shoulder region
                    z_ref = shoulder_center[2]
                elif v > 0.7:  # Hip region
                    z_ref = hip_center[2]
                else:  # Mid-torso
                    z_ref = (shoulder_center[2] + hip_center[2]) / 2

                # Apply subtle warping at edges for arm contours
                edge_warp = 0
                if u < 0.3:  # Left edge near left arm
                    # Pull toward left elbow
                    warp_strength = (0.3 - u) / 0.3  # 1.0 at edge, 0.0 at u=0.3
                    edge_warp = (l_elbow - center_point) * warp_strength * 0.3
                elif u > 0.7:  # Right edge near right arm
                    # Pull toward right elbow
                    warp_strength = (u - 0.7) / 0.3
                    edge_warp = (r_elbow - center_point) * warp_strength * 0.3

                # Final vertex position (in normalized coords)
                vertex_norm = vertex_2d + edge_warp
                vertex_3d = np.array([vertex_norm[0], vertex_norm[1], z_ref])

                vertices.append(vertex_3d)
                uvs.append([u, v])

        vertices = np.array(vertices, dtype=np.float32)
        uvs = np.array(uvs, dtype=np.float32)

        # Create triangle faces from grid
        faces = []
        for row in range(res - 1):
            for col in range(res - 1):
                # Grid indices
                tl = row * res + col
                tr = row * res + (col + 1)
                bl = (row + 1) * res + col
                br = (row + 1) * res + (col + 1)

                # Two triangles per quad
                faces.append([tl, tr, bl])
                faces.append([tr, br, bl])

        faces = np.array(faces, dtype=np.int32)

        return vertices, faces, uvs

    def convert_to_camera_space(self,
                                 vertices_normalized: np.ndarray,
                                 frame_shape: Tuple[int, int],
                                 camera_intrinsics: dict,
                                 avg_depth_m: float = 1.5) -> np.ndarray:
        """
        Convert normalized MediaPipe coordinates to camera 3D space

        Args:
            vertices_normalized: Nx3 normalized coords (x,y,z in [0,1])
            frame_shape: (height, width) of frame
            camera_intrinsics: Camera parameters (fx, fy, cx, cy)
            avg_depth_m: Average depth in meters for z-scale

        Returns:
            vertices_3d: Nx3 camera space coordinates (meters)
        """
        h, w = frame_shape
        fx = camera_intrinsics['fx']
        fy = camera_intrinsics['fy']
        cx = camera_intrinsics['cx']
        cy = camera_intrinsics['cy']

        vertices_3d = vertices_normalized.copy()

        for i in range(len(vertices_3d)):
            x_norm, y_norm, z_norm = vertices_normalized[i]

            # Convert to pixel coordinates
            x_px = x_norm * w
            y_px = y_norm * h

            # Depth from normalized z (scale to realistic range)
            z_m = avg_depth_m + (z_norm - 0.5) * 0.5  # ±25cm around avg depth

            # Back-project to 3D camera space
            X = (x_px - cx) * z_m / fx
            Y = (y_px - cy) * z_m / fy
            Z = z_m

            vertices_3d[i] = [X, Y, Z]

        return vertices_3d


def main():
    """Test billboard overlay generation"""
    print("="*70)
    print("Testing Billboard Overlay")
    print("="*70)

    # Create fake MediaPipe landmarks (T-pose)
    landmarks = np.array([
        [0.5, 0.3, 0.0],  # Nose
        [0.5, 0.3, 0.0],  # Left eye inner
        [0.5, 0.3, 0.0],  # Left eye
        [0.5, 0.3, 0.0],  # Left eye outer
        [0.5, 0.3, 0.0],  # Right eye inner
        [0.5, 0.3, 0.0],  # Right eye
        [0.5, 0.3, 0.0],  # Right eye outer
        [0.5, 0.3, 0.0],  # Left ear
        [0.5, 0.3, 0.0],  # Right ear
        [0.5, 0.3, 0.0],  # Mouth left
        [0.5, 0.3, 0.0],  # Mouth right
        [0.3, 0.4, 0.0],  # Left shoulder
        [0.7, 0.4, 0.0],  # Right shoulder
        [0.2, 0.5, 0.0],  # Left elbow
        [0.8, 0.5, 0.0],  # Right elbow
        [0.1, 0.6, 0.0],  # Left wrist
        [0.9, 0.6, 0.0],  # Right wrist
        [0.1, 0.6, 0.0],  # Left pinky
        [0.9, 0.6, 0.0],  # Right pinky
        [0.1, 0.6, 0.0],  # Left index
        [0.9, 0.6, 0.0],  # Right index
        [0.1, 0.6, 0.0],  # Left thumb
        [0.9, 0.6, 0.0],  # Right thumb
        [0.35, 0.65, 0.0],  # Left hip
        [0.65, 0.65, 0.0],  # Right hip
        [0.35, 0.8, 0.0],  # Left knee
        [0.65, 0.8, 0.0],  # Right knee
    ], dtype=np.float32)

    # Pad to 33 landmarks
    landmarks = np.vstack([landmarks, np.zeros((33 - len(landmarks), 3))])

    landmark_indices = {
        'LEFT_SHOULDER': 11,
        'RIGHT_SHOULDER': 12,
        'LEFT_ELBOW': 13,
        'RIGHT_ELBOW': 14,
        'LEFT_HIP': 23,
        'RIGHT_HIP': 24,
    }

    billboard = BillboardOverlay(mesh_resolution=15)

    vertices, faces, uvs = billboard.create_body_billboard(landmarks, landmark_indices)

    print(f"\n✓ Generated billboard mesh:")
    print(f"  Vertices: {len(vertices)}")
    print(f"  Faces: {len(faces)}")
    print(f"  UV coords: {len(uvs)}")
    print(f"  UV range: u=[{uvs[:, 0].min():.2f}, {uvs[:, 0].max():.2f}], "
          f"v=[{uvs[:, 1].min():.2f}, {uvs[:, 1].max():.2f}]")

    # Convert to camera space
    camera_intrinsics = {
        'fx': 573.0,
        'fy': 572.8,
        'cx': 320.0,
        'cy': 200.0
    }

    vertices_3d = billboard.convert_to_camera_space(
        vertices, (400, 640), camera_intrinsics
    )

    print(f"\n✓ Converted to camera space:")
    print(f"  X range: [{vertices_3d[:, 0].min():.3f}, {vertices_3d[:, 0].max():.3f}] m")
    print(f"  Y range: [{vertices_3d[:, 1].min():.3f}, {vertices_3d[:, 1].max():.3f}] m")
    print(f"  Z range: [{vertices_3d[:, 2].min():.3f}, {vertices_3d[:, 2].max():.3f}] m")

    print("\n" + "="*70)
    print("Test complete!")
    print("="*70)


if __name__ == "__main__":
    main()
