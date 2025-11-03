"""
Body-Part Warping - Map clothing to body segments

This creates separate mesh quads for each body part (torso, arms)
and warps them independently based on keypoint correspondence.

Two approaches:
- Option A: Reference pose → Current pose warping
- Option C: BodyPix segmentation-based warping
"""

import numpy as np
import cv2
from typing import Tuple, Dict, List, Optional
from dataclasses import dataclass


@dataclass
class BodyPartMesh:
    """Mesh data for one body part"""
    name: str
    vertices: np.ndarray  # Nx3 positions
    faces: np.ndarray     # Mx3 triangle indices
    uvs: np.ndarray       # Nx2 texture coordinates
    colors: np.ndarray    # Nx3 RGB colors


class BodyPartWarper:
    """Warp clothing texture by body parts"""

    def __init__(self, quad_resolution: int = 15):
        """
        Initialize body part warper

        Args:
            quad_resolution: Grid density per body part
        """
        self.quad_res = quad_resolution
        print(f"BodyPartWarper initialized (resolution: {quad_resolution}x{quad_resolution} per part)")

    def create_body_part_quads(self,
                                current_landmarks: np.ndarray,
                                reference_landmarks: np.ndarray,
                                clothing_image: np.ndarray,
                                mask_image: np.ndarray,
                                landmark_indices: dict,
                                mode: str = 'reference',
                                debug: bool = False) -> List[BodyPartMesh]:
        """
        Create textured mesh quads for each body part

        Args:
            current_landmarks: 33x3 current pose (normalized x,y,z)
            reference_landmarks: 33x3 reference pose (normalized x,y,z from metadata)
            clothing_image: Generated clothing texture (H x W x 3/4)
            mask_image: Segmentation mask (H x W) - 255 for clothing, 0 for background
            landmark_indices: Dict mapping landmark names to indices
            mode: 'reference' (Option A) or 'bodypix' (Option C)
            debug: Print debug information

        Returns:
            List of BodyPartMesh objects (one per body part)
        """
        if debug:
            print(f"  [WARPER] Current landmarks range: x=[{current_landmarks[:, 0].min():.3f}, {current_landmarks[:, 0].max():.3f}], "
                  f"y=[{current_landmarks[:, 1].min():.3f}, {current_landmarks[:, 1].max():.3f}]")
            print(f"  [WARPER] Reference landmarks range: x=[{reference_landmarks[:, 0].min():.3f}, {reference_landmarks[:, 0].max():.3f}], "
                  f"y=[{reference_landmarks[:, 1].min():.3f}, {reference_landmarks[:, 1].max():.3f}]")
            print(f"  [WARPER] Clothing image: {clothing_image.shape}, Mask: {mask_image.shape}")

        body_parts = []

        # Define body part regions by keypoints
        part_definitions = {
            'torso': {
                'corners': ['LEFT_SHOULDER', 'RIGHT_SHOULDER', 'RIGHT_HIP', 'LEFT_HIP'],
                'order': [0, 1, 2, 3]  # Quad winding order
            },
            'left_upper_arm': {
                'corners': ['LEFT_SHOULDER', 'LEFT_SHOULDER', 'LEFT_ELBOW', 'LEFT_ELBOW'],
                'order': [0, 1, 2, 3],
                'width_offset': 0.05  # Arm width in normalized coords
            },
            'right_upper_arm': {
                'corners': ['RIGHT_SHOULDER', 'RIGHT_SHOULDER', 'RIGHT_ELBOW', 'RIGHT_ELBOW'],
                'order': [0, 1, 2, 3],
                'width_offset': 0.05
            },
        }

        for part_name, definition in part_definitions.items():
            try:
                if debug:
                    print(f"  [WARPER] Creating mesh for: {part_name}")

                mesh = self._create_single_part_mesh(
                    part_name,
                    definition,
                    current_landmarks,
                    reference_landmarks,
                    clothing_image,
                    mask_image,
                    landmark_indices,
                    mode,
                    debug=debug
                )
                if mesh is not None:
                    body_parts.append(mesh)
                elif debug:
                    print(f"  [WARPER] {part_name} returned None (no visible pixels)")
            except Exception as e:
                print(f"Warning: Failed to create mesh for {part_name}: {e}")
                if debug:
                    import traceback
                    traceback.print_exc()

        return body_parts

    def _create_single_part_mesh(self,
                                  part_name: str,
                                  definition: dict,
                                  current_landmarks: np.ndarray,
                                  reference_landmarks: np.ndarray,
                                  clothing_image: np.ndarray,
                                  mask_image: np.ndarray,
                                  landmark_indices: dict,
                                  mode: str,
                                  debug: bool = False) -> Optional[BodyPartMesh]:
        """Create mesh for one body part"""

        # Get corner keypoints for CURRENT pose
        corner_names = definition['corners']
        current_corners = []
        for name in corner_names:
            idx = landmark_indices[name]
            current_corners.append(current_landmarks[idx])
        current_corners = np.array(current_corners)

        # Get corner keypoints for REFERENCE pose (where clothing was generated)
        reference_corners = []
        for name in corner_names:
            idx = landmark_indices[name]
            reference_corners.append(reference_landmarks[idx])
        reference_corners = np.array(reference_corners)

        # Handle arm width (arms are lines, not quads in keypoints)
        if 'width_offset' in definition:
            width = definition['width_offset']
            # Create perpendicular offset for arm width
            # Current pose
            arm_vec_curr = current_corners[2] - current_corners[0]  # Shoulder to elbow
            perp_curr = np.array([-arm_vec_curr[1], arm_vec_curr[0], 0]) * width
            current_corners[1] = current_corners[0] + perp_curr
            current_corners[3] = current_corners[2] + perp_curr

            # Reference pose
            arm_vec_ref = reference_corners[2] - reference_corners[0]
            perp_ref = np.array([-arm_vec_ref[1], arm_vec_ref[0], 0]) * width
            reference_corners[1] = reference_corners[0] + perp_ref
            reference_corners[3] = reference_corners[2] + perp_ref

        # Create mesh grid
        res = self.quad_res
        vertices_2d = []
        uvs = []

        for row in range(res):
            for col in range(res):
                # Normalized position in quad [0, 1]
                u = col / (res - 1)
                v = row / (res - 1)

                # Bilinear interpolation for CURRENT pose position
                top = current_corners[0] * (1 - u) + current_corners[1] * u
                bottom = current_corners[3] * (1 - u) + current_corners[2] * u
                vertex_current = top * (1 - v) + bottom * v

                vertices_2d.append(vertex_current)

                # Bilinear interpolation for REFERENCE pose position
                # This tells us where in the ORIGINAL IMAGE this vertex should sample from
                top_ref = reference_corners[0] * (1 - u) + reference_corners[1] * u
                bottom_ref = reference_corners[3] * (1 - u) + reference_corners[2] * u
                vertex_ref = top_ref * (1 - v) + bottom_ref * v

                # Reference landmarks are NORMALIZED [0, 1] (saved from metadata)
                # These map directly to texture coordinates!
                tex_x = vertex_ref[0]
                tex_y = vertex_ref[1]

                uvs.append([tex_x, tex_y])

        vertices_2d = np.array(vertices_2d, dtype=np.float32)
        uvs = np.array(uvs, dtype=np.float32)

        # Create faces
        faces = []
        for row in range(res - 1):
            for col in range(res - 1):
                tl = row * res + col
                tr = row * res + (col + 1)
                bl = (row + 1) * res + col
                br = (row + 1) * res + (col + 1)

                faces.append([tl, tr, bl])
                faces.append([tr, br, bl])

        faces = np.array(faces, dtype=np.int32)

        # Sample texture colors at UV coordinates
        colors = self._sample_texture_masked(uvs, clothing_image, mask_image, debug=debug)

        # Check if this body part is actually visible (has non-background pixels)
        if np.all(colors == 0):
            return None  # Skip invisible parts

        return BodyPartMesh(
            name=part_name,
            vertices=vertices_2d,
            faces=faces,
            uvs=uvs,
            colors=colors
        )

    def _sample_texture_masked(self,
                                uvs: np.ndarray,
                                texture: np.ndarray,
                                mask: np.ndarray,
                                debug: bool = False) -> np.ndarray:
        """
        Sample texture at UV coordinates, respecting mask

        Args:
            uvs: Nx2 UV coordinates (normalized [0, 1])
            texture: Clothing image (H x W x 3/4)
            mask: Segmentation mask (H x W), 255=clothing, 0=background
            debug: Print debug info

        Returns:
            colors: Nx3 RGB colors (0-255), black where mask is 0
        """
        h, w = texture.shape[:2]
        colors = np.zeros((len(uvs), 3), dtype=np.uint8)

        if debug:
            print(f"    [TEXTURE SAMPLE] Texture: {texture.shape}, Mask: {mask.shape}")
            print(f"    [TEXTURE SAMPLE] UV range: u=[{uvs[:, 0].min():.3f}, {uvs[:, 0].max():.3f}], "
                  f"v=[{uvs[:, 1].min():.3f}, {uvs[:, 1].max():.3f}]")

        valid_count = 0
        for i, (u, v) in enumerate(uvs):
            # Clamp UV to [0, 1]
            u = np.clip(u, 0, 1)
            v = np.clip(v, 0, 1)

            # Convert to pixel coordinates
            x = int(u * (w - 1))
            y = int(v * (h - 1))

            # Check mask
            if mask[y, x] > 0:
                color = texture[y, x]
                if len(color) == 4:
                    color = color[:3]
                colors[i] = color
                valid_count += 1
            # else: leave as black (background)

        if debug:
            print(f"    [TEXTURE SAMPLE] Valid pixels: {valid_count}/{len(uvs)} ({valid_count/len(uvs)*100:.1f}%)")

        return colors

    def convert_to_camera_space(self,
                                 meshes: List[BodyPartMesh],
                                 frame_shape: Tuple[int, int],
                                 camera_intrinsics: dict,
                                 avg_depth_m: float = 1.5) -> List[BodyPartMesh]:
        """
        Convert normalized coordinates to 3D camera space

        Args:
            meshes: List of BodyPartMesh with normalized vertices
            frame_shape: (height, width)
            camera_intrinsics: Camera parameters
            avg_depth_m: Average depth in meters

        Returns:
            List of BodyPartMesh with 3D vertices
        """
        h, w = frame_shape
        fx = camera_intrinsics['fx']
        fy = camera_intrinsics['fy']
        cx = camera_intrinsics['cx']
        cy = camera_intrinsics['cy']

        meshes_3d = []

        for mesh in meshes:
            vertices_3d = np.zeros_like(mesh.vertices)

            for i, vertex in enumerate(mesh.vertices):
                x_norm, y_norm, z_norm = vertex

                # Convert to pixel coords
                x_px = x_norm * w
                y_px = y_norm * h

                # Depth from z-coordinate
                z_m = avg_depth_m + (z_norm - 0.5) * 0.5

                # Back-project
                X = (x_px - cx) * z_m / fx
                Y = (y_px - cy) * z_m / fy
                Z = z_m

                vertices_3d[i] = [X, Y, Z]

            mesh_3d = BodyPartMesh(
                name=mesh.name,
                vertices=vertices_3d,
                faces=mesh.faces,
                uvs=mesh.uvs,
                colors=mesh.colors
            )
            meshes_3d.append(mesh_3d)

        return meshes_3d


def main():
    """Test body part warping"""
    print("="*70)
    print("Testing Body Part Warping")
    print("="*70)

    # Create fake landmarks
    current_landmarks = np.random.rand(33, 3).astype(np.float32)
    reference_landmarks = np.random.rand(33, 3).astype(np.float32)

    landmark_indices = {
        'LEFT_SHOULDER': 11,
        'RIGHT_SHOULDER': 12,
        'LEFT_ELBOW': 13,
        'RIGHT_ELBOW': 14,
        'LEFT_HIP': 23,
        'RIGHT_HIP': 24,
    }

    # Create fake texture and mask
    clothing_image = np.random.randint(0, 255, (648, 1152, 3), dtype=np.uint8)
    mask_image = np.ones((648, 1152), dtype=np.uint8) * 255

    warper = BodyPartWarper(quad_resolution=10)

    meshes = warper.create_body_part_quads(
        current_landmarks,
        reference_landmarks,
        clothing_image,
        mask_image,
        landmark_indices,
        mode='reference'
    )

    print(f"\n✓ Generated {len(meshes)} body part meshes:")
    for mesh in meshes:
        print(f"  {mesh.name}: {len(mesh.vertices)} vertices, {len(mesh.faces)} faces")

    print("\n" + "="*70)
    print("Test complete!")
    print("="*70)


if __name__ == "__main__":
    main()
