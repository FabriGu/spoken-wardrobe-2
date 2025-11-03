"""
Depth Mesh Generator - Convert OAK-D stereo depth to 3D body surface mesh

This module handles:
1. Capturing aligned RGB + Depth from OAK-D Pro
2. Converting depth map to 3D point cloud
3. Filtering to person region (using BlazePose bounding box)
4. Triangulating point cloud into renderable mesh
5. Segmenting specific body parts (torso, arms, etc.)

Key Innovation: Uses REAL stereo depth (not guessed/estimated)
"""

import numpy as np
import cv2
from scipy.spatial import Delaunay
from typing import Tuple, Optional, Dict, List
import time


class DepthMeshGenerator:
    """Generates 3D body surface mesh from OAK-D stereo depth"""

    def __init__(self, camera_intrinsics: Optional[Dict] = None):
        """
        Initialize depth mesh generator

        Args:
            camera_intrinsics: Optional dict with 'fx', 'fy', 'cx', 'cy'
                             If None, will use default OAK-D values
        """
        # OAK-D Pro default intrinsics (approximate, can be calibrated)
        if camera_intrinsics is None:
            # These are typical values for OAK-D at 640x400 depth resolution
            self.intrinsics = {
                'fx': 440.0,  # Focal length X
                'fy': 440.0,  # Focal length Y
                'cx': 320.0,  # Principal point X (image center)
                'cy': 200.0   # Principal point Y
            }
        else:
            self.intrinsics = camera_intrinsics

        # Depth processing parameters
        self.min_depth_mm = 300    # Minimum valid depth (30cm)
        self.max_depth_mm = 3000   # Maximum valid depth (3m)

        # Mesh decimation (reduce point density for performance)
        self.depth_downsample = 4  # Use every 4th pixel (huge speedup)

        print(f"DepthMeshGenerator initialized")
        print(f"  Intrinsics: fx={self.intrinsics['fx']}, fy={self.intrinsics['fy']}")
        print(f"  Depth range: {self.min_depth_mm}-{self.max_depth_mm}mm")
        print(f"  Downsample: {self.depth_downsample}x")

    def depth_frame_to_pointcloud(self,
                                   depth_frame: np.ndarray,
                                   mask: Optional[np.ndarray] = None) -> np.ndarray:
        """
        Convert depth frame to 3D point cloud (KEEP GRID STRUCTURE)

        Args:
            depth_frame: Depth map (H x W) in millimeters
            mask: Optional binary mask (H x W) - only convert masked pixels

        Returns:
            points_grid: (grid_h, grid_w, 3) array - NaN for invalid points
                   Camera coordinate system: X=right, Y=down, Z=forward
        """
        h, w = depth_frame.shape

        # Downsample depth and mask
        depth_down = depth_frame[::self.depth_downsample, ::self.depth_downsample]

        if mask is not None:
            mask_down = mask[::self.depth_downsample, ::self.depth_downsample]
        else:
            mask_down = np.ones_like(depth_down, dtype=np.uint8) * 255

        grid_h, grid_w = depth_down.shape

        # Create pixel coordinate grids
        y_coords, x_coords = np.mgrid[0:h:self.depth_downsample,
                                       0:w:self.depth_downsample]

        # Validity mask
        valid = (depth_down > self.min_depth_mm) & \
                (depth_down < self.max_depth_mm) & \
                (mask_down > 0)

        # Convert depth to meters
        z = depth_down.astype(np.float32) / 1000.0

        # Back-project to 3D
        x = (x_coords - self.intrinsics['cx']) * z / self.intrinsics['fx']
        y = (y_coords - self.intrinsics['cy']) * z / self.intrinsics['fy']

        # Create points grid
        points_grid = np.stack([x, y, z], axis=-1)

        # Set invalid points to NaN
        points_grid[~valid] = np.nan

        return points_grid

    def create_person_mask_from_bbox(self,
                                      frame_shape: Tuple[int, int],
                                      bbox: Tuple[int, int, int, int],
                                      padding: int = 20) -> np.ndarray:
        """
        Create binary mask from bounding box

        Args:
            frame_shape: (height, width)
            bbox: (x_min, y_min, x_max, y_max) in pixels
            padding: Pixels to expand bbox

        Returns:
            mask: Binary mask (H x W), 255 inside bbox, 0 outside
        """
        h, w = frame_shape
        mask = np.zeros((h, w), dtype=np.uint8)

        x_min, y_min, x_max, y_max = bbox
        x_min = max(0, x_min - padding)
        y_min = max(0, y_min - padding)
        x_max = min(w, x_max + padding)
        y_max = min(h, y_max + padding)

        mask[y_min:y_max, x_min:x_max] = 255

        return mask

    def triangulate_pointcloud_grid(self,
                                     points_grid: np.ndarray,
                                     grid_shape: Tuple[int, int] = None) -> Tuple[np.ndarray, np.ndarray]:
        """
        Triangulate point cloud grid (fast, handles NaN holes)

        Args:
            points_grid: (grid_h, grid_w, 3) grid of points (NaN for invalid)
            grid_shape: Ignored (kept for API compatibility)

        Returns:
            vertices: Nx3 vertex positions
            faces: Mx3 triangle indices
        """
        grid_h, grid_w, _ = points_grid.shape

        # Flatten grid and create vertex index map
        points_flat = points_grid.reshape(-1, 3)
        valid_mask = ~np.isnan(points_flat[:, 0])  # Check X coord for NaN

        # Create mapping: grid_index -> vertex_index
        vertex_indices = np.full(grid_h * grid_w, -1, dtype=np.int32)
        vertex_indices[valid_mask] = np.arange(valid_mask.sum())

        # Extract valid vertices
        vertices = points_flat[valid_mask]

        if len(vertices) < 3:
            raise ValueError(f"Not enough valid vertices: {len(vertices)}")

        # Generate faces from grid topology
        faces = []
        for r in range(grid_h - 1):
            for c in range(grid_w - 1):
                # Get grid indices for this cell's 4 corners
                idx_tl = r * grid_w + c           # top-left
                idx_tr = r * grid_w + (c + 1)     # top-right
                idx_bl = (r + 1) * grid_w + c     # bottom-left
                idx_br = (r + 1) * grid_w + (c + 1)  # bottom-right

                # Get vertex indices (will be -1 if invalid)
                v_tl = vertex_indices[idx_tl]
                v_tr = vertex_indices[idx_tr]
                v_bl = vertex_indices[idx_bl]
                v_br = vertex_indices[idx_br]

                # Triangle 1: top-left, top-right, bottom-left
                if v_tl >= 0 and v_tr >= 0 and v_bl >= 0:
                    faces.append([v_tl, v_tr, v_bl])

                # Triangle 2: top-right, bottom-right, bottom-left
                if v_tr >= 0 and v_br >= 0 and v_bl >= 0:
                    faces.append([v_tr, v_br, v_bl])

        if len(faces) == 0:
            raise ValueError("No valid triangles generated")

        faces = np.array(faces, dtype=np.int32)

        return vertices, faces

    def filter_mesh_by_depth_bounds(self,
                                     vertices: np.ndarray,
                                     faces: np.ndarray,
                                     z_min: float,
                                     z_max: float) -> Tuple[np.ndarray, np.ndarray]:
        """
        Filter mesh to keep only vertices/faces within depth bounds

        Args:
            vertices: Nx3 vertex positions
            faces: Mx3 triangle indices
            z_min: Minimum Z (depth) in meters
            z_max: Maximum Z (depth) in meters

        Returns:
            filtered_vertices: Reduced vertex array
            filtered_faces: Reduced face array with updated indices
        """
        # Find vertices within depth bounds
        valid_mask = (vertices[:, 2] >= z_min) & (vertices[:, 2] <= z_max)
        valid_indices = np.where(valid_mask)[0]

        # Create index mapping (old index -> new index)
        index_map = np.full(len(vertices), -1, dtype=np.int32)
        index_map[valid_indices] = np.arange(len(valid_indices))

        # Filter vertices
        filtered_vertices = vertices[valid_mask]

        # Filter faces (keep only if all 3 vertices are valid)
        valid_faces = []
        for face in faces:
            if all(index_map[face] >= 0):
                # Remap indices
                new_face = [index_map[face[0]],
                           index_map[face[1]],
                           index_map[face[2]]]
                valid_faces.append(new_face)

        filtered_faces = np.array(valid_faces, dtype=np.int32) if valid_faces else np.array([], dtype=np.int32).reshape(0, 3)

        return filtered_vertices, filtered_faces

    def segment_torso_mesh(self,
                          vertices: np.ndarray,
                          faces: np.ndarray,
                          landmarks_world: np.ndarray,
                          landmark_indices: Dict[str, int]) -> Tuple[np.ndarray, np.ndarray]:
        """
        Segment torso region from body mesh using world landmarks

        Args:
            vertices: Nx3 vertex positions (in camera coords)
            faces: Mx3 triangle indices
            landmarks_world: 33x3 BlazePose world landmarks (in world coords, meters)
            landmark_indices: Dict mapping landmark names to indices
                            (e.g., {'LEFT_SHOULDER': 11, ...})

        Returns:
            torso_vertices: Filtered vertices
            torso_faces: Filtered faces

        Note: This is a spatial filtering based on bounding box.
              More sophisticated methods (geodesic distance, etc.) can be added later.
        """
        # Define torso bounding box in world coordinates
        # World coords: origin at mid-hips, Y=up, X=right, Z=back

        left_shoulder = landmarks_world[landmark_indices['LEFT_SHOULDER']]
        right_shoulder = landmarks_world[landmark_indices['RIGHT_SHOULDER']]
        left_hip = landmarks_world[landmark_indices['LEFT_HIP']]
        right_hip = landmarks_world[landmark_indices['RIGHT_HIP']]

        # Compute bounds with some padding
        padding = 0.05  # 5cm padding

        x_min = min(left_shoulder[0], left_hip[0]) - padding
        x_max = max(right_shoulder[0], right_hip[0]) + padding

        y_min = min(left_hip[1], right_hip[1]) - padding
        y_max = max(left_shoulder[1], right_shoulder[1]) + padding

        z_min = min(left_shoulder[2], right_shoulder[2], left_hip[2], right_hip[2]) - padding
        z_max = max(left_shoulder[2], right_shoulder[2], left_hip[2], right_hip[2]) + padding

        # IMPORTANT: vertices are in camera coordinates, landmarks are in world coordinates
        # We need to transform one to match the other
        # For now, we'll use a simple heuristic based on Y (vertical) bounds only
        # TODO: Proper coordinate transform

        # Filter vertices by Y bounds (vertical extent)
        valid_mask = (vertices[:, 1] >= y_min) & (vertices[:, 1] <= y_max)
        valid_indices = np.where(valid_mask)[0]

        # Create index mapping
        index_map = np.full(len(vertices), -1, dtype=np.int32)
        index_map[valid_indices] = np.arange(len(valid_indices))

        # Filter vertices
        torso_vertices = vertices[valid_mask]

        # Filter faces
        torso_faces = []
        for face in faces:
            if all(index_map[face] >= 0):
                new_face = [index_map[face[0]],
                           index_map[face[1]],
                           index_map[face[2]]]
                torso_faces.append(new_face)

        torso_faces = np.array(torso_faces, dtype=np.int32) if torso_faces else np.array([], dtype=np.int32).reshape(0, 3)

        return torso_vertices, torso_faces

    def compute_vertex_colors_from_rgb(self,
                                        vertices: np.ndarray,
                                        rgb_frame: np.ndarray,
                                        intrinsics: Optional[Dict] = None) -> np.ndarray:
        """
        Project 3D vertices back to RGB frame and extract colors

        Args:
            vertices: Nx3 vertex positions (camera coords, meters)
            rgb_frame: RGB image (H x W x 3)
            intrinsics: Optional camera intrinsics (uses self.intrinsics if None)

        Returns:
            colors: Nx3 RGB colors (0-255)
        """
        if intrinsics is None:
            intrinsics = self.intrinsics

        # Project 3D points to 2D pixel coordinates
        x_3d = vertices[:, 0]
        y_3d = vertices[:, 1]
        z_3d = vertices[:, 2]

        # Pinhole projection
        x_2d = (x_3d * intrinsics['fx'] / z_3d + intrinsics['cx']).astype(np.int32)
        y_2d = (y_3d * intrinsics['fy'] / z_3d + intrinsics['cy']).astype(np.int32)

        # Clamp to frame bounds
        h, w = rgb_frame.shape[:2]
        x_2d = np.clip(x_2d, 0, w - 1)
        y_2d = np.clip(y_2d, 0, h - 1)

        # Sample RGB values
        colors = rgb_frame[y_2d, x_2d]

        return colors


class MeshData:
    """Simple mesh container"""
    def __init__(self, vertices: np.ndarray, faces: np.ndarray,
                 colors: Optional[np.ndarray] = None,
                 uv: Optional[np.ndarray] = None):
        self.vertices = vertices  # Nx3
        self.faces = faces        # Mx3
        self.colors = colors      # Nx3 (optional)
        self.uv = uv             # Nx2 (optional)

    def __repr__(self):
        return f"MeshData({len(self.vertices)} vertices, {len(self.faces)} faces)"


def main():
    """Test depth mesh generation with dummy data"""
    print("="*70)
    print("Testing DepthMeshGenerator")
    print("="*70)

    generator = DepthMeshGenerator()

    # Create fake depth frame
    h, w = 400, 640
    depth_frame = np.random.randint(500, 2000, (h, w), dtype=np.uint16)

    # Create fake person mask (center region)
    mask = generator.create_person_mask_from_bbox(
        (h, w),
        bbox=(100, 50, 540, 350),
        padding=20
    )

    print("\nConverting depth to point cloud...")
    start = time.time()
    points = generator.depth_frame_to_pointcloud(depth_frame, mask)
    print(f"✓ Generated {len(points)} points in {time.time()-start:.3f}s")

    print("\nTriangulating point cloud...")
    start = time.time()

    # Calculate grid shape (based on downsampling)
    grid_h = h // generator.depth_downsample
    grid_w = w // generator.depth_downsample

    vertices, faces = generator.triangulate_pointcloud_grid(points, (grid_h, grid_w))
    print(f"✓ Generated {len(vertices)} vertices, {len(faces)} faces in {time.time()-start:.3f}s")

    mesh = MeshData(vertices, faces)
    print(f"\n{mesh}")

    print("\n" + "="*70)
    print("Test complete!")
    print("="*70)


if __name__ == "__main__":
    main()
