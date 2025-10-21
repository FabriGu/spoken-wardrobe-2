"""
TEST SCRIPT 3: Real-time Mesh Warping
======================================
Warps mesh to follow user's body movements using simple
vertex displacement based on keypoint motion.

Run from root: python tests/test_03_mesh_warping.py

Dependencies: trimesh, mediapipe, opencv, numpy, scipy
"""

import cv2
import numpy as np
import trimesh
from pathlib import Path
import pickle
import time
import mediapipe as mp
from scipy.spatial import cKDTree


class MeshWarpingEngine:
    """
    Handles real-time mesh warping based on body keypoint motion.
    Uses simple distance-based vertex displacement for speed.
    """
    
    def __init__(self, mesh_path, calibration_path):
        """
        Initialize warping engine.
        
        Args:
            mesh_path: Path to corrected mesh (.obj)
            calibration_path: Path to calibration data (.pkl)
        """
        print("="*60)
        print("INITIALIZING MESH WARPING ENGINE")
        print("="*60)
        
        # Load mesh
        print(f"Loading mesh: {mesh_path.name}")
        self.mesh = trimesh.load(mesh_path, process=False)
        
        # Store original vertices for reference
        self.vertices_rest = self.mesh.vertices.copy()
        self.vertices_current = self.mesh.vertices.copy()
        
        print(f"✓ Mesh loaded: {len(self.vertices_rest):,} vertices")
        
        # Load calibration
        print(f"Loading calibration: {calibration_path.name}")
        with open(calibration_path, 'rb') as f:
            self.calibration = pickle.load(f)
        
        # Use calibrated mesh keypoints (with user depth)
        self.keypoints_rest = self.calibration['keypoints_3d_mesh_calibrated']
        print(f"✓ Calibration loaded: {len(self.keypoints_rest)} calibrated keypoints")
        
        # Store user reference for debugging
        self.keypoints_user_reference = self.calibration['keypoints_3d_user']
        
        # Initialize MediaPipe
        print("Initializing MediaPipe Pose...")
        self.mp_pose = mp.solutions.pose
        self.pose_detector = self.mp_pose.Pose(
            static_image_mode=False,
            model_complexity=1,
            smooth_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        print("✓ MediaPipe initialized")
        
        # Keypoint indices
        self.KEYPOINT_INDICES = {
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
        
        # Warping parameters - SET BEFORE building influence map
        self.influence_radius = self._compute_influence_radius()
        self.smoothing_iterations = 0
        
        # Build vertex-keypoint relationship (uses self.influence_radius)
        print("\nBuilding vertex influence map...")
        self._build_vertex_influence_map()
        
        # Performance tracking
        self.warp_time_ms = 0
        self.render_time_ms = 0
        
        print("✓ Warping engine initialized\n")
    
    def _compute_influence_radius(self):
        """
        Compute appropriate influence radius based on mesh size.
        Vertices within this distance are affected by keypoint motion.
        """
        bounds = self.mesh.bounds
        mesh_size = np.linalg.norm(bounds[1] - bounds[0])
        
        # Influence radius is ~15% of mesh size
        radius = mesh_size * 0.15
        
        print(f"  Influence radius: {radius:.3f}")
        
        return radius
    
    def _build_vertex_influence_map(self):
        """
        Pre-compute which keypoints influence each vertex.
        Uses spatial tree for efficient nearest-neighbor lookup.
        """
        # Get keypoint positions as array
        keypoint_names = list(self.keypoints_rest.keys())
        keypoint_positions = np.array([self.keypoints_rest[name] for name in keypoint_names])
        
        # Build KD-tree for fast spatial queries
        self.keypoint_tree = cKDTree(keypoint_positions)
        self.keypoint_names_array = keypoint_names
        
        # For each vertex, find nearby keypoints
        self.vertex_influences = []
        
        for i, vertex in enumerate(self.vertices_rest):
            # Find keypoints within influence radius
            indices = self.keypoint_tree.query_ball_point(vertex, r=self.influence_radius)
            
            if len(indices) == 0:
                # No keypoints nearby - use closest keypoint with very low weight
                dist, idx = self.keypoint_tree.query(vertex, k=1)
                influences = {
                    keypoint_names[idx]: np.exp(-dist / self.influence_radius)
                }
            else:
                # Compute weights based on distance (inverse distance weighting)
                influences = {}
                for idx in indices:
                    kp_pos = keypoint_positions[idx]
                    dist = np.linalg.norm(vertex - kp_pos)
                    
                    # Weight falls off with distance (Gaussian-like)
                    weight = np.exp(-(dist ** 2) / (2 * (self.influence_radius / 2) ** 2))
                    influences[keypoint_names[idx]] = weight
                
                # Normalize weights
                total_weight = sum(influences.values())
                if total_weight > 0:
                    influences = {k: v / total_weight for k, v in influences.items()}
            
            self.vertex_influences.append(influences)
        
        print(f"✓ Influence map built for {len(self.vertices_rest)} vertices")
        
        # Debug: Show average influences per vertex
        avg_influences = np.mean([len(infl) for infl in self.vertex_influences])
        print(f"  Average keypoints per vertex: {avg_influences:.1f}")
    
    def extract_live_keypoints(self, frame):
        """
        Extract MediaPipe keypoints from video frame.
        
        Args:
            frame: OpenCV frame (BGR)
            
        Returns:
            Dict of 3D keypoint positions {name: (x, y, z)}
        """
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.pose_detector.process(rgb)
        
        if not results.pose_landmarks:
            return None
        
        h, w = frame.shape[:2]
        keypoints = {}
        
        for name, idx in self.KEYPOINT_INDICES.items():
            landmark = results.pose_landmarks.landmark[idx]
            
            # Get normalized coordinates
            x = landmark.x
            y = landmark.y
            z = landmark.z
            
            keypoints[name] = (x, y, z)
        
        return keypoints
    
    def compute_keypoint_displacements(self, live_keypoints_normalized, frame_shape):
        """
        Compute how much each keypoint has moved from rest pose.
        
        Args:
            live_keypoints_normalized: Current keypoints (normalized 0-1)
            frame_shape: (height, width) for scaling
            
        Returns:
            Dict of displacement vectors {name: (dx, dy, dz)}
        """
        if live_keypoints_normalized is None:
            return None
        
        h, w = frame_shape[:2]
        displacements = {}
        
        # Get reference points for alignment (shoulders)
        if ('left_shoulder' not in self.keypoints_rest or
            'right_shoulder' not in self.keypoints_rest or
            'left_shoulder' not in live_keypoints_normalized or
            'right_shoulder' not in live_keypoints_normalized):
            return None
        
        # Convert rest keypoints to normalized space
        bounds = self.mesh.bounds
        mesh_center = bounds.mean(axis=0)
        mesh_size = bounds[1] - bounds[0]
        
        keypoints_rest_norm = {}
        for name, (x, y, z) in self.keypoints_rest.items():
            x_norm = (x - bounds[0, 0]) / mesh_size[0]
            y_norm = (y - bounds[0, 1]) / mesh_size[1]
            z_norm = (z - bounds[0, 2]) / (mesh_size[2] + 1e-6)
            keypoints_rest_norm[name] = (x_norm, y_norm, z_norm)
        
        # Compute alignment transform (simple translation + scale)
        # Use shoulders as reference
        rest_l_shoulder = np.array(keypoints_rest_norm['left_shoulder'])
        rest_r_shoulder = np.array(keypoints_rest_norm['right_shoulder'])
        rest_shoulder_dist = np.linalg.norm(rest_r_shoulder - rest_l_shoulder)
        rest_center = (rest_l_shoulder + rest_r_shoulder) / 2
        
        live_l_shoulder = np.array(live_keypoints_normalized['left_shoulder'])
        live_r_shoulder = np.array(live_keypoints_normalized['right_shoulder'])
        live_shoulder_dist = np.linalg.norm(live_r_shoulder - live_l_shoulder)
        live_center = (live_l_shoulder + live_r_shoulder) / 2
        
        # Scale factor
        scale = live_shoulder_dist / (rest_shoulder_dist + 1e-6)
        
        # Translation
        translation = live_center - rest_center * scale
        
        # Compute displacements for all keypoints
        for name in self.keypoints_rest.keys():
            if name not in live_keypoints_normalized:
                continue
            
            # Rest position (scaled and translated)
            rest_pos = np.array(keypoints_rest_norm[name])
            rest_transformed = rest_pos * scale + translation
            
            # Current position
            live_pos = np.array(live_keypoints_normalized[name])
            
            # Displacement
            displacement = live_pos - rest_transformed
            
            displacements[name] = displacement
        
        return displacements
    
    def warp_mesh(self, keypoint_displacements):
        """
        Warp mesh vertices based on keypoint displacements.
        Uses pre-computed influence weights for efficient deformation.
        
        Args:
            keypoint_displacements: Dict of displacement vectors
            
        Returns:
            Warped vertex positions
        """
        if keypoint_displacements is None:
            return self.vertices_rest
        
        warp_start = time.time()
        
        # Start with rest pose
        vertices_warped = self.vertices_rest.copy()
        
        # Get mesh bounds for normalization
        bounds = self.mesh.bounds
        mesh_size = bounds[1] - bounds[0]
        
        # Apply displacements
        for i, influences in enumerate(self.vertex_influences):
            # Compute weighted displacement for this vertex
            total_displacement = np.zeros(3)
            
            for kp_name, weight in influences.items():
                if kp_name in keypoint_displacements:
                    disp_norm = keypoint_displacements[kp_name]
                    
                    # Convert normalized displacement back to mesh space
                    disp_mesh = disp_norm * mesh_size
                    
                    # Apply weighted displacement
                    total_displacement += disp_mesh * weight
            
            # Apply displacement to vertex
            vertices_warped[i] += total_displacement
        
        # Smoothing pass to prevent mesh tearing
        if self.smoothing_iterations > 0:
            vertices_warped = self._smooth_vertices(vertices_warped)
        
        self.warp_time_ms = (time.time() - warp_start) * 1000
        
        return vertices_warped
    
    def _smooth_vertices(self, vertices):
        """
        Apply Laplacian smoothing to prevent mesh artifacts.
        
        Args:
            vertices: Current vertex positions
            
        Returns:
            Smoothed vertices
        """
        # Get vertex adjacency from mesh
        # For speed, we use a simplified approach
        smoothed = vertices.copy()
        
        for iteration in range(self.smoothing_iterations):
            new_positions = smoothed.copy()
            
            # For each vertex, average with neighbors
            for i in range(len(vertices)):
                # Find faces containing this vertex
                face_indices = np.where(np.any(self.mesh.faces == i, axis=1))[0]
                
                if len(face_indices) == 0:
                    continue
                
                # Get neighbor vertices
                neighbor_vertices = set()
                for face_idx in face_indices:
                    face = self.mesh.faces[face_idx]
                    neighbor_vertices.update(face[face != i])
                
                if len(neighbor_vertices) == 0:
                    continue
                
                # Average position with neighbors
                neighbor_positions = smoothed[list(neighbor_vertices)]
                avg_position = neighbor_positions.mean(axis=0)
                
                # Blend with current position (preserve overall shape)
                blend_factor = 0.3
                new_positions[i] = (1 - blend_factor) * smoothed[i] + blend_factor * avg_position
            
            smoothed = new_positions
        
        return smoothed
    
    def render_warped_mesh(self, frame_shape, vertices_warped, scale, offset_x, offset_y):
        """
        Render warped mesh to 2D image.
        
        Args:
            frame_shape: (height, width)
            vertices_warped: Warped vertex positions
            scale, offset_x, offset_y: Transformation parameters
            
        Returns:
            Rendered image (RGBA)
        """
        render_start = time.time()
        
        h, w = frame_shape[:2]
        image = np.zeros((h, w, 4), dtype=np.uint8)
        
        # Get bounds
        bounds = self.mesh.bounds
        mesh_width = bounds[1, 0] - bounds[0, 0]
        mesh_height = bounds[1, 1] - bounds[0, 1]
        
        # Project vertices to 2D
        vertices_2d = vertices_warped[:, :2]
        
        # Normalize
        vertices_norm = (vertices_2d - bounds[0, :2]) / np.array([mesh_width, mesh_height])
        
        # Scale and position
        vertices_scaled = vertices_norm * scale
        vertices_centered = vertices_scaled - 0.5
        vertices_final = vertices_centered * min(w, h)
        
        vertices_final[:, 0] += w / 2 + offset_x
        vertices_final[:, 1] += h / 2 + offset_y
        
        # Flip Y
        vertices_final[:, 1] = h - vertices_final[:, 1]
        
        vertices_int = vertices_final.astype(np.int32)
        
        # Draw faces with simple shading
        for face in self.mesh.faces:
            pts = vertices_int[face]
            
            if np.all((pts >= 0) & (pts < np.array([w, h]))):
                # Simple flat shading based on face normal
                # (For speed, we use a constant color)
                cv2.fillPoly(image, [pts], (180, 180, 180, 255))
                cv2.polylines(image, [pts], True, (100, 100, 100, 255), 1)
        
        self.render_time_ms = (time.time() - render_start) * 1000
        
        return image


def main():
    """
    Main warping test loop.
    """
    print("="*60)
    print("TEST SCRIPT 3: REAL-TIME MESH WARPING")
    print("="*60)
    print("\nThis script:")
    print("1. Loads calibrated mesh")
    print("2. Tracks your body movements")
    print("3. Warps mesh to follow your pose")
    print("4. Renders warped mesh in real-time")
    print("\nControls:")
    print("  Q - Quit")
    print("  W - Toggle wireframe mode")
    print("  S - Toggle smoothing")
    print("  + / - - Adjust influence radius")
    print("  1-3 - Set smoothing iterations")
    print("="*60)
    
    # Find calibration
    calib_dir = Path("calibration_data")
    if not calib_dir.exists():
        print(f"\n✗ Directory not found: {calib_dir}")
        print("Run test_01_calibration_keypoints.py first!")
        return
    
    calib_files = sorted(list(calib_dir.glob("*_calibration.pkl")))
    
    if len(calib_files) == 0:
        print(f"\n✗ No calibration files in {calib_dir}")
        return
    
    print(f"\nFound {len(calib_files)} calibrated meshes:")
    for i, calib_file in enumerate(calib_files, 1):
        print(f"  {i}. {calib_file.stem}")
    
    # Select
    choice = input("\nWarp which mesh? (1-N): ").strip()
    
    try:
        idx = int(choice) - 1
        if not (0 <= idx < len(calib_files)):
            idx = 0
    except:
        idx = 0
    
    calib_path = calib_files[idx]
    mesh_path = calib_dir / f"{calib_path.stem.replace('_calibration', '_corrected')}.obj"
    
    if not mesh_path.exists():
        print(f"✗ Mesh not found: {mesh_path}")
        return
    
    print(f"\nLoading: {mesh_path.name}")
    
    # Initialize warping engine
    try:
        warper = MeshWarpingEngine(mesh_path, calib_path)
    except Exception as e:
        print(f"✗ Error initializing warper: {e}")
        import traceback
        traceback.print_exc()
        return
    
    # Initialize camera
    print("\nInitializing camera...")
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    
    if not cap.isOpened():
        print("✗ Could not open camera")
        return
    
    print("✓ Camera initialized")
    
    # Settings
    show_wireframe = False
    use_smoothing = True
    scale = 300.0
    offset_x = 0
    offset_y = 0
    
    # Performance tracking
    fps = 0
    frame_count = 0
    fps_start = time.time()
    
    print("\n" + "="*60)
    print("RUNNING - Move around and watch the mesh warp!")
    print("="*60)
    
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                continue
            
            # Mirror
            frame = cv2.flip(frame, 1)
            h, w = frame.shape[:2]
            
            # Extract live keypoints
            live_keypoints = warper.extract_live_keypoints(frame)
            
            if live_keypoints:
                # Compute keypoint displacements
                displacements = warper.compute_keypoint_displacements(
                    live_keypoints,
                    frame.shape
                )
                
                # Warp mesh
                if displacements:
                    vertices_warped = warper.warp_mesh(displacements)
                else:
                    vertices_warped = warper.vertices_rest
                
                # Render warped mesh
                mesh_image = warper.render_warped_mesh(
                    frame.shape,
                    vertices_warped,
                    scale,
                    offset_x,
                    offset_y
                )
                
                # Composite
                mesh_rgb = mesh_image[:, :, :3]
                mesh_alpha = mesh_image[:, :, 3:] / 255.0 * 0.7
                mesh_alpha_3ch = np.repeat(mesh_alpha, 3, axis=2)
                
                result = (mesh_rgb * mesh_alpha_3ch + 
                         frame * (1 - mesh_alpha_3ch)).astype(np.uint8)
                
            else:
                result = frame
            
            # Add info overlay
            status = "Warping active" if live_keypoints else "No pose detected"
            color = (0, 255, 0) if live_keypoints else (0, 0, 255)
            
            cv2.putText(result, f"FPS: {fps:.1f}", (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            
            cv2.putText(result, status, (10, 70),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
            
            if live_keypoints:
                total_time = warper.warp_time_ms + warper.render_time_ms
                cv2.putText(result, f"Warp: {warper.warp_time_ms:.1f}ms", (10, 100),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
                
                cv2.putText(result, f"Render: {warper.render_time_ms:.1f}ms", (10, 130),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
                
                cv2.putText(result, f"Total: {total_time:.1f}ms", (10, 160),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
            
            smoothing_text = f"Smoothing: {warper.smoothing_iterations} iters"
            cv2.putText(result, smoothing_text, (10, 190),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
            
            # Display
            cv2.imshow("Mesh Warping Test", result)
            
            # Update FPS
            frame_count += 1
            elapsed = time.time() - fps_start
            if elapsed >= 1.0:
                fps = frame_count / elapsed
                frame_count = 0
                fps_start = time.time()
            
            # Handle keys
            key = cv2.waitKey(1) & 0xFF
            
            if key == ord('q') or key == ord('Q'):
                break
            
            elif key == ord('w') or key == ord('W'):
                show_wireframe = not show_wireframe
            
            elif key == ord('s') or key == ord('S'):
                use_smoothing = not use_smoothing
                warper.smoothing_iterations = 2 if use_smoothing else 0
            
            elif key == ord('+') or key == ord('='):
                warper.influence_radius *= 1.1
                print(f"Influence radius: {warper.influence_radius:.3f}")
                warper._build_vertex_influence_map()
            
            elif key == ord('-') or key == ord('_'):
                warper.influence_radius *= 0.9
                print(f"Influence radius: {warper.influence_radius:.3f}")
                warper._build_vertex_influence_map()
            
            elif key == ord('1'):
                warper.smoothing_iterations = 1
            
            elif key == ord('2'):
                warper.smoothing_iterations = 2
            
            elif key == ord('3'):
                warper.smoothing_iterations = 3
    
    except KeyboardInterrupt:
        print("\nInterrupted")
    
    finally:
        cap.release()
        cv2.destroyAllWindows()
        print("\n" + "="*60)
        print("TEST COMPLETE")
        print("="*60)
        print("\nEvaluation checklist:")
        print("✓ Does mesh warp follow your movements?")
        print("✓ Is FPS acceptable (aim for 15-25)?")
        print("✓ Are there mesh tears or artifacts?")
        print("✓ Does smoothing help or hurt?")
        print("\nIf performance is good, integrate into main application!")


if __name__ == "__main__":
    main()