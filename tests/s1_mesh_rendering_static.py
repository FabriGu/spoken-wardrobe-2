"""
TEST SCRIPT 2: Mesh Rendering & Overlay
========================================
Loads calibrated mesh with keypoints, renders it, and overlays
on live camera feed with alignment based on MediaPipe tracking.

Run from root: python tests/test_02_mesh_rendering.py

Dependencies: open3d, trimesh, mediapipe, opencv, numpy
"""

import cv2
import numpy as np
import trimesh
import open3d as o3d
from pathlib import Path
import pickle
import time
import mediapipe as mp


class MeshRenderer:
    """
    Handles 3D mesh rendering and compositing onto video frame.
    """
    
    def __init__(self, mesh_path, calibration_path):
        """
        Initialize renderer with mesh and calibration data.
        
        Args:
            mesh_path: Path to mesh file (.obj)
            calibration_path: Path to calibration data (.pkl)
        """
        print("="*60)
        print("INITIALIZING MESH RENDERER")
        print("="*60)
        
        # Load mesh
        print(f"Loading mesh: {mesh_path.name}")
        self.mesh_trimesh = trimesh.load(mesh_path, process=False)
        
        # Convert to Open3D for rendering
        self.mesh_o3d = self._trimesh_to_o3d(self.mesh_trimesh)
        
        print(f"✓ Mesh loaded: {len(self.mesh_trimesh.vertices):,} vertices")
        
        # Load calibration data
        print(f"Loading calibration: {calibration_path.name}")
        with open(calibration_path, 'rb') as f:
            self.calibration = pickle.load(f)
        
        # Use calibrated mesh keypoints (with user depth applied)
        self.keypoints_3d = self.calibration['keypoints_3d_mesh_calibrated']
        print(f"✓ Calibration loaded: {len(self.keypoints_3d)} calibrated keypoints")
        
        # Store user reference keypoints for debugging
        self.keypoints_3d_user = self.calibration['keypoints_3d_user']
        self.keypoints_2d_user = self.calibration['keypoints_2d_user']
        
        # Initialize MediaPipe for live tracking
        print("Initializing MediaPipe Pose...")
        self.mp_pose = mp.solutions.pose
        self.mp_drawing = mp.solutions.drawing_utils
        self.pose_detector = self.mp_pose.Pose(
            static_image_mode=False,
            model_complexity=1,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        print("✓ MediaPipe initialized")
        
        # Keypoint mapping
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
        
        # Rendering state
        self.scale = 1.0
        self.offset_x = 0.0
        self.offset_y = 0.0
        
        print("✓ Renderer initialized\n")
    
    def _trimesh_to_o3d(self, mesh_trimesh):
        """Convert trimesh to Open3D mesh"""
        mesh_o3d = o3d.geometry.TriangleMesh()
        mesh_o3d.vertices = o3d.utility.Vector3dVector(mesh_trimesh.vertices)
        mesh_o3d.triangles = o3d.utility.Vector3iVector(mesh_trimesh.faces)
        
        # Add vertex colors if available
        if mesh_trimesh.visual.kind == 'vertex':
            colors = mesh_trimesh.visual.vertex_colors[:, :3] / 255.0
            mesh_o3d.vertex_colors = o3d.utility.Vector3dVector(colors)
        else:
            # Default gray color
            colors = np.ones((len(mesh_trimesh.vertices), 3)) * 0.7
            mesh_o3d.vertex_colors = o3d.utility.Vector3dVector(colors)
        
        mesh_o3d.compute_vertex_normals()
        
        return mesh_o3d
    
    def extract_live_keypoints(self, frame):
        """
        Extract MediaPipe keypoints from live video frame.
        
        Args:
            frame: OpenCV frame (BGR)
            
        Returns:
            Dict of 3D keypoint positions {name: (x, y, z)}
        """
        # Convert to RGB
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Process with MediaPipe
        results = self.pose_detector.process(rgb)
        
        if not results.pose_landmarks:
            return None
        
        # Extract keypoints
        h, w = frame.shape[:2]
        keypoints = {}
        
        for name, idx in self.KEYPOINT_INDICES.items():
            landmark = results.pose_landmarks.landmark[idx]
            
            # Pixel coordinates
            x = landmark.x * w
            y = landmark.y * h
            z = landmark.z  # Relative depth from MediaPipe
            
            keypoints[name] = (x, y, z)
        
        return keypoints
    
    def compute_alignment_transform(self, mesh_keypoints, live_keypoints, frame_shape):
        """
        Compute transformation to align mesh with live body.
        SIMPLIFIED: Just center mesh on user's nose position for testing.
        
        Args:
            mesh_keypoints: Dict of mesh keypoints from calibration
            live_keypoints: Dict of live keypoints from camera
            frame_shape: (height, width) of video frame
            
        Returns:
            scale, offset_x, offset_y, rotation_angle
        """
        h, w = frame_shape[:2]
        
        # Default values
        scale = 300.0  # Fixed scale for testing
        offset_x = 0.0
        offset_y = 0.0
        rotation_angle = 0.0
        
        # Simple approach: just center on nose
        if 'nose' in live_keypoints:
            nose_x, nose_y, _ = live_keypoints['nose']
            
            # Center mesh at nose position
            offset_x = nose_x - w / 2
            offset_y = nose_y - h / 3  # Slightly above center
            
            print(f"  Centering mesh at nose: ({nose_x:.0f}, {nose_y:.0f})")
        
        # Try to scale based on shoulders if available
        if ('left_shoulder' in live_keypoints and 
            'right_shoulder' in live_keypoints and
            'left_shoulder' in mesh_keypoints and
            'right_shoulder' in mesh_keypoints):
            
            # Live shoulder distance
            live_l = np.array(live_keypoints['left_shoulder'][:2])
            live_r = np.array(live_keypoints['right_shoulder'][:2])
            live_dist = np.linalg.norm(live_r - live_l)
            
            # Mesh shoulder distance (need to convert from mesh space)
            mesh_l = np.array(mesh_keypoints['left_shoulder'][:2])
            mesh_r = np.array(mesh_keypoints['right_shoulder'][:2])
            mesh_dist = np.linalg.norm(mesh_r - mesh_l)
            
            if mesh_dist > 0:
                # Scale to match shoulder width
                scale = live_dist / mesh_dist * 200  # Base scale factor
                print(f"  Scaling based on shoulders: {scale:.1f}")
        
        return scale, offset_x, offset_y, rotation_angle
    
    def render_mesh_to_2d(self, frame_shape, scale, offset_x, offset_y, rotation_angle=0.0):
        """
        Render 3D mesh to 2D image with given transformation.
        
        Args:
            frame_shape: (height, width) of target frame
            scale: Scale factor
            offset_x, offset_y: Translation offsets
            rotation_angle: Rotation around Z-axis (radians)
            
        Returns:
            rendered_image: RGBA numpy array
            depth_buffer: Depth values for each pixel
        """
        h, w = frame_shape[:2]
        
        # Create renderer
        vis = o3d.visualization.Visualizer()
        vis.create_window(visible=False, width=w, height=h)
        
        # Add mesh
        vis.add_geometry(self.mesh_o3d)
        
        # Set view control
        ctr = vis.get_view_control()
        
        # Compute camera parameters for orthographic projection
        # Position camera in front of mesh
        mesh_center = self.mesh_trimesh.bounds.mean(axis=0)
        mesh_size = np.linalg.norm(self.mesh_trimesh.bounds[1] - self.mesh_trimesh.bounds[0])
        
        # Set camera to look at mesh from front
        camera_pos = mesh_center + np.array([0, 0, mesh_size * 2])
        
        # Apply transformations
        # Note: Open3D transformations are applied in mesh space
        transform = np.eye(4)
        
        # Scale
        transform[:3, :3] *= scale
        
        # Rotation (around Y-axis for shoulder tilt)
        if abs(rotation_angle) > 0.01:
            c, s = np.cos(rotation_angle), np.sin(rotation_angle)
            R = np.array([
                [c, 0, s],
                [0, 1, 0],
                [-s, 0, c]
            ])
            transform[:3, :3] = R @ transform[:3, :3]
        
        # Translation
        transform[0, 3] = offset_x / w  # Normalize to [-1, 1]
        transform[1, 3] = offset_y / h
        
        # Note: This is a simplified rendering approach
        # For production, you'd use proper camera projection matrices
        
        # Render
        vis.poll_events()
        vis.update_renderer()
        
        # Capture image
        image = vis.capture_screen_float_buffer(do_render=True)
        image_np = np.asarray(image)
        
        # Convert to RGBA
        if image_np.shape[2] == 3:
            alpha = np.ones((h, w, 1))
            image_np = np.concatenate([image_np, alpha], axis=2)
        
        # Capture depth
        depth = vis.capture_depth_float_buffer(do_render=True)
        depth_np = np.asarray(depth)
        
        vis.destroy_window()
        
        # Convert to uint8
        rendered = (image_np * 255).astype(np.uint8)
        
        return rendered, depth_np
    
    def simple_mesh_projection(self, frame_shape, scale, offset_x, offset_y):
        """
        Simple orthographic projection of mesh (fallback if Open3D rendering fails).
        
        Args:
            frame_shape: (height, width)
            scale: Scale factor
            offset_x, offset_y: Translation
            
        Returns:
            projected_image: RGBA numpy array
        """
        h, w = frame_shape[:2]
        
        # Create blank RGBA image
        image = np.zeros((h, w, 4), dtype=np.uint8)
        
        # Get mesh vertices
        vertices = self.mesh_trimesh.vertices.copy()
        
        # Project to 2D (orthographic: just drop Z)
        vertices_2d = vertices[:, :2]
        
        # Get mesh bounds for normalization
        bounds = self.mesh_trimesh.bounds
        mesh_width = bounds[1, 0] - bounds[0, 0]
        mesh_height = bounds[1, 1] - bounds[0, 1]
        mesh_center = (bounds[0, :2] + bounds[1, :2]) / 2
        
        # Normalize to [0, 1]
        vertices_norm = (vertices_2d - bounds[0, :2]) / np.array([mesh_width, mesh_height])
        
        # Apply scale
        vertices_scaled = vertices_norm * scale
        
        # Center and add offset
        vertices_centered = vertices_scaled - 0.5  # Center around origin
        vertices_final = vertices_centered * min(w, h)  # Scale to image size
        
        # Add translation
        vertices_final[:, 0] += w / 2 + offset_x
        vertices_final[:, 1] += h / 2 + offset_y
        
        # Flip Y (image coordinates)
        vertices_final[:, 1] = h - vertices_final[:, 1]
        
        # Convert to integers
        vertices_int = vertices_final.astype(np.int32)
        
        # Draw mesh faces
        for face in self.mesh_trimesh.faces:
            pts = vertices_int[face]
            
            # Check if triangle is on screen
            if np.all((pts >= 0) & (pts < np.array([w, h]))):
                # Draw filled triangle
                cv2.fillPoly(image, [pts], (180, 180, 180, 255))
                # Draw outline
                cv2.polylines(image, [pts], True, (100, 100, 100, 255), 1)
        
        return image
    
    def composite_mesh_on_frame(self, frame, mesh_image, alpha=0.8):
        """
        Composite rendered mesh onto video frame.
        
        Args:
            frame: Background frame (BGR)
            mesh_image: Rendered mesh (RGBA)
            alpha: Opacity of mesh
            
        Returns:
            Composited frame (BGR)
        """
        # Extract alpha channel
        mesh_rgb = mesh_image[:, :, :3]
        mesh_alpha = mesh_image[:, :, 3:] / 255.0 * alpha
        
        # Expand alpha to 3 channels
        mesh_alpha_3ch = np.repeat(mesh_alpha, 3, axis=2)
        
        # Alpha blend
        result = (mesh_rgb * mesh_alpha_3ch + 
                 frame * (1 - mesh_alpha_3ch)).astype(np.uint8)
        
        return result
    
    def draw_debug_keypoints(self, frame, mesh_keypoints, live_keypoints, 
                            scale, offset_x, offset_y):
        """
        Draw debug visualization of keypoint alignment.
        
        Args:
            frame: Frame to draw on
            mesh_keypoints: Calibrated mesh keypoints
            live_keypoints: Live body keypoints
            scale, offset_x, offset_y: Current transformation
            
        Returns:
            Frame with debug overlay
        """
        debug_frame = frame.copy()
        h, w = frame.shape[:2]
        
        # Draw live keypoints (purple)
        if live_keypoints:
            for name, (x, y, z) in live_keypoints.items():
                cv2.circle(debug_frame, (int(x), int(y)), 6, (255, 0, 255), -1)
                cv2.putText(debug_frame, name[:3], (int(x)+8, int(y)),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255, 255, 255), 1)
        
        # Draw mesh keypoints transformed to frame space (red)
        if mesh_keypoints:
            bounds = self.mesh_trimesh.bounds
            mesh_width = bounds[1, 0] - bounds[0, 0]
            mesh_height = bounds[1, 1] - bounds[0, 1]
            
            for name, (mx, my, mz) in mesh_keypoints.items():
                # Normalize mesh coordinates
                mx_norm = (mx - bounds[0, 0]) / mesh_width
                my_norm = (my - bounds[0, 1]) / mesh_height
                
                # Apply transformation
                x = (mx_norm - 0.5) * scale * min(w, h) + w/2 + offset_x
                y = h - ((my_norm - 0.5) * scale * min(w, h) + h/2 + offset_y)
                
                cv2.circle(debug_frame, (int(x), int(y)), 6, (0, 0, 255), -1)
        
        # Draw skeleton connections for live keypoints
        connections = [
            ('left_shoulder', 'right_shoulder'),
            ('left_shoulder', 'left_elbow'),
            ('left_elbow', 'left_wrist'),
            ('right_shoulder', 'right_elbow'),
            ('right_elbow', 'right_wrist'),
            ('left_shoulder', 'left_hip'),
            ('right_shoulder', 'right_hip'),
            ('left_hip', 'right_hip'),
        ]
        
        if live_keypoints:
            for pt1_name, pt2_name in connections:
                if pt1_name in live_keypoints and pt2_name in live_keypoints:
                    pt1 = tuple(map(int, live_keypoints[pt1_name][:2]))
                    pt2 = tuple(map(int, live_keypoints[pt2_name][:2]))
                    cv2.line(debug_frame, pt1, pt2, (255, 0, 255), 2)
        
        return debug_frame


def main():
    """
    Main rendering and overlay test.
    """
    print("="*60)
    print("TEST SCRIPT 2: MESH RENDERING & OVERLAY")
    print("="*60)
    print("\nThis script:")
    print("1. Loads calibrated mesh and keypoints")
    print("2. Starts live camera feed")
    print("3. Tracks body with MediaPipe")
    print("4. Aligns and renders mesh on body")
    print("5. Shows debug view with keypoints")
    print("\nControls:")
    print("  Q - Quit")
    print("  D - Toggle debug view (show keypoints)")
    print("  R - Remove camera feed (mesh keypoints only)")
    print("  + / - - Adjust mesh scale")
    print("  Arrow keys - Adjust position")
    print("="*60)
    
    # Find calibration files
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
    choice = input("\nRender which mesh? (1-N): ").strip()
    
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
    
    # Initialize renderer
    try:
        renderer = MeshRenderer(mesh_path, calib_path)
    except Exception as e:
        print(f"✗ Error initializing renderer: {e}")
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
    show_debug = False
    show_camera = True
    scale_adjust = 1.0
    pos_x_adjust = 0
    pos_y_adjust = 0
    
    # Performance tracking
    fps = 0
    frame_count = 0
    fps_start = time.time()
    
    print("\n" + "="*60)
    print("RUNNING - Stand in front of camera!")
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
            live_keypoints = renderer.extract_live_keypoints(frame)
            
            if live_keypoints:
                # Compute alignment
                scale, offset_x, offset_y, rotation = renderer.compute_alignment_transform(
                    renderer.keypoints_3d,
                    live_keypoints,
                    frame.shape
                )
                
                # Apply manual adjustments
                scale *= scale_adjust
                offset_x += pos_x_adjust
                offset_y += pos_y_adjust
                
                # Render mesh (use simple projection for now, Open3D can be temperamental)
                try:
                    mesh_image = renderer.simple_mesh_projection(
                        frame.shape,
                        scale,
                        offset_x,
                        offset_y
                    )
                    
                    # Composite onto frame
                    if show_camera:
                        result = renderer.composite_mesh_on_frame(frame, mesh_image, alpha=0.7)
                    else:
                        # Show mesh only
                        result = cv2.cvtColor(mesh_image[:, :, :3], cv2.COLOR_RGB2BGR)
                    
                except Exception as e:
                    print(f"Rendering error: {e}")
                    result = frame
                
                # Add debug overlay if enabled
                if show_debug:
                    result = renderer.draw_debug_keypoints(
                        result,
                        renderer.keypoints_3d,
                        live_keypoints,
                        scale,
                        offset_x,
                        offset_y
                    )
            else:
                result = frame
            
            # Add info overlay
            status = "Pose detected - Mesh rendered" if live_keypoints else "No pose detected"
            color = (0, 255, 0) if live_keypoints else (0, 0, 255)
            
            cv2.putText(result, f"FPS: {fps:.1f}", (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            
            cv2.putText(result, status, (10, 70),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
            
            cv2.putText(result, f"Scale: {scale_adjust:.2f}", (10, 100),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
            
            mode = "Debug: ON" if show_debug else "Debug: OFF"
            cv2.putText(result, mode, (10, 130),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
            
            # Display
            cv2.imshow("Mesh Rendering Test", result)
            
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
            
            elif key == ord('d') or key == ord('D'):
                show_debug = not show_debug
            
            elif key == ord('r') or key == ord('R'):
                show_camera = not show_camera
            
            elif key == ord('+') or key == ord('='):
                scale_adjust += 0.1
            
            elif key == ord('-') or key == ord('_'):
                scale_adjust = max(0.1, scale_adjust - 0.1)
            
            elif key == 82:  # Up arrow
                pos_y_adjust -= 10
            
            elif key == 84:  # Down arrow
                pos_y_adjust += 10
            
            elif key == 81:  # Left arrow
                pos_x_adjust -= 10
            
            elif key == 83:  # Right arrow
                pos_x_adjust += 10
    
    except KeyboardInterrupt:
        print("\nInterrupted")
    
    finally:
        cap.release()
        cv2.destroyAllWindows()
        print("\n" + "="*60)
        print("TEST COMPLETE")
        print("="*60)
        print("\nNext step:")
        print("  python tests/test_03_mesh_warping.py")


if __name__ == "__main__":
    main()