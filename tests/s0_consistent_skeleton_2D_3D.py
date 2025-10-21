"""
TEST SCRIPT 1: Calibration & Keypoint Extraction
================================================
Loads 3D mesh, detects orientation, extracts MediaPipe keypoints,
creates 3D skeleton using depth estimation for calibration.

Run from root: python tests/test_01_calibration_keypoints.py

Dependencies: trimesh, mediapipe, transformers, torch, numpy, opencv
"""

import cv2
import numpy as np
import torch
import time
import trimesh
from pathlib import Path
from PIL import Image
import mediapipe as mp
from transformers import pipeline
import pickle


class MeshKeypointCalibrator:
    """
    Handles mesh loading, orientation detection, and keypoint extraction.
    
    NEW WORKFLOW:
    1. Capture background depth (empty scene) for reference
    2. Capture user with countdown for keypoint + depth calibration
    3. Extract keypoints from both mesh and user
    4. Apply calibrated user depth to mesh keypoints
    """
    
    def __init__(self, depth_model="Intel/dpt-hybrid-midas"):
        """Initialize calibrator with depth estimation model"""
        
        print("="*60)
        print("INITIALIZING MESH KEYPOINT CALIBRATOR")
        print("="*60)
        
        # Device setup
        self.device = "cuda" if torch.cuda.is_available() else (
            "mps" if torch.backends.mps.is_available() else "cpu"
        )
        print(f"Using device: {self.device}")
        
        # Initialize depth estimator
        print("Loading depth estimation model...")
        self.depth_estimator = pipeline(
            task="depth-estimation",
            model=depth_model,
            device=0 if self.device == "cuda" else -1
        )
        print("✓ Depth model loaded")
        
        # Initialize MediaPipe Pose
        print("Loading MediaPipe Pose...")
        self.mp_pose = mp.solutions.pose
        self.mp_drawing = mp.solutions.drawing_utils
        self.pose_detector = self.mp_pose.Pose(
            static_image_mode=True,  # For single image processing
            model_complexity=1,
            min_detection_confidence=0.5
        )
        print("✓ MediaPipe Pose loaded")
        
        # Keypoint mapping for body parts
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
        
        print("✓ Calibrator initialized\n")
    
    def load_mesh(self, mesh_path):
        """
        Load 3D mesh from file (OBJ, PLY, etc.)
        
        Args:
            mesh_path: Path to mesh file
            
        Returns:
            trimesh.Trimesh object
        """
        print(f"Loading mesh: {mesh_path}")
        
        try:
            mesh = trimesh.load(mesh_path, process=False)
            
            print(f"✓ Mesh loaded successfully")
            print(f"  Vertices: {len(mesh.vertices):,}")
            print(f"  Faces: {len(mesh.faces):,}")
            print(f"  Bounds: {mesh.bounds}")
            
            return mesh
            
        except Exception as e:
            print(f"✗ Error loading mesh: {e}")
            return None
    
    def detect_mesh_orientation(self, mesh):
        """
        Detect if mesh is upright or needs rotation.
        Uses PCA to find principal axes.
        
        Args:
            mesh: trimesh.Trimesh object
            
        Returns:
            rotation_matrix: 4x4 transformation matrix to make mesh upright
            orientation_info: Dict with orientation details
        """
        print("\nDetecting mesh orientation...")
        
        vertices = mesh.vertices
        
        # Center vertices
        centered = vertices - vertices.mean(axis=0)
        
        # PCA to find principal axes
        cov_matrix = np.cov(centered.T)
        eigenvalues, eigenvectors = np.linalg.eig(cov_matrix)
        
        # Sort by eigenvalue (largest = main axis)
        idx = eigenvalues.argsort()[::-1]
        eigenvalues = eigenvalues[idx]
        eigenvectors = eigenvectors[:, idx]
        
        # Largest variance should be vertical (Y-axis) for upright mesh
        main_axis = eigenvectors[:, 0]
        
        # Check if main axis is aligned with Y
        y_alignment = np.abs(np.dot(main_axis, [0, 1, 0]))
        
        print(f"  Main axis: {main_axis}")
        print(f"  Y-alignment: {y_alignment:.3f}")
        
        # Create rotation matrix if needed
        if y_alignment < 0.8:  # Not aligned with Y
            print("  ⚠ Mesh needs rotation to be upright")
            
            # Find rotation to align main axis with Y
            target = np.array([0, 1, 0])
            
            # Rotation axis (cross product)
            axis = np.cross(main_axis, target)
            axis = axis / (np.linalg.norm(axis) + 1e-8)
            
            # Rotation angle
            angle = np.arccos(np.clip(np.dot(main_axis, target), -1, 1))
            
            # Create rotation matrix using Rodrigues' formula
            K = np.array([
                [0, -axis[2], axis[1]],
                [axis[2], 0, -axis[0]],
                [-axis[1], axis[0], 0]
            ])
            
            R = np.eye(3) + np.sin(angle) * K + (1 - np.cos(angle)) * np.dot(K, K)
            
            # Create 4x4 transformation matrix
            transform = np.eye(4)
            transform[:3, :3] = R
            
            print(f"  Rotation angle: {np.degrees(angle):.1f}°")
            
        else:
            print("  ✓ Mesh is already upright")
            transform = np.eye(4)
        
        orientation_info = {
            'main_axis': main_axis,
            'y_alignment': y_alignment,
            'needs_rotation': y_alignment < 0.8,
            'eigenvalues': eigenvalues
        }
        
        return transform, orientation_info
    
    def render_mesh_view(self, mesh, view='front', resolution=512):
        """
        Render 2D view of 3D mesh for MediaPipe processing.
        
        Args:
            mesh: trimesh.Trimesh object
            view: 'front', 'side', or 'top'
            resolution: Output image size
            
        Returns:
            PIL Image (RGB)
        """
        print(f"\nRendering {view} view for MediaPipe...")
        
        # Create scene
        scene = mesh.scene()
        
        # Set camera based on view
        if view == 'front':
            # Look at mesh from front (negative Z)
            camera_transform = trimesh.transformations.translation_matrix([0, 0, -3])
        elif view == 'side':
            # Look from side (negative X)
            camera_transform = trimesh.transformations.translation_matrix([-3, 0, 0])
            rot = trimesh.transformations.rotation_matrix(np.pi/2, [0, 1, 0])
            camera_transform = np.dot(camera_transform, rot)
        else:  # top
            # Look from top (positive Y)
            camera_transform = trimesh.transformations.translation_matrix([0, 3, 0])
            rot = trimesh.transformations.rotation_matrix(-np.pi/2, [1, 0, 0])
            camera_transform = np.dot(camera_transform, rot)
        
        # Render using trimesh's built-in renderer
        try:
            # Set up scene camera
            scene.camera_transform = camera_transform
            
            # Render to PNG bytes
            png_bytes = scene.save_image(resolution=(resolution, resolution))
            
            # Convert to PIL Image
            from io import BytesIO
            image = Image.open(BytesIO(png_bytes))
            
            # Ensure RGB mode
            if image.mode != 'RGB':
                image = image.convert('RGB')
            
            print(f"✓ Rendered {resolution}x{resolution} {view} view")
            
            return image
            
        except Exception as e:
            print(f"⚠ Rendering error: {e}")
            print("  Creating simple projection instead...")
            
            # Fallback: Simple orthographic projection
            vertices = mesh.vertices
            
            if view == 'front':
                # Project to XY plane
                x, y = vertices[:, 0], vertices[:, 1]
            elif view == 'side':
                # Project to YZ plane
                x, y = vertices[:, 2], vertices[:, 1]
            else:  # top
                # Project to XZ plane
                x, y = vertices[:, 0], vertices[:, 2]
            
            # Normalize to image coordinates
            x_norm = ((x - x.min()) / (x.max() - x.min()) * resolution).astype(int)
            y_norm = ((y - y.min()) / (y.max() - y.min()) * resolution).astype(int)
            
            # Create image
            img_array = np.ones((resolution, resolution, 3), dtype=np.uint8) * 255
            
            # Draw points
            for px, py in zip(x_norm, y_norm):
                if 0 <= px < resolution and 0 <= py < resolution:
                    img_array[resolution - py - 1, px] = [100, 100, 100]
            
            image = Image.fromarray(img_array)
            print(f"✓ Created projection view")
            
            return image
    
    def extract_2d_keypoints(self, image_pil):
        """
        Extract MediaPipe keypoints from 2D image.
        
        Args:
            image_pil: PIL Image (RGB)
            
        Returns:
            Dict of keypoint positions {name: (x, y)}
        """
        print("\nExtracting MediaPipe keypoints from rendered view...")
        
        # Convert to numpy array
        image_np = np.array(image_pil)
        
        # Run MediaPipe
        results = self.pose_detector.process(image_np)
        
        if not results.pose_landmarks:
            print("✗ No pose detected in image!")
            return None
        
        # Extract keypoints
        h, w = image_np.shape[:2]
        keypoints_2d = {}
        
        for name, idx in self.KEYPOINT_INDICES.items():
            landmark = results.pose_landmarks.landmark[idx]
            x = int(landmark.x * w)
            y = int(landmark.y * h)
            keypoints_2d[name] = (x, y)
        
        print(f"✓ Extracted {len(keypoints_2d)} keypoints")
        
        return keypoints_2d
    
    def create_3d_skeleton(self, image_pil, keypoints_2d):
        """
        Create 3D skeleton from 2D keypoints + depth estimation.
        
        Args:
            image_pil: PIL Image used for depth estimation
            keypoints_2d: Dict of 2D keypoint positions
            
        Returns:
            Dict of 3D keypoint positions {name: (x, y, z)}
        """
        print("\nCreating 3D skeleton using depth estimation...")
        
        # Get depth map
        depth_result = self.depth_estimator(image_pil)
        depth_map = np.array(depth_result["depth"], dtype=np.float32)
        
        # Normalize depth to 0-1 range (0=far, 1=close)
        depth_map = (depth_map - depth_map.min()) / (depth_map.max() - depth_map.min() + 1e-8)
        
        # Invert so 1=close, 0=far
        depth_map = 1.0 - depth_map
        
        print(f"  Depth map: {depth_map.shape}, range: [{depth_map.min():.3f}, {depth_map.max():.3f}]")
        
        # Create 3D keypoints
        keypoints_3d = {}
        h, w = depth_map.shape
        
        for name, (x, y) in keypoints_2d.items():
            # Get depth at keypoint location
            if 0 <= x < w and 0 <= y < h:
                z = float(depth_map[y, x])
            else:
                z = 0.5  # Default middle depth
            
            # Normalize coordinates to [-1, 1] range
            x_norm = (x / w) * 2 - 1
            y_norm = (y / h) * 2 - 1
            z_norm = z * 2 - 1  # Z in [-1, 1]
            
            keypoints_3d[name] = (x_norm, y_norm, z_norm)
        
        print(f"✓ Created 3D skeleton with {len(keypoints_3d)} keypoints")
        
        return keypoints_3d, depth_map
    
    def map_keypoints_to_mesh(self, mesh, keypoints_2d, image_size=512):
        """
        Map 2D keypoints to 3D mesh surface.
        Places keypoints at middle of mesh in Z, matching X and Y.
        
        NOTE: Z-depth will be replaced with calibrated user depth in main workflow.
        
        Args:
            mesh: trimesh.Trimesh object
            keypoints_2d: Dict of 2D keypoints from rendered view
            image_size: Size of rendered image
            
        Returns:
            Dict of 3D keypoint positions on mesh {name: (x, y, z)}
        """
        print("\nMapping keypoints to mesh surface...")
        
        vertices = mesh.vertices
        bounds = mesh.bounds
        
        # Get mesh dimensions
        x_min, y_min, z_min = bounds[0]
        x_max, y_max, z_max = bounds[1]
        z_mid = (z_min + z_max) / 2
        
        print(f"  Mesh bounds: X[{x_min:.2f}, {x_max:.2f}], Y[{y_min:.2f}, {y_max:.2f}], Z[{z_min:.2f}, {z_max:.2f}]")
        print(f"  Z middle: {z_mid:.2f}")
        
        keypoints_3d = {}
        
        for name, (x_2d, y_2d) in keypoints_2d.items():
            # Convert 2D image coordinates to mesh coordinates
            # Image: (0,0) is top-left, Y increases downward
            # Mesh: Y increases upward
            
            x_norm = x_2d / image_size  # 0 to 1
            y_norm = 1.0 - (y_2d / image_size)  # 0 to 1, flip Y
            
            # Map to mesh bounds
            x_3d = x_min + x_norm * (x_max - x_min)
            y_3d = y_min + y_norm * (y_max - y_min)
            z_3d = z_mid  # Place at middle depth for now
            
            keypoints_3d[name] = (x_3d, y_3d, z_3d)
        
        print(f"✓ Mapped {len(keypoints_3d)} keypoints to mesh")
        
        return keypoints_3d
    
    def visualize_keypoints(self, image_pil, keypoints_2d, depth_map=None):
        """
        Create visualization of keypoints on rendered mesh view.
        
        Args:
            image_pil: PIL Image
            keypoints_2d: Dict of 2D keypoints
            depth_map: Optional depth map for overlay
            
        Returns:
            numpy array (BGR) for OpenCV display
        """
        # Convert to numpy BGR
        image_np = np.array(image_pil)
        image_bgr = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)
        
        # Draw keypoints
        for name, (x, y) in keypoints_2d.items():
            # Draw circle
            cv2.circle(image_bgr, (x, y), 8, (0, 255, 0), -1)
            # Draw label
            cv2.putText(image_bgr, name.replace('_', ' '), (x + 10, y),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
        
        # Draw skeleton connections
        connections = [
            ('left_shoulder', 'right_shoulder'),
            ('left_shoulder', 'left_elbow'),
            ('left_elbow', 'left_wrist'),
            ('right_shoulder', 'right_elbow'),
            ('right_elbow', 'right_wrist'),
            ('left_shoulder', 'left_hip'),
            ('right_shoulder', 'right_hip'),
            ('left_hip', 'right_hip'),
            ('left_hip', 'left_knee'),
            ('left_knee', 'left_ankle'),
            ('right_hip', 'right_knee'),
            ('right_knee', 'right_ankle'),
        ]
        
        for pt1_name, pt2_name in connections:
            if pt1_name in keypoints_2d and pt2_name in keypoints_2d:
                pt1 = keypoints_2d[pt1_name]
                pt2 = keypoints_2d[pt2_name]
                cv2.line(image_bgr, pt1, pt2, (0, 255, 255), 2)
        
        # Add depth map overlay if provided
        if depth_map is not None:
            depth_colored = cv2.applyColorMap(
                (depth_map * 255).astype(np.uint8), 
                cv2.COLORMAP_TURBO
            )
            depth_colored = cv2.resize(depth_colored, (image_bgr.shape[1], image_bgr.shape[0]))
            image_bgr = cv2.addWeighted(image_bgr, 0.7, depth_colored, 0.3, 0)
        
        return image_bgr


def capture_with_countdown(cap, countdown_seconds=5, message="Get ready!"):
    """
    Capture frame after countdown with visual feedback.
    
    Args:
        cap: OpenCV VideoCapture object
        countdown_seconds: Countdown duration
        message: Message to display during countdown
        
    Returns:
        Captured frame
    """
    print(f"\n{message}")
    print(f"Countdown: {countdown_seconds} seconds...")
    
    for i in range(countdown_seconds, 0, -1):
        ret, frame = cap.read()
        if not ret:
            time.sleep(1)
            continue
        
        # Mirror for natural feel
        frame = cv2.flip(frame, 1)
        
        # Create large countdown overlay
        overlay = frame.copy()
        h, w = frame.shape[:2]
        
        # Semi-transparent background
        cv2.rectangle(overlay, (0, 0), (w, h), (0, 0, 0), -1)
        frame = cv2.addWeighted(frame, 0.5, overlay, 0.5, 0)
        
        # Countdown number
        text = str(i)
        font_scale = 10
        thickness = 20
        text_size = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness)[0]
        
        text_x = (w - text_size[0]) // 2
        text_y = (h + text_size[1]) // 2
        
        # Draw countdown with shadow
        cv2.putText(frame, text, (text_x + 5, text_y + 5),
                   cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 0, 0), thickness + 5)
        cv2.putText(frame, text, (text_x, text_y),
                   cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 255, 255), thickness)
        
        # Message
        msg_y = 100
        cv2.putText(frame, message, (50, msg_y),
                   cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 255, 255), 3)
        
        cv2.imshow("Calibration", frame)
        cv2.waitKey(1000)  # Wait 1 second
    
    # Capture final frame
    ret, frame = cap.read()
    if ret:
        frame = cv2.flip(frame, 1)
        
        # # Show "CAPTURING" message briefly
        # overlay = frame.copy()
        # cv2.rectangle(overlay, (0, h//2 - 50), (w, h//2 + 50), (0, 255, 0), -1)
        # frame = cv2.addWeighted(frame, 0.7, overlay, 0.3, 0)
        
        # cv2.putText(frame, "CAPTURING!", (w//2 - 200, h//2 + 20),
        #            cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255), 5)
        
        # # cv2.imshow("Calibration", frame)
        # cv2.waitKey(500)
    
    return frame


def main():
    """
    Main calibration workflow with two-stage depth capture.
    """
    print("="*60)
    print("TEST SCRIPT 1: MESH CALIBRATION & KEYPOINT EXTRACTION")
    print("="*60)
    print("\nThis script:")
    print("1. Loads 3D mesh from TripoSR output")
    print("2. Detects and corrects mesh orientation")
    print("3. TWO-STAGE CALIBRATION:")
    print("   - Stage 1: Capture background depth (empty scene)")
    print("   - Stage 2: Capture user keypoints with depth")
    print("4. Maps keypoints to mesh surface with calibrated depth")
    print("5. Saves calibration data for next script")
    print("\nIMPORTANT: You'll need to step OUT then IN of frame")
    print("="*60)
    
    # Find meshes
    mesh_dir = Path("generated_meshes")
    if not mesh_dir.exists():
        print(f"\n✗ Directory not found: {mesh_dir}")
        print("Run TripoSR generation first!")
        return
    
    mesh_files = sorted(list(mesh_dir.glob("*_triposr.obj")))
    
    if len(mesh_files) == 0:
        print(f"\n✗ No meshes found in {mesh_dir}")
        return
    
    print(f"\nFound {len(mesh_files)} meshes:")
    for i, mesh_file in enumerate(mesh_files, 1):
        print(f"  {i}. {mesh_file.name}")
    
    # Select mesh
    choice = input("\nCalibrate which mesh? (1-N or 'all'): ").strip().lower()
    
    if choice == 'all':
        selected = mesh_files
    else:
        try:
            idx = int(choice) - 1
            selected = [mesh_files[idx]] if 0 <= idx < len(mesh_files) else [mesh_files[0]]
        except:
            selected = [mesh_files[0]]
    
    print(f"\nWill calibrate {len(selected)} mesh(es)")
    
    # Initialize calibrator (this loads models)
    print("\n" + "="*60)
    print("INITIALIZING CALIBRATION SYSTEM")
    print("="*60)
    calibrator = MeshKeypointCalibrator()
    
    # Initialize camera EARLY for depth calibration
    print("\n" + "="*60)
    print("INITIALIZING CAMERA FOR DEPTH CALIBRATION")
    print("="*60)
    print("Initializing camera...")
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    
    if not cap.isOpened():
        print("✗ Could not open camera")
        return
    
    print("✓ Camera initialized")
    
    # Create window for calibration
    cv2.namedWindow("Calibration", cv2.WINDOW_NORMAL)
    
    # ========================================================================
    # STAGE 1: CAPTURE REFERENCE DEPTH (EMPTY SCENE)
    # ========================================================================
    print("\n" + "="*60)
    print("STAGE 1: BACKGROUND DEPTH CALIBRATION")
    print("="*60)
    print("\nThis captures the depth of your space WITHOUT you in it.")
    print("This creates a reference so we can accurately measure YOUR depth.")
    print("\n⚠️  IMPORTANT: Step OUT of the camera frame completely!")
    
    input("\nPress ENTER when you are OUT of frame...")
    
    # Show preview
    print("\nShowing camera preview...")
    print("Make sure you are NOT visible in the frame.")
    print("Press SPACE to capture, or ESC to skip this mesh")
    
    background_frame = None
    skip_mesh = False
    
    while True:
        ret, frame = cap.read()
        if not ret:
            continue
        
        frame = cv2.flip(frame, 1)
        
        # Add overlay instructions
        overlay = frame.copy()
        cv2.putText(overlay, "STEP OUT OF FRAME", (50, 50),
                   cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 255), 3)
        cv2.putText(overlay, "Press SPACE when ready", (50, 100),
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        cv2.putText(overlay, "Press ESC to skip", (50, 150),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (200, 200, 200), 2)
        
        cv2.imshow("Calibration", overlay)
        
        key = cv2.waitKey(1) & 0xFF
        if key == 32:  # SPACE
            background_frame = frame
            break
        elif key == 27:  # ESC
            skip_mesh = True
            break
    
    if skip_mesh:
        print("\n⚠️  Skipping calibration")
        cap.release()
        cv2.destroyAllWindows()
        return
    
    print("✓ Background frame captured")
    
    # Get reference depth map
    print("Computing reference depth map...")
    background_pil = Image.fromarray(cv2.cvtColor(background_frame, cv2.COLOR_BGR2RGB))
    depth_result_bg = calibrator.depth_estimator(background_pil)
    depth_map_background = np.array(depth_result_bg["depth"], dtype=np.float32)
    
    # Normalize
    depth_map_background = (depth_map_background - depth_map_background.min()) / \
                          (depth_map_background.max() - depth_map_background.min() + 1e-8)
    depth_map_background = 1.0 - depth_map_background  # Invert: 1=close, 0=far
    
    print(f"✓ Reference depth: min={depth_map_background.min():.3f}, max={depth_map_background.max():.3f}")
    
    # ========================================================================
    # STAGE 2: CAPTURE USER WITH KEYPOINTS
    # ========================================================================
    print("\n" + "="*60)
    print("STAGE 2: USER DEPTH & KEYPOINT CALIBRATION")
    print("="*60)
    print("\n⚠️  NOW step INTO frame and stand in your clothing pose!")
    print("Stand where you'll be when trying on clothes.")
    
    input("\nPress ENTER when you're ready for countdown...")
    
    # Countdown and capture
    user_frame = capture_with_countdown(
        cap, 
        countdown_seconds=5,
        message="Stand still in your clothing pose!"
    )
    
    if user_frame is None:
        print("✗ Failed to capture user frame")
        cap.release()
        cv2.destroyAllWindows()
        return
    
    print("✓ User frame captured")
    
    # Get user depth map
    print("Computing user depth map...")
    user_pil = Image.fromarray(cv2.cvtColor(user_frame, cv2.COLOR_BGR2RGB))
    depth_result_user = calibrator.depth_estimator(user_pil)
    depth_map_user = np.array(depth_result_user["depth"], dtype=np.float32)
    
    # Normalize
    depth_map_user = (depth_map_user - depth_map_user.min()) / \
                     (depth_map_user.max() - depth_map_user.min() + 1e-8)
    depth_map_user = 1.0 - depth_map_user
    
    print(f"✓ User depth: min={depth_map_user.min():.3f}, max={depth_map_user.max():.3f}")
    
    # Compute relative depth (user depth minus background)
    # This isolates the user's depth from the environment
    depth_map_relative = np.maximum(depth_map_user - depth_map_background, 0)
    depth_map_relative = depth_map_relative / (depth_map_relative.max() + 1e-8)
    
    print(f"✓ Relative depth computed: range [0.0, 1.0]")
    
    # Extract MediaPipe keypoints from user frame
    print("\nExtracting MediaPipe keypoints from user...")
    rgb_frame = cv2.cvtColor(user_frame, cv2.COLOR_BGR2RGB)
    results = calibrator.pose_detector.process(rgb_frame)
    
    if not results.pose_landmarks:
        print("✗ No pose detected! Make sure you're fully visible.")
        print("   Try again with better lighting and full body in frame.")
        cap.release()
        cv2.destroyAllWindows()
        return
    
    # Extract 2D keypoints
    h, w = user_frame.shape[:2]
    keypoints_2d_user = {}
    
    for name, idx in calibrator.KEYPOINT_INDICES.items():
        landmark = results.pose_landmarks.landmark[idx]
        x = int(landmark.x * w)
        y = int(landmark.y * h)
        keypoints_2d_user[name] = (x, y)
    
    print(f"✓ Extracted {len(keypoints_2d_user)} keypoints from user")
    
    # Create 3D keypoints using RELATIVE depth at keypoint positions
    print("Creating 3D skeleton with calibrated depth...")
    keypoints_3d_user = {}
    
    for name, (x, y) in keypoints_2d_user.items():
        # Get relative depth at this keypoint
        if 0 <= x < w and 0 <= y < h:
            z_relative = float(depth_map_relative[y, x])
        else:
            z_relative = 0.5
        
        # Normalize coordinates
        x_norm = (x / w) * 2 - 1  # [-1, 1]
        y_norm = (y / h) * 2 - 1
        z_norm = z_relative * 2 - 1  # [-1, 1], calibrated relative depth
        
        keypoints_3d_user[name] = (x_norm, y_norm, z_norm)
    
    print(f"✓ Created 3D skeleton with calibrated depth")
    
    # Visualize calibration results
    print("\nGenerating calibration visualizations...")
    
    # Create visualization with all three depth maps
    vis_combined = np.hstack([
        cv2.applyColorMap((depth_map_background * 255).astype(np.uint8), cv2.COLORMAP_TURBO),
        cv2.applyColorMap((depth_map_user * 255).astype(np.uint8), cv2.COLORMAP_TURBO),
        cv2.applyColorMap((depth_map_relative * 255).astype(np.uint8), cv2.COLORMAP_TURBO)
    ])
    
    # Add labels
    cv2.putText(vis_combined, "Background", (50, 50),
               cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    cv2.putText(vis_combined, "User", (w + 50, 50),
               cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    cv2.putText(vis_combined, "Relative (Calibrated)", (2*w + 50, 50),
               cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    
    cv2.imshow("Depth Calibration", vis_combined)
    
    # Draw keypoints on user frame
    vis_keypoints = user_frame.copy()
    for name, (x, y) in keypoints_2d_user.items():
        # Get depth for color coding
        if 0 <= x < w and 0 <= y < h:
            z = depth_map_relative[y, x]
            color = (0, int((1 - z) * 255), int(z * 255))  # Blue=far, Red=close
        else:
            color = (128, 128, 128)
        
        cv2.circle(vis_keypoints, (x, y), 8, color, -1)
        cv2.putText(vis_keypoints, name[:4], (x + 10, y),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
    
    # Draw skeleton connections
    connections = [
        ('left_shoulder', 'right_shoulder'),
        ('left_shoulder', 'left_elbow'),
        ('left_elbow', 'left_wrist'),
        ('right_shoulder', 'right_elbow'),
        ('right_elbow', 'right_wrist'),
        ('left_shoulder', 'left_hip'),
        ('right_shoulder', 'right_hip'),
        ('left_hip', 'right_hip'),
        ('left_hip', 'left_knee'),
        ('left_knee', 'left_ankle'),
        ('right_hip', 'right_knee'),
        ('right_knee', 'right_ankle'),
    ]
    
    for pt1_name, pt2_name in connections:
        if pt1_name in keypoints_2d_user and pt2_name in keypoints_2d_user:
            pt1 = keypoints_2d_user[pt1_name]
            pt2 = keypoints_2d_user[pt2_name]
            cv2.line(vis_keypoints, pt1, pt2, (0, 255, 255), 2)
    
    cv2.imshow("User Keypoints", vis_keypoints)
    
    print("\n✓ Calibration visualizations ready")
    print("  - 'Depth Calibration' window shows 3 depth maps")
    print("  - 'User Keypoints' window shows detected pose")
    print("\nPress any key to continue with mesh processing...")
    cv2.waitKey(0)
    
    # Close camera (we're done with live capture)
    cap.release()
    cv2.destroyWindow("Calibration")
    
    # Output directory
    calib_dir = Path("calibration_data")
    calib_dir.mkdir(exist_ok=True)
    
    # Process each mesh
    for i, mesh_path in enumerate(selected, 1):
        print(f"\n{'='*60}")
        print(f"[{i}/{len(selected)}] Processing: {mesh_path.name}")
        print('='*60)
        
        # Load mesh
        mesh = calibrator.load_mesh(mesh_path)
        if mesh is None:
            continue
        
        # Detect orientation and correct if needed
        transform, orientation_info = calibrator.detect_mesh_orientation(mesh)
        
        if orientation_info['needs_rotation']:
            print("Applying orientation correction...")
            mesh.apply_transform(transform)
            print("✓ Mesh reoriented to upright position")
        
        # Render front view for mesh keypoint extraction
        print("\nRendering mesh front view...")
        front_view = calibrator.render_mesh_view(mesh, view='front', resolution=512)
        
        # Extract 2D keypoints from MESH rendering
        print("Extracting keypoints from mesh rendering...")
        keypoints_2d_mesh = calibrator.extract_2d_keypoints(front_view)
        
        if keypoints_2d_mesh is None:
            print("✗ Skipping - no keypoints detected on mesh")
            continue
        
        print(f"✓ Extracted {len(keypoints_2d_mesh)} keypoints from mesh")
        
        # Map mesh keypoints to 3D mesh surface
        print("Mapping keypoints to mesh surface...")
        keypoints_3d_mesh = calibrator.map_keypoints_to_mesh(mesh, keypoints_2d_mesh, image_size=512)
        
        # NOW: Apply calibrated USER depth to mesh keypoints
        # This aligns the mesh's Z-depth with the user's actual depth
        print("\nApplying calibrated depth to mesh keypoints...")
        
        keypoints_3d_mesh_calibrated = {}
        
        for name in keypoints_3d_mesh.keys():
            if name in keypoints_3d_user:
                # Get mesh X, Y coordinates
                mesh_x, mesh_y, mesh_z = keypoints_3d_mesh[name]
                
                # Get calibrated user depth
                user_x, user_y, user_z_calibrated = keypoints_3d_user[name]
                
                # Replace mesh Z with calibrated user Z
                # This makes the mesh keypoints have realistic depth
                keypoints_3d_mesh_calibrated[name] = (mesh_x, mesh_y, user_z_calibrated)
            else:
                # Keep original if no user keypoint
                keypoints_3d_mesh_calibrated[name] = keypoints_3d_mesh[name]
        
        print(f"✓ Applied calibrated depth to {len(keypoints_3d_mesh_calibrated)} keypoints")
        
        # Visualize mesh keypoints
        vis_mesh = calibrator.visualize_keypoints(front_view, keypoints_2d_mesh, None)
        
        # Show visualization
        cv2.imshow("Mesh Keypoints - Press any key to save", vis_mesh)
        cv2.waitKey(0)
        
        # Save calibration data with BOTH sets of keypoints and depth maps
        calib_path = calib_dir / f"{mesh_path.stem}_calibration.pkl"
        
        calibration_data = {
            'mesh_path': str(mesh_path),
            'mesh_transform': transform,
            'orientation_info': orientation_info,
            
            # Mesh keypoints (from rendered view)
            'keypoints_2d_mesh': keypoints_2d_mesh,
            'keypoints_3d_mesh': keypoints_3d_mesh,
            'keypoints_3d_mesh_calibrated': keypoints_3d_mesh_calibrated,  # With user depth
            
            # User keypoints (from live capture)
            'keypoints_2d_user': keypoints_2d_user,
            'keypoints_3d_user': keypoints_3d_user,
            
            # Depth maps
            'depth_map_background': depth_map_background,
            'depth_map_user': depth_map_user,
            'depth_map_relative': depth_map_relative,
            
            # Frames for reference
            'background_frame_shape': background_frame.shape,
            'user_frame_shape': user_frame.shape,
            
            'calibration_time': time.time()
        }
        
        with open(calib_path, 'wb') as f:
            pickle.dump(calibration_data, f)
        
        print(f"✓ Calibration saved: {calib_path.name}")
        
        # Save visualizations
        vis_path = calib_dir / f"{mesh_path.stem}_mesh_keypoints.png"
        cv2.imwrite(str(vis_path), vis_mesh)
        print(f"✓ Saved mesh visualization: {vis_path.name}")
        
        user_vis_path = calib_dir / f"{mesh_path.stem}_user_keypoints.png"
        cv2.imwrite(str(user_vis_path), vis_keypoints)
        print(f"✓ Saved user visualization: {user_vis_path.name}")
        
        depth_vis_path = calib_dir / f"{mesh_path.stem}_depth_calibration.png"
        cv2.imwrite(str(depth_vis_path), vis_combined)
        print(f"✓ Saved depth visualization: {depth_vis_path.name}")
        
        # Save corrected mesh
        corrected_mesh_path = calib_dir / f"{mesh_path.stem}_corrected.obj"
        mesh.export(str(corrected_mesh_path))
        print(f"✓ Saved corrected mesh: {corrected_mesh_path.name}")
    
    cv2.destroyAllWindows()
    
    print("\n" + "="*60)
    print("CALIBRATION COMPLETE!")
    print("="*60)
    print(f"\nCalibration data saved to: {calib_dir}/")
    print("\nWhat was saved:")
    print("  ✓ Background depth map (scene without user)")
    print("  ✓ User depth map (with your pose)")
    print("  ✓ Relative depth map (calibrated user depth)")
    print("  ✓ Mesh keypoints (from 3D clothing)")
    print("  ✓ User keypoints (from your body)")
    print("  ✓ Calibrated mesh with your depth applied")
    print("\nNext step:")
    print("  python tests/test_02_mesh_rendering.py")


if __name__ == "__main__":
    main()