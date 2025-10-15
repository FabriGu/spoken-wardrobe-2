# Save as: tests/test_smpl_mesh_overlay.py

"""
SMPL Mesh Overlay Test
======================
This test takes MediaPipe keypoints and generates SMPL meshes in real-time.
It measures the FPS to evaluate if this approach is viable for your application.
"""

import cv2
import numpy as np
import time
import sys
from pathlib import Path
import mediapipe as mp
import torch
import smplx
import trimesh
from scipy.spatial.transform import Rotation as R

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))


class SMPLMeshRenderer:
    """
    Takes MediaPipe keypoints and generates SMPL mesh for overlay.
    This class handles the conversion from 2D pose to 3D mesh.
    """
    
    # MediaPipe to SMPL joint mapping
    # MediaPipe has 33 joints, SMPL-X has 22 body joints
    # This mapping connects corresponding joints between the two systems
    MEDIAPIPE_TO_SMPL_JOINTS = {
        # MediaPipe index: SMPL-X joint name
        0: 'nose',          # Head approximation
        11: 'left_shoulder',
        12: 'right_shoulder',
        13: 'left_elbow',
        14: 'right_elbow',
        15: 'left_wrist',
        16: 'right_wrist',
        23: 'left_hip',
        24: 'right_hip',
        25: 'left_knee',
        26: 'right_knee',
        27: 'left_ankle',
        28: 'right_ankle',
    }
    
    def __init__(self, model_path='models/smplx', device='cpu', use_optimization=False):
        """
        Initialize SMPL-X model and renderer.
        
        Args:
            model_path: Path to SMPL-X model files
            device: 'cuda', 'mps', or 'cpu' 
            use_optimization: If True, use SMPLify-style optimization (slower but more accurate)
        """
        
        print(f"Initializing SMPL-X model on {device}...")
        
        # For Mac with MPS, we need to use CPU for SMPL-X as it doesn't support MPS yet
        if device == 'mps':
            print("Note: SMPL-X doesn't support MPS, falling back to CPU")
            device = 'cpu'
        
        self.device = device
        self.use_optimization = use_optimization
        
        # Initialize SMPL-X model
        # We use SMPL-X in a simplified mode (only body, no hands/face for speed)
        self.body_model = smplx.create(
            model_path,
            model_type='smplx',
            gender='neutral',
            use_face_contour=False,
            use_pca=False,  # Don't use PCA for hands (faster)
            num_betas=10,   # Shape parameters
            num_expression_coeffs=10,
            dtype=torch.float32
        ).to(device)
        
        # print(f"SMPL-X model loaded with {self.body_model.num_joints} joints")
        print(f"SMPL-X model loaded: {self.body_model}")
        print(f"Output vertices shape: {self.body_model.faces.shape}")  # mesh triangles info
# Optionally: print("Body model output example:", self.body_model)

        
        # Initialize pose parameters (these get updated each frame)
        # self.current_pose = torch.zeros(1, 72).to(device)  # 72 = 23 joints * 3 (axis-angle)
        self.current_pose = torch.zeros(1, 63).to(device)  # 69 = 23 joints * 3 (axis-angle), excluding global orient
        self.current_betas = torch.zeros(1, 10).to(device)  # Shape stays constant
        self.current_transl = torch.zeros(1, 3).to(device)  # Translation in 3D space
        
        # Mesh rendering setup
        self.vertices = None
        self.faces = self.body_model.faces
        
        # Performance tracking
        self.pose_estimation_ms = 0
        self.mesh_generation_ms = 0
        self.rendering_ms = 0
        
    def keypoints_to_smpl(self, mediapipe_keypoints_3d, image_width, image_height):
        """
        Convert MediaPipe 3D keypoints to SMPL mesh.
        
        This is the core function that takes 2D/3D pose and generates a mesh.
        We use a simplified approach for speed: direct pose estimation without optimization.
        
        Args:
            mediapipe_keypoints_3d: MediaPipe pose landmarks (with x, y, z coordinates)
            image_width: Width of the image (for scaling)
            image_height: Height of the image (for scaling)
            
        Returns:
            vertices: Mesh vertices in image coordinates
            faces: Mesh faces (triangles)
        """
        
        if mediapipe_keypoints_3d is None:
            return None, None
        
        pose_start = time.time()
        
        # Extract relevant joints from MediaPipe
        # We'll estimate SMPL pose from these key joints
        joints_3d = []
        for mp_idx in self.MEDIAPIPE_TO_SMPL_JOINTS.keys():
            landmark = mediapipe_keypoints_3d.landmark[mp_idx]
            # MediaPipe gives normalized coordinates, convert to image space
            x = landmark.x * image_width - image_width/2  # Center at origin
            y = landmark.y * image_height - image_height/2
            z = landmark.z * 100  # Scale depth (MediaPipe z is quite small)
            joints_3d.append([x, y, z])
        
        joints_3d = np.array(joints_3d)
        
        # Simple pose estimation: compute joint rotations from positions
        # This is a simplified version - full SMPLify would optimize these
        if self.use_optimization:
            # Would implement full optimization here (slower but more accurate)
            # For now, we'll use a heuristic approach
            pass
        
        # Heuristic pose estimation (fast but approximate)
        # Estimate global orientation from shoulders and hips
        left_shoulder = joints_3d[1]  # Index 1 in our extracted joints
        right_shoulder = joints_3d[2]
        left_hip = joints_3d[7]
        right_hip = joints_3d[8]
        
        # Compute torso orientation
        shoulder_vec = right_shoulder - left_shoulder
        hip_vec = right_hip - left_hip
        forward_vec = np.cross(shoulder_vec, [0, -1, 0])  # Assuming Y-up
        forward_vec = forward_vec / (np.linalg.norm(forward_vec) + 1e-8)
        
        # Convert to rotation matrix then axis-angle
        # This is simplified - just setting global orientation
        # angle = np.arctan2(forward_vec[0], forward_vec[2])
        # global_orient = torch.tensor([[0, angle, 0]], dtype=torch.float32).to(self.device)

        # --- FIX 1: Make the mesh face the camera (flip Z-axis) ---
        angle = np.arctan2(forward_vec[0], forward_vec[2])
        global_orient = torch.tensor([[0, angle + np.pi, 0]], dtype=torch.float32).to(self.device)

        
        # Set translation to center of hips
        transl = torch.tensor([[(left_hip[0] + right_hip[0])/2, 
                                (left_hip[1] + right_hip[1])/2,
                                (left_hip[2] + right_hip[2])/2]], 
                             dtype=torch.float32).to(self.device)
        
        self.pose_estimation_ms = (time.time() - pose_start) * 1000
        
        # Generate SMPL mesh
        mesh_start = time.time()
        
        with torch.no_grad():
            # output = self.body_model(
            #     global_orient=global_orient,
            #     body_pose=self.current_pose[:, 3:],  # Keep rest of pose neutral
            #     betas=self.current_betas,
            #     transl=transl,
            #     return_verts=True
            # )
            output = self.body_model(
                global_orient=global_orient,              # (1, 3), axis-angle
                body_pose=self.current_pose,              # (1, 63), 21*3 axis-angle
                betas=self.current_betas,
                transl=transl,
                return_verts=True
            )
            
            vertices = output.vertices[0].cpu().numpy()

            # --- DEBUG SCALE + TRANSLATION FIX ---
            # --- Recenter and scale mesh properly (FIX FOR OFFSCREEN MESH) ---

            # Move mesh so its mean is near origin before scaling
            vertices -= np.mean(vertices, axis=0)

            # Scale to roughly image size (meters → pixels)
            scale = image_height / 2.0
            vertices *= scale

            # Flip Y to match OpenCV coordinates (Y-down)
            vertices[:, 1] *= -1

            # Center mesh on image
            center_x = image_width / 2
            center_y = image_height / 1.1  # a bit lower to simulate standing on ground
            vertices[:, 0] += center_x
            vertices[:, 1] += center_y




            
        self.mesh_generation_ms = (time.time() - mesh_start) * 1000
        
        # Convert back to image coordinates
        # vertices[:, 0] += image_width/2
        # vertices[:, 1] += image_height/2

        # # --- DEBUG TRANSLATE TO IMAGE CENTER ---
        # vertices[:, 0] += image_width / 2
        # vertices[:, 1] += image_height / 1.1  # shift vertically for better placement
        # # ---------------------------------

        
        self.vertices = vertices
        # trimesh.Trimesh(vertices, self.faces).show()
        # print(f"Vertices shape: {vertices.shape}")
        # print(f"Vertices range X: {vertices[:,0].min():.1f}–{vertices[:,0].max():.1f}")
        # print(f"Vertices range Y: {vertices[:,1].min():.1f}–{vertices[:,1].max():.1f}")
        # print(f"Vertices range Z: {vertices[:,2].min():.3f}–{vertices[:,2].max():.3f}")


        return vertices, self.faces
    
    def render_mesh_overlay(self, frame, vertices, faces, alpha=0.9):
        """
        Render the SMPL mesh as an overlay on the video frame.
        Uses simple wireframe rendering for speed.
        
        Args:
            frame: Video frame to overlay on
            vertices: Mesh vertices in image space
            faces: Mesh faces
            alpha: Transparency of overlay
            
        Returns:
            Frame with mesh overlay
        """
        
        
        if vertices is None or faces is None:
            return frame
        
            
        
        render_start = time.time()
        
        overlay = frame.copy()
        
        # Project 3D vertices to 2D (they're already in image space)
        verts_2d = vertices[:, :2].astype(np.int32)

        for pt in verts_2d:
            cv2.circle(overlay, tuple(pt), 1, (0, 0, 255), -1)
        
        # Draw wireframe mesh
        # We'll draw edges of triangles for visualization
        for face in faces:
            # Get vertices of this triangle
            pts = verts_2d[face]
            
            # Check if triangle is visible (not behind camera)
            if np.all(pts >= 0) and np.all(pts[:, 0] < frame.shape[1]) and np.all(pts[:, 1] < frame.shape[0]):
                # Draw triangle edges
                # cv2.polylines(overlay, [pts], True, (0, 255, 255), 1, cv2.LINE_AA) // not visible wireframe
                cv2.polylines(overlay, [pts], True, (0, 255, 255), 2, cv2.LINE_AA)

        
        # Blend with original frame
        result = cv2.addWeighted(frame, 1 - alpha, overlay, alpha, 0)
        
        self.rendering_ms = (time.time() - render_start) * 1000

        # print("Vertices min/max:", vertices.min(axis=0), vertices.max(axis=0))
        # print("First face verts:", verts_2d[faces[0]])  
        
        return result


class MediaPipePoseTracker:
    """
    Extracts 3D pose from video using MediaPipe.
    This provides the keypoints that drive the SMPL mesh.
    """
    
    def __init__(self):
        self.mp_pose = mp.solutions.pose
        self.pose = self.mp_pose.Pose(
            static_image_mode=False,
            model_complexity=1,  # 1 for balance of speed and accuracy
            enable_segmentation=False,  # We don't need segmentation for this test
            smooth_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        
        self.processing_time_ms = 0
        self.latest_landmarks = None
        
    def process_frame(self, frame):
        """Extract 3D pose landmarks from frame"""
        
        start = time.time()
        
        # Convert BGR to RGB for MediaPipe
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Process the frame
        results = self.pose.process(rgb)
        
        self.processing_time_ms = (time.time() - start) * 1000
        
        self.latest_landmarks = results.pose_landmarks
        
        return results.pose_landmarks


def main():
    """
    Main test loop: Real-time SMPL mesh overlay on video.
    This tests the feasibility and performance of the approach.
    """
    
    print("="*60)
    print("SMPL MESH OVERLAY TEST")
    print("="*60)
    print("\nThis test will:")
    print("1. Extract pose keypoints using MediaPipe")
    print("2. Generate SMPL mesh from those keypoints")
    print("3. Overlay the mesh on your video in real-time")
    print("4. Show FPS to evaluate performance")
    print("\nControls:")
    print("  Q - Quit")
    print("  W - Toggle wireframe/solid rendering")
    print("  O - Toggle optimization (slower but more accurate)")
    print("="*60)
    
    # Initialize camera
    print("\nInitializing camera...")
    cap = cv2.VideoCapture(0)  # Change to 1 if needed
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    
    if not cap.isOpened():
        print("Error: Could not open camera")
        return
    
    # Get frame dimensions
    ret, test_frame = cap.read()
    if not ret:
        print("Error: Could not read from camera")
        return
    h, w = test_frame.shape[:2]
    
    # Initialize pose tracker
    print("\nInitializing MediaPipe pose tracker...")
    pose_tracker = MediaPipePoseTracker()
    
    # Initialize SMPL renderer
    print("\nInitializing SMPL-X model...")
    try:
        # Use CPU for compatibility (change to 'cuda' if you have NVIDIA GPU)
        mesh_renderer = SMPLMeshRenderer(
            model_path='models/smplx',
            device='cpu',
            use_optimization=False  # Start with fast mode
        )
    except Exception as e:
        print(f"Error initializing SMPL-X: {e}")
        print("\nMake sure you have downloaded the SMPL-X model files to models/smplx/")
        print("Download from: https://smpl-x.is.tue.mpg.de/")
        return
    
    # Settings
    show_wireframe = True
    
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
            
            # Mirror for natural interaction
            frame = cv2.flip(frame, 1)
            
            # Extract pose keypoints
            pose_landmarks = pose_tracker.process_frame(frame)

            # --- DEBUG: draw MediaPipe keypoints ---
            if pose_landmarks:
                for lm in pose_landmarks.landmark:
                    cx, cy = int(lm.x * w), int(lm.y * h)
                    cv2.circle(frame, (cx, cy), 3, (0, 255, 0), -1)
            # --------------------------------------

            
            # Generate SMPL mesh from pose
            if pose_landmarks:
                vertices, faces = mesh_renderer.keypoints_to_smpl(
                    pose_landmarks, w, h
                )
                
                # Render mesh overlay
                if vertices is not None:
                    result = mesh_renderer.render_mesh_overlay(
                        frame, vertices, faces, alpha=0.9
                    )
                else:
                    result = frame
            else:
                result = frame
            
            # Calculate total processing time
            total_time = (pose_tracker.processing_time_ms + 
                         mesh_renderer.pose_estimation_ms +
                         mesh_renderer.mesh_generation_ms +
                         mesh_renderer.rendering_ms)
            
            # Add performance overlay
            cv2.putText(result, f"FPS: {fps:.1f}", (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            
            cv2.putText(result, f"Total: {total_time:.1f}ms", (10, 70),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            
            # Breakdown of processing times
            cv2.putText(result, f"Pose Detection: {pose_tracker.processing_time_ms:.1f}ms", 
                       (10, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
            
            cv2.putText(result, f"Pose->SMPL: {mesh_renderer.pose_estimation_ms:.1f}ms", 
                       (10, 125), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
            
            cv2.putText(result, f"Mesh Generation: {mesh_renderer.mesh_generation_ms:.1f}ms", 
                       (10, 150), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
            
            cv2.putText(result, f"Rendering: {mesh_renderer.rendering_ms:.1f}ms", 
                       (10, 175), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
            
            # Status indicator
            status = "Pose detected - Mesh active" if pose_landmarks else "No pose detected"
            color = (0, 255, 0) if pose_landmarks else (0, 0, 255)
            cv2.putText(result, status, (10, 210),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
            
            # Display
            cv2.imshow("SMPL Mesh Overlay Test", result)
            # continue
            
            # Update FPS counter
            frame_count += 1
            elapsed = time.time() - fps_start
            if elapsed >= 1.0:
                fps = frame_count / elapsed
                frame_count = 0
                fps_start = time.time()
            
            # Handle keyboard input
            key = cv2.waitKey(1) & 0xFF
            
            if key == ord('q') or key == ord('Q'):
                break
            
            elif key == ord('w') or key == ord('W'):
                show_wireframe = not show_wireframe
                print(f"Wireframe: {show_wireframe}")
            
            elif key == ord('o') or key == ord('O'):
                mesh_renderer.use_optimization = not mesh_renderer.use_optimization
                print(f"Optimization: {mesh_renderer.use_optimization}")
                if mesh_renderer.use_optimization:
                    print("Note: This will be slower but more accurate")
    
    except KeyboardInterrupt:
        print("\nInterrupted")
    
    finally:
        cap.release()
        cv2.destroyAllWindows()
        print("\n" + "="*60)
        print("TEST COMPLETE")
        print("="*60)
        print("\nPerformance Summary:")
        print(f"- Final FPS: {fps:.1f}")
        print(f"- Pose detection: ~{pose_tracker.processing_time_ms:.1f}ms")
        print(f"- SMPL generation: ~{mesh_renderer.mesh_generation_ms:.1f}ms")
        
        # print(f"- Total pipeline: ~{total_time:.1f}ms")
        print("\nIs this fast enough for your application?")
        print("Target: 15+ FPS for smooth real-time experience")
        print("If too slow, consider:")
        print("- Reducing MediaPipe model complexity")
        print("- Simplifying SMPL model (use SMPL instead of SMPL-X)")
        print("- Using GPU acceleration if available")


if __name__ == "__main__":
    main()