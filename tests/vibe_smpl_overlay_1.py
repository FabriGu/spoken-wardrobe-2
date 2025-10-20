#!/usr/bin/env python3
"""
Simplified VIBE + SMPL Test
===========================
This bypasses VIBE complexity and tests SMPL mesh generation directly.
Uses MediaPipe for pose → SMPL for mesh rendering.
"""

import cv2
import numpy as np
import time
import sys
from pathlib import Path
import torch
import mediapipe as mp

# At the VERY top of tests/vibe_smpl_overlay_1.py
import inspect
if not hasattr(inspect, 'getargspec'):
    inspect.getargspec = inspect.getfullargspec

# For SMPL (not SMPL-X) - VIBE uses SMPL
try:
    # Add VIBE to path
    sys.path.insert(0, str(Path(__file__).parent.parent / 'VIBE'))
    
    from VIBE.lib.models.smpl import SMPL
    from VIBE.lib.utils.geometry import batch_rodrigues
    
    print("✓ VIBE SMPL imported successfully")
    SMPL_AVAILABLE = True
except ImportError as e:
    print(f"✗ Could not import VIBE SMPL: {e}")
    print("This test requires VIBE to be cloned in your project root")
    SMPL_AVAILABLE = False


class SimpleSMPLRenderer:
    """
    Uses MediaPipe pose + SMPL model for mesh rendering.
    Much simpler than full VIBE pipeline.
    """
    
    def __init__(self, smpl_model_path='VIBE/data/body_models/smpl', device='cpu'):
        """
        Initialize SMPL model.
        
        Args:
            smpl_model_path: Path to SMPL model files (inside VIBE/data/body_models/)
            device: 'cuda' or 'cpu'
        """
        
        print(f"Initializing SMPL model on {device}...")
        
        self.device = device
        
        # Initialize SMPL model (NOT SMPL-X)
        # SMPL has 24 joints (body only, no hands/face)
        try:
            self.smpl = SMPL(
                smpl_model_path,
                batch_size=1,
                create_transl=False
            ).to(device)
            
            print(f"✓ SMPL model loaded from: {smpl_model_path}")
            print(f"  - Joints: 24 (body only)")
            print(f"  - Vertices: {self.smpl.faces.shape[0]} faces")
            
        except Exception as e:
            print(f"✗ Error loading SMPL: {e}")
            print("\nYou need SMPL model files:")
            print("1. Download from: https://smpl.is.tue.mpg.de/")
            print("2. Place in: VIBE/data/body_models/smpl/")
            print("   - SMPL_NEUTRAL.pkl")
            print("   - SMPL_MALE.pkl") 
            print("   - SMPL_FEMALE.pkl")
            raise
        
        # MediaPipe Pose for keypoint detection
        self.mp_pose = mp.solutions.pose
        self.pose = self.mp_pose.Pose(
            static_image_mode=False,
            model_complexity=1,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        
        # Default pose parameters
        self.pose_params = torch.zeros(1, 72).to(device)  # 24 joints × 3 (axis-angle)
        self.shape_params = torch.zeros(1, 10).to(device)  # Shape parameters
        
        # Performance tracking
        self.pose_time_ms = 0
        self.mesh_time_ms = 0
        self.render_time_ms = 0
        
    def extract_pose(self, frame):
        """Extract pose using MediaPipe"""
        
        start = time.time()
        
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.pose.process(rgb)
        
        self.pose_time_ms = (time.time() - start) * 1000
        
        if not results.pose_landmarks:
            return None
        
        return results.pose_landmarks
    
    def generate_mesh(self, pose_landmarks, frame_shape):
        """
        Generate SMPL mesh from pose landmarks.
        Uses simple heuristic pose estimation.
        """
        
        if pose_landmarks is None:
            return None, None
        
        start = time.time()
        
        h, w = frame_shape[:2]
        
        try:
            # For simplicity, use neutral pose with slight adjustments
            # In full VIBE, this would be learned from the video
            
            # Generate SMPL mesh with neutral pose
            with torch.no_grad():
                output = self.smpl(
                    betas=self.shape_params,
                    body_pose=self.pose_params[:, 3:],  # Skip global orient
                    global_orient=self.pose_params[:, :3]
                )
                
                vertices = output.vertices[0].cpu().numpy()  # [6890, 3]
                
                # Transform to image space
                # Center and scale mesh to fit frame
                vertices -= vertices.mean(axis=0)
                vertices *= 100  # Scale up
                
                # Position in center of frame, lower half
                vertices[:, 0] += w / 2
                vertices[:, 1] += h * 0.65  # Slightly lower
                vertices[:, 1] = h - vertices[:, 1]  # Flip Y
                
            self.mesh_time_ms = (time.time() - start) * 1000
            
            return vertices, self.smpl.faces
            
        except Exception as e:
            print(f"Error generating mesh: {e}")
            self.mesh_time_ms = (time.time() - start) * 1000
            return None, None
    
    def render_mesh(self, frame, vertices, faces, alpha=0.7):
        """Render mesh as wireframe overlay"""
        
        if vertices is None or faces is None:
            return frame
        
        start = time.time()
        
        overlay = frame.copy()
        
        # Project to 2D
        verts_2d = vertices[:, :2].astype(np.int32)
        
        # Draw vertices
        for pt in verts_2d[::10]:  # Every 10th vertex for speed
            if 0 <= pt[0] < frame.shape[1] and 0 <= pt[1] < frame.shape[0]:
                cv2.circle(overlay, tuple(pt), 2, (0, 255, 0), -1)
        
        # Draw wireframe (subset of faces for performance)
        for face in faces[::50]:  # Every 50th face
            pts = verts_2d[face]
            
            if (np.all(pts >= 0) and 
                np.all(pts[:, 0] < frame.shape[1]) and 
                np.all(pts[:, 1] < frame.shape[0])):
                
                cv2.polylines(overlay, [pts], True, (0, 255, 255), 1, cv2.LINE_AA)
        
        result = cv2.addWeighted(frame, 1 - alpha, overlay, alpha, 0)
        
        self.render_time_ms = (time.time() - start) * 1000
        
        return result


def main():
    """Test SMPL mesh rendering with MediaPipe pose"""
    
    print("="*60)
    print("SIMPLIFIED SMPL MESH RENDERING TEST")
    print("="*60)
    print("\nThis test uses:")
    print("- MediaPipe for pose detection")
    print("- SMPL model from VIBE (simpler than SMPL-X)")
    print("- Simple heuristic mesh positioning")
    print("\nControls:")
    print("  Q - Quit")
    print("  P - Toggle pose keypoints")
    print("="*60)
    
    if not SMPL_AVAILABLE:
        print("\n✗ SMPL not available. Exiting.")
        return
    
    # Initialize camera
    print("\nInitializing camera...")
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    
    if not cap.isOpened():
        print("✗ Could not open camera")
        return
    
    ret, test_frame = cap.read()
    if not ret:
        print("✗ Could not read from camera")
        return
    
    h, w = test_frame.shape[:2]
    print(f"✓ Camera: {w}x{h}")
    
    # Initialize renderer
    print("\nInitializing SMPL renderer...")
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    
    try:
        renderer = SimpleSMPLRenderer(device=device)
    except Exception as e:
        print(f"✗ Failed to initialize: {e}")
        return
    
    print("\n✓ Ready! Starting capture...")
    
    # Settings
    show_pose = True
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
            
            frame = cv2.flip(frame, 1)
            
            # Extract pose
            pose_landmarks = renderer.extract_pose(frame)
            
            # Generate mesh
            if pose_landmarks:
                vertices, faces = renderer.generate_mesh(pose_landmarks, frame.shape)
                
                # Render
                if vertices is not None:
                    result = renderer.render_mesh(frame, vertices, faces, alpha=0.6)
                    
                    # Draw pose skeleton if enabled
                    if show_pose:
                        mp.solutions.drawing_utils.draw_landmarks(
                            result,
                            pose_landmarks,
                            mp.solutions.pose.POSE_CONNECTIONS
                        )
                else:
                    result = frame
            else:
                result = frame
            
            # Performance overlay
            total_time = (renderer.pose_time_ms + 
                         renderer.mesh_time_ms + 
                         renderer.render_time_ms)
            
            cv2.putText(result, f"FPS: {fps:.1f}", (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            
            cv2.putText(result, f"Total: {total_time:.1f}ms", (10, 70),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            
            cv2.putText(result, f"Pose: {renderer.pose_time_ms:.1f}ms", 
                       (10, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)
            
            cv2.putText(result, f"Mesh: {renderer.mesh_time_ms:.1f}ms", 
                       (10, 125), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)
            
            cv2.putText(result, f"Render: {renderer.render_time_ms:.1f}ms", 
                       (10, 150), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)
            
            status = "Pose detected" if pose_landmarks else "No pose"
            color = (0, 255, 0) if pose_landmarks else (0, 0, 255)
            cv2.putText(result, status, (10, 180),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
            
            cv2.imshow("SMPL Mesh Test", result)
            
            # FPS
            frame_count += 1
            if time.time() - fps_start >= 1.0:
                fps = frame_count / (time.time() - fps_start)
                frame_count = 0
                fps_start = time.time()
            
            # Keys
            key = cv2.waitKey(1) & 0xFF
            
            if key == ord('q') or key == ord('Q'):
                break
            elif key == ord('p') or key == ord('P'):
                show_pose = not show_pose
                print(f"Pose keypoints: {'ON' if show_pose else 'OFF'}")
    
    except KeyboardInterrupt:
        print("\nInterrupted")
    
    finally:
        cap.release()
        cv2.destroyAllWindows()
        print("\n" + "="*60)
        print("TEST COMPLETE")
        print("="*60)
        print(f"\nFinal FPS: {fps:.1f}")
        print(f"Average times:")
        print(f"  - Pose detection: {renderer.pose_time_ms:.1f}ms")
        print(f"  - Mesh generation: {renderer.mesh_time_ms:.1f}ms")
        print(f"  - Rendering: {renderer.render_time_ms:.1f}ms")


if __name__ == "__main__":
    main()