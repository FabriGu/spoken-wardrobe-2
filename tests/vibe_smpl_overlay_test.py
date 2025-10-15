# Save as: tests/vibe_smpl_overlay_test.py

"""
VIBE SMPL Mesh Overlay Test
===========================
This test uses VIBE to extract 3D human pose and shape from real-time video
and generates SMPL meshes as overlay. It measures FPS to evaluate performance.
"""

import cv2
import numpy as np
import time
import sys
from pathlib import Path
import torch
import smplx
import trimesh
from PIL import Image
import torchvision.transforms as transforms

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

# VIBE imports (you may need to adjust these based on your VIBE installation)
try:
    # Assuming VIBE is installed or cloned in your environment
    from vibe.models.vibe import VIBE_Demo
    from vibe.utils.demo_utils import convert_crop_cam_to_orig_img
    from vibe.core.config import VIBE_DATA_DIR
    from vibe.models.smpl import SMPL, SMPL_MODEL_DIR
    from vibe.utils.geometry import batch_rodrigues, perspective_projection
except ImportError:
    print("VIBE not found. Please install VIBE or adjust import paths.")
    print("Clone from: https://github.com/mkocabas/VIBE")
    sys.exit(1)


class VIBETracker:
    """
    Uses VIBE to extract 3D pose and shape from video frames.
    This replaces MediaPipe and provides direct SMPL parameters.
    """
    
    def __init__(self, device='cpu', checkpoint_path=None):
        """
        Initialize VIBE model for real-time inference.
        
        Args:
            device: 'cuda', 'cpu', or 'mps'
            checkpoint_path: Path to VIBE checkpoint (None for default)
        """
        
        print(f"Initializing VIBE model on {device}...")
        
        self.device = device
        
        # Initialize VIBE model
        try:
            self.model = VIBE_Demo(
                seqlen=16,  # Sequence length for temporal modeling
                n_layers=1,
                hidden_size=1024,
                add_linear=True,
                use_residual=True,
            ).to(device)
            
            # Load pretrained weights
            if checkpoint_path is None:
                # Use default VIBE checkpoint
                checkpoint_path = f"{VIBE_DATA_DIR}/vibe_model_w_3dpw.pth.tar"
            
            checkpoint = torch.load(checkpoint_path, map_location=device)
            self.model.load_state_dict(checkpoint['gen_state_dict'])
            self.model.eval()
            
            print(f"VIBE model loaded from: {checkpoint_path}")
            
        except Exception as e:
            print(f"Error loading VIBE model: {e}")
            print("Make sure you have downloaded VIBE pretrained weights.")
            raise
        
        # Image preprocessing for VIBE
        self.normalize_img = transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
        
        # Frame buffer for temporal modeling
        self.frame_buffer = []
        self.max_buffer_size = 16
        
        # Performance tracking
        self.processing_time_ms = 0
        
        # Latest predictions
        self.latest_pose = None
        self.latest_betas = None
        self.latest_cam = None
        
    def preprocess_frame(self, frame):
        """
        Preprocess frame for VIBE input.
        
        Args:
            frame: OpenCV frame (BGR)
            
        Returns:
            Preprocessed tensor
        """
        
        # Convert BGR to RGB
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Resize to VIBE input size (224x224)
        resized = cv2.resize(rgb_frame, (224, 224))
        
        # Convert to PIL Image then to tensor
        pil_img = Image.fromarray(resized)
        img_tensor = transforms.ToTensor()(pil_img)
        
        # Normalize
        img_tensor = self.normalize_img(img_tensor)
        
        return img_tensor
    
    def process_frame(self, frame):
        """
        Process frame through VIBE to get SMPL parameters.
        
        Args:
            frame: OpenCV frame
            
        Returns:
            Dictionary with pose, betas, cam parameters or None
        """
        
        start_time = time.time()
        
        try:
            # Preprocess frame
            img_tensor = self.preprocess_frame(frame)
            
            # Add to frame buffer
            self.frame_buffer.append(img_tensor)
            
            # Keep buffer at max size
            if len(self.frame_buffer) > self.max_buffer_size:
                self.frame_buffer.pop(0)
            
            # Need at least some frames for temporal modeling
            if len(self.frame_buffer) < 8:
                self.processing_time_ms = (time.time() - start_time) * 1000
                return None
            
            # Create batch from buffer
            # For real-time, we'll use the last N frames
            seq_len = min(len(self.frame_buffer), 16)
            input_seq = torch.stack(self.frame_buffer[-seq_len:]).unsqueeze(0).to(self.device)
            
            # VIBE inference
            with torch.no_grad():
                pred_rotmat, pred_betas, pred_camera = self.model(input_seq)
                
                # Get the latest frame predictions (last in sequence)
                latest_rotmat = pred_rotmat[0, -1]  # [24, 3, 3]
                latest_betas = pred_betas[0, -1]    # [10]
                latest_camera = pred_camera[0, -1]  # [3]
                
                # Convert rotation matrices to axis-angle
                pred_pose = batch_rodrigues(latest_rotmat.view(-1, 3, 3)).view(1, -1)
                
                # Store latest predictions
                self.latest_pose = pred_pose
                self.latest_betas = latest_betas.unsqueeze(0)
                self.latest_cam = latest_camera.unsqueeze(0)
                
                self.processing_time_ms = (time.time() - start_time) * 1000
                
                return {
                    'pose': self.latest_pose,
                    'betas': self.latest_betas,
                    'cam': self.latest_cam
                }
                
        except Exception as e:
            print(f"VIBE processing error: {e}")
            self.processing_time_ms = (time.time() - start_time) * 1000
            return None


class SMPLMeshRenderer:
    """
    Takes VIBE output and generates SMPL mesh for overlay.
    This is simplified compared to the MediaPipe version since VIBE provides direct SMPL parameters.
    """
    
    def __init__(self, model_path='models/smplx', device='cpu'):
        """
        Initialize SMPL-X model and renderer.
        
        Args:
            model_path: Path to SMPL-X model files
            device: 'cuda', 'mps', or 'cpu'
        """
        
        print(f"Initializing SMPL-X model on {device}...")
        
        # For Mac with MPS, we need to use CPU for SMPL-X as it doesn't support MPS yet
        if device == 'mps':
            print("Note: SMPL-X doesn't support MPS, falling back to CPU")
            device = 'cpu'
        
        self.device = device
        
        # Initialize SMPL-X model
        self.body_model = smplx.create(
            model_path,
            model_type='smplx',
            gender='neutral',
            use_face_contour=False,
            use_pca=False,
            num_betas=10,
            num_expression_coeffs=10,
            dtype=torch.float32
        ).to(device)
        
        print(f"SMPL-X model loaded: {self.body_model}")
        
        # Mesh rendering setup
        self.vertices = None
        self.faces = self.body_model.faces
        
        # Performance tracking
        self.mesh_generation_ms = 0
        self.rendering_ms = 0
        
    def vibe_to_smpl(self, vibe_output, image_width, image_height):
        """
        Convert VIBE output to SMPL mesh vertices.
        
        Args:
            vibe_output: Dictionary with pose, betas, cam from VIBE
            image_width: Width of the image
            image_height: Height of the image
            
        Returns:
            vertices: Mesh vertices in image coordinates
            faces: Mesh faces (triangles)
        """
        
        if vibe_output is None:
            return None, None
        
        mesh_start = time.time()
        
        try:
            pose = vibe_output['pose']      # [1, 72] - axis-angle rotations
            betas = vibe_output['betas']    # [1, 10] - shape parameters
            cam = vibe_output['cam']        # [1, 3] - camera parameters [scale, tx, ty]
            
            # Split pose into global orientation and body pose
            global_orient = pose[:, :3]      # First 3 values are global orientation
            body_pose = pose[:, 3:66]        # Next 63 values are body pose (21 joints * 3)
            
            # Generate SMPL mesh
            with torch.no_grad():
                output = self.body_model(
                    global_orient=global_orient,
                    body_pose=body_pose,
                    betas=betas,
                    return_verts=True
                )
                
                vertices = output.vertices[0].cpu().numpy()  # [6890, 3]
                
                # Convert from VIBE camera space to image space
                # VIBE cam format: [scale, tx, ty] in normalized coordinates
                scale = cam[0, 0].item()
                tx = cam[0, 1].item()
                ty = cam[0, 2].item()
                
                # Apply camera transformation
                # Scale vertices
                vertices *= scale * 100  # Scale up for visibility
                
                # Apply translation and convert to image coordinates
                vertices[:, 0] += tx * image_width + image_width / 2
                vertices[:, 1] += ty * image_height + image_height / 2
                
                # Flip Y to match OpenCV coordinates (Y-down)
                vertices[:, 1] = image_height - vertices[:, 1]
                
                # Additional scaling and positioning adjustments for better alignment
                # Center the mesh vertically
                center_y = image_height * 0.7  # Place a bit lower for natural standing pose
                vertices[:, 1] += (center_y - np.mean(vertices[:, 1]))
                
                self.mesh_generation_ms = (time.time() - mesh_start) * 1000
                
                return vertices, self.faces
                
        except Exception as e:
            print(f"Error in SMPL mesh generation: {e}")
            self.mesh_generation_ms = (time.time() - mesh_start) * 1000
            return None, None
    
    def render_mesh_overlay(self, frame, vertices, faces, alpha=0.7):
        """
        Render the SMPL mesh as an overlay on the video frame.
        Uses wireframe rendering for speed.
        
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
        
        # Draw mesh vertices as points
        for pt in verts_2d:
            if 0 <= pt[0] < frame.shape[1] and 0 <= pt[1] < frame.shape[0]:
                cv2.circle(overlay, tuple(pt), 1, (0, 0, 255), -1)
        
        # Draw wireframe mesh
        face_subset = faces[::20]  # Use every 20th face for performance
        for face in face_subset:
            # Get vertices of this triangle
            pts = verts_2d[face]
            
            # Check if triangle is visible (within image bounds)
            if (np.all(pts >= 0) and 
                np.all(pts[:, 0] < frame.shape[1]) and 
                np.all(pts[:, 1] < frame.shape[0])):
                
                # Draw triangle edges
                cv2.polylines(overlay, [pts], True, (0, 255, 255), 2, cv2.LINE_AA)
        
        # Blend with original frame
        result = cv2.addWeighted(frame, 1 - alpha, overlay, alpha, 0)
        
        self.rendering_ms = (time.time() - render_start) * 1000
        
        return result


def main():
    """
    Main test loop: Real-time VIBE SMPL mesh overlay on video.
    This tests VIBE integration for real-time human pose and shape estimation.
    """
    
    print("="*60)
    print("VIBE SMPL MESH OVERLAY TEST")
    print("="*60)
    print("\nThis test will:")
    print("1. Extract 3D pose and shape using VIBE")
    print("2. Generate SMPL mesh from VIBE output")
    print("3. Overlay the mesh on your video in real-time")
    print("4. Show FPS to evaluate performance")
    print("\nControls:")
    print("  Q - Quit")
    print("  A - Toggle alpha/transparency")
    print("="*60)
    
    # Check for GPU availability
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"\nUsing device: {device}")
    
    # Initialize camera
    print("\nInitializing camera...")
    cap = cv2.VideoCapture(0)
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
    print(f"Camera resolution: {w}x{h}")
    
    # Initialize VIBE tracker
    print("\nInitializing VIBE tracker...")
    try:
        vibe_tracker = VIBETracker(device=device)
    except Exception as e:
        print(f"Error initializing VIBE: {e}")
        print("\nMake sure you have:")
        print("- Downloaded VIBE pretrained weights")
        print("- Installed VIBE dependencies")
        return
    
    # Initialize SMPL renderer
    print("\nInitializing SMPL-X model...")
    try:
        mesh_renderer = SMPLMeshRenderer(
            model_path='models/smplx',
            device=device
        )
    except Exception as e:
        print(f"Error initializing SMPL-X: {e}")
        print("\nMake sure you have downloaded the SMPL-X model files to models/smplx/")
        print("Download from: https://smpl-x.is.tue.mpg.de/")
        return
    
    # Settings
    alpha = 0.7
    
    # Performance tracking
    fps = 0
    frame_count = 0
    fps_start = time.time()
    
    print("\n" + "="*60)
    print("RUNNING - Stand in front of camera!")
    print("VIBE needs a few frames to initialize...")
    print("="*60)
    
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                continue
            
            # Mirror for natural interaction
            frame = cv2.flip(frame, 1)
            
            # Process frame through VIBE
            vibe_output = vibe_tracker.process_frame(frame)
            
            # Generate SMPL mesh from VIBE output
            if vibe_output is not None:
                vertices, faces = mesh_renderer.vibe_to_smpl(
                    vibe_output, w, h
                )
                
                # Render mesh overlay
                if vertices is not None:
                    result = mesh_renderer.render_mesh_overlay(
                        frame, vertices, faces, alpha=alpha
                    )
                else:
                    result = frame
            else:
                result = frame
            
            # Calculate total processing time
            total_time = (vibe_tracker.processing_time_ms + 
                         mesh_renderer.mesh_generation_ms +
                         mesh_renderer.rendering_ms)
            
            # Add performance overlay
            cv2.putText(result, f"FPS: {fps:.1f}", (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            
            cv2.putText(result, f"Total: {total_time:.1f}ms", (10, 70),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            
            # Breakdown of processing times
            cv2.putText(result, f"VIBE Processing: {vibe_tracker.processing_time_ms:.1f}ms", 
                       (10, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
            
            cv2.putText(result, f"Mesh Generation: {mesh_renderer.mesh_generation_ms:.1f}ms", 
                       (10, 125), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
            
            cv2.putText(result, f"Rendering: {mesh_renderer.rendering_ms:.1f}ms", 
                       (10, 150), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
            
            # Status indicator
            status = "VIBE active - Mesh rendering" if vibe_output else "VIBE initializing..."
            color = (0, 255, 0) if vibe_output else (0, 165, 255)
            cv2.putText(result, status, (10, 180),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
            
            # Buffer status
            buffer_status = f"Frame buffer: {len(vibe_tracker.frame_buffer)}/16"
            cv2.putText(result, buffer_status, (10, 210),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (150, 150, 150), 1)
            
            # Display
            cv2.imshow("VIBE SMPL Mesh Overlay Test", result)
            
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
            
            elif key == ord('a') or key == ord('A'):
                alpha = 0.3 if alpha > 0.5 else 0.7
                print(f"Alpha: {alpha}")
    
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
        print(f"- VIBE processing: ~{vibe_tracker.processing_time_ms:.1f}ms")
        print(f"- SMPL generation: ~{mesh_renderer.mesh_generation_ms:.1f}ms")
        print(f"- Total pipeline: ~{total_time:.1f}ms")
        print("\nVIBE Real-time Performance:")
        if fps >= 15:
            print("✓ Good performance for real-time use")
        else:
            print("⚠ Performance may be too slow for smooth real-time")
            print("Consider:")
            print("- Using GPU acceleration")
            print("- Reducing camera resolution")
            print("- Optimizing VIBE sequence length")


if __name__ == "__main__":
    main()