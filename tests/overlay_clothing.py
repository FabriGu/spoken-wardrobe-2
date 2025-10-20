"""
Test 3: 3D Clothing Overlay on Live Video
==========================================
Overlays the 3D clothing mesh on your body in real-time.
Uses body keypoints to position and orient the mesh.

Run from root: python tests/test_03_3d_clothing_overlay.py
"""

import cv2
import numpy as np
import trimesh
from pathlib import Path
import sys
import time

sys.path.append(str(Path(__file__).parent.parent))
from src.modules.body_tracking import BodySegmenter


class Simple3DRenderer:
    """
    Renders 3D mesh using OpenCV projection (no GPU needed for testing).
    For production, you'd use ModernGL or similar.
    """
    
    def __init__(self):
        self.focal_length = 800  # Camera focal length (pixels)
        
    def project_3d_to_2d(self, points_3d, camera_matrix):
        """
        Project 3D points to 2D image coordinates.
        
        Args:
            points_3d: Nx3 array of 3D points
            camera_matrix: 3x3 camera intrinsic matrix
            
        Returns:
            Nx2 array of 2D points
        """
        # Homogeneous coordinates
        points_h = np.hstack([points_3d, np.ones((len(points_3d), 1))])
        
        # Project
        points_2d_h = camera_matrix @ points_3d.T
        points_2d = points_2d_h[:2, :] / points_2d_h[2, :]
        
        return points_2d.T
    
    def render_mesh_wireframe(self, frame, mesh, position, rotation, scale):
        """
        Render mesh as wireframe overlay (fast for testing).
        
        Args:
            frame: Video frame
            mesh: trimesh.Trimesh object
            position: [x, y, z] translation
            rotation: [rx, ry, rz] rotation in degrees
            scale: Uniform scale factor
            
        Returns:
            Frame with mesh rendered
        """
        h, w = frame.shape[:2]
        
        # Create camera matrix
        cx, cy = w / 2, h / 2
        camera_matrix = np.array([
            [self.focal_length, 0, cx],
            [0, self.focal_length, cy],
            [0, 0, 1]
        ])
        
        # Transform mesh vertices
        vertices = mesh.vertices.copy()
        
        # Scale
        vertices *= scale
        
        # Rotate
        from scipy.spatial.transform import Rotation as R
        r = R.from_euler('xyz', rotation, degrees=True)
        vertices = r.apply(vertices)
        
        # Translate
        vertices += position
        
        # Project to 2D
        points_2d = self.project_3d_to_2d(vertices, camera_matrix)
        
        # Convert to integer pixel coordinates
        points_2d = points_2d.astype(np.int32)
        
        # Draw mesh faces
        overlay = frame.copy()
        
        for face in mesh.faces[::10]:  # Draw every 10th face for speed
            pts = points_2d[face]
            
            # Check if triangle is visible
            if (np.all(pts[:, 0] >= 0) and np.all(pts[:, 0] < w) and
                np.all(pts[:, 1] >= 0) and np.all(pts[:, 1] < h)):
                
                # Draw triangle
                cv2.polylines(overlay, [pts], True, (0, 255, 255), 1, cv2.LINE_AA)
        
        # Blend
        result = cv2.addWeighted(frame, 0.5, overlay, 0.5, 0)
        
        return result


class BodyAlignedMeshOverlay:
    """
    Positions and orients 3D mesh based on body tracking.
    This is the key component that makes clothing follow your body.
    """
    
    def __init__(self, mesh_path):
        self.mesh = trimesh.load(mesh_path)
        self.renderer = Simple3DRenderer()
        
        # Center mesh at origin
        self.mesh.vertices -= self.mesh.vertices.mean(axis=0)
        
        print(f"Loaded mesh: {len(self.mesh.vertices)} vertices")
        
    def estimate_body_transform(self, mask, frame_shape):
        """
        Estimate position, rotation, and scale of body from mask.
        
        Returns: (position, rotation, scale) or None
        """
        h, w = frame_shape[:2]
        
        # Find bounding box of body
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if len(contours) == 0:
            return None
        
        largest_contour = max(contours, key=cv2.contourArea)
        x, y, box_w, box_h = cv2.boundingRect(largest_contour)
        
        # Position: center of bounding box
        center_x = x + box_w / 2
        center_y = y + box_h / 2
        
        # Depth: estimate based on box size (larger = closer)
        # Assume person is roughly 1.7m tall and ~2m from camera
        reference_height = 400  # pixels for 2m distance
        estimated_distance = 2000 * (reference_height / max(box_h, 1))
        
        position = np.array([
            center_x - w/2,  # Center at origin
            center_y - h/2,
            estimated_distance
        ])
        
        # Rotation: assume upright for now
        rotation = np.array([0, 0, 0])
        
        # Scale: based on body size
        scale = box_h / 500  # Normalize to expected height
        
        return position, rotation, scale
    
    def render(self, frame, mask):
        """Main render function"""
        
        # Get body transform
        transform = self.estimate_body_transform(mask, frame.shape)
        
        if transform is None:
            return frame
        
        position, rotation, scale = transform
        
        # Render mesh
        result = self.renderer.render_mesh_wireframe(
            frame, self.mesh, position, rotation, scale
        )
        
        return result


def main():
    print("="*60)
    print("TEST 3: 3D CLOTHING OVERLAY ON LIVE VIDEO")
    print("="*60)
    print("\nThis combines everything:")
    print("1. Tracks your body")
    print("2. Positions 3D clothing mesh to match")
    print("3. Renders in real-time")
    print("\nControls:")
    print("  Q - Quit")
    print("  +/- - Adjust scale")
    print("="*60)
    
    # Find generated meshes
    mesh_dir = Path("generated_meshes")
    if not mesh_dir.exists():
        print("\nError: generated_meshes/ not found!")
        print("Run Test 2 first to generate meshes.")
        return
    
    mesh_files = sorted(list(mesh_dir.glob("*.obj")))
    
    if len(mesh_files) == 0:
        print("\nNo mesh files found! Run Test 2 first.")
        return
    
    print(f"\nFound {len(mesh_files)} meshes:")
    for i, mesh in enumerate(mesh_files, 1):
        print(f"  {i}. {mesh.name}")
    
    # Use first mesh
    selected_mesh = mesh_files[0]
    print(f"\nUsing: {selected_mesh.name}")
    
    # Initialize
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    
    segmenter = BodySegmenter(model_type='mobilenet_50')
    segmenter.load_model()
    segmenter.set_preset('torso_and_arms')
    
    overlay_system = BodyAlignedMeshOverlay(str(selected_mesh))
    
    # Settings
    scale_adjustment = 1.0
    
    # Performance
    fps = 0
    frame_count = 0
    fps_start = time.time()
    
    print("\n" + "="*60)
    print("RUNNING - Stand still while mesh aligns!")
    print("="*60)
    
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                continue
            
            frame = cv2.flip(frame, 1)
            
            # Get body mask
            mask = segmenter.get_mask_for_inpainting(frame)
            
            # Render 3D mesh overlay
            result = overlay_system.render(frame, mask)
            
            # Add info
            cv2.putText(result, f"FPS: {fps:.1f}", (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            
            cv2.putText(result, f"Scale: {scale_adjustment:.2f}", (10, 70),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            
            if np.any(mask > 0):
                cv2.putText(result, "Body tracked - Mesh aligned", (10, 110),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            else:
                cv2.putText(result, "No body detected", (10, 110),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            
            cv2.imshow("Test 3: 3D Clothing Overlay", result)
            
            # FPS
            frame_count += 1
            if time.time() - fps_start >= 1.0:
                fps = frame_count / (time.time() - fps_start)
                frame_count = 0
                fps_start = time.time()
            
            # Keys
            key = cv2.waitKey(1) & 0xFF
            
            if key == ord('q'):
                break
            elif key == ord('+') or key == ord('='):
                scale_adjustment *= 1.1
            elif key == ord('-') or key == ord('_'):
                scale_adjustment *= 0.9
    
    finally:
        cap.release()
        cv2.destroyAllWindows()
        print("\n" + "="*60)
        print("TEST 3 COMPLETE!")
        print("="*60)
        print(f"\nFinal FPS: {fps:.1f}")
        print("\nWhat you should see:")
        print("  - Wireframe mesh appearing over your body")
        print("  - Mesh following your movement")
        print("  - Mesh scaling with distance from camera")
        print("\nIf this works, you're ready for production integration!")


if __name__ == "__main__":
    main()