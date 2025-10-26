# combined_pipeline.py
# Integrates your existing BodyPix with new cage deformation

import cv2
import numpy as np
import mediapipe as mp
import time
try:
    from body_segmenter import BodySegmenter  # Your existing class
except Exception:
    # Fallback stub for development/testing when body_segmenter is not available.
    class BodySegmenter:
        def __init__(self, model_type=None):
            self.model_type = model_type
            self.current_preset = None

        def load_model(self):
            # No-op for stub
            print("Warning: BodySegmenter stub in use; load_model() is a no-op.")

        def set_preset(self, preset):
            self.current_preset = preset

        def get_mask_for_inpainting(self, frame, preset=None):
            # Return an empty single-channel mask matching frame size
            h, w = frame.shape[:2]
            return np.zeros((h, w), dtype=np.uint8)

from cage_utils import SimpleCageGenerator, MeanValueCoordinates
from keypoint_mapper import KeypointToCageMapper


class IntegratedClothingOverlay:
    """
    Combines your existing BodyPix segmentation with new cage deformation.
    """
    
    def __init__(self, mesh_path):
        # Your existing BodyPix setup
        self.segmenter = BodySegmenter(model_type='mobilenet_50')
        self.segmenter.load_model()
        self.segmenter.set_preset('torso_and_arms')
        
        # New MediaPipe setup (for real-time pose)
        self.mp_pose = mp.solutions.pose
        self.pose = self.mp_pose.Pose(
            static_image_mode=False,
            model_complexity=1,
            smooth_landmarks=True
        )
        
        # Load your 3D mesh (from your pipeline's output)
        import trimesh
        self.mesh = trimesh.load(mesh_path)
        
        # Generate cage (can use BodyPix to guide this later!)
        cage_gen = SimpleCageGenerator(self.mesh)
        self.cage = cage_gen.generate_simple_box_cage(subdivisions=3)
        
        # Compute MVC weights
        print("Computing MVC weights (one-time setup)...")
        self.mvc = MeanValueCoordinates(self.mesh.vertices, self.cage)
        self.mvc.compute_weights()
        print("Setup complete!")
        
        # Keypoint mapper
        self.mapper = KeypointToCageMapper()
    
    def process_frame(self, frame):
        """
        Process one frame through the complete pipeline.
        """
        # 1. Get BodyPix segmentation (your existing code)
        bodypix_mask = self.segmenter.get_mask_for_inpainting(
            frame, 
            preset=self.segmenter.current_preset
        )
        
        # 2. Get MediaPipe pose
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        pose_results = self.pose.process(frame_rgb)
        
        if not pose_results.pose_landmarks:
            return frame, None  # No person detected
        
        # 3. Map pose to cage positions
        deformed_cage_verts = self.mapper.simple_anatomical_mapping(
            pose_results.pose_landmarks,
            self.cage,
            frame.shape
        )
        
        # 4. Deform mesh using cage
        deformed_mesh_verts = self.mvc.deform_mesh(deformed_cage_verts)
        
        # 5. Return results
        return frame, deformed_mesh_verts


def main():
    """
    Main function following your existing patterns.
    """
    # Initialize (following your BodyPix test pattern)
    overlay = IntegratedClothingOverlay(mesh_path='generated_meshes/3dMesh_1_clothing.obj')
    
    # Camera setup (same as your BodyPix code)
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    
    # FPS tracking (same as your BodyPix code)
    fps = 0
    frame_count = 0
    fps_start_time = time.time()
    
    print("\nStarting integrated pipeline...")
    print("Press 'q' to quit, 's' to save mesh")
    
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                continue
            
            frame = cv2.flip(frame, 1)
            
            # Process frame
            display_frame, deformed_mesh = overlay.process_frame(frame)
            
            # Display (following your pattern)
            if deformed_mesh is not None:
                cv2.putText(display_frame, "Mesh deformed", (10, 30),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            else:
                cv2.putText(display_frame, "No pose detected", (10, 30),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            
            # FPS (same as your BodyPix code)
            frame_count += 1
            elapsed = time.time() - fps_start_time
            if elapsed >= 1.0:
                fps = frame_count / elapsed
                frame_count = 0
                fps_start_time = time.time()
            
            cv2.putText(display_frame, f"FPS: {fps:.1f}", (10, 60),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
            
            cv2.imshow("Integrated Clothing Overlay", display_frame)
            
            # Key handling (same as your BodyPix code)
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q') or key == ord('Q'):
                break
            elif key == ord('s') or key == ord('S'):
                if deformed_mesh is not None:
                    # Save mesh
                    import trimesh
                    mesh = trimesh.Trimesh(
                        vertices=deformed_mesh,
                        faces=overlay.mesh.faces
                    )
                    mesh.export('deformed_output.obj')
                    print("Saved: deformed_output.obj")
    
    finally:
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()