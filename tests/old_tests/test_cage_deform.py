# test_cage_deformation.py
# Simple test script for cage-based deformation with MediaPipe
# This creates a complete working prototype

import cv2
import numpy as np
import mediapipe as mp
import time
import trimesh
from cage_utils import SimpleCageGenerator, MeanValueCoordinates
from keypoint_mapper import KeypointToCageMapper


class CageDeformationTester:
    """
    Test cage-based deformation with live camera feed and MediaPipe.
    """
    
    def __init__(self, mesh_path=None):
        """
        Initialize the tester.
        
        Args:
            mesh_path: Path to 3D mesh file (OBJ, STL, etc.)
                      If None, creates a simple test mesh
        """
        # Initialize MediaPipe Pose
        self.mp_pose = mp.solutions.pose
        self.pose = self.mp_pose.Pose(
            static_image_mode=False,
            model_complexity=1,  # 0, 1, or 2 (higher = more accurate but slower)
            smooth_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        
        # Load or create mesh
        if mesh_path is not None:
            print(f"Loading mesh from {mesh_path}...")
            self.mesh = trimesh.load(mesh_path)
        else:
            print("Creating simple test mesh...")
            self.mesh = self.create_simple_tshirt_mesh()
        
        print(f"Mesh loaded: {len(self.mesh.vertices)} vertices, {len(self.mesh.faces)} faces")
        
        # Generate cage
        print("\nGenerating cage...")
        cage_generator = SimpleCageGenerator(self.mesh)
        self.cage = cage_generator.generate_simple_box_cage(subdivisions=3)
        
        # Compute MVC weights (one-time setup)
        print("\nComputing Mean Value Coordinates...")
        self.mvc = MeanValueCoordinates(self.mesh.vertices, self.cage)
        self.mvc.compute_weights()
        
        # Initialize keypoint mapper
        self.keypoint_mapper = KeypointToCageMapper()
        
        # Store original cage for reference
        self.original_cage_vertices = np.array(self.cage.vertices).copy()
        
        print("\n✓ Initialization complete!")
    
    def create_simple_tshirt_mesh(self):
        """
        Create a simple t-shirt-like mesh for testing.
        """
        # Create a simple box-like shape
        vertices = []
        
        # Torso (rectangular prism)
        width, height, depth = 0.5, 0.7, 0.2
        
        # Front face vertices
        vertices.extend([
            [-width/2, height, -depth/2],
            [width/2, height, -depth/2],
            [width/2, 0, -depth/2],
            [-width/2, 0, -depth/2],
        ])
        
        # Back face vertices
        vertices.extend([
            [-width/2, height, depth/2],
            [width/2, height, depth/2],
            [width/2, 0, depth/2],
            [-width/2, 0, depth/2],
        ])
        
        # Add some sleeve vertices (simplified)
        sleeve_length = 0.3
        for side in [-1, 1]:  # left and right
            x_offset = side * width/2
            vertices.extend([
                [x_offset, height - 0.1, -depth/2],
                [x_offset + side * sleeve_length, height - 0.1, -depth/2],
                [x_offset + side * sleeve_length, height - 0.3, -depth/2],
                [x_offset, height - 0.3, -depth/2],
            ])
        
        vertices = np.array(vertices)
        
        # Create simple faces (just enough to make it renderable)
        # For prototype, we'll just create convex hull
        from scipy.spatial import ConvexHull
        hull = ConvexHull(vertices)
        faces = hull.simplices
        
        mesh = trimesh.Trimesh(vertices=vertices, faces=faces)
        return mesh
    
    def run_test(self):
        """
        Run the cage deformation test with live camera.
        """
        # Initialize camera
        cap = cv2.VideoCapture(0)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        
        if not cap.isOpened():
            print("Error: Could not open camera")
            return
        
        print("\n" + "="*60)
        print("CAGE DEFORMATION TEST")
        print("="*60)
        print("Press 'q' to quit")
        print("Press 's' to save deformed mesh")
        print("="*60 + "\n")
        
        # Performance tracking
        fps = 0
        frame_count = 0
        fps_start_time = time.time()
        
        save_counter = 0
        
        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    print("Failed to read frame")
                    continue
                
                # Mirror frame
                frame = cv2.flip(frame, 1)
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                
                # Process with MediaPipe
                start_time = time.time()
                results = self.pose.process(frame_rgb)
                mediapipe_time = (time.time() - start_time) * 1000
                
                # If pose detected, do deformation
                deform_time = 0
                if results.pose_landmarks:
                    start_time = time.time()
                    
                    # Map keypoints to cage
                    # Using simplified version without depth map for prototype
                    deformed_cage_verts = self.keypoint_mapper.simple_anatomical_mapping(
                        results.pose_landmarks,
                        self.cage,
                        frame.shape
                    )
                    
                    # Deform mesh using cage
                    deformed_mesh_verts = self.mvc.deform_mesh(deformed_cage_verts)
                    
                    deform_time = (time.time() - start_time) * 1000
                    
                    # Draw MediaPipe skeleton
                    mp.solutions.drawing_utils.draw_landmarks(
                        frame,
                        results.pose_landmarks,
                        self.mp_pose.POSE_CONNECTIONS
                    )
                    
                    # Visualize cage projection (simple 2D projection)
                    self.visualize_cage_2d(frame, deformed_cage_verts)
                    
                    status_color = (0, 255, 0)
                    status_text = "Person detected - Deforming mesh"
                else:
                    status_color = (0, 0, 255)
                    status_text = "No person detected"
                
                # Display info
                cv2.putText(frame, status_text, (10, 30),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, status_color, 2)
                
                cv2.putText(frame, f"MediaPipe: {mediapipe_time:.1f}ms", (10, 60),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
                
                if deform_time > 0:
                    cv2.putText(frame, f"Deformation: {deform_time:.1f}ms", (10, 90),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
                
                # Calculate FPS
                frame_count += 1
                elapsed = time.time() - fps_start_time
                if elapsed >= 1.0:
                    fps = frame_count / elapsed
                    frame_count = 0
                    fps_start_time = time.time()
                
                cv2.putText(frame, f"FPS: {fps:.1f}", (10, 120),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
                
                # Show frame
                cv2.imshow("Cage Deformation Test", frame)
                
                # Handle keys
                key = cv2.waitKey(1) & 0xFF
                
                if key == ord('q') or key == ord('Q'):
                    break
                
                elif key == ord('s') or key == ord('S'):
                    if results.pose_landmarks:
                        # Save deformed mesh
                        deformed_mesh = trimesh.Trimesh(
                            vertices=deformed_mesh_verts,
                            faces=self.mesh.faces
                        )
                        filename = f"deformed_mesh_{save_counter}.obj"
                        deformed_mesh.export(filename)
                        print(f"Saved: {filename}")
                        save_counter += 1
        
        except KeyboardInterrupt:
            print("\nInterrupted by user")
        
        finally:
            cap.release()
            cv2.destroyAllWindows()
            print("Test complete!")
    
    def visualize_cage_2d(self, frame, cage_vertices):
        """
        Visualize cage by projecting to 2D and drawing.
        This is a simple orthographic projection for visualization.
        """
        height, width = frame.shape[:2]
        
        # Simple orthographic projection (just use x, y and ignore z)
        # Scale and center for display
        cage_2d = cage_vertices[:, :2].copy()
        
        # Center and scale to fit frame
        cage_center = cage_2d.mean(axis=0)
        cage_2d -= cage_center
        
        # Scale to reasonable size
        scale = min(width, height) * 0.3
        cage_2d *= scale
        
        # Translate to center of frame
        cage_2d[:, 0] += width / 2
        cage_2d[:, 1] += height / 2
        
        # Draw cage vertices
        for vertex in cage_2d:
            x, y = int(vertex[0]), int(vertex[1])
            if 0 <= x < width and 0 <= y < height:
                cv2.circle(frame, (x, y), 3, (0, 255, 255), -1)


def main():
    """
    Main function to run the test.
    """
    import argparse
    
    parser = argparse.ArgumentParser(description="Test cage-based deformation")
    parser.add_argument('--mesh', type=str, default=None,
                       help='Path to 3D mesh file (OBJ, STL, etc.)')
    
    args = parser.parse_args()
    
    # Create tester
    tester = CageDeformationTester(mesh_path=args.mesh)
    
    # Run test
    tester.run_test()


if __name__ == "__main__":
    main()
