"""
BodyPix Cage Deformation Integration Script
===========================================
Simple script that integrates all components for testing the complete system.

This script:
1. Starts the WebSocket server
2. Runs the BodyPix cage deformation
3. Streams results to the web viewer
4. Provides debugging tools

Run: python tests/test_integration.py
"""

import cv2
import numpy as np
import time
import trimesh
import asyncio
import threading
import json
from pathlib import Path
import sys

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent))

# BodyPix imports
from tf_bodypix.api import download_model, load_model, BodyPixModelPaths

# Local imports
from enhanced_cage_utils import BodyPixCageGenerator, EnhancedMeanValueCoordinates
from keypoint_mapper import KeypointToCageMapper
from enhanced_websocket_server import EnhancedMeshStreamServer


class IntegratedBodyPixSystem:
    """
    Integrated system that combines all components for testing.
    """
    
    def __init__(self, mesh_path=None):
        """
        Initialize the integrated system.
        
        Args:
            mesh_path: Path to 3D mesh file (OBJ, STL, etc.)
                      If None, creates a simple test mesh
        """
        print("="*70)
        print("INITIALIZING INTEGRATED BODYPIX CAGE DEFORMATION SYSTEM")
        print("="*70)
        
        # Initialize BodyPix model
        print("Loading BodyPix model...")
        self.bodypix_model = load_model(download_model(
            BodyPixModelPaths.MOBILENET_FLOAT_50_STRIDE_16
        ))
        print("✓ BodyPix model loaded")
        
        # Initialize MediaPipe for real-time keypoints
        print("Loading MediaPipe Pose...")
        import mediapipe as mp
        self.mp_pose = mp.solutions.pose
        self.mp_drawing = mp.solutions.drawing_utils
        self.pose = self.mp_pose.Pose(
            static_image_mode=False,
            model_complexity=1,
            smooth_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        print("✓ MediaPipe Pose loaded")
        
        # Load or create mesh
        if mesh_path and Path(mesh_path).exists():
            print(f"Loading mesh from {mesh_path}...")
            self.mesh = trimesh.load(mesh_path)
        else:
            print("Creating simple test mesh...")
            self.mesh = self.create_simple_clothing_mesh()
        
        print(f"✓ Mesh loaded: {len(self.mesh.vertices)} vertices, {len(self.mesh.faces)} faces")
        
        # Initialize cage system
        print("\nSetting up enhanced cage system...")
        self.cage_generator = BodyPixCageGenerator(self.mesh)
        self.mvc = None  # Will be initialized after first cage generation
        
        # Initialize keypoint mapper
        self.keypoint_mapper = KeypointToCageMapper()
        
        # WebSocket server
        self.ws_server = None
        self.loop = None
        self.server_thread = None
        
        # Performance tracking
        self.fps = 0
        self.frame_count = 0
        self.fps_start_time = time.time()
        
        # Debug mode
        self.debug_mode = False
        self.show_segmentation = True
        self.show_window = True  # Display Python viewer window
        
        # State
        self.cage_initialized = False
        self.bodypix_used = False
        
        # Store original cage for reference
        self.cage = None
        self.cage_structure = None  # NEW: Store anatomical structure
        self.original_cage_vertices = None
        
        print("✓ System ready!")
        print("="*70)
    
    def create_simple_clothing_mesh(self):
        """
        Create a simple t-shirt-like mesh for testing.
        """
        # Create a more realistic t-shirt shape
        vertices = []
        
        # Torso dimensions
        width, height, depth = 0.4, 0.6, 0.15
        
        # Create torso vertices (front and back)
        for z_offset in [-depth/2, depth/2]:  # front and back
            for y in np.linspace(0, height, 8):  # vertical subdivisions
                for x in np.linspace(-width/2, width/2, 6):  # horizontal subdivisions
                    # Add some curvature to make it more t-shirt like
                    x_curved = x * (1 + 0.1 * np.sin(y * np.pi / height))
                    vertices.append([x_curved, y, z_offset])
        
        # Add sleeve vertices
        sleeve_length = 0.25
        sleeve_width = 0.08
        
        for side in [-1, 1]:  # left and right sleeves
            x_base = side * width/2
            for z_offset in [-depth/2, depth/2]:
                for y in np.linspace(height * 0.7, height * 0.9, 4):
                    for x in np.linspace(x_base, x_base + side * sleeve_length, 4):
                        vertices.append([x, y, z_offset])
        
        vertices = np.array(vertices)
        
        # Create faces using convex hull for simplicity
        from scipy.spatial import ConvexHull
        hull = ConvexHull(vertices)
        faces = hull.simplices
        
        mesh = trimesh.Trimesh(vertices=vertices, faces=faces)
        
        # Center and normalize
        mesh.vertices -= mesh.vertices.mean(axis=0)
        mesh.vertices /= np.max(np.linalg.norm(mesh.vertices, axis=1))
        
        return mesh
    
    def process_bodypix_segmentation(self, frame):
        """
        Process frame with BodyPix to get body part segmentation.
        """
        # Run BodyPix segmentation
        result = self.bodypix_model.predict_single(frame)
        
        # Get binary person mask
        person_mask = result.get_mask(threshold=0.75)
        
        # Get colored visualization
        colored_mask = result.get_colored_part_mask(person_mask)
        
        # Extract specific body parts for clothing
        body_parts = {
            'torso': result.get_part_mask(
                person_mask,
                part_names=['torso_front', 'torso_back']
            ),
            'left_upper_arm': result.get_part_mask(
                person_mask,
                part_names=['left_upper_arm_front', 'left_upper_arm_back']
            ),
            'right_upper_arm': result.get_part_mask(
                person_mask,
                part_names=['right_upper_arm_front', 'right_upper_arm_back']
            ),
            'left_lower_arm': result.get_part_mask(
                person_mask,
                part_names=['left_lower_arm_front', 'left_lower_arm_back']
            ),
            'right_lower_arm': result.get_part_mask(
                person_mask,
                part_names=['right_lower_arm_front', 'right_lower_arm_back']
            ),
        }
        
        # Convert to numpy arrays and fix data types
        for part_name, mask in body_parts.items():
            if hasattr(mask, 'numpy'):
                mask_array = mask.numpy()
            else:
                mask_array = np.array(mask)
            
            # Ensure uint8 data type
            if mask_array.dtype != np.uint8:
                mask_array = mask_array.astype(np.uint8)
            
            body_parts[part_name] = mask_array
        
        # Convert main masks
        person_mask_array = person_mask.numpy() if hasattr(person_mask, 'numpy') else np.array(person_mask)
        colored_mask_array = colored_mask.numpy() if hasattr(colored_mask, 'numpy') else np.array(colored_mask)
        
        # Ensure uint8 data type
        if person_mask_array.dtype != np.uint8:
            person_mask_array = person_mask_array.astype(np.uint8)
        if colored_mask_array.dtype != np.uint8:
            colored_mask_array = colored_mask_array.astype(np.uint8)
        
        return {
            'person_mask': person_mask_array,
            'colored_mask': colored_mask_array,
            'body_parts': body_parts
        }
    
    def get_mediapipe_keypoints(self, frame):
        """
        Get MediaPipe keypoints for real-time deformation.
        This is fast and runs every frame.
        """
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.pose.process(rgb_frame)
        
        if not results.pose_landmarks:
            return None, None
        
        # Extract keypoints
        h, w = frame.shape[:2]
        keypoints = {}
        
        # MediaPipe pose landmark indices
        KEYPOINT_INDICES = {
            'left_shoulder': 11,
            'right_shoulder': 12,
            'left_elbow': 13,
            'right_elbow': 14,
            'left_wrist': 15,
            'right_wrist': 16,
            'left_hip': 23,
            'right_hip': 24,
        }
        
        for name, idx in KEYPOINT_INDICES.items():
            if idx < len(results.pose_landmarks.landmark):
                landmark = results.pose_landmarks.landmark[idx]
                keypoints[name] = np.array([
                    (landmark.x - 0.5) * w,
                    (landmark.y - 0.5) * h,
                    landmark.z * 1000
                ])
        
        return keypoints, results.pose_landmarks
    
    def deform_mesh_from_keypoints(self, keypoints, landmarks, frame_shape):
        """
        Deform mesh using MediaPipe keypoints (REAL-TIME).
        This is fast and runs every frame.
        """
        if not self.cage_initialized:
            return self.mesh.vertices
        
        # Use keypoint mapper to deform cage WITH cage_structure for section-wise deformation
        deformed_cage_vertices = self.keypoint_mapper.simple_anatomical_mapping(
            landmarks, self.cage, frame_shape, self.cage_structure  # Pass cage_structure
        )
        
        # Deform mesh using MVC
        deformed_vertices = self.mvc.deform_mesh(deformed_cage_vertices)
        
        # Ensure mesh is properly sized and positioned for web viewer
        deformed_vertices = self.normalize_mesh_for_web(deformed_vertices, keypoints)
        
        return deformed_vertices
    
    def normalize_mesh_for_web(self, vertices, keypoints):
        """
        Normalize mesh vertices for proper display in web viewer.
        Ensures mesh is visible and properly aligned with keypoints.
        """
        # Always normalize the mesh to a reasonable size regardless of keypoints
        mesh_center = vertices.mean(axis=0)
        centered_vertices = vertices - mesh_center
        
        # Scale to reasonable size for web viewer
        max_dim = np.max(np.abs(centered_vertices))
        if max_dim > 0:
            scale_factor = 5.0 / max_dim  # Target size of 5.0 units (very large)
            scaled_vertices = centered_vertices * scale_factor
        else:
            scaled_vertices = centered_vertices
        
        # If we have keypoints, try to align with them (but keep mesh small)
        if keypoints is not None and len(keypoints) > 0:
            # Convert keypoints to normalized coordinates
            keypoint_positions = np.array(list(keypoints.values()))
            
            # Get bounding box of keypoints
            kp_min = keypoint_positions.min(axis=0)
            kp_max = keypoint_positions.max(axis=0)
            kp_center = (kp_min + kp_max) / 2
            
            # Normalize keypoint center to [-1, 1] range
            # Assuming keypoints are in pixel coordinates
            frame_width = 640  # Approximate frame width
            frame_height = 480  # Approximate frame height
            
            kp_x_norm = (kp_center[0] / frame_width - 0.5) * 2  # [-1, 1]
            kp_y_norm = -(kp_center[1] / frame_height - 0.5) * 2  # [-1, 1], flip Y
            
            # Apply small offset to align with keypoints
            offset = np.array([kp_x_norm * 0.1, kp_y_norm * 0.1, 0])  # Small offset
            scaled_vertices += offset
        
        return scaled_vertices
    
    def initialize_cage_from_segmentation(self, segmentation_data, frame_shape):
        """
        Initialize cage system based on first frame segmentation.
        """
        if self.cage_initialized:
            return
        
        print("\nInitializing cage from BodyPix segmentation...")
        
        # Generate anatomical cage WITH structure
        self.cage, self.cage_structure = self.cage_generator.generate_anatomical_cage(
            segmentation_data, frame_shape, subdivisions=3
        )
        
        # Store original cage vertices for deformation reference
        self.original_cage_vertices = self.cage.vertices.copy()
        
        # Initialize MVC
        self.mvc = EnhancedMeanValueCoordinates(self.mesh.vertices, self.cage)
        self.mvc.compute_weights()
        
        self.cage_initialized = True
        print("✓ Cage system initialized")
        print(f"   Cage: {len(self.cage.vertices)} vertices, {len(self.cage.faces)} faces")
        print(f"   Structure: {len(self.cage_structure)} body parts")
    
    def deform_mesh_from_segmentation(self, segmentation_data, frame_shape):
        """
        Deform mesh based on current segmentation.
        """
        if not self.cage_initialized:
            return self.mesh.vertices
        
        # DON'T regenerate cage every frame - this causes dimension mismatches!
        # Instead, use the existing cage and just apply transformations
        cage_vertices = self.original_cage_vertices.copy()
        
        # Apply simple transformations based on segmentation
        # This is much faster and avoids dimension issues
        body_parts = segmentation_data['body_parts']
        
        # Calculate scale and translation from torso
        if 'torso' in body_parts:
            torso_mask = body_parts['torso']
            torso_coords = np.where(torso_mask > 0)
            
            if len(torso_coords[0]) > 0:
                y_min, y_max = torso_coords[0].min(), torso_coords[0].max()
                x_min, x_max = torso_coords[1].min(), torso_coords[1].max()
                
                # Calculate torso center and size
                torso_center_x = (x_min + x_max) / 2
                torso_center_y = (y_min + y_max) / 2
                torso_width = x_max - x_min
                torso_height = y_max - y_min
                
                # Convert to normalized coordinates [-1, 1]
                torso_x_norm = (torso_center_x / frame_shape[1] - 0.5) * 2
                torso_y_norm = -(torso_center_y / frame_shape[0] - 0.5) * 2  # Flip Y
                
                # Apply translation
                translation = np.array([torso_x_norm * 0.5, torso_y_norm * 0.5, 0])
                cage_center = cage_vertices.mean(axis=0)
                cage_vertices += translation - cage_center
        
        # Deform mesh using MVC with consistent cage size
        deformed_vertices = self.mvc.deform_mesh(cage_vertices)
        
        return deformed_vertices
    
    def start_websocket_server(self):
        """Start WebSocket server using the existing EnhancedMeshStreamServer."""
        import threading
        
        def run_server():
            """Run the WebSocket server in a separate thread."""
            import websockets
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            
            # Create server instance inside the thread
            self.ws_server = EnhancedMeshStreamServer()
            self.ws_server.debug_mode = self.debug_mode
            self.loop = loop
            
            async def main():
                """Main server function."""
                print("🔄 Starting WebSocket server...")
                # Use the existing server's start method
                await self.ws_server.start()
            
            try:
                loop.run_until_complete(main())
            except Exception as e:
                print(f"❌ WebSocket server error: {e}")
        
        # Start server thread
        self.server_thread = threading.Thread(target=run_server, daemon=True)
        self.server_thread.start()
        
        # Wait for server to start
        time.sleep(2)
    
    def send_mesh_to_web(self, vertices, faces, frame_counter, cage_vertices=None, cage_faces=None, segmentation_data=None):
        """Send mesh data and optional cage to web client via WebSocket."""
        # Check if WebSocket server is properly initialized
        if not self.ws_server:
            print("⚠️ WebSocket server not initialized yet, skipping mesh send")
            return
            
        if not self.loop:
            print("⚠️ WebSocket event loop not available, skipping mesh send")
            return
            
        try:
            # Debug: Print mesh info (only occasionally to reduce spam)
            if frame_counter % 30 == 0:  # Every 30 frames
                print(f"📤 Sending mesh: {len(vertices)} vertices, {len(faces)} faces")
                if len(vertices) > 0:
                    print(f"   Vertex range: X[{vertices[:, 0].min():.3f}, {vertices[:, 0].max():.3f}], "
                          f"Y[{vertices[:, 1].min():.3f}, {vertices[:, 1].max():.3f}], "
                          f"Z[{vertices[:, 2].min():.3f}, {vertices[:, 2].max():.3f}]")
                if cage_vertices is not None:
                    print(f"   Cage: {len(cage_vertices)} vertices")
            
            # Prepare metadata
            metadata = {
                'timestamp': time.time(),
                'vertex_count': len(vertices),
                'face_count': len(faces),
                'debug_mode': self.debug_mode,
                'bodypix_used': self.bodypix_used,
                'has_cage': cage_vertices is not None
            }
            
            # Add segmentation info if available
            if segmentation_data:
                metadata['segmentation'] = {
                    'body_parts_detected': list(segmentation_data['body_parts'].keys()),
                    'person_detected': bool(np.any(segmentation_data['person_mask'] > 0))
                }
            
            # Prepare data to send
            data = {
                'mesh_vertices': vertices,
                'mesh_faces': faces,
                'metadata': metadata
            }
            
            # Add cage data if available
            if cage_vertices is not None and cage_faces is not None:
                data['cage_vertices'] = cage_vertices
                data['cage_faces'] = cage_faces
            
            # Send via WebSocket directly (will handle no clients internally)
            fut = asyncio.run_coroutine_threadsafe(
                self.ws_server.send_mesh_data(
                    vertices, faces, metadata,
                    cage_vertices=cage_vertices,
                    cage_faces=cage_faces
                ),
                self.loop
            )
            try:
                fut.result(timeout=0.1)  # Short timeout for real-time
            except Exception as e:
                pass  # Ignore timeout errors for real-time operation
            
        except Exception as e:
            print(f"❌ Error sending mesh data: {e}")
    
    def run(self):
        """Main processing loop."""
        print("\n" + "="*70)
        print("STARTING INTEGRATED BODYPIX CAGE DEFORMATION")
        print("="*70)
        print("Controls:")
        print("  Q - Quit")
        print("  D - Toggle debug mode")
        print("  S - Toggle segmentation display")
        print("  R - Reset cage")
        print("  O - Open web viewer")
        print("  C - Toggle cage visualization (in web viewer)")
        print("="*70)
        
        # Start WebSocket server
        self.start_websocket_server()
        
        # Initialize camera
        cap = cv2.VideoCapture(0)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        
        if not cap.isOpened():
            print("✗ Could not open camera")
            return
        
        print("✓ Camera initialized")
        print("\nOpening web viewer...")
        print("Please open: tests/enhanced_mesh_viewer.html in your browser")
        print("="*70)
        
        # Send interval for WebSocket (every N frames)
        send_interval = 2
        frame_counter = 0
        
        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    continue
                
                # Mirror frame
                frame = cv2.flip(frame, 1)
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                
                # STEP 1: Run BodyPix ONCE to create cage
                if not self.bodypix_used:
                    print("\n🔄 Running BodyPix segmentation (one time only)...")
                    start_time = time.time()
                    segmentation_data = self.process_bodypix_segmentation(frame_rgb)
                    bodypix_time = (time.time() - start_time) * 1000
                    
                    # Initialize cage from BodyPix
                    self.initialize_cage_from_segmentation(segmentation_data, frame.shape)
                    
                    print(f"✓ BodyPix completed in {bodypix_time:.1f}ms")
                    print("✓ Now using MediaPipe for real-time deformation")
                    self.bodypix_used = True  # Mark as used
                    bodypix_time = 0  # Don't show BodyPix time after first run
                else:
                    bodypix_time = 0
                
                # STEP 2: Get MediaPipe keypoints (REAL-TIME)
                start_time = time.time()
                keypoints, landmarks = self.get_mediapipe_keypoints(frame)
                mediapipe_time = (time.time() - start_time) * 1000
                
                # STEP 3: Deform mesh using keypoints (REAL-TIME)
                start_time = time.time()
                if keypoints and landmarks and self.cage_initialized:
                    # Get deformed cage vertices
                    deformed_cage_vertices = self.keypoint_mapper.simple_anatomical_mapping(
                        landmarks, self.cage, frame.shape
                    )
                    
                    # Deform mesh using MVC
                    deformed_vertices = self.mvc.deform_mesh(deformed_cage_vertices)
                    
                    # Normalize for web display
                    deformed_vertices = self.normalize_mesh_for_web(deformed_vertices, keypoints)
                else:
                    deformed_vertices = self.mesh.vertices
                    deformed_cage_vertices = self.original_cage_vertices if self.cage_initialized else None
                
                deform_time = (time.time() - start_time) * 1000
                
                # Send to web client (with cage for debugging)
                frame_counter += 1
                if frame_counter % send_interval == 0:
                    cage_verts = deformed_cage_vertices if self.cage_initialized else None
                    cage_faces_to_send = self.cage.faces if self.cage_initialized else None
                    
                    self.send_mesh_to_web(
                        deformed_vertices, 
                        self.mesh.faces,
                        frame_counter,
                        cage_vertices=cage_verts,
                        cage_faces=cage_faces_to_send,
                        segmentation_data=segmentation_data if not self.bodypix_used else None
                    )
                
                # Create visualization
                vis_frame = self.create_visualization(frame, landmarks)
                
                # Add performance info
                self.add_performance_info(
                    vis_frame, bodypix_time, mediapipe_time, deform_time
                )
                
                # Show frame
                if self.show_window:
                    cv2.imshow("Integrated BodyPix Cage Deformation", vis_frame)
                
                # Handle keyboard input
                key = cv2.waitKey(1) & 0xFF if self.show_window else 0
                if key == ord('q'):
                    break
                elif key == ord('d'):
                    self.debug_mode = not self.debug_mode
                    print(f"Debug mode: {'ON' if self.debug_mode else 'OFF'}")
                elif key == ord('s'):
                    self.show_segmentation = not self.show_segmentation
                elif key == ord('r'):
                    print("Resetting system...")
                    self.cage_initialized = False
                    self.bodypix_used = False
                elif key == ord('o'):
                    print("Please open: tests/enhanced_mesh_viewer.html in your browser")
        
        except KeyboardInterrupt:
            print("\nInterrupted by user")
        
        finally:
            cap.release()
            cv2.destroyAllWindows()
            print("System stopped")
    
    def create_visualization(self, frame, landmarks):
        """Create visualization frame."""
        vis_frame = frame.copy()
        
        # Draw MediaPipe skeleton
        if landmarks:
            self.mp_drawing.draw_landmarks(
                vis_frame, landmarks, self.mp_pose.POSE_CONNECTIONS,
                self.mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=3),
                self.mp_drawing.DrawingSpec(color=(0, 255, 255), thickness=2)
            )
        
        return vis_frame
    
    def add_performance_info(self, frame, bodypix_time, mediapipe_time, deform_time):
        """Add performance information to frame."""
        total_time = mediapipe_time + deform_time
        
        # Calculate FPS
        self.frame_count += 1
        elapsed = time.time() - self.fps_start_time
        if elapsed >= 1.0:
            self.fps = self.frame_count / elapsed
            self.frame_count = 0
            self.fps_start_time = time.time()
        
        # Draw info
        y = 30
        cv2.putText(frame, f"FPS: {self.fps:.1f}", (10, y), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        
        y += 25
        if bodypix_time > 0:
            cv2.putText(frame, f"BodyPix: {bodypix_time:.1f}ms", (10, y),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            y += 20
        
        cv2.putText(frame, f"MediaPipe: {mediapipe_time:.1f}ms", (10, y),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        y += 20
        cv2.putText(frame, f"Deform: {deform_time:.1f}ms", (10, y),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        y += 20
        cv2.putText(frame, f"Total: {total_time:.1f}ms", (10, y),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
        
        # Status
        if self.debug_mode:
            cv2.putText(frame, "DEBUG MODE", (10, frame.shape[0] - 20),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
        
        if self.bodypix_used:
            cv2.putText(frame, "BODYPIX CAGE ACTIVE", (10, frame.shape[0] - 50),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        else:
            cv2.putText(frame, "INITIALIZING BODYPIX...", (10, frame.shape[0] - 50),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)


def main():
    """Main function."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Integrated BodyPix Cage Deformation")
    parser.add_argument('--mesh', type=str, default=None,
                       help='Path to 3D mesh file (OBJ, STL, etc.)')
    parser.add_argument('--debug', action='store_true',
                       help='Start in debug mode')
    parser.add_argument('--headless', action='store_true',
                       help='Run without displaying Python viewer window')
    
    args = parser.parse_args()
    
    # Create and run system
    system = IntegratedBodyPixSystem(mesh_path=args.mesh)
    
    if args.debug:
        system.debug_mode = True
    
    if args.headless:
        system.show_window = False
    
    system.run()


if __name__ == "__main__":
    main()
