"""
Optimized BodyPix Cage Deformation - Real-Time Version
=====================================================
Uses BodyPix ONCE to create intelligent cage, then MediaPipe keypoints for real-time deformation.

Key optimizations:
- BodyPix runs only once during initialization
- MediaPipe keypoints used for real-time cage deformation
- Fixed WebSocket server threading issue
- Much faster performance for real-time use
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

# MediaPipe imports
import mediapipe as mp

# Local imports
from enhanced_cage_utils import BodyPixCageGenerator, EnhancedMeanValueCoordinates
from keypoint_mapper import KeypointToCageMapper
from enhanced_websocket_server import EnhancedMeshStreamServer


class OptimizedBodyPixSystem:
    """
    Optimized system: BodyPix once, MediaPipe for real-time.
    """
    
    def __init__(self, mesh_path=None):
        """
        Initialize the optimized system.
        """
        print("="*70)
        print("INITIALIZING OPTIMIZED BODYPIX CAGE DEFORMATION SYSTEM")
        print("="*70)
        
        # Initialize BodyPix model (will be used only once)
        print("Loading BodyPix model...")
        self.bodypix_model = load_model(download_model(
            BodyPixModelPaths.MOBILENET_FLOAT_50_STRIDE_16
        ))
        print("✓ BodyPix model loaded")
        
        # Initialize MediaPipe for real-time keypoints
        print("Loading MediaPipe Pose...")
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
        print("\nSetting up cage system...")
        self.cage_generator = BodyPixCageGenerator(self.mesh)
        self.mvc = None  # Will be initialized after BodyPix cage creation
        self.original_cage_vertices = None
        
        # Initialize keypoint mapper for real-time deformation
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
        self.show_segmentation = False
        
        # State
        self.cage_initialized = False
        self.bodypix_used = False
        
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
    
    def process_bodypix_once(self, frame):
        """
        Process frame with BodyPix ONCE to create initial cage.
        This is the expensive operation that we only do once.
        """
        print("\n" + "="*50)
        print("RUNNING BODYPIX SEGMENTATION (ONE TIME ONLY)")
        print("="*50)
        
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
        
        segmentation_data = {
            'person_mask': person_mask_array,
            'colored_mask': colored_mask_array,
            'body_parts': body_parts
        }
        
        print("✓ BodyPix segmentation complete")
        print("✓ Now switching to MediaPipe for real-time deformation")
        print("="*50)
        
        return segmentation_data
    
    def initialize_cage_from_bodypix(self, segmentation_data, frame_shape):
        """
        Initialize cage system based on BodyPix segmentation (ONE TIME ONLY).
        """
        if self.cage_initialized:
            return
        
        print("\nInitializing cage from BodyPix segmentation...")
        
        # Generate anatomical cage from BodyPix
        self.cage = self.cage_generator.generate_anatomical_cage(
            segmentation_data, frame_shape, subdivisions=3
        )
        
        # Store original cage vertices
        self.original_cage_vertices = np.array(self.cage.vertices).copy()
        
        # Initialize MVC
        self.mvc = EnhancedMeanValueCoordinates(self.mesh.vertices, self.cage)
        self.mvc.compute_weights()
        
        self.cage_initialized = True
        self.bodypix_used = True
        print("✓ Cage system initialized from BodyPix")
    
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
    
    def deform_mesh_from_keypoints(self, keypoints, frame_shape):
        """
        Deform mesh using MediaPipe keypoints (REAL-TIME).
        This is fast and runs every frame.
        """
        if not self.cage_initialized:
            return self.mesh.vertices
        
        # Use keypoint mapper to deform cage
        deformed_cage_vertices = self.keypoint_mapper.simple_anatomical_mapping(
            keypoints, self.cage, frame_shape
        )
        
        # Deform mesh using MVC
        deformed_vertices = self.mvc.deform_mesh(deformed_cage_vertices)
        
        return deformed_vertices
    
    def start_websocket_server(self):
        """Start WebSocket server in separate thread."""
        def run_server():
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            self.loop = loop
            
            self.ws_server = EnhancedMeshStreamServer()
            self.ws_server.debug_mode = self.debug_mode
            
            try:
                loop.run_until_complete(self.ws_server.start())
            except Exception as e:
                print(f"WebSocket server error: {e}")
        
        self.server_thread = threading.Thread(target=run_server, daemon=True)
        self.server_thread.start()
        
        # Wait a moment for server to start
        time.sleep(1)
        print("✓ WebSocket server started on ws://localhost:8765")
    
    def send_mesh_to_web(self, vertices, faces, metadata=None):
        """Send mesh data to web client via WebSocket."""
        if self.ws_server and self.loop:
            try:
                # Prepare metadata
                if metadata is None:
                    metadata = {}
                
                metadata.update({
                    'timestamp': time.time(),
                    'vertex_count': len(vertices),
                    'face_count': len(faces),
                    'debug_mode': self.debug_mode,
                    'bodypix_used': self.bodypix_used
                })
                
                # Send via WebSocket
                fut = asyncio.run_coroutine_threadsafe(
                    self.ws_server.send_mesh_data(vertices, faces, metadata),
                    self.loop
                )
                fut.result(timeout=0.1)  # Short timeout for real-time
                
            except Exception as e:
                if self.debug_mode:
                    print(f"Error sending mesh data: {e}")
    
    def run(self):
        """Main processing loop."""
        print("\n" + "="*70)
        print("STARTING OPTIMIZED BODYPIX CAGE DEFORMATION")
        print("="*70)
        print("Workflow:")
        print("  1. BodyPix runs ONCE to create intelligent cage")
        print("  2. MediaPipe runs EVERY FRAME for real-time deformation")
        print("  3. WebSocket streams results to web browser")
        print("\nControls:")
        print("  Q - Quit")
        print("  D - Toggle debug mode")
        print("  S - Toggle segmentation display")
        print("  R - Reset system")
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
                
                # STEP 1: Run BodyPix ONCE to create cage
                if not self.bodypix_used:
                    print("\n🔄 Running BodyPix segmentation (one time only)...")
                    start_time = time.time()
                    segmentation_data = self.process_bodypix_once(frame)
                    bodypix_time = (time.time() - start_time) * 1000
                    
                    # Initialize cage from BodyPix
                    self.initialize_cage_from_bodypix(segmentation_data, frame.shape)
                    
                    print(f"✓ BodyPix completed in {bodypix_time:.1f}ms")
                    print("✓ Now using MediaPipe for real-time deformation")
                
                # STEP 2: Get MediaPipe keypoints (REAL-TIME)
                start_time = time.time()
                keypoints, landmarks = self.get_mediapipe_keypoints(frame)
                mediapipe_time = (time.time() - start_time) * 1000
                
                # STEP 3: Deform mesh using keypoints (REAL-TIME)
                start_time = time.time()
                if keypoints:
                    deformed_vertices = self.deform_mesh_from_keypoints(keypoints, frame.shape)
                else:
                    deformed_vertices = self.mesh.vertices
                deform_time = (time.time() - start_time) * 1000
                
                # Send to web client
                frame_counter += 1
                if frame_counter % send_interval == 0:
                    self.send_mesh_to_web(deformed_vertices, self.mesh.faces)
                
                # Create visualization
                vis_frame = self.create_visualization(frame, landmarks)
                
                # Add performance info
                self.add_performance_info(vis_frame, mediapipe_time, deform_time)
                
                # Show frame
                cv2.imshow("Optimized BodyPix Cage Deformation", vis_frame)
                
                # Handle keyboard input
                key = cv2.waitKey(1) & 0xFF
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
    
    def add_performance_info(self, frame, mediapipe_time, deform_time):
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
    
    parser = argparse.ArgumentParser(description="Optimized BodyPix Cage Deformation")
    parser.add_argument('--mesh', type=str, default=None,
                       help='Path to 3D mesh file (OBJ, STL, etc.)')
    parser.add_argument('--debug', action='store_true',
                       help='Start in debug mode')
    
    args = parser.parse_args()
    
    # Create and run system
    system = OptimizedBodyPixSystem(mesh_path=args.mesh)
    
    if args.debug:
        system.debug_mode = True
    
    system.run()


if __name__ == "__main__":
    main()
