"""
BodyPix Cage-Based Deformation Test
===================================
Integrates BodyPix segmentation with cage-based 3D mesh deformation.
Uses BodyPix body part segmentation to create intelligent cage placement.

Key Features:
- BodyPix segmentation for precise body part detection
- Cage generation based on segmented body parts
- Real-time mesh deformation using Mean Value Coordinates
- WebSocket streaming to Three.js for rendering
- Debugging tools for both Python and web

Run: python tests/test_bodypix_cage_deformation.py
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
from cage_utils import SimpleCageGenerator, MeanValueCoordinates
from keypoint_mapper import KeypointToCageMapper


class BodyPixCageDeformation:
    """
    Main class that integrates BodyPix segmentation with cage-based deformation.
    
    Workflow:
    1. Capture video frame
    2. Run BodyPix segmentation to get body part masks
    3. Generate cage based on segmented body parts
    4. Deform mesh using cage movement
    5. Stream results to web browser via WebSocket
    """
    
    def __init__(self, mesh_path=None):
        """
        Initialize the deformation system.
        
        Args:
            mesh_path: Path to 3D mesh file (OBJ, STL, etc.)
                      If None, creates a simple test mesh
        """
        print("="*60)
        print("INITIALIZING BODYPIX CAGE DEFORMATION SYSTEM")
        print("="*60)
        
        # Initialize BodyPix model
        print("Loading BodyPix model...")
        self.bodypix_model = load_model(download_model(
            BodyPixModelPaths.MOBILENET_FLOAT_50_STRIDE_16
        ))
        print("✓ BodyPix model loaded")
        
        # Load or create mesh
        if mesh_path and Path(mesh_path).exists():
            print(f"Loading mesh from {mesh_path}...")
            self.mesh = trimesh.load(mesh_path)
        else:
            print("Creating simple test mesh...")
            self.mesh = self.create_simple_clothing_mesh()
        
        print(f"✓ Mesh loaded: {len(self.mesh.vertices)} vertices, {len(self.mesh.faces)} faces")
        
        # Initialize cage system
        print("\nSetting up cage deformation system...")
        self.setup_cage_system()
        
        # Initialize keypoint mapper
        self.keypoint_mapper = KeypointToCageMapper()
        
        # WebSocket server
        self.ws_server = None
        self.loop = None
        self.server_ready = threading.Event()
        
        # Performance tracking
        self.fps = 0
        self.frame_count = 0
        self.fps_start_time = time.time()
        
        # Debug mode
        self.debug_mode = False
        self.show_segmentation = True
        
        print("✓ System ready!")
        print("="*60)
    
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
    
    def setup_cage_system(self):
        """
        Set up the cage deformation system based on BodyPix body parts.
        """
        # Generate cage based on mesh
        cage_generator = SimpleCageGenerator(self.mesh)
        self.cage = cage_generator.generate_simple_box_cage(subdivisions=4)
        
        # Compute Mean Value Coordinates
        print("Computing Mean Value Coordinates...")
        self.mvc = MeanValueCoordinates(self.mesh.vertices, self.cage)
        self.mvc.compute_weights()
        
        # Store original cage for reference
        self.original_cage_vertices = np.array(self.cage.vertices).copy()
        
        print(f"✓ Cage system ready: {len(self.cage.vertices)} cage vertices")
    
    def process_bodypix_segmentation(self, frame):
        """
        Process frame with BodyPix to get body part segmentation.
        
        Args:
            frame: Input video frame (BGR)
            
        Returns:
            Dict containing segmentation masks and metadata
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
    
    def generate_cage_from_segmentation(self, segmentation_data, frame_shape):
        """
        Generate cage vertices based on BodyPix segmentation.
        
        Args:
            segmentation_data: Output from process_bodypix_segmentation
            frame_shape: Shape of the video frame
            
        Returns:
            deformed_cage_vertices: New cage vertex positions
        """
        height, width = frame_shape[:2]
        cage_vertices = self.original_cage_vertices.copy()
        
        # Get body part masks
        body_parts = segmentation_data['body_parts']
        
        # Calculate scale and translation from torso
        if 'torso' in body_parts:
            torso_mask = body_parts['torso']
            
            # Find torso bounding box
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
                torso_x_norm = (torso_center_x / width - 0.5) * 2
                torso_y_norm = -(torso_center_y / height - 0.5) * 2  # Flip Y
                
                # Calculate scale factor
                # Assume person is ~50cm wide at shoulders
                estimated_torso_width_3d = 0.5  # meters
                scale_factor = estimated_torso_width_3d / (torso_width / width)
                
                # Apply transformation to cage
                translation = np.array([torso_x_norm * scale_factor, torso_y_norm * scale_factor, 0])
                cage_center = cage_vertices.mean(axis=0)
                cage_vertices += translation - cage_center
                
                # Apply scaling
                cage_vertices *= scale_factor
        
        # Apply arm-specific deformations
        for arm_name in ['left_upper_arm', 'right_upper_arm', 'left_lower_arm', 'right_lower_arm']:
            if arm_name in body_parts:
                arm_mask = body_parts[arm_name]
                arm_coords = np.where(arm_mask > 0)
                
                if len(arm_coords[0]) > 0:
                    # Find arm center
                    arm_center_x = np.mean(arm_coords[1])
                    arm_center_y = np.mean(arm_coords[0])
                    
                    # Convert to normalized coordinates
                    arm_x_norm = (arm_center_x / width - 0.5) * 2
                    arm_y_norm = -(arm_center_y / height - 0.5) * 2
                    
                    # Find cage vertices that should move with this arm
                    # This is simplified - in practice you'd have more sophisticated mapping
                    arm_position = np.array([arm_x_norm, arm_y_norm, 0])
                    
                    # Move cage vertices that are closest to arm position
                    distances = np.linalg.norm(cage_vertices - arm_position, axis=1)
                    closest_indices = np.argsort(distances)[:len(cage_vertices)//8]  # Move top 12.5%
                    
                    # Blend arm position with current cage position
                    blend_factor = 0.3
                    for idx in closest_indices:
                        cage_vertices[idx] = (1 - blend_factor) * cage_vertices[idx] + blend_factor * arm_position
        
        return cage_vertices
    
    def start_websocket_server(self):
        """Start WebSocket server in separate thread."""
        def run_server():
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            self.loop = loop
            
            # Import here to avoid circular imports
            from websocket_server import MeshStreamServer
            self.ws_server = MeshStreamServer()
            
            try:
                loop.run_until_complete(self.ws_server.start())
            except Exception as e:
                print(f"WebSocket server error: {e}")
            finally:
                self.server_ready.set()
        
        thread = threading.Thread(target=run_server, daemon=True)
        thread.start()
        
        # Wait for server to be ready
        if self.server_ready.wait(timeout=5):
            print("✓ WebSocket server started on ws://localhost:8765")
        else:
            print("⚠ WebSocket server failed to start")
    
    def send_mesh_to_web(self, vertices, faces, segmentation_data=None):
        """Send mesh data to web client via WebSocket."""
        if self.ws_server and self.loop:
            try:
                # Prepare data
                data = {
                    'type': 'mesh_update',
                    'vertices': vertices.tolist(),
                    'faces': faces.tolist(),
                    'timestamp': time.time()
                }
                
                # Add segmentation data for debugging
                if segmentation_data and self.debug_mode:
                    data['segmentation'] = {
                        'person_mask_shape': segmentation_data['person_mask'].shape,
                        'body_parts_detected': list(segmentation_data['body_parts'].keys())
                    }
                
                # Send via WebSocket
                fut = asyncio.run_coroutine_threadsafe(
                    self.ws_server.send_mesh_data(vertices, faces),
                    self.loop
                )
                fut.result(timeout=0.1)  # Short timeout for real-time
                
            except Exception as e:
                if self.debug_mode:
                    print(f"Error sending mesh data: {e}")
    
    def run(self):
        """Main processing loop."""
        print("\n" + "="*60)
        print("STARTING BODYPIX CAGE DEFORMATION")
        print("="*60)
        print("Controls:")
        print("  Q - Quit")
        print("  D - Toggle debug mode")
        print("  S - Toggle segmentation display")
        print("  R - Reset cage")
        print("="*60)
        
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
        
        # Send interval for WebSocket (every N frames)
        send_interval = 3
        frame_counter = 0
        
        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    continue
                
                # Mirror frame
                frame = cv2.flip(frame, 1)
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                
                # Process with BodyPix
                start_time = time.time()
                segmentation_data = self.process_bodypix_segmentation(frame_rgb)
                bodypix_time = (time.time() - start_time) * 1000
                
                # Generate cage from segmentation
                start_time = time.time()
                deformed_cage_vertices = self.generate_cage_from_segmentation(
                    segmentation_data, frame.shape
                )
                cage_time = (time.time() - start_time) * 1000
                
                # Deform mesh using cage
                start_time = time.time()
                deformed_mesh_vertices = self.mvc.deform_mesh(deformed_cage_vertices)
                deform_time = (time.time() - start_time) * 1000
                
                # Send to web client
                frame_counter += 1
                if frame_counter % send_interval == 0:
                    self.send_mesh_to_web(
                        deformed_mesh_vertices, 
                        self.mesh.faces,
                        segmentation_data
                    )
                
                # Create visualization
                vis_frame = self.create_visualization(
                    frame, segmentation_data, deformed_cage_vertices
                )
                
                # Add performance info
                self.add_performance_info(
                    vis_frame, bodypix_time, cage_time, deform_time
                )
                
                # Show frame
                cv2.imshow("BodyPix Cage Deformation", vis_frame)
                
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
                    print("Resetting cage...")
                    self.original_cage_vertices = np.array(self.cage.vertices).copy()
        
        except KeyboardInterrupt:
            print("\nInterrupted by user")
        
        finally:
            cap.release()
            cv2.destroyAllWindows()
            print("System stopped")
    
    def create_visualization(self, frame, segmentation_data, cage_vertices):
        """Create visualization frame."""
        vis_frame = frame.copy()
        
        if self.show_segmentation:
            # Overlay segmentation
            colored_mask = segmentation_data['colored_mask']
            
            # Fix data type issues
            if colored_mask.dtype != np.uint8:
                colored_mask = colored_mask.astype(np.uint8)
            
            if colored_mask.ndim == 3:
                # Ensure the mask is the right shape and type
                if colored_mask.shape[2] == 3:
                    colored_mask = cv2.cvtColor(colored_mask, cv2.COLOR_RGB2BGR)
                elif colored_mask.shape[2] == 4:
                    colored_mask = cv2.cvtColor(colored_mask, cv2.COLOR_RGBA2BGR)
            
            # Resize mask to match frame if needed
            if colored_mask.shape[:2] != frame.shape[:2]:
                colored_mask = cv2.resize(colored_mask, (frame.shape[1], frame.shape[0]))
            
            # Blend with original frame
            vis_frame = cv2.addWeighted(vis_frame, 0.7, colored_mask, 0.3, 0)
        
        # Draw cage projection (2D visualization)
        self.draw_cage_2d(vis_frame, cage_vertices)
        
        return vis_frame
    
    def draw_cage_2d(self, frame, cage_vertices):
        """Draw cage vertices as 2D projection."""
        height, width = frame.shape[:2]
        
        # Simple orthographic projection
        cage_2d = cage_vertices[:, :2].copy()
        
        # Center and scale
        cage_center = cage_2d.mean(axis=0)
        cage_2d -= cage_center
        
        # Scale to fit frame
        scale = min(width, height) * 0.2
        cage_2d *= scale
        
        # Translate to center
        cage_2d[:, 0] += width / 2
        cage_2d[:, 1] += height / 2
        
        # Draw cage vertices
        for vertex in cage_2d:
            x, y = int(vertex[0]), int(vertex[1])
            if 0 <= x < width and 0 <= y < height:
                cv2.circle(frame, (x, y), 2, (0, 255, 255), -1)
    
    def add_performance_info(self, frame, bodypix_time, cage_time, deform_time):
        """Add performance information to frame."""
        total_time = bodypix_time + cage_time + deform_time
        
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
        cv2.putText(frame, f"BodyPix: {bodypix_time:.1f}ms", (10, y),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        y += 20
        cv2.putText(frame, f"Cage: {cage_time:.1f}ms", (10, y),
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


def main():
    """Main function."""
    import argparse
    
    parser = argparse.ArgumentParser(description="BodyPix Cage Deformation Test")
    parser.add_argument('--mesh', type=str, default=None,
                       help='Path to 3D mesh file (OBJ, STL, etc.)')
    parser.add_argument('--debug', action='store_true',
                       help='Start in debug mode')
    
    args = parser.parse_args()
    
    # Create and run system
    system = BodyPixCageDeformation(mesh_path=args.mesh)
    
    if args.debug:
        system.debug_mode = True
    
    system.run()


if __name__ == "__main__":
    main()
