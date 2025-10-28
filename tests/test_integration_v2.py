"""
Test Integration V2 - Corrected Cage-Based Mesh Deformation
===========================================================

This version implements the corrected pipeline with:
1. Cage from SAME image as mesh generation (reference data)
2. Proper coordinate system handling
3. Section-wise cage deformation
4. Toggle for 2D vs 3D warping
5. Debug logging every 2 seconds
6. No automatic scaling (manual camera adjustment)

Usage:
    python tests/test_integration_v2.py \\
        --mesh generated_meshes/0/mesh.obj \\
        --reference generated_images/0_reference.pkl \\
        [--enable-z-warp] \\
        [--headless]

Author: AI Assistant
Date: October 26, 2025
"""

import cv2
import numpy as np
import trimesh
import mediapipe as mp
import time
import threading
import asyncio
import argparse
from pathlib import Path

# Import V2 modules
from enhanced_cage_utils_v2 import (
    CageGeneratorV2,
    EnhancedMVCCoordinatesV2,
    load_reference_data
)
from keypoint_mapper_v2 import KeypointMapperV2
from enhanced_websocket_server_v2 import EnhancedMeshStreamServerV2


class IntegratedDeformationSystemV2:
    """
    Main system integrating all components with corrected pipeline.
    """
    
    def __init__(
        self,
        mesh_path: str,
        reference_path: str,
        enable_z_warp: bool = False,
        headless: bool = False
    ):
        """
        Initialize the integrated system.
        
        Args:
            mesh_path: Path to .obj mesh file
            reference_path: Path to .pkl reference data
            enable_z_warp: Whether to use Z-axis deformation
            headless: Run without Python viewer window
        """
        self.mesh_path = mesh_path
        self.reference_path = reference_path
        self.enable_z_warp = enable_z_warp
        self.headless = headless
        
        # Components (initialized in setup())
        self.mesh = None
        self.cage = None
        self.cage_structure = None
        self.mvc = None
        self.keypoint_mapper = None
        self.pose_detector = None
        
        # WebSocket server
        self.ws_server = None
        self.loop = None
        
        # Video capture
        self.cap = None
        
        # State
        self.original_cage_vertices = None
        self.show_debug = True
        self.show_cage = True
        self.last_debug_time = 0
        self.debug_interval = 2.0  # Log every 2 seconds
        
        print(f"\n{'='*70}")
        print("INTEGRATED DEFORMATION SYSTEM V2")
        print(f"{'='*70}")
        print(f"Mesh: {mesh_path}")
        print(f"Reference: {reference_path}")
        print(f"Z-axis warping: {'ENABLED' if enable_z_warp else 'DISABLED (2D only)'}")
        print(f"Headless mode: {headless}")
        print(f"{'='*70}\n")
    
    def setup(self):
        """Initialize all components."""
        print("\n" + "="*70)
        print("SETUP PHASE")
        print("="*70)
        
        # Load mesh and reference data
        print("\n1. Loading mesh and reference data...")
        self.mesh = trimesh.load(self.mesh_path, process=False)
        reference_data = load_reference_data(self.reference_path)
        print(f"✓ Mesh loaded: {len(self.mesh.vertices)} vertices")
        print(f"✓ Reference data loaded")
        
        # Generate cage
        print("\n2. Generating cage from reference data...")
        generator = CageGeneratorV2(self.mesh, reference_data)
        self.cage, self.cage_structure = generator.generate_cage()
        self.original_cage_vertices = self.cage.vertices.copy()
        print(f"✓ Cage generated: {len(self.cage.vertices)} vertices")
        
        # Compute MVC weights
        print("\n3. Computing MVC weights...")
        self.mvc = EnhancedMVCCoordinatesV2(self.mesh.vertices, self.cage.vertices)
        print(f"✓ MVC weights computed")
        
        # Initialize MediaPipe
        print("\n4. Initializing MediaPipe Pose...")
        mp_pose = mp.solutions.pose
        self.pose_detector = mp_pose.Pose(
            static_image_mode=False,
            model_complexity=1,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        print(f"✓ MediaPipe initialized")
        
        # Initialize keypoint mapper
        print("\n5. Initializing keypoint mapper...")
        # Extract reference keypoints from reference data
        ref_keypoints = reference_data.get('mediapipe_keypoints_2d', {})
        frame_shape = reference_data.get('frame_shape', (720, 1280))
        
        self.keypoint_mapper = KeypointMapperV2(
            mesh_bounds=self.mesh.bounds,
            reference_keypoints=ref_keypoints,
            frame_shape=frame_shape,
            enable_z_warp=self.enable_z_warp
        )
        print(f"✓ Keypoint mapper initialized")
        
        # Initialize camera
        print("\n6. Initializing camera...")
        self.cap = cv2.VideoCapture(0)
        if not self.cap.isOpened():
            raise Exception("Could not open camera")
        
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        self.cap.set(cv2.CAP_PROP_FPS, 30)
        print(f"✓ Camera initialized")
        
        # Start WebSocket server
        print("\n7. Starting WebSocket server...")
        self.start_websocket_server()
        time.sleep(1)  # Give server time to start
        print(f"✓ WebSocket server started")
        
        print("\n" + "="*70)
        print("SETUP COMPLETE")
        print("="*70 + "\n")
    
    def start_websocket_server(self):
        """Start WebSocket server in separate thread."""
        self.ws_server = EnhancedMeshStreamServerV2(port=8765)
        
        def run_server():
            self.loop = asyncio.new_event_loop()
            asyncio.set_event_loop(self.loop)
            self.loop.run_until_complete(self.ws_server.start())
            self.loop.run_forever()
        
        server_thread = threading.Thread(target=run_server, daemon=True)
        server_thread.start()
    
    def send_mesh_to_web(
        self,
        vertices: np.ndarray,
        faces: np.ndarray,
        cage_vertices: np.ndarray = None,
        cage_faces: np.ndarray = None
    ):
        """Send mesh and cage data to web viewer."""
        if self.ws_server is None or self.loop is None:
            return
        
        try:
            # Schedule in server's event loop
            asyncio.run_coroutine_threadsafe(
                self.ws_server.send_mesh_data(
                    vertices, faces,
                    cage_vertices if self.show_cage else None,
                    cage_faces if self.show_cage else None
                ),
                self.loop
            )
        except Exception as e:
            pass  # Silently fail to avoid spam
    
    def run(self):
        """Main loop."""
        print("\n" + "="*70)
        print("STARTING REAL-TIME DEFORMATION")
        print("="*70)
        print("\nControls:")
        print("  Q - Quit")
        print("  D - Toggle debug display")
        print("  C - Toggle cage visualization (in web viewer)")
        print(f"  Z-axis warping: {'ENABLED' if self.enable_z_warp else 'DISABLED'}")
        print("\nOpen tests/enhanced_mesh_viewer_v2.html in your browser")
        print("="*70 + "\n")
        
        frame_counter = 0
        fps_start_time = time.time()
        fps_frame_count = 0
        fps = 0
        
        try:
            while True:
                ret, frame = self.cap.read()
                if not ret:
                    continue
                
                frame_counter += 1
                fps_frame_count += 1
                
                # Compute FPS
                current_time = time.time()
                if current_time - fps_start_time >= 1.0:
                    fps = fps_frame_count / (current_time - fps_start_time)
                    fps_frame_count = 0
                    fps_start_time = current_time
                
                # Convert to RGB for MediaPipe
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                
                # Run MediaPipe
                results = self.pose_detector.process(frame_rgb)
                
                # Deform cage based on keypoints
                if results.pose_landmarks:
                    deformed_cage_vertices = self.keypoint_mapper.deform_cage(
                        self.original_cage_vertices,
                        self.cage_structure,
                        results.pose_landmarks
                    )
                else:
                    deformed_cage_vertices = self.original_cage_vertices.copy()
                
                # Deform mesh using MVC
                deformed_mesh_vertices = self.mvc.deform_mesh(deformed_cage_vertices)
                
                # Send to web viewer
                self.send_mesh_to_web(
                    deformed_mesh_vertices,
                    self.mesh.faces,
                    deformed_cage_vertices,
                    self.cage.faces
                )
                
                # Debug logging (every 2 seconds)
                if self.show_debug and (current_time - self.last_debug_time) >= self.debug_interval:
                    self.log_debug_info(
                        frame_counter,
                        fps,
                        deformed_mesh_vertices,
                        deformed_cage_vertices,
                        results.pose_landmarks
                    )
                    self.last_debug_time = current_time
                
                # Display frame (if not headless)
                if not self.headless:
                    display_frame = frame.copy()
                    
                    # Draw MediaPipe skeleton
                    if results.pose_landmarks:
                        mp.solutions.drawing_utils.draw_landmarks(
                            display_frame,
                            results.pose_landmarks,
                            mp.solutions.pose.POSE_CONNECTIONS
                        )
                    
                    # Overlay info
                    cv2.putText(display_frame, f"FPS: {fps:.1f}", (10, 30),
                               cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                    cv2.putText(display_frame, f"Frame: {frame_counter}", (10, 70),
                               cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                    cv2.putText(display_frame, f"Z-warp: {'ON' if self.enable_z_warp else 'OFF'}", (10, 110),
                               cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                    
                    cv2.imshow("Camera Feed", display_frame)
                
                # Handle key presses
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    break
                elif key == ord('d'):
                    self.show_debug = not self.show_debug
                    print(f"Debug logging: {'ON' if self.show_debug else 'OFF'}")
                elif key == ord('c'):
                    self.show_cage = not self.show_cage
                    print(f"Cage visualization: {'ON' if self.show_cage else 'OFF'}")
        
        finally:
            self.cleanup()
    
    def log_debug_info(
        self,
        frame_counter: int,
        fps: float,
        mesh_vertices: np.ndarray,
        cage_vertices: np.ndarray,
        landmarks
    ):
        """Log debug information."""
        print(f"\n{'='*70}")
        print(f"DEBUG INFO - Frame {frame_counter} ({fps:.1f} FPS)")
        print(f"{'='*70}")
        
        # Mesh bounds
        mesh_min = mesh_vertices.min(axis=0)
        mesh_max = mesh_vertices.max(axis=0)
        mesh_center = (mesh_min + mesh_max) / 2
        mesh_size = mesh_max - mesh_min
        
        print(f"\nMesh:")
        print(f"  Position: ({mesh_center[0]:.3f}, {mesh_center[1]:.3f}, {mesh_center[2]:.3f})")
        print(f"  Size: ({mesh_size[0]:.3f}, {mesh_size[1]:.3f}, {mesh_size[2]:.3f})")
        print(f"  X range: [{mesh_min[0]:.3f}, {mesh_max[0]:.3f}]")
        print(f"  Y range: [{mesh_min[1]:.3f}, {mesh_max[1]:.3f}]")
        print(f"  Z range: [{mesh_min[2]:.3f}, {mesh_max[2]:.3f}]")
        
        # Cage deformation
        cage_delta = cage_vertices - self.original_cage_vertices
        mean_delta = cage_delta.mean(axis=0)
        max_delta = np.abs(cage_delta).max()
        
        print(f"\nCage Deformation:")
        print(f"  Mean delta: ({mean_delta[0]:.3f}, {mean_delta[1]:.3f}, {mean_delta[2]:.3f})")
        print(f"  Max delta: {max_delta:.3f}")
        
        # Keypoint info
        if landmarks:
            debug_info = self.keypoint_mapper.get_debug_info(landmarks)
            print(f"\nKeypoints:")
            print(f"  Detected: {debug_info['keypoints_detected']}/{len(KeypointMapperV2.KEYPOINT_INDICES)}")
            print(f"  Delta magnitude: {debug_info['delta_magnitude']:.3f}")
            print(f"  Max delta: {debug_info['max_delta']:.3f}")
            
            # Show largest movers
            if 'delta_per_keypoint' in debug_info:
                sorted_deltas = sorted(debug_info['delta_per_keypoint'].items(),
                                      key=lambda x: x[1], reverse=True)
                print(f"  Top movers:")
                for name, mag in sorted_deltas[:3]:
                    print(f"    {name}: {mag:.3f}")
        else:
            print(f"\nKeypoints: No pose detected")
        
        # Visibility check (simple heuristic)
        # Assume visible if within reasonable bounds
        is_visible = (
            np.abs(mesh_center[0]) < 10 and
            np.abs(mesh_center[1]) < 10 and
            np.abs(mesh_center[2]) < 10
        )
        print(f"\nMesh likely {'VISIBLE' if is_visible else 'OFFSCREEN'}")
        if not is_visible:
            print(f"  ⚠ Mesh center far from origin, may be offscreen")
            print(f"  ⚠ Try adjusting camera in web viewer (WASD + mouse)")
        
        print(f"{'='*70}\n")
    
    def cleanup(self):
        """Clean up resources."""
        print("\nCleaning up...")
        
        if self.cap:
            self.cap.release()
        
        if self.pose_detector:
            self.pose_detector.close()
        
        if not self.headless:
            cv2.destroyAllWindows()
        
        if self.loop:
            self.loop.call_soon_threadsafe(self.loop.stop)
        
        print("✓ Cleanup complete")


def main():
    """Parse arguments and run system."""
    parser = argparse.ArgumentParser(
        description="Test Integration V2 - Corrected Cage-Based Mesh Deformation",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # 2D warping (default)
  python tests/test_integration_v2.py \\
      --mesh generated_meshes/0/mesh.obj \\
      --reference generated_images/0_reference.pkl
  
  # 3D warping (with Z-axis)
  python tests/test_integration_v2.py \\
      --mesh generated_meshes/0/mesh.obj \\
      --reference generated_images/0_reference.pkl \\
      --enable-z-warp
  
  # Headless mode (no Python window)
  python tests/test_integration_v2.py \\
      --mesh generated_meshes/0/mesh.obj \\
      --reference generated_images/0_reference.pkl \\
      --headless
        """
    )
    
    parser.add_argument(
        '--mesh',
        type=str,
        required=True,
        help='Path to .obj mesh file'
    )
    
    parser.add_argument(
        '--reference',
        type=str,
        required=True,
        help='Path to .pkl reference data file'
    )
    
    parser.add_argument(
        '--enable-z-warp',
        action='store_true',
        help='Enable Z-axis warping (default: 2D-only warping)'
    )
    
    parser.add_argument(
        '--headless',
        action='store_true',
        help='Run without Python viewer window'
    )
    
    args = parser.parse_args()
    
    # Validate paths
    if not Path(args.mesh).exists():
        print(f"Error: Mesh file not found: {args.mesh}")
        return
    
    if not Path(args.reference).exists():
        print(f"Error: Reference file not found: {args.reference}")
        return
    
    # Create and run system
    system = IntegratedDeformationSystemV2(
        mesh_path=args.mesh,
        reference_path=args.reference,
        enable_z_warp=args.enable_z_warp,
        headless=args.headless
    )
    
    system.setup()
    system.run()


if __name__ == "__main__":
    main()

