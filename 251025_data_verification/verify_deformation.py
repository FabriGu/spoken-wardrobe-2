"""
Real-Time Deformation Verification Script
==========================================
Visualizes cage and mesh deformation in real-time using WebSocket.
Reuses logic from test_integration.py without modifying it.

Usage:
    python 251025_data_verification/verify_deformation.py
    Then open: 251025_data_verification/verification_viewer.html
"""

import sys
from pathlib import Path
import numpy as np
import cv2
import time
import trimesh
import asyncio
import threading

# Add parent directory and tests directory to path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))
sys.path.append(str(project_root / "tests"))

# Import from main system (NO MODIFICATIONS)
from enhanced_cage_utils import BodyPixCageGenerator, EnhancedMeanValueCoordinates
from keypoint_mapper import KeypointToCageMapper
from enhanced_websocket_server import EnhancedMeshStreamServer
from tf_bodypix.api import download_model, load_model, BodyPixModelPaths
import mediapipe as mp


class DeformationVerificationSystem:
    """
    Verification system that mirrors test_integration.py logic
    but adds extra visualization and analysis.
    """
    
    def __init__(self, mesh_path=None):
        """Initialize verification system."""
        print("\n" + "="*70)
        print("DEFORMATION VERIFICATION SYSTEM")
        print("="*70)
        
        # Load BodyPix
        print("\n Loading BodyPix...")
        self.bodypix_model = load_model(download_model(
            BodyPixModelPaths.MOBILENET_FLOAT_50_STRIDE_16
        ))
        print("✓ BodyPix loaded")
        
        # Load MediaPipe
        print("Loading MediaPipe...")
        self.mp_pose = mp.solutions.pose
        self.mp_drawing = mp.solutions.drawing_utils
        self.pose = self.mp_pose.Pose(
            static_image_mode=False,
            model_complexity=1,
            smooth_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        print("✓ MediaPipe loaded")
        
        # Load mesh
        if mesh_path and Path(mesh_path).exists():
            print(f"Loading mesh from {mesh_path}...")
            self.mesh = trimesh.load(mesh_path)
        else:
            print("Creating test mesh...")
            self.mesh = self.create_test_mesh()
        
        print(f"✓ Mesh: {len(self.mesh.vertices)} vertices, {len(self.mesh.faces)} faces")
        
        # Initialize components
        self.cage_generator = BodyPixCageGenerator(self.mesh)
        self.keypoint_mapper = KeypointToCageMapper()
        self.mvc = None
        self.cage = None
        self.original_cage_vertices = None
        
        # State
        self.cage_initialized = False
        self.bodypix_used = False
        
        # WebSocket
        self.ws_server = None
        self.loop = None
        
        # Analysis data
        self.frame_count = 0
        self.cage_motion_history = []
        self.mesh_deformation_history = []
        
        print("✓ System ready!\n")
    
    def create_test_mesh(self):
        """Create simple test mesh (same as test_integration.py)."""
        vertices = []
        width, height, depth = 0.4, 0.6, 0.15
        
        for z_offset in [-depth/2, depth/2]:
            for y in np.linspace(0, height, 8):
                for x in np.linspace(-width/2, width/2, 6):
                    x_curved = x * (1 + 0.1 * np.sin(y * np.pi / height))
                    vertices.append([x_curved, y, z_offset])
        
        vertices = np.array(vertices)
        from scipy.spatial import ConvexHull
        hull = ConvexHull(vertices)
        mesh = trimesh.Trimesh(vertices=vertices, faces=hull.simplices)
        mesh.vertices -= mesh.vertices.mean(axis=0)
        mesh.vertices /= np.max(np.linalg.norm(mesh.vertices, axis=1))
        
        return mesh
    
    def start_websocket_server(self):
        """Start WebSocket server (same logic as test_integration.py)."""
        def run_server():
            import websockets
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            
            self.ws_server = EnhancedMeshStreamServer(port=8766)  # Different port
            self.ws_server.debug_mode = True
            self.loop = loop
            
            async def main():
                print("🔄 Starting WebSocket server on port 8766...")
                await self.ws_server.start()
            
            try:
                loop.run_until_complete(main())
            except Exception as e:
                print(f"❌ WebSocket error: {e}")
        
        self.server_thread = threading.Thread(target=run_server, daemon=True)
        self.server_thread.start()
        time.sleep(2)
        print("✓ WebSocket server started on ws://localhost:8766\n")
    
    def initialize_cage(self, segmentation_data, frame_shape):
        """Initialize cage (mirrors test_integration.py)."""
        if self.cage_initialized:
            return
        
        print("Initializing cage from BodyPix...")
        
        self.cage = self.cage_generator.generate_anatomical_cage(
            segmentation_data, frame_shape, subdivisions=3
        )
        self.original_cage_vertices = self.cage.vertices.copy()
        
        print(f"✓ Cage: {len(self.cage.vertices)} vertices, {len(self.cage.faces)} faces")
        
        # Compute MVC weights
        print("Computing MVC weights...")
        start_time = time.time()
        self.mvc = EnhancedMeanValueCoordinates(self.mesh.vertices, self.cage)
        self.mvc.compute_weights()
        elapsed = time.time() - start_time
        print(f"✓ MVC weights computed in {elapsed:.2f}s")
        
        self.cage_initialized = True
    
    def analyze_cage_motion(self, deformed_cage_vertices):
        """Analyze how cage vertices move."""
        if self.original_cage_vertices is None:
            return {}
        
        # Compute displacement per vertex
        displacement = deformed_cage_vertices - self.original_cage_vertices
        displacement_magnitude = np.linalg.norm(displacement, axis=1)
        
        analysis = {
            'max_displacement': displacement_magnitude.max(),
            'mean_displacement': displacement_magnitude.mean(),
            'moving_vertices': np.sum(displacement_magnitude > 0.01),  # >1cm
            'stationary_vertices': np.sum(displacement_magnitude <= 0.01)
        }
        
        self.cage_motion_history.append(analysis)
        return analysis
    
    def run(self):
        """Main verification loop."""
        print("="*70)
        print("STARTING VERIFICATION")
        print("="*70)
        print("\nControls:")
        print("  Q - Quit")
        print("  R - Reset")
        print("  A - Show analysis")
        print("\nPlease open: 251025_data_verification/verification_viewer.html")
        print("="*70 + "\n")
        
        # Start WebSocket
        self.start_websocket_server()
        
        # Initialize camera
        cap = cv2.VideoCapture(0)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        
        if not cap.isOpened():
            print("✗ Camera not available")
            return
        
        print("✓ Camera initialized\n")
        
        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    continue
                
                frame = cv2.flip(frame, 1)
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                
                # Initialize cage with BodyPix (once)
                if not self.bodypix_used:
                    print("Running BodyPix segmentation...")
                    segmentation_data = self.process_bodypix(frame_rgb)
                    self.initialize_cage(segmentation_data, frame.shape)
                    self.bodypix_used = True
                    print("✓ Cage initialized\n")
                
                # Get MediaPipe keypoints
                keypoints, landmarks = self.get_mediapipe_keypoints(frame)
                
                # Deform cage and mesh
                if keypoints and landmarks and self.cage_initialized:
                    # Deform cage
                    deformed_cage_vertices = self.keypoint_mapper.simple_anatomical_mapping(
                        landmarks, self.cage, frame.shape
                    )
                    
                    # Analyze cage motion
                    if self.frame_count % 30 == 0:  # Every 30 frames
                        analysis = self.analyze_cage_motion(deformed_cage_vertices)
                        print(f"📊 Frame {self.frame_count}: "
                              f"Max cage displacement: {analysis['max_displacement']:.3f}, "
                              f"Moving vertices: {analysis['moving_vertices']}/{len(self.cage.vertices)}")
                    
                    # Deform mesh
                    deformed_mesh_vertices = self.mvc.deform_mesh(deformed_cage_vertices)
                else:
                    deformed_cage_vertices = self.original_cage_vertices if self.cage_initialized else None
                    deformed_mesh_vertices = self.mesh.vertices
                
                # Send to web viewer
                if self.frame_count % 2 == 0:
                    self.send_to_web(
                        deformed_mesh_vertices,
                        deformed_cage_vertices
                    )
                
                # Show frame
                vis_frame = frame.copy()
                if landmarks:
                    self.mp_drawing.draw_landmarks(
                        vis_frame, landmarks, self.mp_pose.POSE_CONNECTIONS
                    )
                
                cv2.putText(vis_frame, f"Frame: {self.frame_count}", (10, 30),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                cv2.imshow("Deformation Verification", vis_frame)
                
                self.frame_count += 1
                
                # Handle keys
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    break
                elif key == ord('r'):
                    self.cage_initialized = False
                    self.bodypix_used = False
                    self.frame_count = 0
                    print("\n🔄 Reset\n")
                elif key == ord('a'):
                    self.show_analysis()
        
        except KeyboardInterrupt:
            print("\n\nInterrupted")
        finally:
            cap.release()
            cv2.destroyAllWindows()
            print("\n✓ Verification stopped")
    
    def process_bodypix(self, frame_rgb):
        """Process BodyPix segmentation."""
        result = self.bodypix_model.predict_single(frame_rgb)
        person_mask = result.get_mask(threshold=0.75)
        
        # Get masks (handle both tensor and numpy array formats)
        person_mask_np = person_mask.numpy() if hasattr(person_mask, 'numpy') else person_mask
        colored_mask = result.get_colored_part_mask(person_mask)
        colored_mask_np = colored_mask.numpy() if hasattr(colored_mask, 'numpy') else colored_mask
        
        torso_mask = result.get_part_mask(person_mask, part_names=['torso_front', 'torso_back'])
        torso_mask_np = torso_mask.numpy() if hasattr(torso_mask, 'numpy') else torso_mask
        
        left_arm_mask = result.get_part_mask(person_mask, part_names=['left_upper_arm_front', 'left_upper_arm_back'])
        left_arm_mask_np = left_arm_mask.numpy() if hasattr(left_arm_mask, 'numpy') else left_arm_mask
        
        right_arm_mask = result.get_part_mask(person_mask, part_names=['right_upper_arm_front', 'right_upper_arm_back'])
        right_arm_mask_np = right_arm_mask.numpy() if hasattr(right_arm_mask, 'numpy') else right_arm_mask
        
        body_parts = {
            'torso': torso_mask_np.astype(np.uint8),
            'left_upper_arm': left_arm_mask_np.astype(np.uint8),
            'right_upper_arm': right_arm_mask_np.astype(np.uint8),
        }
        
        return {
            'person_mask': person_mask_np.astype(np.uint8),
            'colored_mask': colored_mask_np.astype(np.uint8),
            'body_parts': body_parts
        }
    
    def get_mediapipe_keypoints(self, frame):
        """Get MediaPipe keypoints."""
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.pose.process(rgb)
        
        if not results.pose_landmarks:
            return None, None
        
        h, w = frame.shape[:2]
        keypoints = {}
        
        KEYPOINTS = {
            'left_shoulder': 11, 'right_shoulder': 12,
            'left_elbow': 13, 'right_elbow': 14,
            'left_wrist': 15, 'right_wrist': 16,
            'left_hip': 23, 'right_hip': 24,
        }
        
        for name, idx in KEYPOINTS.items():
            if idx < len(results.pose_landmarks.landmark):
                lm = results.pose_landmarks.landmark[idx]
                keypoints[name] = np.array([
                    (lm.x - 0.5) * w,
                    (lm.y - 0.5) * h,
                    lm.z * 1000
                ])
        
        return keypoints, results.pose_landmarks
    
    def send_to_web(self, mesh_vertices, cage_vertices):
        """Send data to web viewer."""
        if not self.ws_server or not self.loop:
            return
        
        try:
            fut = asyncio.run_coroutine_threadsafe(
                self.ws_server.send_mesh_data(
                    mesh_vertices,
                    self.mesh.faces,
                    metadata={'frame': self.frame_count},
                    cage_vertices=cage_vertices,
                    cage_faces=self.cage.faces if self.cage_initialized else None
                ),
                self.loop
            )
            fut.result(timeout=0.1)
        except:
            pass
    
    def show_analysis(self):
        """Show accumulated analysis."""
        print("\n" + "="*70)
        print("ANALYSIS")
        print("="*70)
        
        if not self.cage_motion_history:
            print("No data collected yet")
            return
        
        history = self.cage_motion_history[-100:]  # Last 100 frames
        
        max_disps = [h['max_displacement'] for h in history]
        mean_disps = [h['mean_displacement'] for h in history]
        moving = [h['moving_vertices'] for h in history]
        
        print(f"\n📊 Cage Motion Statistics (last {len(history)} frames):")
        print(f"   Max displacement: {np.mean(max_disps):.4f} ± {np.std(max_disps):.4f}")
        print(f"   Mean displacement: {np.mean(mean_disps):.4f} ± {np.std(mean_disps):.4f}")
        print(f"   Moving vertices: {np.mean(moving):.1f} / {len(self.cage.vertices)}")
        
        print(f"\n🔍 Interpretation:")
        if np.mean(max_disps) < 0.01:
            print("   ⚠ Cage barely moving - keypoint mapping might be broken!")
        elif np.mean(moving) > len(self.cage.vertices) * 0.8:
            print("   ⚠ Most cage vertices moving - lacking independent sections!")
        else:
            print("   ✓ Some cage motion detected")
        
        print("="*70 + "\n")


def main():
    """Main function."""
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--mesh', type=str, 
                       default='generated_meshes/0/mesh.obj',
                       help='Path to mesh file (default: generated_meshes/0/mesh.obj)')
    args = parser.parse_args()
    
    system = DeformationVerificationSystem(mesh_path=args.mesh)
    system.run()


if __name__ == "__main__":
    main()

