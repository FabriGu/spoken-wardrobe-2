"""
Option A: Unified Cage-Based Deformation with Proper MVC
===========================================================

This implementation fixes the previous cage approach by:
1. Creating a SINGLE unified cage (not multiple independent boxes)
2. Computing MVC weights ONCE (not every frame)
3. Deforming the cage based on MediaPipe keypoints
4. Using the pre-computed weights to deform the mesh

Key differences from previous failed attempt:
- Cage is ONE mesh, not multiple independent sections
- Cage vertices have section labels for targeted deformation
- Deformation is smooth because weights are computed from unified structure

Usage:
    python tests/test_integration_cage.py \\
        --mesh generated_meshes/0/mesh.obj \\
        [--headless]

Author: AI Assistant
Date: October 28, 2025
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
from typing import Dict, List, Tuple
from scipy.spatial import ConvexHull

# Import WebSocket server
from enhanced_websocket_server_v2 import EnhancedMeshStreamServerV2


class UnifiedCageGenerator:
    """
    Generates a single unified cage around a 3D mesh with anatomical sections.
    
    Unlike previous implementations that created multiple independent cages,
    this creates ONE cage mesh with labeled sections for targeted deformation.
    """
    
    def __init__(self, mesh: trimesh.Trimesh):
        """
        Args:
            mesh: The 3D clothing mesh to create a cage around
        """
        self.mesh = mesh
        self.cage = None
        self.section_info = {}  # Maps section names to vertex indices in the cage
        
    def generate_unified_cage(self, subdivisions: int = 3) -> Tuple[trimesh.Trimesh, Dict]:
        """
        Generate a unified humanoid-shaped cage around the mesh.
        
        The cage is a single connected mesh with anatomical sections marked.
        
        Args:
            subdivisions: Number of subdivisions for cage detail
            
        Returns:
            (cage_mesh, section_info) where section_info maps section names to vertex indices
        """
        print(f"\n{'='*60}")
        print("GENERATING UNIFIED CAGE")
        print(f"{'='*60}")
        
        # Get mesh bounding box
        bounds = self.mesh.bounds
        center = self.mesh.centroid
        extents = bounds[1] - bounds[0]
        
        print(f"Mesh bounds: {bounds[0]} to {bounds[1]}")
        print(f"Mesh center: {center}")
        print(f"Mesh extents: {extents}")
        
        # Expand cage slightly beyond mesh
        padding = 0.15  # 15% padding
        cage_min = bounds[0] - extents * padding
        cage_max = bounds[1] + extents * padding
        
        # Define anatomical sections and their relative positions
        # Each section is defined by its Y-range (vertical extent)
        sections = {
            'torso': {'y_range': (0.2, 0.8), 'depth_ratio': 1.0, 'width_ratio': 1.0},
            'left_upper_arm': {'y_range': (0.5, 0.8), 'depth_ratio': 0.4, 'width_ratio': 0.3},
            'right_upper_arm': {'y_range': (0.5, 0.8), 'depth_ratio': 0.4, 'width_ratio': 0.3},
            'left_lower_arm': {'y_range': (0.2, 0.5), 'depth_ratio': 0.3, 'width_ratio': 0.25},
            'right_lower_arm': {'y_range': (0.2, 0.5), 'depth_ratio': 0.3, 'width_ratio': 0.25},
        }
        
        # Generate cage vertices for each section
        all_vertices = []
        vertex_sections = []  # Track which section each vertex belongs to
        
        for section_name, section_params in sections.items():
            y_min_norm, y_max_norm = section_params['y_range']
            depth_ratio = section_params['depth_ratio']
            width_ratio = section_params['width_ratio']
            
            # Compute section bounds
            y_min = cage_min[1] + (cage_max[1] - cage_min[1]) * y_min_norm
            y_max = cage_min[1] + (cage_max[1] - cage_min[1]) * y_max_norm
            
            # Adjust width and depth based on section
            if 'left' in section_name:
                x_min = cage_min[0]
                x_max = center[0]
            elif 'right' in section_name:
                x_min = center[0]
                x_max = cage_max[0]
            else:  # torso
                x_min = cage_min[0]
                x_max = cage_max[0]
            
            z_center = center[2]
            z_extent = (cage_max[2] - cage_min[2]) * depth_ratio / 2
            z_min = z_center - z_extent
            z_max = z_center + z_extent
            
            # Create vertices for this section (box corners + subdivision points)
            section_start_idx = len(all_vertices)
            
            # Add corner vertices
            for x in [x_min, x_max]:
                for y in [y_min, y_max]:
                    for z in [z_min, z_max]:
                        all_vertices.append([x, y, z])
                        vertex_sections.append(section_name)
            
            # Add subdivision vertices along edges
            for i in range(1, subdivisions):
                t = i / subdivisions
                
                # X-edges
                for y in [y_min, y_max]:
                    for z in [z_min, z_max]:
                        x = x_min + (x_max - x_min) * t
                        all_vertices.append([x, y, z])
                        vertex_sections.append(section_name)
                
                # Y-edges
                for x in [x_min, x_max]:
                    for z in [z_min, z_max]:
                        y = y_min + (y_max - y_min) * t
                        all_vertices.append([x, y, z])
                        vertex_sections.append(section_name)
                
                # Z-edges
                for x in [x_min, x_max]:
                    for y in [y_min, y_max]:
                        z = z_min + (z_max - z_min) * t
                        all_vertices.append([x, y, z])
                        vertex_sections.append(section_name)
            
            section_end_idx = len(all_vertices)
            self.section_info[section_name] = list(range(section_start_idx, section_end_idx))
            
            print(f"  Section '{section_name}': {section_end_idx - section_start_idx} vertices")
        
        all_vertices = np.array(all_vertices)
        
        # Create convex hull to form cage surface
        print(f"\nCreating convex hull from {len(all_vertices)} vertices...")
        try:
            hull = ConvexHull(all_vertices)
            
            # IMPORTANT: ConvexHull may reduce vertices (keeps only hull vertices)
            # We need to use hull.vertices to get the actual subset used
            hull_vertex_indices = hull.vertices
            hull_vertices = all_vertices[hull_vertex_indices]
            cage_faces = hull.simplices
            
            print(f"  Generated {len(cage_faces)} faces")
            print(f"  Note: Hull reduced {len(all_vertices)} vertices to {len(hull_vertices)} hull vertices")
            
            # Rebuild section_info to map to hull vertices
            # Create mapping: original index -> hull index
            orig_to_hull = {orig_idx: hull_idx for hull_idx, orig_idx in enumerate(hull_vertex_indices)}
            
            # Update section_info to use hull indices
            updated_section_info = {}
            for section_name, orig_indices in self.section_info.items():
                # Find which original indices are in the hull
                hull_indices = [orig_to_hull[orig_idx] for orig_idx in orig_indices if orig_idx in orig_to_hull]
                updated_section_info[section_name] = hull_indices
                print(f"  Section '{section_name}': {len(orig_indices)} original → {len(hull_indices)} hull vertices")
            
            self.section_info = updated_section_info
            all_vertices = hull_vertices
            
        except Exception as e:
            print(f"  Warning: ConvexHull failed ({e}), using all vertices with simple faces")
            cage_faces = self._create_simple_faces(len(all_vertices))
        
        # Create unified cage mesh
        self.cage = trimesh.Trimesh(vertices=all_vertices, faces=cage_faces)
        
        print(f"\n✓ Unified cage generated")
        print(f"  Total vertices: {len(self.cage.vertices)}")
        print(f"  Total faces: {len(self.cage.faces)}")
        print(f"  Sections: {list(self.section_info.keys())}")
        print(f"{'='*60}\n")
        
        return self.cage, self.section_info
    
    def _create_simple_faces(self, n_vertices: int) -> np.ndarray:
        """Fallback face generation if ConvexHull fails"""
        faces = []
        for i in range(0, n_vertices - 2, 3):
            if i + 2 < n_vertices:
                faces.append([i, i + 1, i + 2])
        return np.array(faces) if faces else np.array([[0, 1, 2]])


class UnifiedMVCCoordinates:
    """
    Mean Value Coordinates for unified cage deformation.
    
    Computes binding weights once, then uses them for real-time deformation.
    """
    
    def __init__(self, mesh_vertices: np.ndarray, cage_vertices: np.ndarray):
        """
        Compute MVC weights binding mesh to cage.
        
        Args:
            mesh_vertices: (M, 3) mesh vertex positions
            cage_vertices: (N, 3) cage vertex positions
        """
        print(f"\n{'='*60}")
        print("COMPUTING MVC WEIGHTS")
        print(f"{'='*60}")
        print(f"Mesh vertices: {len(mesh_vertices)}")
        print(f"Cage vertices: {len(cage_vertices)}")
        
        self.mvc_weights = self._compute_weights(mesh_vertices, cage_vertices)
        
        weight_sum_min = self.mvc_weights.sum(axis=1).min()
        weight_sum_max = self.mvc_weights.sum(axis=1).max()
        
        print(f"\n✓ MVC weights computed: shape {self.mvc_weights.shape}")
        print(f"  Weight sum per vertex: min={weight_sum_min:.4f}, max={weight_sum_max:.4f}")
        print(f"{'='*60}\n")
    
    def _compute_weights(self, mesh_verts: np.ndarray, cage_verts: np.ndarray) -> np.ndarray:
        """
        Compute MVC weights using inverse distance weighting.
        
        This is a simplified MVC implementation optimized for real-time use.
        For production, consider implementing full MVC formula from:
        "Mean Value Coordinates for Closed Triangular Meshes" (Ju et al. 2005)
        """
        n_mesh = len(mesh_verts)
        n_cage = len(cage_verts)
        weights = np.zeros((n_mesh, n_cage))
        
        # Compute weights in batches for better performance
        batch_size = 1000
        for batch_start in range(0, n_mesh, batch_size):
            batch_end = min(batch_start + batch_size, n_mesh)
            
            if batch_start % 5000 == 0:
                print(f"  Processing vertices {batch_start}/{n_mesh}...")
            
            # Vectorized distance computation for batch
            batch_verts = mesh_verts[batch_start:batch_end, np.newaxis, :]  # (batch, 1, 3)
            cage_verts_exp = cage_verts[np.newaxis, :, :]  # (1, n_cage, 3)
            
            # Compute distances: (batch, n_cage)
            distances = np.linalg.norm(batch_verts - cage_verts_exp, axis=2)
            
            # Inverse distance weighting with power 2
            # Add epsilon to avoid division by zero
            inv_dist = 1.0 / (distances ** 2 + 1e-8)
            
            # Normalize weights to sum to 1 for each vertex
            weight_sums = inv_dist.sum(axis=1, keepdims=True)
            weights[batch_start:batch_end] = inv_dist / weight_sums
        
        return weights
    
    def deform_mesh(self, deformed_cage_vertices: np.ndarray) -> np.ndarray:
        """
        Deform mesh using deformed cage (fast matrix multiplication).
        
        Args:
            deformed_cage_vertices: (N, 3) deformed cage vertex positions
            
        Returns:
            (M, 3) deformed mesh vertex positions
        """
        # This is the magic: single matrix multiplication
        # Each mesh vertex is a weighted average of cage vertices
        return self.mvc_weights @ deformed_cage_vertices


class MediaPipeCageDeformer:
    """
    Deforms unified cage based on MediaPipe keypoints.
    
    Applies section-wise transformations while maintaining cage connectivity.
    """
    
    # MediaPipe landmark indices
    LANDMARKS = {
        'left_shoulder': 11,
        'right_shoulder': 12,
        'left_elbow': 13,
        'right_elbow': 14,
        'left_wrist': 15,
        'right_wrist': 16,
        'left_hip': 23,
        'right_hip': 24,
    }
    
    # Map cage sections to relevant keypoints
    SECTION_KEYPOINTS = {
        'torso': ['left_shoulder', 'right_shoulder', 'left_hip', 'right_hip'],
        'left_upper_arm': ['left_shoulder', 'left_elbow'],
        'right_upper_arm': ['right_shoulder', 'right_elbow'],
        'left_lower_arm': ['left_elbow', 'left_wrist'],
        'right_lower_arm': ['right_elbow', 'right_wrist'],
    }
    
    def __init__(self, original_cage_vertices: np.ndarray, section_info: Dict,
                 frame_shape: Tuple[int, int]):
        """
        Args:
            original_cage_vertices: Initial cage vertex positions
            section_info: Maps section names to vertex indices
            frame_shape: (height, width) of video frame
        """
        self.original_cage_vertices = original_cage_vertices.copy()
        self.section_info = section_info
        self.frame_shape = frame_shape
        
        # Compute original section centers for reference
        self.original_section_centers = {}
        for section_name, vertex_indices in section_info.items():
            section_verts = original_cage_vertices[vertex_indices]
            self.original_section_centers[section_name] = section_verts.mean(axis=0)
        
        # Temporal smoothing
        self.prev_deformed_cage = original_cage_vertices.copy()
        self.smooth_alpha = 0.3
        
        print(f"\n✓ Cage deformer initialized")
        print(f"  Sections: {list(section_info.keys())}")
        print(f"  Frame shape: {frame_shape}")
    
    def extract_keypoints(self, landmarks) -> Dict[str, np.ndarray]:
        """
        Extract MediaPipe keypoints in mesh-normalized space.
        
        Args:
            landmarks: MediaPipe pose landmarks
            
        Returns:
            Dict of {keypoint_name: np.array([x, y, z])}
        """
        h, w = self.frame_shape
        keypoints = {}
        
        for name, idx in self.LANDMARKS.items():
            if idx < len(landmarks.landmark):
                lm = landmarks.landmark[idx]
                
                # Convert to normalized space centered at origin
                x = (lm.x - 0.5) * 2.0  # [-1, 1]
                y = -(lm.y - 0.5) * 2.0  # Flip Y, [-1, 1]
                z = lm.z * 2.0  # MediaPipe Z (relative depth)
                
                keypoints[name] = np.array([x, y, z])
        
        return keypoints
    
    def deform_cage(self, keypoints: Dict[str, np.ndarray]) -> np.ndarray:
        """
        Deform cage based on current keypoints.
        
        Applies section-wise transformations with hierarchical constraints.
        
        Args:
            keypoints: Dict of {keypoint_name: np.array([x, y, z])}
            
        Returns:
            Deformed cage vertices
        """
        deformed_cage = self.original_cage_vertices.copy()
        
        # Hierarchy: arms depend on torso
        parent_transforms = {}
        
        for section_name, vertex_indices in self.section_info.items():
            if len(vertex_indices) == 0:
                continue
            
            # Get keypoints for this section
            section_kp_names = self.SECTION_KEYPOINTS.get(section_name, [])
            section_kps = [keypoints.get(name) for name in section_kp_names]
            section_kps = [kp for kp in section_kps if kp is not None]
            
            if len(section_kps) < 2:
                # Not enough keypoints, keep section stable
                continue
            
            # Compute current section center from keypoints
            current_center = np.mean(section_kps, axis=0)
            
            # Compute translation
            original_center = self.original_section_centers[section_name]
            translation = current_center - original_center
            
            # Apply hierarchical constraint: arms inherit torso movement
            if 'arm' in section_name and 'torso' in parent_transforms:
                torso_translation = parent_transforms['torso']
                translation += torso_translation * 0.5  # 50% inheritance
            
            # Apply translation to section vertices
            deformed_cage[vertex_indices] += translation
            
            # Store for children
            parent_transforms[section_name] = translation
        
        # Temporal smoothing
        deformed_cage = (self.smooth_alpha * deformed_cage +
                        (1 - self.smooth_alpha) * self.prev_deformed_cage)
        self.prev_deformed_cage = deformed_cage.copy()
        
        return deformed_cage


class IntegratedCageSystem:
    """
    Main system integrating all components for unified cage deformation.
    """
    
    def __init__(self, mesh_path: str, headless: bool = False):
        """
        Args:
            mesh_path: Path to .obj mesh file
            headless: Run without Python viewer window
        """
        self.mesh_path = mesh_path
        self.headless = headless
        
        # Components
        self.mesh = None
        self.cage = None
        self.section_info = None
        self.mvc = None
        self.cage_deformer = None
        self.pose_detector = None
        
        # WebSocket
        self.ws_server = None
        self.loop = None
        
        # Video
        self.cap = None
        
        # State
        self.original_cage_vertices = None
        self.show_cage = True
        self.running = True
        
        print(f"\n{'='*70}")
        print("UNIFIED CAGE DEFORMATION SYSTEM - OPTION A")
        print(f"{'='*70}")
        print(f"Mesh: {mesh_path}")
        print(f"Headless: {headless}")
        print(f"{'='*70}\n")
    
    def setup(self):
        """Initialize all components"""
        # Load mesh
        print("Loading mesh...")
        self.mesh = trimesh.load(self.mesh_path)
        print(f"✓ Mesh loaded: {len(self.mesh.vertices)} vertices, {len(self.mesh.faces)} faces")
        
        # Generate unified cage
        cage_generator = UnifiedCageGenerator(self.mesh)
        self.cage, self.section_info = cage_generator.generate_unified_cage(subdivisions=2)
        self.original_cage_vertices = self.cage.vertices.copy()
        
        # Compute MVC weights
        self.mvc = UnifiedMVCCoordinates(self.mesh.vertices, self.cage.vertices)
        
        # Initialize MediaPipe
        print("Loading MediaPipe Pose...")
        mp_pose = mp.solutions.pose
        self.pose_detector = mp_pose.Pose(
            static_image_mode=False,
            model_complexity=1,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        print("✓ MediaPipe Pose loaded")
        
        # Initialize camera
        print("Initializing camera...")
        self.cap = cv2.VideoCapture(0)
        ret, frame = self.cap.read()
        if not ret:
            raise RuntimeError("Failed to open camera")
        
        frame_shape = frame.shape[:2]
        print(f"✓ Camera initialized: {frame_shape[1]}x{frame_shape[0]}")
        
        # Initialize cage deformer
        self.cage_deformer = MediaPipeCageDeformer(
            self.original_cage_vertices,
            self.section_info,
            frame_shape
        )
        
        # Start WebSocket server
        print("\n🔄 Starting WebSocket server...")
        self.start_websocket_server()
        time.sleep(1)  # Give server time to start
        print("✓ WebSocket server ready")
        
        print(f"\n{'='*70}")
        print("✓ SYSTEM READY")
        print(f"{'='*70}\n")
        print("Controls:")
        print("  Q - Quit")
        print("  C - Toggle cage visualization")
        print("  O - Open web viewer (tests/enhanced_mesh_viewer_v2.html)")
        print()
    
    def start_websocket_server(self):
        """Start WebSocket server in separate thread"""
        def run_server():
            self.loop = asyncio.new_event_loop()
            asyncio.set_event_loop(self.loop)
            
            self.ws_server = EnhancedMeshStreamServerV2(host='localhost', port=8765)
            
            async def serve():
                await self.ws_server.start()
                # Keep running until told to stop
                while self.running:
                    await asyncio.sleep(0.1)
            
            self.loop.run_until_complete(serve())
        
        server_thread = threading.Thread(target=run_server, daemon=True)
        server_thread.start()
    
    def run(self):
        """Main processing loop"""
        frame_count = 0
        fps_update_time = time.time()
        fps = 0
        
        while self.running:
            ret, frame = self.cap.read()
            if not ret:
                break
            
            frame_count += 1
            
            # Update FPS
            if time.time() - fps_update_time > 1.0:
                fps = frame_count
                frame_count = 0
                fps_update_time = time.time()
            
            # Process MediaPipe
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = self.pose_detector.process(rgb)
            
            if results.pose_landmarks:
                # Extract keypoints
                keypoints = self.cage_deformer.extract_keypoints(results.pose_landmarks)
                
                # Deform cage
                deformed_cage_vertices = self.cage_deformer.deform_cage(keypoints)
                
                # Deform mesh using MVC
                deformed_mesh_vertices = self.mvc.deform_mesh(deformed_cage_vertices)
                
                # Send to web viewer
                if self.ws_server and self.loop:
                    asyncio.run_coroutine_threadsafe(
                        self.ws_server.send_mesh_data(
                            vertices=deformed_mesh_vertices,
                            faces=self.mesh.faces,
                            cage_vertices=deformed_cage_vertices if self.show_cage else None,
                            cage_faces=self.cage.faces if self.show_cage else None
                        ),
                        self.loop
                    )
            
            # Display (if not headless)
            if not self.headless:
                display_frame = frame.copy()
                
                # Draw info
                cv2.putText(display_frame, f"FPS: {fps}", (10, 30),
                           cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                cv2.putText(display_frame, "Option A: Unified Cage", (10, 70),
                           cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2)
                cv2.putText(display_frame, f"Mesh: {len(self.mesh.vertices)} verts", (10, 110),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 1)
                cv2.putText(display_frame, f"Cage: {len(self.cage.vertices)} verts", (10, 140),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 1)
                
                # Draw MediaPipe skeleton
                if results.pose_landmarks:
                    mp.solutions.drawing_utils.draw_landmarks(
                        display_frame,
                        results.pose_landmarks,
                        mp.solutions.pose.POSE_CONNECTIONS
                    )
                
                cv2.imshow("Unified Cage Deformation", display_frame)
            
            # Handle keys
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                self.running = False
            elif key == ord('c'):
                self.show_cage = not self.show_cage
                print(f"Cage visualization: {'ON' if self.show_cage else 'OFF'}")
            elif key == ord('o'):
                print("\nPlease open: tests/enhanced_mesh_viewer_v2.html in your browser")
        
        # Cleanup
        self.cap.release()
        cv2.destroyAllWindows()
        print("\n✓ System stopped")


def main():
    parser = argparse.ArgumentParser(description="Option A: Unified Cage Deformation")
    parser.add_argument('--mesh', type=str, required=True,
                       help='Path to .obj mesh file')
    parser.add_argument('--headless', action='store_true',
                       help='Run without Python viewer window')
    
    args = parser.parse_args()
    
    # Validate mesh path
    mesh_path = Path(args.mesh)
    if not mesh_path.exists():
        print(f"Error: Mesh file not found: {mesh_path}")
        return
    
    # Run system
    system = IntegratedCageSystem(str(mesh_path), headless=args.headless)
    system.setup()
    system.run()


if __name__ == '__main__':
    main()

