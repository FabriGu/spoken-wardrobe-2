"""
Option A V2: Articulated Cage with OBBs and Regional MVC
=========================================================

This is the PROPERLY IMPLEMENTED version of Option A, based on research literature:
- Le & Deng (2017): "Interactive Cage Generation for Mesh Deformation"
- Xian et al. (2012): "Automatic cage generation by improved OBBs"

Key Improvements Over V1:
1. OBBs (not ConvexHull) for distinct anatomical sections
2. Regional MVC (not global) to prevent pinching
3. Hierarchical parent-child connections (no detachment)
4. Angle-based articulated rotation (foundation ready)

Usage:
    python tests/test_integration_cage_v2.py \\
        --mesh generated_meshes/0/mesh.obj \\
        [--headless]

Controls:
    Q - Quit
    SPACE - Capture T-pose for cage generation
    R - Reset
    C - Toggle cage visualization

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
from typing import Dict, List, Tuple, Optional

# Import our new articulated cage system
from articulated_cage_generator import ArticulatedCageGenerator
from articulated_deformer import ArticulatedDeformer

# Import WebSocket server
from enhanced_websocket_server_v2 import EnhancedMeshStreamServerV2


class ArticulatedCageSystem:
    """
    Main system integrating articulated cage generation and deformation.
    
    This is the correct implementation of cage-based deformation.
    """
    
    def __init__(self, mesh_path: str, headless: bool = False):
        """
        Initialize the articulated cage system.
        
        Args:
            mesh_path: Path to 3D mesh file (.obj)
            headless: If True, don't show OpenCV windows
        """
        self.headless = headless
        
        # Load mesh
        print(f"\n{'='*70}")
        print("LOADING 3D MESH")
        print(f"{'='*70}")
        self.mesh = trimesh.load(mesh_path)
        print(f"✓ Loaded mesh: {len(self.mesh.vertices)} vertices, {len(self.mesh.faces)} faces")
        print(f"  Bounds: {self.mesh.bounds[0]} to {self.mesh.bounds[1]}")
        print(f"{'='*70}\n")
        
        # Initialize MediaPipe Pose
        self.mp_pose = mp.solutions.pose
        self.pose = self.mp_pose.Pose(
            static_image_mode=False,
            model_complexity=1,
            enable_segmentation=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        self.mp_drawing = mp.solutions.drawing_utils
        
        # Camera
        self.cap = cv2.VideoCapture(0)
        self.frame_shape = None
        
        # Cage system
        self.cage_generator = None
        self.deformer = None
        self.cage_mesh = None
        self.section_info = None
        self.joint_info = None
        
        # State
        self.cage_initialized = False
        self.show_cage = True
        
        # WebSocket
        self.ws_server = None
        self.ws_loop = None
        self.ws_thread = None
        
        # Keypoint mapping
        self.LANDMARK_NAMES = {
            11: 'left_shoulder',
            12: 'right_shoulder',
            13: 'left_elbow',
            14: 'right_elbow',
            15: 'left_wrist',
            16: 'right_wrist',
            23: 'left_hip',
            24: 'right_hip',
            25: 'left_knee',
            26: 'right_knee',
            27: 'left_ankle',
            28: 'right_ankle',
            0: 'nose',
        }
    
    def start_websocket_server(self):
        """Start WebSocket server in background thread"""
        def run_server():
            self.ws_loop = asyncio.new_event_loop()
            asyncio.set_event_loop(self.ws_loop)
            self.ws_server = EnhancedMeshStreamServerV2(host='localhost', port=8765)
            self.ws_loop.run_until_complete(self.ws_server.start())
            self.ws_loop.run_forever()
        
        self.ws_thread = threading.Thread(target=run_server, daemon=True)
        self.ws_thread.start()
        time.sleep(1)  # Wait for server to start
        print("✓ WebSocket server started on ws://localhost:8765")
    
    def extract_keypoints_2d(self, landmarks) -> Dict[str, Tuple[float, float]]:
        """
        Extract 2D keypoints from MediaPipe landmarks.
        
        Returns:
            Dict mapping keypoint names to (x, y) in pixel coordinates
        """
        h, w = self.frame_shape
        keypoints_2d = {}
        
        for idx, name in self.LANDMARK_NAMES.items():
            if idx < len(landmarks.landmark):
                lm = landmarks.landmark[idx]
                keypoints_2d[name] = (lm.x * w, lm.y * h)
        
        return keypoints_2d
    
    def extract_keypoints_3d(self, landmarks) -> Dict[str, np.ndarray]:
        """
        Extract 3D keypoints from MediaPipe landmarks (normalized to mesh space).
        
        Returns:
            Dict mapping keypoint names to (x, y, z) in mesh space
        """
        mesh_bounds = self.mesh.bounds
        mesh_center = self.mesh.centroid
        mesh_size = mesh_bounds[1] - mesh_bounds[0]
        
        keypoints_3d = {}
        
        for idx, name in self.LANDMARK_NAMES.items():
            if idx < len(landmarks.landmark):
                lm = landmarks.landmark[idx]
                
                # Normalize to [-1, 1]
                x_norm = lm.x * 2 - 1
                y_norm = 1 - lm.y * 2  # Flip Y
                z_norm = lm.z * 2  # MediaPipe Z is already relative
                
                # Scale to mesh dimensions
                x_mesh = x_norm * mesh_size[0] / 2 + mesh_center[0]
                y_mesh = y_norm * mesh_size[1] / 2 + mesh_center[1]
                z_mesh = z_norm * mesh_size[2] / 2 + mesh_center[2]
                
                keypoints_3d[name] = np.array([x_mesh, y_mesh, z_mesh])
        
        return keypoints_3d
    
    def extract_bodypix_masks(self, segmentation_mask: np.ndarray) -> Dict[str, np.ndarray]:
        """
        Extract body part masks from MediaPipe segmentation.
        
        For now, use MediaPipe's single-mask segmentation.
        In full pipeline, this would use BodyPix with 24 parts.
        
        Args:
            segmentation_mask: MediaPipe segmentation mask
            
        Returns:
            Dict mapping section names to binary masks
        """
        # For demo: create simple masks from segmentation
        # In production: use actual BodyPix masks
        
        h, w = segmentation_mask.shape
        masks = {}
        
        # Simple heuristic segmentation based on Y-position
        body_mask = (segmentation_mask > 0.5).astype(np.uint8) * 255
        
        # Torso: middle 40%-80% vertical
        torso_mask = body_mask.copy()
        torso_mask[:int(h*0.2), :] = 0
        torso_mask[int(h*0.8):, :] = 0
        masks['torso'] = torso_mask
        
        # Arms: top 40%-70%, left/right halves
        arm_mask = body_mask.copy()
        arm_mask[:int(h*0.4), :] = 0
        arm_mask[int(h*0.7):, :] = 0
        
        left_arm = arm_mask.copy()
        left_arm[:, w//2:] = 0
        masks['left_upper_arm'] = left_arm
        
        right_arm = arm_mask.copy()
        right_arm[:, :w//2] = 0
        masks['right_upper_arm'] = right_arm
        
        return masks
    
    def initialize_cage(self, frame: np.ndarray, landmarks):
        """
        Initialize cage from current frame (T-pose).
        
        Args:
            frame: Camera frame (RGB)
            landmarks: MediaPipe pose landmarks
        """
        if self.cage_initialized:
            print("⚠ Cage already initialized")
            return
        
        print(f"\n{'='*70}")
        print("INITIALIZING ARTICULATED CAGE SYSTEM")
        print(f"{'='*70}")
        
        # Extract keypoints
        keypoints_2d = self.extract_keypoints_2d(landmarks)
        print(f"✓ Extracted {len(keypoints_2d)} keypoints (2D)")
        
        # Get MediaPipe segmentation as proxy for BodyPix
        segmentation_mask = self.pose.process(frame).segmentation_mask
        if segmentation_mask is None:
            print("✗ No segmentation mask available")
            return
        
        # Extract body part masks (simplified)
        bodypix_masks = self.extract_bodypix_masks(segmentation_mask)
        print(f"✓ Extracted {len(bodypix_masks)} body part masks")
        
        # Generate articulated cage
        self.cage_generator = ArticulatedCageGenerator(self.mesh)
        self.cage_mesh, self.section_info, self.joint_info = self.cage_generator.generate_cage(
            bodypix_masks, keypoints_2d, self.frame_shape, padding=0.15
        )
        
        # Initialize deformer
        self.deformer = ArticulatedDeformer(
            self.mesh.vertices,
            self.cage_mesh.vertices,
            self.section_info,
            self.joint_info
        )
        
        # Set reference pose (T-pose)
        keypoints_3d = self.extract_keypoints_3d(landmarks)
        self.deformer.set_reference_pose(keypoints_3d)
        print(f"✓ Reference T-pose captured")
        
        self.cage_initialized = True
        
        print(f"\n{'='*70}")
        print("✓ ARTICULATED CAGE SYSTEM READY")
        print(f"{'='*70}\n")
    
    def run(self):
        """Main loop"""
        print(f"\n{'='*70}")
        print("STARTING ARTICULATED CAGE DEMO")
        print(f"{'='*70}")
        print("\nControls:")
        print("  Q - Quit")
        print("  SPACE - Capture T-pose and initialize cage")
        print("  R - Reset cage")
        print("  C - Toggle cage visualization")
        print(f"{'='*70}\n")
        
        # Start WebSocket server
        self.start_websocket_server()
        
        frame_count = 0
        fps_start = time.time()
        
        try:
            while True:
                ret, frame = self.cap.read()
                if not ret:
                    break
                
                if self.frame_shape is None:
                    self.frame_shape = (frame.shape[0], frame.shape[1])
                
                # Convert to RGB for MediaPipe
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                
                # Process with MediaPipe
                results = self.pose.process(frame_rgb)
                
                # Draw pose on frame
                if results.pose_landmarks:
                    if not self.headless:
                        self.mp_drawing.draw_landmarks(
                            frame, results.pose_landmarks, self.mp_pose.POSE_CONNECTIONS
                        )
                    
                    # If cage initialized, deform mesh
                    if self.cage_initialized:
                        keypoints_3d = self.extract_keypoints_3d(results.pose_landmarks)
                        deformed_mesh, deformed_cage = self.deformer.deform(keypoints_3d)
                        
                        # Send to web viewer
                        if self.ws_server:
                            asyncio.run_coroutine_threadsafe(
                                self.ws_server.send_mesh_data(
                                    vertices=deformed_mesh,
                                    faces=self.mesh.faces,
                                    cage_vertices=deformed_cage if self.show_cage else None,
                                    cage_faces=self.cage_mesh.faces if self.show_cage else None
                                ),
                                self.ws_loop
                            )
                
                # Show instructions on frame
                if not self.cage_initialized:
                    cv2.putText(frame, "Press SPACE in T-pose to initialize cage",
                                (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                else:
                    cv2.putText(frame, "Cage Active - Move around!",
                                (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                
                # Show FPS
                frame_count += 1
                if frame_count % 30 == 0:
                    fps = 30 / (time.time() - fps_start)
                    fps_start = time.time()
                    print(f"FPS: {fps:.1f}")
                
                # Display frame
                if not self.headless:
                    cv2.imshow('Articulated Cage Demo', frame)
                
                # Handle keys
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    break
                elif key == ord(' ') and results.pose_landmarks:
                    self.initialize_cage(frame_rgb, results.pose_landmarks)
                elif key == ord('r'):
                    self.cage_initialized = False
                    print("✓ Cage reset")
                elif key == ord('c'):
                    self.show_cage = not self.show_cage
                    print(f"✓ Cage visualization: {'ON' if self.show_cage else 'OFF'}")
        
        finally:
            self.cap.release()
            cv2.destroyAllWindows()
            if self.ws_loop:
                self.ws_loop.call_soon_threadsafe(self.ws_loop.stop)
            print("\n✓ System shutdown complete")


def main():
    parser = argparse.ArgumentParser(description='Articulated Cage Deformation Demo')
    parser.add_argument('--mesh', type=str, default='generated_meshes/0/mesh.obj',
                        help='Path to 3D mesh file')
    parser.add_argument('--headless', action='store_true',
                        help='Run without OpenCV windows')
    args = parser.parse_args()
    
    # Check if mesh exists
    mesh_path = Path(args.mesh)
    if not mesh_path.exists():
        print(f"✗ Mesh not found: {mesh_path}")
        print(f"  Please provide a valid mesh path")
        return
    
    # Run system
    system = ArticulatedCageSystem(str(mesh_path), headless=args.headless)
    system.run()


if __name__ == "__main__":
    main()

