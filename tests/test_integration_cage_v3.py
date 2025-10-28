"""
Option A V3: Articulated Cage with REAL BodyPix Data
=====================================================

This version FIXES the issues in V2:
1. Loads REAL BodyPix masks from reference_data.pkl (not fake MediaPipe masks)
2. Fixes mesh orientation (90° Y-axis rotation)
3. Uses keypoint-based mask partitioning as fallback if no reference data

Key Fixes:
- Load reference_data.pkl with 24-part BodyPix segmentation
- Proper mesh orientation correction
- Fallback to intelligent mask partitioning using MediaPipe keypoints

Usage:
    python tests/test_integration_cage_v3.py \\
        --mesh generated_meshes/0/mesh.obj \\
        [--reference generated_meshes/0/reference_data.pkl] \\
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
import pickle
from pathlib import Path
from typing import Dict, List, Tuple, Optional

# Import our articulated cage system
from articulated_cage_generator import ArticulatedCageGenerator
from articulated_deformer import ArticulatedDeformer

# Import WebSocket server
from enhanced_websocket_server_v2 import EnhancedMeshStreamServerV2


class ArticulatedCageSystemV3:
    """
    V3: Uses REAL BodyPix data from reference_data.pkl
    
    Major improvements over V2:
    - Loads actual 24-part BodyPix segmentation
    - Proper mesh orientation
    - Keypoint-based mask partitioning fallback
    """
    
    def __init__(self, mesh_path: str, reference_path: Optional[str] = None, headless: bool = False):
        """
        Initialize the articulated cage system.
        
        Args:
            mesh_path: Path to 3D mesh file (.obj)
            reference_path: Path to reference_data.pkl with BodyPix masks
            headless: If True, don't show OpenCV windows
        """
        self.headless = headless
        self.reference_path = reference_path
        
        # Load mesh
        print(f"\n{'='*70}")
        print("LOADING 3D MESH")
        print(f"{'='*70}")
        self.mesh = trimesh.load(mesh_path)
        
        # FIX: Apply mesh orientation correction (same as clothing_to_3d_triposr_2.py)
        print("Applying mesh orientation correction...")
        # 180° flip around X-axis (fix upside-down)
        flip_transform = np.eye(4)
        angle_x = np.pi
        c_x, s_x = np.cos(angle_x), np.sin(angle_x)
        R_flip = np.array([
            [1, 0, 0],
            [0, c_x, -s_x],
            [0, s_x, c_x]
        ])
        flip_transform[:3, :3] = R_flip
        self.mesh.apply_transform(flip_transform)
        
        # 90° rotation around Y-axis (face forward)
        forward_transform = np.eye(4)
        angle_y = np.pi / 2
        c_y, s_y = np.cos(angle_y), np.sin(angle_y)
        R_forward = np.array([
            [c_y, 0, s_y],
            [0, 1, 0],
            [-s_y, 0, c_y]
        ])
        forward_transform[:3, :3] = R_forward
        self.mesh.apply_transform(forward_transform)
        
        print(f"✓ Loaded mesh: {len(self.mesh.vertices)} vertices, {len(self.mesh.faces)} faces")
        print(f"  Bounds: {self.mesh.bounds[0]} to {self.mesh.bounds[1]}")
        print(f"  ✓ Orientation corrected (180° X-flip, 90° Y-rotation)")
        print(f"{'='*70}\n")
        
        # Load reference data if provided
        self.reference_data = None
        if reference_path:
            self.load_reference_data(reference_path)
        
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
    
    def load_reference_data(self, reference_path: str):
        """Load reference data with BodyPix masks."""
        print(f"\n{'='*70}")
        print("LOADING REFERENCE DATA")
        print(f"{'='*70}")
        
        try:
            with open(reference_path, 'rb') as f:
                self.reference_data = pickle.load(f)
            
            print(f"✓ Loaded reference data from: {reference_path}")
            print(f"  BodyPix masks: {len(self.reference_data.get('bodypix_masks', {}))} parts")
            print(f"  Keypoints: {len(self.reference_data.get('mediapipe_keypoints_2d', {}))} points")
            print(f"  Frame shape: {self.reference_data.get('frame_shape', 'unknown')}")
            print(f"{'='*70}\n")
        except Exception as e:
            print(f"⚠ Could not load reference data: {e}")
            print("  Will use keypoint-based mask partitioning instead")
            self.reference_data = None
    
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
    
    def partition_mask_by_keypoints(
        self,
        full_mask: np.ndarray,
        keypoints_2d: Dict[str, Tuple[float, float]]
    ) -> Dict[str, np.ndarray]:
        """
        Partition a single segmentation mask into body parts using keypoint positions.
        
        This is the FALLBACK when we don't have real BodyPix data.
        
        Args:
            full_mask: Single binary mask (H, W)
            keypoints_2d: MediaPipe keypoint positions in pixels
            
        Returns:
            Dict mapping section names to binary masks
        """
        h, w = full_mask.shape
        masks = {}
        
        # Get key Y-coordinates
        nose_y = keypoints_2d.get('nose', (0, h*0.1))[1]
        shoulder_y = (keypoints_2d.get('left_shoulder', (0, h*0.25))[1] + 
                     keypoints_2d.get('right_shoulder', (0, h*0.25))[1]) / 2
        hip_y = (keypoints_2d.get('left_hip', (0, h*0.55))[1] + 
                 keypoints_2d.get('right_hip', (0, h*0.55))[1]) / 2
        knee_y = (keypoints_2d.get('left_knee', (0, h*0.75))[1] + 
                  keypoints_2d.get('right_knee', (0, h*0.75))[1]) / 2
        
        # Get key X-coordinates
        left_shoulder_x = keypoints_2d.get('left_shoulder', (w*0.35, 0))[0]
        right_shoulder_x = keypoints_2d.get('right_shoulder', (w*0.65, 0))[0]
        center_x = w / 2
        
        # HEAD: Above shoulders
        head_mask = full_mask.copy()
        head_mask[int(shoulder_y):, :] = 0
        if np.sum(head_mask > 0) > 100:
            masks['head'] = head_mask
        
        # TORSO: Shoulders to hips
        torso_mask = full_mask.copy()
        torso_mask[:int(shoulder_y), :] = 0
        torso_mask[int(hip_y):, :] = 0
        if np.sum(torso_mask > 0) > 100:
            masks['torso'] = torso_mask
        
        # ARMS: Shoulders to hips, left/right of center
        arm_top = int(shoulder_y)
        arm_bottom = int(hip_y)
        
        # Left arm
        left_arm_mask = full_mask.copy()
        left_arm_mask[:arm_top, :] = 0
        left_arm_mask[arm_bottom:, :] = 0
        left_arm_mask[:, int(center_x):] = 0  # Only left half
        if np.sum(left_arm_mask > 0) > 100:
            masks['left_upper_arm'] = left_arm_mask
        
        # Right arm
        right_arm_mask = full_mask.copy()
        right_arm_mask[:arm_top, :] = 0
        right_arm_mask[arm_bottom:, :] = 0
        right_arm_mask[:, :int(center_x)] = 0  # Only right half
        if np.sum(right_arm_mask > 0) > 100:
            masks['right_upper_arm'] = right_arm_mask
        
        # LEGS: Below hips
        leg_top = int(hip_y)
        leg_bottom = int(knee_y)
        
        # Left upper leg
        left_upper_leg_mask = full_mask.copy()
        left_upper_leg_mask[:leg_top, :] = 0
        left_upper_leg_mask[leg_bottom:, :] = 0
        left_upper_leg_mask[:, int(center_x):] = 0
        if np.sum(left_upper_leg_mask > 0) > 100:
            masks['left_upper_leg'] = left_upper_leg_mask
        
        # Right upper leg
        right_upper_leg_mask = full_mask.copy()
        right_upper_leg_mask[:leg_top, :] = 0
        right_upper_leg_mask[leg_bottom:, :] = 0
        right_upper_leg_mask[:, :int(center_x)] = 0
        if np.sum(right_upper_leg_mask > 0) > 100:
            masks['right_upper_leg'] = right_upper_leg_mask
        
        print(f"  ✓ Partitioned mask into {len(masks)} sections: {list(masks.keys())}")
        
        return masks
    
    def get_bodypix_masks(self, frame: np.ndarray, landmarks) -> Dict[str, np.ndarray]:
        """
        Get BodyPix masks either from reference data or by partitioning MediaPipe mask.
        
        Args:
            frame: Camera frame
            landmarks: MediaPipe pose landmarks
            
        Returns:
            Dict mapping section names to binary masks
        """
        # Option 1: Use reference data if available
        if self.reference_data and 'bodypix_masks' in self.reference_data:
            print("  Using BodyPix masks from reference data")
            return self.reference_data['bodypix_masks']
        
        # Option 2: Partition MediaPipe segmentation
        print("  Using keypoint-based mask partitioning (fallback)")
        segmentation_mask = self.pose.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)).segmentation_mask
        
        if segmentation_mask is None:
            print("  ⚠ No segmentation mask available")
            return {}
        
        # Convert to binary mask
        binary_mask = (segmentation_mask > 0.5).astype(np.uint8) * 255
        
        # Extract keypoints
        keypoints_2d = self.extract_keypoints_2d(landmarks)
        
        # Partition mask
        return self.partition_mask_by_keypoints(binary_mask, keypoints_2d)
    
    def initialize_cage(self, frame: np.ndarray, landmarks):
        """
        Initialize cage from current frame (T-pose) after 5-second countdown.
        
        Args:
            frame: Camera frame (RGB)
            landmarks: MediaPipe pose landmarks
        """
        if self.cage_initialized:
            print("⚠ Cage already initialized")
            return
        
        # 5-SECOND COUNTDOWN
        print(f"\n{'='*70}")
        print("T-POSE CALIBRATION COUNTDOWN")
        print(f"{'='*70}")
        print("Get into T-pose position (arms extended horizontally)...")
        
        for i in range(5, 0, -1):
            print(f"  {i}...")
            time.sleep(1)
        
        print("✓ Capturing T-pose!\n")
        
        print(f"{'='*70}")
        print("INITIALIZING ARTICULATED CAGE SYSTEM")
        print(f"{'='*70}")
        
        # Extract keypoints
        keypoints_2d = self.extract_keypoints_2d(landmarks)
        print(f"✓ Extracted {len(keypoints_2d)} keypoints (2D)")
        
        # Get BodyPix masks (from reference data or partitioning)
        bodypix_masks = self.get_bodypix_masks(frame, landmarks)
        print(f"✓ Got {len(bodypix_masks)} body part masks")
        
        if len(bodypix_masks) == 0:
            print("✗ No body part masks available - cannot initialize cage")
            return
        
        # Generate articulated cage
        self.cage_generator = ArticulatedCageGenerator(self.mesh)
        try:
            self.cage_mesh, self.section_info, self.joint_info = self.cage_generator.generate_cage(
                bodypix_masks, keypoints_2d, self.frame_shape, padding=0.15
            )
        except Exception as e:
            print(f"✗ Cage generation failed: {e}")
            import traceback
            traceback.print_exc()
            return
        
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
        print("STARTING ARTICULATED CAGE DEMO (V3)")
        print(f"{'='*70}")
        print("\nControls:")
        print("  Q - Quit")
        print("  SPACE - Start T-pose calibration (5-second countdown)")
        print("  R - Reset cage")
        print("  C - Toggle cage visualization")
        print("\nCalibration Process:")
        print("  1. Press SPACE to start countdown")
        print("  2. Get into T-pose during 5-second countdown")
        print("  3. System captures cage at end of countdown")
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
                    cv2.putText(frame, "Press SPACE for T-pose calibration (5s countdown)",
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
                    cv2.imshow('Articulated Cage Demo V3', frame)
                
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
    parser = argparse.ArgumentParser(description='Articulated Cage Deformation Demo V3')
    parser.add_argument('--mesh', type=str, default='generated_meshes/0/mesh.obj',
                        help='Path to 3D mesh file')
    parser.add_argument('--reference', type=str, default=None,
                        help='Path to reference_data.pkl with BodyPix masks')
    parser.add_argument('--headless', action='store_true',
                        help='Run without OpenCV windows')
    args = parser.parse_args()
    
    # Check if mesh exists
    mesh_path = Path(args.mesh)
    if not mesh_path.exists():
        print(f"✗ Mesh not found: {mesh_path}")
        print(f"  Please provide a valid mesh path")
        return
    
    # Auto-detect reference data if not specified
    if args.reference is None:
        reference_path = mesh_path.parent / "reference_data.pkl"
        if reference_path.exists():
            print(f"✓ Auto-detected reference data: {reference_path}")
            args.reference = str(reference_path)
        else:
            print(f"⚠ No reference data found at: {reference_path}")
            print(f"  Will use keypoint-based mask partitioning")
    
    # Run system
    system = ArticulatedCageSystemV3(str(mesh_path), args.reference, headless=args.headless)
    system.run()


if __name__ == "__main__":
    main()

