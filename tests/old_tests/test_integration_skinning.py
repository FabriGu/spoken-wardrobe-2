"""
Option B: Proper Skeletal Skinning with Automatic Rigging
===========================================================

This implementation provides proper Linear Blend Skinning (LBS) with:
1. Automatic skeleton generation from MediaPipe keypoints
2. Automatic skinning weight computation (no manual weight painting)
3. Proper bone transformations (rotation + translation, not just translation)
4. Hierarchical bone system (parent-child relationships)

Key differences from previous failed LBS attempts:
- Proper 4x4 transformation matrices (not just displacement vectors)
- Automatic weight computation using geodesic-like distance falloff
- Bind pose setup with inverse bind matrices
- Hierarchical transformations (arms inherit from torso)

Usage:
    python tests/test_integration_skinning.py \\
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

# Import WebSocket server
from enhanced_websocket_server_v2 import EnhancedMeshStreamServerV2


class AutomaticRigging:
    """
    Automatic rigging system that generates a skeleton from MediaPipe keypoints
    and computes skinning weights automatically.
    """
    
    # Define bone hierarchy (bone_name: (landmark_indices, parent_bone))
    BONE_HIERARCHY = {
        'torso': ([11, 12, 23, 24], None),  # Shoulders and hips
        'left_upper_arm': ([11, 13], 'torso'),  # Shoulder to elbow
        'right_upper_arm': ([12, 14], 'torso'),
        'left_lower_arm': ([13, 15], 'left_upper_arm'),  # Elbow to wrist
        'right_lower_arm': ([14, 16], 'right_upper_arm'),
    }
    
    def __init__(self, mesh: trimesh.Trimesh):
        """
        Args:
            mesh: The 3D clothing mesh to rig
        """
        self.mesh = mesh
        self.bone_names = list(self.BONE_HIERARCHY.keys())
        
        # Bind pose data (set during calibration)
        self.bind_bone_transforms = {}  # 4x4 matrices
        self.inv_bind_bone_transforms = {}  # Inverse bind matrices
        self.skinning_weights = None  # (n_verts, n_bones)
        self.body_scale = 1.0
        self.is_calibrated = False
        
        print(f"\n{'='*60}")
        print("AUTOMATIC RIGGING SYSTEM")
        print(f"{'='*60}")
        print(f"Mesh: {len(mesh.vertices)} vertices, {len(mesh.faces)} faces")
        print(f"Skeleton: {len(self.bone_names)} bones")
        print(f"  Bones: {self.bone_names}")
        print(f"{'='*60}\n")
    
    def calibrate(self, initial_keypoints: np.ndarray, body_scale: float):
        """
        One-time calibration: Set bind pose and compute skinning weights.
        
        This is the "automatic rigging" step - it creates the skeleton and
        computes how much each bone influences each vertex.
        
        Args:
            initial_keypoints: (33, 3) array of NORMALIZED MediaPipe keypoint positions
            body_scale: Body scale in pixels (for reference only, keypoints already normalized)
        """
        print(f"\n{'='*60}")
        print("CALIBRATING BIND POSE")
        print(f"{'='*60}")
        
        # Store body scale (already computed in pixel space before normalization)
        self.body_scale = body_scale
        
        print(f"Body scale: {self.body_scale:.1f} pixels")
        print(f"Keypoints space: Normalized (centered at origin)")
        
        # Compute bind pose bone transforms
        bind_bone_positions = []
        
        for bone_name in self.bone_names:
            # Compute bone transform in bind pose
            transform = self._compute_bone_transform(bone_name, initial_keypoints)
            self.bind_bone_transforms[bone_name] = transform
            
            # Compute inverse for skinning
            self.inv_bind_bone_transforms[bone_name] = np.linalg.inv(transform)
            
            # Extract bone position for weight computation
            bone_pos = transform[:3, 3]
            bind_bone_positions.append(bone_pos)
        
        # Compute skinning weights automatically
        print("\nComputing skinning weights...")
        self.skinning_weights = self._compute_skinning_weights(
            self.mesh.vertices,
            bind_bone_positions
        )
        
        self.is_calibrated = True
        
        print(f"\n✓ Calibration complete")
        print(f"  Skinning weights: {self.skinning_weights.shape}")
        print(f"  Weight sum per vertex: "
              f"min={self.skinning_weights.sum(axis=1).min():.4f}, "
              f"max={self.skinning_weights.sum(axis=1).max():.4f}")
        print(f"{'='*60}\n")
    
    def _compute_bone_transform(self, bone_name: str, keypoints: np.ndarray) -> np.ndarray:
        """
        Compute 4x4 transformation matrix for a bone.
        
        This includes BOTH rotation (bone orientation) AND translation (bone position).
        
        Args:
            bone_name: Name of the bone
            keypoints: (33, 3) MediaPipe keypoint positions
            
        Returns:
            4x4 transformation matrix
        """
        landmark_indices, parent = self.BONE_HIERARCHY[bone_name]
        
        # Compute bone center (average of landmarks)
        bone_landmarks = [keypoints[i] for i in landmark_indices]
        bone_center = np.mean(bone_landmarks, axis=0)
        
        # Compute bone direction for rotation
        if len(landmark_indices) >= 2:
            # Direction from first to last landmark
            bone_start = keypoints[landmark_indices[0]]
            bone_end = keypoints[landmark_indices[-1]]
            bone_direction = bone_end - bone_start
            bone_length = np.linalg.norm(bone_direction)
            
            if bone_length > 1e-6:
                bone_direction = bone_direction / bone_length
            else:
                bone_direction = np.array([0, 1, 0])  # Default: Y-up
        else:
            bone_direction = np.array([0, 1, 0])
        
        # Build rotation matrix to align bone with its direction
        # Default bone orientation is [0, 1, 0] (Y-up in model space)
        default_direction = np.array([0, 1, 0])
        
        # Compute rotation using Rodrigues' rotation formula
        rotation_axis = np.cross(default_direction, bone_direction)
        rotation_axis_length = np.linalg.norm(rotation_axis)
        
        if rotation_axis_length > 1e-6:
            # Normalize rotation axis
            rotation_axis = rotation_axis / rotation_axis_length
            
            # Compute rotation angle
            rotation_angle = np.arccos(np.clip(np.dot(default_direction, bone_direction), -1, 1))
            
            # Rodrigues' rotation formula: R = I + sin(θ)K + (1-cos(θ))K²
            K = np.array([
                [0, -rotation_axis[2], rotation_axis[1]],
                [rotation_axis[2], 0, -rotation_axis[0]],
                [-rotation_axis[1], rotation_axis[0], 0]
            ])
            
            rotation_matrix = (
                np.eye(3) +
                np.sin(rotation_angle) * K +
                (1 - np.cos(rotation_angle)) * (K @ K)
            )
        else:
            # No rotation needed (bone already aligned)
            rotation_matrix = np.eye(3)
        
        # Build 4x4 transformation matrix
        transform = np.eye(4)
        transform[:3, :3] = rotation_matrix
        transform[:3, 3] = bone_center
        
        return transform
    
    def _compute_skinning_weights(self, vertices: np.ndarray,
                                  bone_positions: List[np.ndarray]) -> np.ndarray:
        """
        Automatically compute skinning weights using distance-based falloff.
        
        Each vertex is influenced by nearby bones with smooth falloff.
        This is the "automatic weight painting" step.
        
        Args:
            vertices: (n_verts, 3) mesh vertices
            bone_positions: List of (3,) bone center positions
            
        Returns:
            (n_verts, n_bones) skinning weight matrix
        """
        n_verts = len(vertices)
        n_bones = len(bone_positions)
        weights = np.zeros((n_verts, n_bones))
        
        # Compute influence of each bone on each vertex
        for bone_idx, bone_pos in enumerate(bone_positions):
            bone_name = self.bone_names[bone_idx]
            
            # Compute distances from vertices to bone
            distances = np.linalg.norm(vertices - bone_pos, axis=1)
            
            # Adaptive sigma based on bone type
            # Arms need tighter influence, torso needs wider
            if 'arm' in bone_name:
                sigma = 0.3  # Tight influence for arms
            else:
                sigma = 0.5  # Wider influence for torso
            
            # Gaussian falloff: exp(-d²/2σ²)
            weights[:, bone_idx] = np.exp(-distances**2 / (2 * sigma**2))
        
        # Normalize weights: each vertex's weights sum to 1
        weight_sums = weights.sum(axis=1, keepdims=True)
        weight_sums[weight_sums == 0] = 1.0  # Avoid division by zero
        weights = weights / weight_sums
        
        return weights
    
    def deform_mesh(self, current_keypoints: np.ndarray, return_debug: bool = False):
        """
        Apply Linear Blend Skinning to deform mesh based on current keypoints.
        
        Formula: v' = Σ wᵢ * Mᵢ * Mᵢ_bind_inv * v
        
        Where:
        - wᵢ is the skinning weight for bone i
        - Mᵢ is the current bone transform (4x4 matrix)
        - Mᵢ_bind_inv is the inverse bind pose transform
        - v is the vertex in bind pose
        
        Args:
            current_keypoints: (33, 3) array of current MediaPipe keypoints
            return_debug: If True, return (deformed_vertices, bone_positions)
            
        Returns:
            deformed_vertices: (n_verts, 3) deformed mesh vertices
            bone_positions: (optional) Dict of bone positions for debugging
        """
        if not self.is_calibrated:
            raise RuntimeError("Must calibrate bind pose first!")
        
        # Compute current bone transforms
        current_bone_transforms = {}
        bone_positions = {}  # For debugging
        
        for bone_name in self.bone_names:
            # Compute current transform for this bone
            current_transform = self._compute_bone_transform(bone_name, current_keypoints)
            
            # Apply hierarchical transformation (parent-child)
            landmark_indices, parent_name = self.BONE_HIERARCHY[bone_name]
            if parent_name and parent_name in current_bone_transforms:
                # Child inherits parent's transformation
                parent_transform = current_bone_transforms[parent_name]
                current_transform = parent_transform @ current_transform
            
            current_bone_transforms[bone_name] = current_transform
            
            # Store bone position for debugging
            if return_debug:
                bone_positions[bone_name] = current_transform[:3, 3].tolist()
        
        # Apply LBS: for each vertex, blend bone transformations
        n_verts = len(self.mesh.vertices)
        deformed_vertices = np.zeros((n_verts, 3))
        
        # Convert vertices to homogeneous coordinates for transformation
        verts_homogeneous = np.hstack([self.mesh.vertices, np.ones((n_verts, 1))])
        
        for bone_idx, bone_name in enumerate(self.bone_names):
            # Get skinning weight for this bone
            bone_weights = self.skinning_weights[:, bone_idx][:, np.newaxis]  # (n_verts, 1)
            
            # Compute skinning matrix: M * M_bind_inv
            current_transform = current_bone_transforms[bone_name]
            bind_inv_transform = self.inv_bind_bone_transforms[bone_name]
            skinning_matrix = current_transform @ bind_inv_transform
            
            # Transform vertices: (4x4) @ (n_verts, 4).T = (4, n_verts)
            transformed = (skinning_matrix @ verts_homogeneous.T).T  # (n_verts, 4)
            
            # Accumulate weighted transformation
            deformed_vertices += bone_weights * transformed[:, :3]
        
        if return_debug:
            return deformed_vertices, bone_positions
        return deformed_vertices


class MediaPipeSkeletonDriver:
    """
    Drives the skeleton using MediaPipe pose estimation.
    Handles coordinate system conversions and temporal smoothing.
    """
    
    def __init__(self, frame_shape: Tuple[int, int], body_scale: float):
        """
        Args:
            frame_shape: (height, width) of video frame
            body_scale: Approximate body extent in pixels
        """
        self.frame_shape = frame_shape
        self.body_scale = body_scale
        
        # Temporal smoothing
        self.prev_keypoints = None
        self.smooth_alpha = 0.3
    
    def extract_keypoints(self, landmarks, smooth: bool = True) -> np.ndarray:
        """
        Extract and normalize MediaPipe keypoints.
        
        Args:
            landmarks: MediaPipe pose landmarks
            smooth: Whether to apply temporal smoothing
            
        Returns:
            (33, 3) array of keypoint positions in normalized space
        """
        h, w = self.frame_shape
        keypoints = np.zeros((33, 3))
        
        for i, lm in enumerate(landmarks.landmark):
            # Convert normalized coords [0,1] to pixel coords
            x_px = lm.x * w
            y_px = lm.y * h
            z_px = lm.z * w  # MediaPipe Z is relative depth, scaled by width
            
            # Normalize to model space centered at origin
            x_norm = (x_px - w/2) / (self.body_scale / 2)
            y_norm = (y_px - h/2) / (self.body_scale / 2)
            z_norm = z_px / (self.body_scale / 2)
            
            keypoints[i] = [x_norm, y_norm, z_norm]
        
        # Temporal smoothing
        if smooth and self.prev_keypoints is not None:
            keypoints = (self.smooth_alpha * keypoints +
                        (1 - self.smooth_alpha) * self.prev_keypoints)
        
        self.prev_keypoints = keypoints.copy()
        
        return keypoints


class IntegratedSkinningSystem:
    """
    Main system integrating all components for skeletal skinning deformation.
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
        self.rigging = None
        self.skeleton_driver = None
        self.pose_detector = None
        
        # WebSocket
        self.ws_server = None
        self.loop = None
        
        # Video
        self.cap = None
        
        # State
        self.is_calibrated = False
        self.running = True
        self.calibration_countdown = 0  # 0 = not counting, >0 = counting down
        self.calibration_start_time = 0
        
        print(f"\n{'='*70}")
        print("SKELETAL SKINNING SYSTEM - OPTION B")
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
        
        # Fix mesh orientation
        # Based on clothing_to_3d_triposr_2.py detect_and_correct_orientation()
        print("\nCorrecting mesh orientation...")
        
        bounds = self.mesh.bounds
        size = bounds[1] - bounds[0]
        print(f"  Original size: X={size[0]:.3f}, Y={size[1]:.3f}, Z={size[2]:.3f}")
        
        # Step 1: 180° X-flip (fixes upside-down - from TripoSR script)
        flip_transform = np.eye(4)
        angle_x = np.pi
        c, s = np.cos(angle_x), np.sin(angle_x)
        flip_transform[:3, :3] = np.array([
            [1, 0, 0],
            [0, c, -s],
            [0, s, c]
        ])
        self.mesh.apply_transform(flip_transform)
        print("  ✓ Applied 180° X-flip (upside-down fix)")
        
        # Step 2: NO Y-rotation for now - let's see the result first
        print(f"  Skipping Y-rotation - test orientation first")
        print(f"  If facing wrong direction, will add rotation in next iteration")
        
        print(f"✓ Mesh orientation corrected (partial - testing)")
        
        # Initialize automatic rigging
        self.rigging = AutomaticRigging(self.mesh)
        
        # Initialize MediaPipe
        print("\nLoading MediaPipe Pose...")
        mp_pose = mp.solutions.pose
        self.pose_detector = mp_pose.Pose(
            static_image_mode=False,
            model_complexity=1,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        print("✓ MediaPipe Pose loaded")
        
        # Initialize camera
        print("\nInitializing camera...")
        self.cap = cv2.VideoCapture(0)
        ret, frame = self.cap.read()
        if not ret:
            raise RuntimeError("Failed to open camera")
        
        frame_shape = frame.shape[:2]
        print(f"✓ Camera initialized: {frame_shape[1]}x{frame_shape[0]}")
        
        # Start WebSocket server
        print("\n🔄 Starting WebSocket server...")
        self.start_websocket_server()
        time.sleep(1)
        print("✓ WebSocket server ready")
        
        print(f"\n{'='*70}")
        print("✓ SYSTEM READY - WAITING FOR CALIBRATION")
        print(f"{'='*70}\n")
        print("Controls:")
        print("  SPACE - Calibrate bind pose (stand in T-pose)")
        print("  Q - Quit")
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
                while self.running:
                    await asyncio.sleep(0.1)
            
            self.loop.run_until_complete(serve())
        
        server_thread = threading.Thread(target=run_server, daemon=True)
        server_thread.start()
    
    def calibrate_from_frame(self, frame):
        """Calibrate bind pose from current frame"""
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.pose_detector.process(rgb)
        
        if not results.pose_landmarks:
            print("⚠️ No pose detected! Please ensure you're visible in frame.")
            return False
        
        # Extract keypoints in pixel space first (for body scale computation)
        h, w = frame.shape[:2]
        keypoints_px = np.zeros((33, 3))
        
        for i, lm in enumerate(results.pose_landmarks.landmark):
            keypoints_px[i] = [lm.x * w, lm.y * h, lm.z * w]
        
        # Compute body scale from pixel coordinates
        left_shoulder = keypoints_px[11]
        right_shoulder = keypoints_px[12]
        shoulder_dist = np.linalg.norm(right_shoulder - left_shoulder)
        body_scale = shoulder_dist * 2.5
        
        print(f"\nBody scale: {body_scale:.1f} pixels")
        
        # NOW normalize keypoints to model space for calibration
        keypoints_normalized = np.zeros((33, 3))
        
        for i, lm in enumerate(results.pose_landmarks.landmark):
            x_px = lm.x * w
            y_px = lm.y * h
            z_px = lm.z * w
            
            # Normalize to model space centered at origin
            keypoints_normalized[i, 0] = (x_px - w/2) / (body_scale / 2)
            keypoints_normalized[i, 1] = (y_px - h/2) / (body_scale / 2)
            keypoints_normalized[i, 2] = z_px / (body_scale / 2)
        
        # Calibrate rigging with NORMALIZED keypoints
        self.rigging.calibrate(keypoints_normalized, body_scale)
        
        # Initialize skeleton driver with body scale
        self.skeleton_driver = MediaPipeSkeletonDriver(
            frame_shape=(h, w),
            body_scale=body_scale
        )
        
        self.is_calibrated = True
        print("\n🎉 Calibration successful! System is now tracking your movements.")
        
        return True
    
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
            
            # Handle calibration countdown
            if self.calibration_countdown > 0:
                elapsed = time.time() - self.calibration_start_time
                remaining = 5.0 - elapsed
                
                if remaining <= 0:
                    # Countdown complete, calibrate now!
                    print("\n🎯 Calibrating NOW!")
                    self.calibrate_from_frame(frame)
                    self.calibration_countdown = 0
                else:
                    # Still counting down
                    self.calibration_countdown = int(remaining) + 1
            
            # Process frame
            if self.is_calibrated:
                # Extract keypoints
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                results = self.pose_detector.process(rgb)
                
                if results.pose_landmarks:
                    # Get normalized keypoints
                    keypoints = self.skeleton_driver.extract_keypoints(results.pose_landmarks)
                    
                    # Apply LBS deformation
                    deformed_vertices, bone_positions = self.rigging.deform_mesh(keypoints, return_debug=True)
                    
                    # Debug: Log vertex movement (every 60 frames)
                    if frame_count % 60 == 0:
                        movement = np.linalg.norm(deformed_vertices - self.mesh.vertices, axis=1)
                        print(f"\n[DEBUG] Vertex movement: min={movement.min():.4f}, max={movement.max():.4f}, mean={movement.mean():.4f}")
                        if movement.max() < 0.001:
                            print("  ⚠️  WARNING: Vertices barely moving! LBS may not be working.")
                    
                    # Send to web viewer with bone debug data
                    if self.ws_server and self.loop:
                        asyncio.run_coroutine_threadsafe(
                            self.ws_server.send_mesh_data(
                                vertices=deformed_vertices,
                                faces=self.mesh.faces,
                                debug_data={
                                    'bone_positions': bone_positions,
                                    'keypoints': keypoints.tolist()
                                }
                            ),
                            self.loop
                        )
            
            # Display
            if not self.headless:
                display_frame = frame.copy()
                
                # Draw info
                cv2.putText(display_frame, f"FPS: {fps}", (10, 30),
                           cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                cv2.putText(display_frame, "Option B: Skeletal Skinning", (10, 70),
                           cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2)
                
                if self.is_calibrated:
                    cv2.putText(display_frame, "Status: TRACKING", (10, 110),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                elif self.calibration_countdown > 0:
                    # Show countdown timer
                    countdown_text = f"CALIBRATING IN {self.calibration_countdown}... GET INTO T-POSE!"
                    cv2.putText(display_frame, countdown_text, (10, 110),
                               cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 255), 3)
                else:
                    cv2.putText(display_frame, "Status: PRESS SPACE TO CALIBRATE", (10, 110),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                
                # Draw MediaPipe skeleton
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                results = self.pose_detector.process(rgb)
                if results.pose_landmarks:
                    mp.solutions.drawing_utils.draw_landmarks(
                        display_frame,
                        results.pose_landmarks,
                        mp.solutions.pose.POSE_CONNECTIONS
                    )
                
                cv2.imshow("Skeletal Skinning", display_frame)
            
            # Handle keys
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                self.running = False
            elif key == ord(' ') and not self.is_calibrated and self.calibration_countdown == 0:
                # Start countdown
                print("\n⏱️  Starting 5-second countdown... Get into T-pose!")
                self.calibration_countdown = 5
                self.calibration_start_time = time.time()
            elif key == ord('o'):
                print("\nPlease open: tests/enhanced_mesh_viewer_v2.html in your browser")
        
        # Cleanup
        self.cap.release()
        cv2.destroyAllWindows()
        print("\n✓ System stopped")


def main():
    parser = argparse.ArgumentParser(description="Option B: Skeletal Skinning")
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
    system = IntegratedSkinningSystem(str(mesh_path), headless=args.headless)
    system.setup()
    system.run()


if __name__ == '__main__':
    main()

