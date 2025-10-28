"""
3D Puppet Keypoints Debugger
==============================

Pure debugging tool to visualize articulated skeleton as 3D rectangles.

Purpose: Diagnose coordinate system and transformation issues by:
1. Calibrating with T-pose (eye distance for Z-scale)
2. Creating bone-aligned rectangles for each body part
3. Rendering in real-time with Three.js (wireframe only)
4. NO mesh involved - just skeleton + rectangles

This will reveal if the problem is in:
- Coordinate system conversion
- Rectangle orientation/placement
- Hierarchical transformations

Usage:
    python tests/3d_puppet_keypoints.py

Controls:
    SPACE - Start 5-second T-pose calibration
    Q - Quit
    C - Toggle coordinate axes display

Author: AI Assistant
Date: October 28, 2025
"""

import cv2
import numpy as np
import mediapipe as mp
import time
import threading
import asyncio
import json
from pathlib import Path
from typing import Dict, Tuple, List

# Import WebSocket server
from enhanced_websocket_server_v2 import EnhancedMeshStreamServerV2


class PuppetDebugger:
    """
    Minimal debugger that shows skeleton as articulated rectangles.
    
    No mesh, no cage, no MVC - just pure coordinate system validation.
    """
    
    # Body part definitions (matching BodyPix structure)
    BODY_PARTS = {
        'head': {'parent': None, 'keypoints': ['nose', 'left_ear', 'right_ear']},
        'torso': {'parent': None, 'keypoints': ['left_shoulder', 'right_shoulder', 'left_hip', 'right_hip']},
        'left_upper_arm': {'parent': 'torso', 'keypoints': ['left_shoulder', 'left_elbow']},
        'right_upper_arm': {'parent': 'torso', 'keypoints': ['right_shoulder', 'right_elbow']},
        'left_lower_arm': {'parent': 'left_upper_arm', 'keypoints': ['left_elbow', 'left_wrist']},
        'right_lower_arm': {'parent': 'right_upper_arm', 'keypoints': ['right_elbow', 'right_wrist']},
        'left_upper_leg': {'parent': 'torso', 'keypoints': ['left_hip', 'left_knee']},
        'right_upper_leg': {'parent': 'torso', 'keypoints': ['right_hip', 'right_knee']},
        'left_lower_leg': {'parent': 'left_upper_leg', 'keypoints': ['left_knee', 'left_ankle']},
        'right_lower_leg': {'parent': 'right_upper_leg', 'keypoints': ['right_knee', 'right_ankle']},
    }
    
    # MediaPipe landmark indices
    LANDMARK_MAP = {
        'nose': 0,
        'left_eye': 2,
        'right_eye': 5,
        'left_ear': 7,
        'right_ear': 8,
        'left_shoulder': 11,
        'right_shoulder': 12,
        'left_elbow': 13,
        'right_elbow': 14,
        'left_wrist': 15,
        'right_wrist': 16,
        'left_hip': 23,
        'right_hip': 24,
        'left_knee': 25,
        'right_knee': 26,
        'left_ankle': 27,
        'right_ankle': 28,
    }
    
    def __init__(self):
        """Initialize debugger."""
        print(f"\n{'='*70}")
        print("3D PUPPET KEYPOINTS DEBUGGER")
        print(f"{'='*70}")
        print("\nPurpose: Isolate and debug coordinate system issues")
        print("Shows: Real-time skeleton as articulated rectangles")
        print("No mesh, no cage, no MVC - just pure transforms")
        print(f"{'='*70}\n")
        
        # MediaPipe
        self.mp_pose = mp.solutions.pose
        self.pose = self.mp_pose.Pose(
            static_image_mode=False,
            model_complexity=1,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        self.mp_drawing = mp.solutions.drawing_utils
        
        # Camera
        self.cap = cv2.VideoCapture(0)
        
        # Calibration
        self.calibrated = False
        self.reference_keypoints = {}
        self.eye_distance_pixels = None  # For Z-scale calibration
        self.head_size = 0.15  # Default head size in meters (15cm ear-to-ear)
        
        # WebSocket
        self.ws_server = None
        self.ws_loop = None
        self.ws_thread = None
        
        # State
        self.show_axes = True
    
    def start_websocket_server(self):
        """Start WebSocket server for Three.js visualization."""
        def run_server():
            self.ws_loop = asyncio.new_event_loop()
            asyncio.set_event_loop(self.ws_loop)
            self.ws_server = EnhancedMeshStreamServerV2(host='localhost', port=8766)
            self.ws_loop.run_until_complete(self.ws_server.start())
            self.ws_loop.run_forever()
        
        self.ws_thread = threading.Thread(target=run_server, daemon=True)
        self.ws_thread.start()
        time.sleep(1)
        print("✓ WebSocket server started on ws://localhost:8766")
    
    def extract_keypoints_3d(self, landmarks) -> Dict[str, np.ndarray]:
        """
        Extract 3D keypoints using calibrated Z-scale.
        
        Returns:
            Dict mapping keypoint names to (x, y, z) in meters
        """
        if not landmarks:
            return {}
        
        keypoints = {}
        
        for name, idx in self.LANDMARK_MAP.items():
            if idx < len(landmarks.landmark):
                lm = landmarks.landmark[idx]
                
                # Raw MediaPipe coordinates (normalized)
                x_norm = lm.x * 2 - 1  # [-1, 1]
                y_norm = 1 - lm.y * 2  # [-1, 1], flipped
                z_norm = lm.z  # MediaPipe Z (depth)
                
                # Scale by head size (calibrated from eye distance)
                if self.calibrated and self.eye_distance_pixels:
                    # X and Y scaled by head size
                    x = x_norm * self.head_size * 5  # Roughly 5x head width for full body
                    y = y_norm * self.head_size * 8  # Roughly 8x head height for full body
                    
                    # Z scaled by calibrated eye distance
                    # Convert MediaPipe's arbitrary Z to real-world meters
                    z = z_norm * self.head_size * 0.5  # Half head size for depth variation
                else:
                    # Pre-calibration: just use normalized coords
                    x = x_norm
                    y = y_norm
                    z = z_norm * 0.1
                
                keypoints[name] = np.array([x, y, z])
        
        return keypoints
    
    def calibrate_t_pose(self, landmarks):
        """
        Calibrate using T-pose.
        
        Uses eye distance as reference for Z-scale.
        """
        print(f"\n{'='*70}")
        print("T-POSE CALIBRATION COUNTDOWN")
        print(f"{'='*70}")
        print("Get into T-pose position (arms extended horizontally)...")
        print("Make sure your face is visible (need eye distance)...")
        
        for i in range(5, 0, -1):
            print(f"  {i}...")
            time.sleep(1)
        
        print("✓ Capturing calibration!\n")
        
        # Extract keypoints
        keypoints = self.extract_keypoints_3d(landmarks)
        
        # Calculate eye distance in pixels (for Z-scale)
        if 'left_eye' in keypoints and 'right_eye' in keypoints:
            left_eye = keypoints['left_eye']
            right_eye = keypoints['right_eye']
            self.eye_distance_pixels = np.linalg.norm(left_eye - right_eye)
            print(f"✓ Eye distance captured: {self.eye_distance_pixels:.3f}")
        else:
            print("⚠ Could not detect eyes, using default scale")
            self.eye_distance_pixels = 0.06  # ~6cm default
        
        # Store reference keypoints (T-pose)
        self.reference_keypoints = keypoints.copy()
        self.calibrated = True
        
        print(f"✓ Calibration complete!")
        print(f"  Reference keypoints: {len(self.reference_keypoints)}")
        print(f"  Z-scale factor: {self.eye_distance_pixels:.3f}")
        print(f"{'='*70}\n")
    
    def generate_bounding_box_for_part(
        self,
        part_name: str,
        keypoints: Dict[str, np.ndarray]
    ) -> Tuple[np.ndarray, np.ndarray, Dict]:
        """
        Generate 3D rectangular prism (bounding box) for a body part.
        
        This creates a proper 3D cage segment that:
        1. Encloses the body part in 3D space (8 vertices, 12 edges)
        2. Aligns with the bone/keypoint axis
        3. Uses proportional depth (Z) - later replaced with actual mesh depth
        4. Stores all geometric data for future mesh vertex assignment
        
        PROOF OF CONCEPT: This verifies the geometry of 3D cage deformation
        before integrating with TripoSR mesh + BodyPix alignment.
        
        Args:
            part_name: Name of body part (e.g., 'left_upper_arm')
            keypoints: Current 3D keypoint positions
            
        Returns:
            (vertices, edges, metadata) where:
            - vertices: (8, 3) corners of rectangular prism
                       [near-top-left, near-top-right, near-bottom-right, near-bottom-left,
                        far-top-left, far-top-right, far-bottom-right, far-bottom-left]
            - edges: (12, 2) indices for edges (wireframe)
            - metadata: Complete geometric description for mesh assignment
        """
        part_info = self.BODY_PARTS[part_name]
        part_keypoints = part_info['keypoints']
        
        # Get keypoint positions
        positions = []
        for kp_name in part_keypoints:
            if kp_name in keypoints:
                positions.append(keypoints[kp_name])
        
        if len(positions) < 2:
            return None, None, None
        
        positions = np.array(positions)
        
        # ==========================================
        # STEP 1: Compute bone axis (length)
        # ==========================================
        if len(positions) == 2:
            # Limb (2 keypoints): bone is between them
            start, end = positions[0], positions[1]
            bone_vector = end - start
            bone_length = np.linalg.norm(bone_vector)
            bone_direction = bone_vector / (bone_length + 1e-8)
        else:
            # Torso/head (multiple keypoints): use centroid + extent
            centroid = positions.mean(axis=0)
            
            # For torso: explicitly use shoulders → hips
            if part_name == 'torso':
                shoulder_center = (keypoints.get('left_shoulder', centroid) + 
                                 keypoints.get('right_shoulder', centroid)) / 2
                hip_center = (keypoints.get('left_hip', centroid) + 
                            keypoints.get('right_hip', centroid)) / 2
                start = shoulder_center
                end = hip_center
            else:
                # Head or other: use vertical extent
                min_y = positions[:, 1].min()
                max_y = positions[:, 1].max()
                start = centroid.copy()
                start[1] = max_y
                end = centroid.copy()
                end[1] = min_y
            
            bone_vector = end - start
            bone_length = np.linalg.norm(bone_vector)
            bone_direction = bone_vector / (bone_length + 1e-8)
        
        # ==========================================
        # STEP 2: Compute width (perpendicular to bone, in horizontal plane)
        # ==========================================
        # Use head size as base unit
        width = self.head_size * 1.0
        
        # Part-specific width adjustments
        if part_name == 'torso' and 'left_shoulder' in keypoints and 'right_shoulder' in keypoints:
            shoulder_width = np.linalg.norm(keypoints['left_shoulder'] - keypoints['right_shoulder'])
            width = shoulder_width * 1.15  # 15% padding
        elif part_name == 'head':
            width = self.head_size * 0.8
        elif 'upper' in part_name:
            width = self.head_size * 0.6  # Upper arm/leg
        elif 'lower' in part_name:
            width = self.head_size * 0.5  # Lower arm/leg
        
        # ==========================================
        # STEP 3: Compute depth (Z-axis, perpendicular to both bone and width)
        # ==========================================
        # For now: proportional constant based on width
        # FUTURE: Replace with actual mesh depth from TripoSR + BodyPix
        depth = width * 0.8  # Slightly flatter than width
        
        # ==========================================
        # STEP 4: Use MediaPipe coordinate system directly (NO ROTATION)
        # ==========================================
        # MediaPipe gives us X, Y, Z directly - just use them!
        # X = horizontal (left-right)
        # Y = vertical (up-down)  
        # Z = depth (camera direction)
        
        x_axis = np.array([1, 0, 0])  # Width
        y_axis = np.array([0, 1, 0])  # Height (bone length)
        z_axis = np.array([0, 0, 1])  # Depth
        
        # ==========================================
        # STEP 5: Generate 8 corners - SIMPLE axis-aligned box
        # ==========================================
        # Center of box = midpoint between start and end
        center = (start + end) / 2
        
        # Half-extents
        half_width = width / 2
        half_height = bone_length / 2  # Bone length is the height (Y-axis)
        half_depth = depth / 2
        
        # Generate 8 vertices in standard Three.js BoxGeometry order
        # Using MediaPipe's coordinate system directly:
        # X = horizontal (left-right)
        # Y = vertical (up-down)
        # Z = depth (into/out of screen)
        
        vertices = np.array([
            # Back face (Z-)
            [center[0] - half_width, center[1] - half_height, center[2] - half_depth],  # 0: left-bottom-back
            [center[0] + half_width, center[1] - half_height, center[2] - half_depth],  # 1: right-bottom-back
            [center[0] + half_width, center[1] + half_height, center[2] - half_depth],  # 2: right-top-back
            [center[0] - half_width, center[1] + half_height, center[2] - half_depth],  # 3: left-top-back
            # Front face (Z+)
            [center[0] - half_width, center[1] - half_height, center[2] + half_depth],  # 4: left-bottom-front
            [center[0] + half_width, center[1] - half_height, center[2] + half_depth],  # 5: right-bottom-front
            [center[0] + half_width, center[1] + half_height, center[2] + half_depth],  # 6: right-top-front
            [center[0] - half_width, center[1] + half_height, center[2] + half_depth],  # 7: left-top-front
        ])
        
        # ==========================================
        # STEP 6: Define edges (12 edges of rectangular prism)
        # ==========================================
        # Edges matching Three.js BoxGeometry
        edges = np.array([
            # Back face (Z-)
            [0, 1], [1, 2], [2, 3], [3, 0],
            # Front face (Z+)
            [4, 5], [5, 6], [6, 7], [7, 4],
            # Connecting edges (front to back)
            [0, 4], [1, 5], [2, 6], [3, 7],
        ])
        
        # ==========================================
        # STEP 7: Store complete metadata for future mesh assignment
        # ==========================================
        metadata = {
            'bone_start': start.tolist(),
            'bone_end': end.tolist(),
            'bone_length': float(bone_length),
            
            'width': float(width),
            'height': float(bone_length),
            'depth': float(depth),
            
            'center': center.tolist(),
            
            # Simple axis-aligned bounding box
            'min': [center[0] - half_width, center[1] - half_height, center[2] - half_depth],
            'max': [center[0] + half_width, center[1] + half_height, center[2] + half_depth],
            
            # Half-extents for AABB checks
            'half_extents': [half_width, half_height, half_depth],
        }
        
        return vertices, edges, metadata
    
    def generate_puppet_data(self, keypoints: Dict[str, np.ndarray]) -> Dict:
        """
        Generate all rectangles for puppet visualization.
        
        Returns:
            Dict with 'parts' list, each containing vertices and edges
        """
        puppet_data = {
            'parts': [],
            'keypoints': [],
            'show_axes': self.show_axes
        }
        
        # Add keypoints as small spheres
        for name, pos in keypoints.items():
            puppet_data['keypoints'].append({
                'name': name,
                'position': pos.tolist()
            })
        
        # Generate 3D bounding boxes for each body part
        for part_name in self.BODY_PARTS.keys():
            vertices, edges, metadata = self.generate_bounding_box_for_part(part_name, keypoints)
            
            if vertices is not None:
                puppet_data['parts'].append({
                    'name': part_name,
                    'vertices': vertices.tolist(),
                    'edges': edges.tolist(),
                    'metadata': metadata  # Complete geometric description for mesh assignment
                })
        
        return puppet_data
    
    def send_puppet_data(self, puppet_data: Dict):
        """Send puppet data to Three.js via WebSocket."""
        if not self.ws_server:
            return
        
        # Send via WebSocket using broadcast method
        asyncio.run_coroutine_threadsafe(
            self.ws_server.broadcast('puppet_update', puppet_data),
            self.ws_loop
        )
    
    def run(self):
        """Main loop."""
        print(f"\n{'='*70}")
        print("STARTING PUPPET DEBUGGER")
        print(f"{'='*70}")
        print("\nControls:")
        print("  SPACE - Start T-pose calibration (5 seconds)")
        print("  C - Toggle coordinate axes")
        print("  Q - Quit")
        print("\nImportant:")
        print("  - Stand 1-2 meters from camera")
        print("  - Make sure your face is visible (for eye distance)")
        print("  - Full body should be in frame")
        print(f"{'='*70}\n")
        
        # Start WebSocket
        self.start_websocket_server()
        
        print("\n⚠ IMPORTANT: Open the puppet visualizer HTML:")
        print("   open tests/puppet_visualizer.html")
        print("   (or manually open in browser)\n")
        
        frame_count = 0
        fps_start = time.time()
        
        try:
            while True:
                ret, frame = self.cap.read()
                if not ret:
                    break
                
                # Process with MediaPipe
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                results = self.pose.process(frame_rgb)
                
                # Draw skeleton
                if results.pose_landmarks:
                    self.mp_drawing.draw_landmarks(
                        frame,
                        results.pose_landmarks,
                        self.mp_pose.POSE_CONNECTIONS
                    )
                    
                    # If calibrated, generate and send puppet data
                    if self.calibrated:
                        keypoints = self.extract_keypoints_3d(results.pose_landmarks)
                        puppet_data = self.generate_puppet_data(keypoints)
                        self.send_puppet_data(puppet_data)
                
                # Show instructions
                if not self.calibrated:
                    cv2.putText(frame, "Press SPACE for T-pose calibration (5s)",
                                (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                else:
                    cv2.putText(frame, "Calibrated - Move around!",
                                (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                
                # FPS
                frame_count += 1
                if frame_count % 30 == 0:
                    fps = 30 / (time.time() - fps_start)
                    fps_start = time.time()
                    print(f"FPS: {fps:.1f}")
                
                # Display
                cv2.imshow('Puppet Debugger - Camera Feed', frame)
                
                # Keys
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    break
                elif key == ord(' ') and results.pose_landmarks:
                    self.calibrate_t_pose(results.pose_landmarks)
                elif key == ord('c'):
                    self.show_axes = not self.show_axes
                    print(f"✓ Axes display: {'ON' if self.show_axes else 'OFF'}")
        
        finally:
            self.cap.release()
            cv2.destroyAllWindows()
            if self.ws_loop:
                self.ws_loop.call_soon_threadsafe(self.ws_loop.stop)
            print("\n✓ Debugger shutdown complete")


def main():
    debugger = PuppetDebugger()
    debugger.run()


if __name__ == "__main__":
    main()

