#!/usr/bin/env python3
"""
RealSense + Rigged Clothing Animation (PC Version)
===================================================

This is the PC-compatible version using Intel RealSense instead of OAK-D.

Pipeline:
1. Load pre-rigged human mesh (CAUCASIAN MAN.glb) as weight template
2. Load pre-generated clothing mesh (base.glb from Rodin)
3. Scale and align meshes
4. Transfer skin weights from human → clothing
5. Calibrate in T-pose
6. Real-time animation with MediaPipe from RealSense camera
7. Stream to Three.js viewer

Controls:
- SPACE: Start T-pose calibration (5 second countdown)
- H: Toggle human/clothing mesh visualization
- Q: Quit
"""

import sys
from pathlib import Path
import numpy as np
import cv2
import time
import asyncio
import json
import threading
from queue import Queue

# RealSense
import pyrealsense2 as rs

# MediaPipe (replaces OAK-D's on-device BlazePose)
import mediapipe as mp

# Add paths to parent directory modules
parent_path = Path(__file__).parent.parent
sys.path.insert(0, str(parent_path / "tests"))
sys.path.insert(0, str(parent_path / "src" / "modules"))

from rigged_mesh_loader import RiggedMeshLoader, RiggedMesh
from mediapipe_to_bones import MediaPipeToBones, MEDIAPIPE_LANDMARKS, GLB_BONE_NAME_MAPPING
from simple_weight_transfer import transfer_weights_smooth
from mediapipe_lbs import MediaPipeLBS

# WebSocket
import websockets


class RealSenseMediaPipeWrapper:
    """
    Wrapper that provides OAK-D-like interface using RealSense + MediaPipe

    This mimics the BlazeposeDepthai interface so minimal changes are needed
    in the main code.
    """

    def __init__(self):
        # RealSense pipeline
        self.pipeline = rs.pipeline()
        self.config = rs.config()

        # Configure streams (match OAK-D resolution)
        # Using 1280x720 for good balance of quality and performance
        self.config.enable_stream(rs.stream.color, 1280, 720, rs.format.bgr8, 30)
        self.config.enable_stream(rs.stream.depth, 1280, 720, rs.format.z16, 30)

        # Start pipeline
        print("Starting RealSense pipeline...")
        self.profile = self.pipeline.start(self.config)

        # Get device info
        device = self.profile.get_device()
        print(f"✓ Connected to: {device.get_info(rs.camera_info.name)}")

        # Align depth to color
        self.align = rs.align(rs.stream.color)

        # MediaPipe Pose
        print("Initializing MediaPipe Pose...")
        self.mp_pose = mp.solutions.pose
        self.pose = self.mp_pose.Pose(
            static_image_mode=False,
            model_complexity=1,  # 0=lite, 1=full, 2=heavy (use 1 for balance)
            smooth_landmarks=True,
            enable_segmentation=False,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        print("✓ MediaPipe Pose initialized")

        # Get depth scale (converts depth units to meters)
        depth_sensor = self.profile.get_device().first_depth_sensor()
        self.depth_scale = depth_sensor.get_depth_scale()
        print(f"Depth scale: {self.depth_scale} (meters per unit)")

    def next_frame(self):
        """
        Get next frame with pose landmarks (mimics OAK-D interface)

        Returns:
            frame: BGR image (numpy array)
            body: Object with .landmarks_world attribute (33x3 array)
        """
        # Wait for frames
        frames = self.pipeline.wait_for_frames()

        # Align depth to color
        aligned_frames = self.align.process(frames)

        # Get frames
        color_frame = aligned_frames.get_color_frame()
        depth_frame = aligned_frames.get_depth_frame()

        if not color_frame:
            return None, None

        # Convert to numpy
        frame = np.asanyarray(color_frame.get_data())

        # Process with MediaPipe
        # Convert BGR to RGB for MediaPipe
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.pose.process(frame_rgb)

        # Create body object (mimics OAK-D BlazePose output)
        body = None
        if results.pose_world_landmarks:
            body = type('Body', (), {})()  # Create simple object

            # Convert MediaPipe landmarks to numpy array (33 landmarks x 3 coords)
            landmarks_world = []
            for lm in results.pose_world_landmarks.landmark:
                landmarks_world.append([lm.x, lm.y, lm.z])

            body.landmarks_world = np.array(landmarks_world, dtype=np.float32)

            # Also store 2D normalized landmarks for visualization
            if results.pose_landmarks:
                landmarks_2d = []
                for lm in results.pose_landmarks.landmark:
                    landmarks_2d.append([lm.x, lm.y, lm.z])
                body.landmarks = np.array(landmarks_2d, dtype=np.float32)

        return frame, body

    def exit(self):
        """Cleanup resources"""
        self.pose.close()
        self.pipeline.stop()
        print("✓ RealSense pipeline stopped")


class SimpleRenderer:
    """
    Simple renderer for MediaPipe landmarks (mimics BlazeposeRenderer)
    """

    def __init__(self):
        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_pose = mp.solutions.pose

    def draw(self, frame, body):
        """Draw skeleton on frame"""
        if body is None or not hasattr(body, 'landmarks'):
            return frame

        # Convert landmarks back to MediaPipe format for drawing
        from mediapipe.framework.formats import landmark_pb2

        landmark_list = landmark_pb2.NormalizedLandmarkList()
        for lm in body.landmarks:
            landmark = landmark_list.landmark.add()
            landmark.x = lm[0]
            landmark.y = lm[1]
            landmark.z = lm[2]

        # Draw landmarks and connections
        self.mp_drawing.draw_landmarks(
            frame,
            landmark_list,
            self.mp_pose.POSE_CONNECTIONS,
            self.mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=2),
            self.mp_drawing.DrawingSpec(color=(0, 255, 255), thickness=2)
        )

        return frame


class RiggedClothingTest:
    """Main test class (same as OAK-D version, just uses RealSense wrapper)"""

    def __init__(self):
        # Meshes
        self.human_mesh: RiggedMesh = None
        self.clothing_mesh: RiggedMesh = None
        self.clothing_vertices_base = None
        self.human_vertices_base = None

        # Animation
        self.mediapipe_mapper = MediaPipeToBones()
        self.reference_keypoints = None
        self.is_calibrated = False
        self.scale_factor = 1.0

        # LBS system
        self.lbs_human = None
        self.lbs_clothing = None

        # Rendering
        self.ws_clients = set()
        self.ws_server = None

        # RealSense + MediaPipe (replaces OAK-D)
        self.tracker = None
        self.renderer = None

        # State
        self.calibration_active = False
        self.calibration_start_time = None

        # Threading
        self.frame_queue = Queue(maxsize=2)
        self.running = True

        # Visualization toggle
        self.show_human_mesh = False

    async def load_meshes(self):
        """Load rigged human and clothing meshes"""
        print("\n=== Loading Meshes ===")

        # Load rigged human (weight template)
        human_path = "../rigged_mesh/meshRigged_0.glb"
        print(f"Loading human mesh: {human_path}")
        self.human_mesh = RiggedMeshLoader.load(human_path)
        print(f"✓ Human: {len(self.human_mesh.vertices)} verts, {len(self.human_mesh.bones)} bones")

        # Initialize MediaPipe mapper
        print("\n=== Building Bone Name Mapping ===")
        bone_names = [bone.name for bone in self.human_mesh.bones]
        self.mediapipe_mapper = MediaPipeToBones(bone_names)
        print(f"✓ Mapped {len(self.mediapipe_mapper.glb_bone_mapping)} bones")

        # Initialize LBS systems
        print("\n=== Initializing LBS Systems ===")
        bone_mapping = {}
        for mp_name, glb_candidates in GLB_BONE_NAME_MAPPING.items():
            for candidate in glb_candidates:
                if candidate in bone_names:
                    bone_mapping[mp_name] = candidate
                    break

        self.lbs_human = MediaPipeLBS(self.human_mesh, bone_mapping)
        print("✓ Human mesh LBS initialized")

        # Load clothing mesh
        clothing_path = "../pregenerated_mesh_clothing/base.glb"
        print(f"Loading clothing mesh: {clothing_path}")
        self.clothing_mesh = RiggedMeshLoader.load(clothing_path)
        print(f"✓ Clothing: {len(self.clothing_mesh.vertices)} verts")

    def scale_and_align_meshes(self):
        """Scale and align meshes (same as OAK-D version)"""
        print("\n=== Scaling & Aligning Meshes ===")

        TARGET_HEIGHT = 1.7  # meters

        # Orientation correction
        print("Step 1: Applying orientation corrections...")

        human_bbox = self.human_mesh.vertices.max(axis=0) - self.human_mesh.vertices.min(axis=0)
        print(f"  Human bbox: X={human_bbox[0]:.3f}, Y={human_bbox[1]:.3f}, Z={human_bbox[2]:.3f}")

        if human_bbox[2] > human_bbox[1]:
            print("  Human lying down - rotating 90° around X")
            theta_x = np.pi / 2
            rot_x = np.array([
                [1, 0, 0],
                [0, np.cos(theta_x), -np.sin(theta_x)],
                [0, np.sin(theta_x), np.cos(theta_x)]
            ])
            self.human_mesh.vertices = self.human_mesh.vertices @ rot_x.T
            self.human_mesh.vertices[:, 1] *= -1

        clothing_bbox = self.clothing_mesh.vertices.max(axis=0) - self.clothing_mesh.vertices.min(axis=0)
        print(f"  Clothing bbox: X={clothing_bbox[0]:.3f}, Y={clothing_bbox[1]:.3f}, Z={clothing_bbox[2]:.3f}")

        if clothing_bbox[2] > clothing_bbox[1]:
            print("  Clothing lying down - rotating")
            theta_x = np.pi / 2
            rot_x = np.array([
                [1, 0, 0],
                [0, np.cos(theta_x), -np.sin(theta_x)],
                [0, np.sin(theta_x), np.cos(theta_x)]
            ])
            self.clothing_mesh.vertices = self.clothing_mesh.vertices @ rot_x.T
            self.clothing_mesh.vertices[:, 1] *= -1

        print("  ✓ Applied orientation corrections")

        # Center at origin
        print("Step 2: Centering meshes at origin...")
        human_center = self.human_mesh.vertices.mean(axis=0)
        self.human_mesh.vertices -= human_center

        clothing_center = self.clothing_mesh.vertices.mean(axis=0)
        self.clothing_mesh.vertices -= clothing_center
        print("  ✓ Both meshes centered at (0, 0, 0)")

        # Scale to same height
        print("Step 3: Scaling to target height...")
        human_height = self.human_mesh.vertices[:, 1].max() - self.human_mesh.vertices[:, 1].min()
        clothing_height = self.clothing_mesh.vertices[:, 1].max() - self.clothing_mesh.vertices[:, 1].min()

        print(f"  Human height before scale: {human_height:.3f}m")
        print(f"  Clothing height before scale: {clothing_height:.3f}m")

        human_scale = TARGET_HEIGHT / human_height
        clothing_scale = TARGET_HEIGHT / clothing_height

        self.human_mesh.vertices *= human_scale
        self.clothing_mesh.vertices *= clothing_scale

        print(f"  ✓ Human scaled by {human_scale:.1f}x")
        print(f"  ✓ Clothing scaled by {clothing_scale:.3f}x")

        # Align bottoms to Y=0
        print("Step 4: Aligning bottoms...")
        human_bottom = self.human_mesh.vertices[:, 1].min()
        self.human_mesh.vertices[:, 1] -= human_bottom

        clothing_bottom = self.clothing_mesh.vertices[:, 1].min()
        self.clothing_mesh.vertices[:, 1] -= clothing_bottom
        print("  ✓ Both meshes standing on ground (Y=0)")

        # Update bounds
        self.human_mesh.bounds = np.array([
            self.human_mesh.vertices.min(axis=0),
            self.human_mesh.vertices.max(axis=0)
        ])
        self.clothing_mesh.bounds = np.array([
            self.clothing_mesh.vertices.min(axis=0),
            self.clothing_mesh.vertices.max(axis=0)
        ])

        # Store bind pose
        self.human_vertices_base = self.human_mesh.vertices.copy()
        self.clothing_vertices_base = self.clothing_mesh.vertices.copy()

        print(f"\n✓ Alignment complete!")

    def transfer_weights(self):
        """Transfer skin weights from human to clothing (same as OAK-D version)"""
        print("\n=== Transferring Skin Weights ===")

        if self.human_mesh.skin_weights is None:
            print("ERROR: Human mesh has no skin weights!")
            sys.exit(1)

        clothing_weights, clothing_indices = transfer_weights_smooth(
            source_vertices=self.human_mesh.vertices,
            source_weights=self.human_mesh.skin_weights,
            source_indices=self.human_mesh.skin_indices,
            target_vertices=self.clothing_mesh.vertices,
            k_neighbors=5,
            max_distance=0.2
        )

        self.clothing_mesh.skin_weights = clothing_weights
        self.clothing_mesh.skin_indices = clothing_indices
        self.clothing_mesh.bones = self.human_mesh.bones
        self.clothing_mesh.bone_name_to_idx = self.human_mesh.bone_name_to_idx
        self.clothing_mesh.root_bone_idx = self.human_mesh.root_bone_idx

        print("✓ Weights transferred and skeleton copied")

        # Initialize LBS for clothing
        bone_mapping = {}
        bone_names = [bone.name for bone in self.human_mesh.bones]
        for mp_name, glb_candidates in GLB_BONE_NAME_MAPPING.items():
            for candidate in glb_candidates:
                if candidate in bone_names:
                    bone_mapping[mp_name] = candidate
                    break

        self.lbs_clothing = MediaPipeLBS(self.clothing_mesh, bone_mapping)
        print("✓ Clothing mesh LBS initialized")

    def start_calibration(self):
        """Start T-pose calibration countdown"""
        print("\n=== Starting T-Pose Calibration ===")
        print("Stand in T-pose: arms straight out to sides")
        self.calibration_active = True
        self.calibration_start_time = time.time()

    def check_calibration(self, frame, body):
        """Check calibration countdown and capture T-pose"""
        if not self.calibration_active:
            return frame

        if body is None or not hasattr(body, 'landmarks_world') or body.landmarks_world is None:
            h, w = frame.shape[:2]
            cv2.putText(frame, "NO BODY DETECTED", (w//2 - 200, h//2),
                       cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 255), 3)
            return frame

        elapsed = time.time() - self.calibration_start_time
        remaining = 5 - int(elapsed)

        if remaining > 0:
            # Draw countdown
            h, w = frame.shape[:2]
            text = f"T-POSE CALIBRATION: {remaining}"
            font = cv2.FONT_HERSHEY_SIMPLEX
            (tw, th), _ = cv2.getTextSize(text, font, 2, 3)
            x = (w - tw) // 2
            y = (h + th) // 2

            overlay = frame.copy()
            cv2.rectangle(overlay, (0, 0), (w, h), (0, 0, 0), -1)
            frame = cv2.addWeighted(frame, 0.3, overlay, 0.7, 0)

            cv2.putText(frame, text, (x, y), font, 2, (0, 255, 255), 3)

        else:
            # Capture T-pose
            if len(body.landmarks_world) > 0:
                keypoints = {}
                for name, idx in MEDIAPIPE_LANDMARKS.items():
                    if idx < len(body.landmarks_world):
                        lm = body.landmarks_world[idx].astype(np.float32)
                        lm[1] = -lm[1]  # Flip Y-axis
                        keypoints[name] = lm

                # Compute scale factor
                ref_mid_hip = (keypoints['left_hip'] + keypoints['right_hip']) / 2
                ref_mid_shoulder = (keypoints['left_shoulder'] + keypoints['right_shoulder']) / 2
                mediapipe_torso_height = np.linalg.norm(ref_mid_shoulder - ref_mid_hip)

                mesh_hip_y = self.human_vertices_base[:, 1].max() * 0.55
                mesh_shoulder_y = self.human_vertices_base[:, 1].max() * 0.85
                mesh_torso_height = mesh_shoulder_y - mesh_hip_y

                self.scale_factor = mesh_torso_height / (mediapipe_torso_height + 1e-6)

                # Scale keypoints
                for name in keypoints:
                    keypoints[name] *= self.scale_factor

                # Align hip position
                ref_mid_hip = (keypoints['left_hip'] + keypoints['right_hip']) / 2
                hip_y_offset = mesh_hip_y - ref_mid_hip[1]

                for name in keypoints:
                    keypoints[name][1] += hip_y_offset

                # Set reference
                self.mediapipe_mapper.set_reference_pose(keypoints)
                self.reference_keypoints = keypoints

                # Compute inverse bind matrices
                print("\n=== Computing Inverse Bind Matrices ===")
                if self.lbs_human is not None:
                    self.lbs_human.set_bind_pose(self.human_vertices_base, keypoints)
                if self.lbs_clothing is not None:
                    self.lbs_clothing.set_bind_pose(self.clothing_vertices_base, keypoints)

                self.is_calibrated = True

                print("\n✓ T-pose calibrated!")
                print(f"  Scale factor: {self.scale_factor:.3f}")
                print(f"  Hip Y offset: {hip_y_offset:.3f}")
            else:
                print("⚠️  No body detected - calibration failed")

            self.calibration_active = False

        return frame

    def animate_clothing(self, body):
        """Update clothing mesh based on current pose"""
        if not self.is_calibrated:
            return

        if body is None or not hasattr(body, 'landmarks_world') or body.landmarks_world is None:
            return

        if len(body.landmarks_world) == 0:
            return

        # Get current keypoints
        current_keypoints = {}
        for name, idx in MEDIAPIPE_LANDMARKS.items():
            if idx < len(body.landmarks_world):
                lm = body.landmarks_world[idx].astype(np.float32)
                lm[1] = -lm[1]
                lm *= self.scale_factor
                current_keypoints[name] = lm

        # Align hip Y position
        mesh_hip_y = self.human_vertices_base[:, 1].max() * 0.55
        curr_mid_hip = (current_keypoints['left_hip'] + current_keypoints['right_hip']) / 2
        hip_y_offset = mesh_hip_y - curr_mid_hip[1]

        for name in current_keypoints:
            current_keypoints[name][1] += hip_y_offset

        # Apply LBS deformation
        if self.lbs_human is not None and self.lbs_clothing is not None:
            self.human_mesh.vertices = self.lbs_human.deform(
                self.human_vertices_base,
                self.reference_keypoints,
                current_keypoints
            )

            self.clothing_mesh.vertices = self.lbs_clothing.deform(
                self.clothing_vertices_base,
                self.reference_keypoints,
                current_keypoints
            )

    async def websocket_handler(self, websocket):
        """Handle WebSocket client connections"""
        self.ws_clients.add(websocket)
        print(f"✓ WebSocket client connected: {websocket.remote_address}")

        await self.send_mesh_update()

        try:
            async for message in websocket:
                pass
        except websockets.exceptions.ConnectionClosed:
            pass
        finally:
            self.ws_clients.discard(websocket)
            print(f"✗ WebSocket client disconnected")

    async def send_mesh_update(self, send_human=False, body=None):
        """Send current mesh state to all connected clients"""
        if not self.ws_clients:
            return

        mesh_to_send = self.human_mesh if send_human else self.clothing_mesh

        # Generate vertex colors
        y_coords = mesh_to_send.vertices[:, 1]
        y_min, y_max = y_coords.min(), y_coords.max()
        y_norm = (y_coords - y_min) / (y_max - y_min + 1e-6)

        if send_human:
            colors = np.zeros((len(mesh_to_send.vertices), 3))
            colors[:, 0] = 50 + y_norm * 50
            colors[:, 1] = 100 + y_norm * 155
            colors[:, 2] = 50 + y_norm * 50
        else:
            colors = np.zeros((len(mesh_to_send.vertices), 3))
            colors[:, 0] = 50 + y_norm * 100
            colors[:, 1] = 100 + y_norm * 155
            colors[:, 2] = 200 + y_norm * 55

        data = {
            "type": "mesh_update",
            "data": {
                "vertices": mesh_to_send.vertices.tolist(),
                "faces": mesh_to_send.faces.tolist(),
                "colors": colors.tolist(),
            }
        }

        # Add keypoints if available
        if body is not None and hasattr(body, 'landmarks_world') and body.landmarks_world is not None and len(body.landmarks_world) > 0:
            keypoints = []
            kp_dict = {}
            for name, idx in MEDIAPIPE_LANDMARKS.items():
                if idx < len(body.landmarks_world):
                    lm = body.landmarks_world[idx].astype(np.float32)
                    lm[1] = -lm[1]
                    if self.is_calibrated:
                        lm *= self.scale_factor
                    kp_dict[name] = lm

            if 'left_hip' in kp_dict and 'right_hip' in kp_dict:
                mesh_hip_y = self.human_vertices_base[:, 1].max() * 0.55
                curr_mid_hip = (kp_dict['left_hip'] + kp_dict['right_hip']) / 2
                hip_y_offset = mesh_hip_y - curr_mid_hip[1]

                for name in kp_dict:
                    kp_dict[name][1] += hip_y_offset

            for name, idx in MEDIAPIPE_LANDMARKS.items():
                if name in kp_dict:
                    keypoints.append(kp_dict[name].tolist())

            data["keypoints"] = keypoints

        message = json.dumps(data)
        await asyncio.gather(
            *[client.send(message) for client in self.ws_clients],
            return_exceptions=True
        )

    def camera_thread(self):
        """Run camera capture in separate thread"""
        while self.running:
            frame, body = self.tracker.next_frame()
            if frame is not None:
                if self.frame_queue.full():
                    try:
                        self.frame_queue.get_nowait()
                    except:
                        pass
                self.frame_queue.put((frame, body))

    async def run(self):
        """Main async run loop"""
        # Load meshes
        await self.load_meshes()
        self.scale_and_align_meshes()
        self.transfer_weights()

        # Start WebSocket server
        print(f"\n=== Starting WebSocket Server ===")
        self.ws_server = await websockets.serve(
            self.websocket_handler,
            "localhost",
            8765
        )
        print("✓ WebSocket server started on ws://localhost:8765")
        print("Open: ../tests/clothing_viewer.html")

        # Initialize RealSense + MediaPipe
        print("\n=== Initializing RealSense + MediaPipe ===")
        self.tracker = RealSenseMediaPipeWrapper()
        self.renderer = SimpleRenderer()
        print("✓ RealSense + MediaPipe ready")

        # Start camera thread
        camera_thread = threading.Thread(target=self.camera_thread, daemon=True)
        camera_thread.start()
        print("✓ Camera thread started")

        print("\n=== Controls ===")
        print("SPACE: Start T-pose calibration")
        print("H: Toggle human/clothing mesh visualization")
        print("Q: Quit")

        # Main loop
        fps_tracker = []
        last_update_time = time.time()
        update_interval = 1.0 / 30

        while True:
            loop_start = time.time()

            try:
                frame, body = self.frame_queue.get_nowait()
            except:
                await asyncio.sleep(0.001)
                continue

            # Draw skeleton
            frame = self.renderer.draw(frame, body)

            # Handle calibration
            if self.calibration_active:
                frame = self.check_calibration(frame, body)

            # Animate clothing
            if self.is_calibrated and body is not None:
                self.animate_clothing(body)

            # Send mesh update
            current_time = time.time()
            if current_time - last_update_time >= update_interval:
                if self.ws_clients:
                    await self.send_mesh_update(send_human=self.show_human_mesh, body=body)
                last_update_time = current_time

            # FPS calculation
            loop_time = time.time() - loop_start
            fps_tracker.append(loop_time)
            if len(fps_tracker) > 30:
                fps_tracker.pop(0)
            avg_fps = 1.0 / (sum(fps_tracker) / len(fps_tracker)) if fps_tracker else 0

            # Draw status
            status = f"FPS: {avg_fps:.1f} | "
            status += "CALIBRATED" if self.is_calibrated else "Press SPACE to calibrate"
            cv2.putText(frame, status, (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
                        0.7, (0, 255, 0) if self.is_calibrated else (0, 165, 255), 2)

            # Show frame
            cv2.imshow("RealSense Rigged Clothing Test", frame)

            # Handle keys
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord(' ') and not self.calibration_active:
                self.start_calibration()
            elif key == ord('h') or key == ord('H'):
                self.show_human_mesh = not self.show_human_mesh
                mesh_type = "HUMAN" if self.show_human_mesh else "CLOTHING"
                print(f"→ Switched to {mesh_type} mesh")

        # Cleanup
        self.running = False
        cv2.destroyAllWindows()
        self.tracker.exit()
        self.ws_server.close()
        await self.ws_server.wait_closed()


def main():
    """Entry point"""
    test = RiggedClothingTest()
    asyncio.run(test.run())


if __name__ == "__main__":
    main()
