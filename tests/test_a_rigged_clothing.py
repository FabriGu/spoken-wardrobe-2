"""
Test A: Rigged Clothing Animation with Pre-Generated Mesh

This test demonstrates the complete pipeline:
1. Load pre-rigged human mesh (CAUCASIAN MAN.glb) as weight template
2. Load pre-generated clothing mesh (base.glb from Rodin)
3. Scale and align meshes based on bounding boxes
4. Transfer skin weights from human → clothing
5. Calibrate in T-pose
6. Real-time animation with MediaPipe from OAK-D
7. Stream to Three.js viewer

Controls:
- SPACE: Start T-pose calibration (5 second countdown)
- 'q': Quit
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

# Add paths
blazepose_path = Path(__file__).parent.parent / "external" / "depthai_blazepose"
sys.path.insert(0, str(blazepose_path))

src_path = Path(__file__).parent.parent / "src" / "modules"
sys.path.insert(0, str(src_path))

from BlazeposeDepthaiEdge import BlazeposeDepthai
from BlazeposeRenderer import BlazeposeRenderer

# Import our new modules
from rigged_mesh_loader import RiggedMeshLoader, RiggedMesh
from mediapipe_to_bones import MediaPipeToBones, MEDIAPIPE_LANDMARKS, GLB_BONE_NAME_MAPPING
from simple_weight_transfer import transfer_weights_smooth
from mediapipe_lbs import MediaPipeLBS

# WebSocket
import websockets


class RiggedClothingTest:
    """Main test class"""

    def __init__(self):
        # Meshes
        self.human_mesh: RiggedMesh = None
        self.clothing_mesh: RiggedMesh = None  # Will get rigging transferred
        self.clothing_vertices_base = None  # Base pose vertices (AFTER alignment)
        self.human_vertices_base = None  # Base pose vertices (AFTER alignment)

        # Animation
        self.mediapipe_mapper = MediaPipeToBones()
        self.reference_keypoints = None  # T-pose calibration
        self.is_calibrated = False
        self.scale_factor = 1.0  # MediaPipe → mesh scale factor (computed during calibration)

        # LBS system (initialized after meshes loaded)
        self.lbs_human = None
        self.lbs_clothing = None

        # Rendering
        self.ws_clients = set()
        self.ws_server = None

        # OAK-D
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
        human_path = "rigged_mesh/CAUCASIAN MAN.glb"
        print(f"Loading human mesh: {human_path}")
        self.human_mesh = RiggedMeshLoader.load(human_path)
        print(f"✓ Human: {len(self.human_mesh.vertices)} verts, {len(self.human_mesh.bones)} bones")

        # Initialize MediaPipe mapper with actual GLB bone names
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

        # Load clothing mesh (no rigging yet)
        clothing_path = "pregenerated_mesh_clothing/base.glb"
        print(f"Loading clothing mesh: {clothing_path}")
        self.clothing_mesh = RiggedMeshLoader.load(clothing_path)
        print(f"✓ Clothing: {len(self.clothing_mesh.vertices)} verts")

        # DON'T store base pose yet - need to align first

    def scale_and_align_meshes(self):
        """
        Hybrid Option 1+4: Proper alignment with orientation correction

        Steps:
        1. Apply known orientation fixes (Option 4)
        2. Center both meshes at origin
        3. Scale to same height
        4. Store as bind pose
        """
        print("\n=== Scaling & Aligning Meshes ===")

        TARGET_HEIGHT = 1.7  # meters - standard human height

        # ===== STEP 1: Orientation Correction (Option 4) =====
        print("Step 1: Applying orientation corrections...")

        # Human mesh: Analyze its orientation first
        human_bbox = self.human_mesh.vertices.max(axis=0) - self.human_mesh.vertices.min(axis=0)
        print(f"  Human bbox: X={human_bbox[0]:.3f}, Y={human_bbox[1]:.3f}, Z={human_bbox[2]:.3f}")

        # If height is along Z (Z > Y), rotate to make it along Y
        if human_bbox[2] > human_bbox[1]:
            print("  Human lying down (height along Z) - rotating 90° around X")
            theta_x = np.pi / 2  # +90 degrees brings Z → -Y (hanging upside down)
            # Then flip Y to correct
            rot_x = np.array([
                [1, 0, 0],
                [0, np.cos(theta_x), -np.sin(theta_x)],
                [0, np.sin(theta_x), np.cos(theta_x)]
            ])
            self.human_mesh.vertices = self.human_mesh.vertices @ rot_x.T
            # Flip Y
            self.human_mesh.vertices[:, 1] *= -1

        # Clothing mesh: Check orientation
        clothing_bbox = self.clothing_mesh.vertices.max(axis=0) - self.clothing_mesh.vertices.min(axis=0)
        print(f"  Clothing bbox: X={clothing_bbox[0]:.3f}, Y={clothing_bbox[1]:.3f}, Z={clothing_bbox[2]:.3f}")

        # If clothing is lying down or upside down, fix it
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

        # ===== STEP 2: Center at Origin (Option 1) =====
        print("Step 2: Centering meshes at origin...")

        human_center = self.human_mesh.vertices.mean(axis=0)
        self.human_mesh.vertices -= human_center

        clothing_center = self.clothing_mesh.vertices.mean(axis=0)
        self.clothing_mesh.vertices -= clothing_center

        print("  ✓ Both meshes centered at (0, 0, 0)")

        # ===== STEP 3: Scale to Same Height (Option 1) =====
        print("Step 3: Scaling to target height...")

        # Get heights (Y-axis range)
        human_min_y = self.human_mesh.vertices[:, 1].min()
        human_max_y = self.human_mesh.vertices[:, 1].max()
        human_height = human_max_y - human_min_y

        clothing_min_y = self.clothing_mesh.vertices[:, 1].min()
        clothing_max_y = self.clothing_mesh.vertices[:, 1].max()
        clothing_height = clothing_max_y - clothing_min_y

        print(f"  Human height before scale: {human_height:.3f}m")
        print(f"  Clothing height before scale: {clothing_height:.3f}m")

        # Scale both to target height
        human_scale = TARGET_HEIGHT / human_height
        clothing_scale = TARGET_HEIGHT / clothing_height

        self.human_mesh.vertices *= human_scale
        self.clothing_mesh.vertices *= clothing_scale

        print(f"  ✓ Human scaled by {human_scale:.1f}x")
        print(f"  ✓ Clothing scaled by {clothing_scale:.3f}x")

        # ===== STEP 4: Align Bottoms to Y=0 =====
        print("Step 4: Aligning bottoms...")

        # Move so bottom of meshes is at Y=0
        human_bottom = self.human_mesh.vertices[:, 1].min()
        self.human_mesh.vertices[:, 1] -= human_bottom

        clothing_bottom = self.clothing_mesh.vertices[:, 1].min()
        self.clothing_mesh.vertices[:, 1] -= clothing_bottom

        print("  ✓ Both meshes standing on ground (Y=0)")

        # ===== STEP 5: Update Bounds =====
        self.human_mesh.bounds = np.array([
            self.human_mesh.vertices.min(axis=0),
            self.human_mesh.vertices.max(axis=0)
        ])
        self.clothing_mesh.bounds = np.array([
            self.clothing_mesh.vertices.min(axis=0),
            self.clothing_mesh.vertices.max(axis=0)
        ])

        # ===== STEP 6: Store Bind Pose =====
        self.human_vertices_base = self.human_mesh.vertices.copy()
        self.clothing_vertices_base = self.clothing_mesh.vertices.copy()

        print(f"\n✓ Alignment complete!")
        print(f"  Human: {self.human_mesh.bounds[0]} to {self.human_mesh.bounds[1]}")
        print(f"  Clothing: {self.clothing_mesh.bounds[0]} to {self.clothing_mesh.bounds[1]}")

    def transfer_weights(self):
        """
        Transfer skin weights from human mesh to clothing mesh
        """
        print("\n=== Transferring Skin Weights ===")

        if self.human_mesh.skin_weights is None:
            print("ERROR: Human mesh has no skin weights!")
            sys.exit(1)

        # Use smooth K-NN transfer
        # After scaling, meshes are ~1.7m tall, so 20cm search radius is reasonable
        clothing_weights, clothing_indices = transfer_weights_smooth(
            source_vertices=self.human_mesh.vertices,
            source_weights=self.human_mesh.skin_weights,
            source_indices=self.human_mesh.skin_indices,
            target_vertices=self.clothing_mesh.vertices,
            k_neighbors=5,
            max_distance=0.2  # 20cm search radius
        )

        # Store weights in clothing mesh
        self.clothing_mesh.skin_weights = clothing_weights
        self.clothing_mesh.skin_indices = clothing_indices

        # Copy skeleton from human
        self.clothing_mesh.bones = self.human_mesh.bones
        self.clothing_mesh.bone_name_to_idx = self.human_mesh.bone_name_to_idx
        self.clothing_mesh.root_bone_idx = self.human_mesh.root_bone_idx

        print("✓ Weights transferred and skeleton copied")

        # Initialize LBS for clothing (uses same bone mapping as human)
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

        # Check if body is detected
        if body is None or not hasattr(body, 'landmarks_world') or body.landmarks_world is None:
            # No body detected - show warning
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

            # Background
            cv2.rectangle(frame, (0, 0), (w, h), (0, 0, 0), -1)
            frame = cv2.addWeighted(frame, 0.3, frame, 0.7, 0)

            cv2.putText(frame, text, (x, y), font, 2, (0, 255, 255), 3)

        else:
            # Capture T-pose
            if len(body.landmarks_world) > 0:
                # Convert landmarks to dict
                keypoints = {}
                for name, idx in MEDIAPIPE_LANDMARKS.items():
                    if idx < len(body.landmarks_world):
                        # landmarks_world is (33, 3) array
                        lm = body.landmarks_world[idx].astype(np.float32)

                        # FIX 1: Flip Y-axis
                        lm[1] = -lm[1]

                        keypoints[name] = lm

                # FIX 2: Compute scale factor from MediaPipe to mesh coordinates
                # MediaPipe torso height (after flip)
                ref_mid_hip = (keypoints['left_hip'] + keypoints['right_hip']) / 2
                ref_mid_shoulder = (keypoints['left_shoulder'] + keypoints['right_shoulder']) / 2
                mediapipe_torso_height = np.linalg.norm(ref_mid_shoulder - ref_mid_hip)

                # Mesh torso height
                mesh_hip_y = self.human_vertices_base[:, 1].max() * 0.55
                mesh_shoulder_y = self.human_vertices_base[:, 1].max() * 0.85
                mesh_torso_height = mesh_shoulder_y - mesh_hip_y

                # Scale factor to go from MediaPipe → mesh
                self.scale_factor = mesh_torso_height / (mediapipe_torso_height + 1e-6)

                # Scale ALL keypoint coordinates
                for name in keypoints:
                    keypoints[name] *= self.scale_factor

                # FIX 3: Align hip position after scaling
                ref_mid_hip = (keypoints['left_hip'] + keypoints['right_hip']) / 2
                hip_y_offset = mesh_hip_y - ref_mid_hip[1]

                # Apply Y offset only (X and Z should be centered already)
                for name in keypoints:
                    keypoints[name][1] += hip_y_offset

                # Set as reference
                self.mediapipe_mapper.set_reference_pose(keypoints)
                self.reference_keypoints = keypoints

                # OPTION C: Compute and store inverse bind matrices
                print("\n=== Computing Inverse Bind Matrices (Option C) ===")
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
        """Update clothing mesh based on current pose using Linear Blend Skinning"""
        # Check if calibrated and body is detected
        if not self.is_calibrated:
            return

        if body is None or not hasattr(body, 'landmarks_world') or body.landmarks_world is None:
            return

        if len(body.landmarks_world) == 0:
            return

        # Get current keypoints and transform to mesh coordinate system
        current_keypoints = {}
        for name, idx in MEDIAPIPE_LANDMARKS.items():
            if idx < len(body.landmarks_world):
                # landmarks_world is (33, 3) array
                lm = body.landmarks_world[idx].astype(np.float32)

                # FIX 1: Flip Y-axis (MediaPipe has Y pointing down, we need Y up)
                lm[1] = -lm[1]

                # FIX 2: Scale to mesh coordinates (using factor from calibration)
                lm *= self.scale_factor

                current_keypoints[name] = lm

        # FIX 3: Align hip Y position (same offset as calibration)
        mesh_hip_y = self.human_vertices_base[:, 1].max() * 0.55
        curr_mid_hip = (current_keypoints['left_hip'] + current_keypoints['right_hip']) / 2
        hip_y_offset = mesh_hip_y - curr_mid_hip[1]

        # Apply Y offset to ALL keypoints
        for name in current_keypoints:
            current_keypoints[name][1] += hip_y_offset

        # Apply LBS deformation using gpytoolbox
        if self.lbs_human is not None and self.lbs_clothing is not None:
            # Deform human mesh
            self.human_mesh.vertices = self.lbs_human.deform(
                self.human_vertices_base,
                self.reference_keypoints,
                current_keypoints
            )

            # Deform clothing mesh
            self.clothing_mesh.vertices = self.lbs_clothing.deform(
                self.clothing_vertices_base,
                self.reference_keypoints,
                current_keypoints
            )
        else:
            # Fallback: simple translation if LBS not initialized
            ref_mid_hip = (self.reference_keypoints['left_hip'] + self.reference_keypoints['right_hip']) / 2
            curr_mid_hip = (current_keypoints['left_hip'] + current_keypoints['right_hip']) / 2
            delta_position = curr_mid_hip - ref_mid_hip

            self.clothing_mesh.vertices = self.clothing_vertices_base.copy() + delta_position
            self.human_mesh.vertices = self.human_vertices_base.copy() + delta_position

    async def websocket_handler(self, websocket):
        """Handle WebSocket client connections"""
        self.ws_clients.add(websocket)
        print(f"✓ WebSocket client connected: {websocket.remote_address}")

        # Send initial mesh immediately
        await self.send_mesh_update()

        try:
            async for message in websocket:
                pass  # No incoming messages expected
        except websockets.exceptions.ConnectionClosed:
            pass
        finally:
            self.ws_clients.discard(websocket)
            print(f"✗ WebSocket client disconnected")

    async def send_mesh_update(self, send_human=False, body=None):
        """Send current mesh state to all connected clients, plus keypoints and bones"""
        if not self.ws_clients:
            return

        # Choose which mesh to send
        mesh_to_send = self.human_mesh if send_human else self.clothing_mesh

        # Generate vertex colors
        y_coords = mesh_to_send.vertices[:, 1]
        y_min, y_max = y_coords.min(), y_coords.max()
        y_norm = (y_coords - y_min) / (y_max - y_min + 1e-6)

        if send_human:
            # Green gradient for human
            colors = np.zeros((len(mesh_to_send.vertices), 3))
            colors[:, 0] = 50 + y_norm * 50   # R
            colors[:, 1] = 100 + y_norm * 155  # G
            colors[:, 2] = 50 + y_norm * 50    # B
        else:
            # Blue to cyan gradient for clothing
            colors = np.zeros((len(mesh_to_send.vertices), 3))
            colors[:, 0] = 50 + y_norm * 100  # R
            colors[:, 1] = 100 + y_norm * 155  # G
            colors[:, 2] = 200 + y_norm * 55   # B

        # Prepare data (match viewer's expected format)
        data = {
            "type": "mesh_update",
            "data": {
                "vertices": mesh_to_send.vertices.tolist(),
                "faces": mesh_to_send.faces.tolist(),
                "colors": colors.tolist(),
            }
        }

        # Add MediaPipe keypoints if available (transform same as animate_clothing)
        if body is not None and hasattr(body, 'landmarks_world') and body.landmarks_world is not None and len(body.landmarks_world) > 0:
            keypoints = []

            # Transform keypoints to mesh coordinate system (same as animate_clothing)
            kp_dict = {}
            for name, idx in MEDIAPIPE_LANDMARKS.items():
                if idx < len(body.landmarks_world):
                    lm = body.landmarks_world[idx].astype(np.float32)

                    # FIX 1: Flip Y-axis
                    lm[1] = -lm[1]

                    # FIX 2: Scale to mesh coordinates (if calibrated)
                    if self.is_calibrated:
                        lm *= self.scale_factor

                    kp_dict[name] = lm

            # FIX 3: Align hip Y position
            if 'left_hip' in kp_dict and 'right_hip' in kp_dict:
                mesh_hip_y = self.human_vertices_base[:, 1].max() * 0.55
                curr_mid_hip = (kp_dict['left_hip'] + kp_dict['right_hip']) / 2
                hip_y_offset = mesh_hip_y - curr_mid_hip[1]

                # Apply Y offset
                for name in kp_dict:
                    kp_dict[name][1] += hip_y_offset

            # Convert to list
            for name, idx in MEDIAPIPE_LANDMARKS.items():
                if name in kp_dict:
                    keypoints.append(kp_dict[name].tolist())

            data["keypoints"] = keypoints

        # Send to all clients
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
                # Put in queue, drop old frames if full
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
        print("Open: tests/clothing_viewer.html")

        # Initialize OAK-D
        print("\n=== Initializing OAK-D ===")
        # self.tracker = BlazeposeDepthai(
        #     input_src="rgb",
        #     pd_model="external/depthai_blazepose/models/pose_detection_sh4.blob",
        #     lm_model="external/depthai_blazepose/models/pose_landmark_full_sh4.blob",
        # )

        self.tracker = BlazeposeDepthai(
        input_src='rgb',
        lm_model='lite',
        xyz=True,
        smoothing=True,
        internal_fps=30,
        internal_frame_height=640,
        stats=False,
        trace=False
        )   
        self.renderer = BlazeposeRenderer(self.tracker, output=None)
        print("✓ OAK-D ready")

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
        update_interval = 1.0 / 30  # 30 FPS WebSocket updates

        while True:
            loop_start = time.time()

            # Get frame from queue (non-blocking)
            try:
                frame, body = self.frame_queue.get_nowait()
            except:
                # No frame available, yield to event loop
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

            # Send mesh update to WebSocket clients (with keypoints for visualization)
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
            cv2.imshow("Rigged Clothing Test", frame)

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
        self.ws_server.close()
        await self.ws_server.wait_closed()


def main():
    """Entry point"""
    test = RiggedClothingTest()

    # Run async event loop
    asyncio.run(test.run())


if __name__ == "__main__":
    main()
