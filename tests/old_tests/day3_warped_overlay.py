"""
Day 3 FINAL: Body-Part Warped Clothing Overlay

This is the CORRECT implementation:
1. Load clothing image + reference pose from Day 2 metadata
2. Capture current pose with MediaPipe
3. Create separate mesh quads for torso and arms
4. Warp each part independently based on keypoint correspondence
5. Stream to Three.js web viewer

Perfect body alignment - arms move independently!

Controls:
- 'q' - Quit
- 'r' - Change resolution
- 'd' - Toggle debug
"""

import sys
from pathlib import Path
import json
import asyncio
import websockets
import threading
import time

# Add paths
src_path = Path(__file__).parent.parent / "src"
sys.path.insert(0, str(src_path))

import cv2
import numpy as np
import depthai as dai
import mediapipe as mp

from modules.body_part_warping import BodyPartWarper


# Landmark indices
LANDMARK_INDICES = {
    'LEFT_SHOULDER': 11,
    'RIGHT_SHOULDER': 12,
    'LEFT_ELBOW': 13,
    'RIGHT_ELBOW': 14,
    'LEFT_WRIST': 15,
    'RIGHT_WRIST': 16,
    'LEFT_HIP': 23,
    'RIGHT_HIP': 24,
    'LEFT_KNEE': 25,
    'RIGHT_KNEE': 26,
}


class WarpedClothingOverlay:
    """Body-part warped clothing overlay system"""

    def __init__(self, clothing_dir: Path, quad_resolution: int = 15):
        print("Initializing Warped Clothing Overlay System...")
        print("="*70)

        # Load clothing and metadata
        print("\n1. Loading clothing and reference pose...")
        self.load_clothing_data(clothing_dir)

        # === OAK-D Setup ===
        print("\n2. Setting up OAK-D...")
        self.setup_oakd()

        # === MediaPipe Pose ===
        print("\n3. Loading MediaPipe Pose...")
        mp_pose = mp.solutions.pose
        self.pose = mp_pose.Pose(
            static_image_mode=False,
            model_complexity=0,
            smooth_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )

        # === Body Part Warper ===
        print("\n4. Initializing body part warper...")
        self.warper = BodyPartWarper(quad_resolution=quad_resolution)
        self.quad_resolution = quad_resolution

        # === WebSocket Server ===
        print("\n5. WebSocket server will start on port 8765")
        self.websocket_clients = set()

        print("\n" + "="*70)
        print("✓ System initialized successfully!")
        print("="*70)

    def load_clothing_data(self, clothing_dir: Path):
        """Load clothing image, mask, and reference pose from Day 2"""
        # Find latest metadata file
        metadata_files = list(clothing_dir.glob("metadata_*.json"))
        if not metadata_files:
            raise FileNotFoundError(f"No metadata found in {clothing_dir}")

        latest_metadata = sorted(metadata_files, key=lambda p: p.stem.split('_')[1], reverse=True)[0]

        with open(latest_metadata, 'r') as f:
            self.metadata = json.load(f)

        print(f"✓ Loaded metadata: {latest_metadata.name}")
        print(f"  Clothing type: {self.metadata['clothing_type']}")
        print(f"  Prompt: {self.metadata['prompt']}")

        # Load images
        clothing_file = clothing_dir / self.metadata['files']['clothing']
        mask_file = clothing_dir / self.metadata['files']['mask']

        self.clothing_image = cv2.imread(str(clothing_file), cv2.IMREAD_UNCHANGED)
        self.mask_image = cv2.imread(str(mask_file), cv2.IMREAD_GRAYSCALE)

        if self.clothing_image is None or self.mask_image is None:
            raise FileNotFoundError("Could not load clothing or mask images")

        # Convert BGR to RGB for consistency
        if len(self.clothing_image.shape) == 3 and self.clothing_image.shape[2] == 3:
            self.clothing_image = cv2.cvtColor(self.clothing_image, cv2.COLOR_BGR2RGB)

        print(f"✓ Loaded clothing image: {self.clothing_image.shape}")
        print(f"✓ Loaded mask: {self.mask_image.shape}")

        # Load reference landmarks (normalized)
        if 'landmarks_normalized' not in self.metadata:
            raise ValueError("Metadata missing 'landmarks_normalized' - please regenerate with updated Day 2!")

        self.reference_landmarks = np.array(self.metadata['landmarks_normalized'], dtype=np.float32)
        print(f"✓ Loaded reference landmarks: {self.reference_landmarks.shape}")

    def setup_oakd(self):
        """Create OAK-D pipeline (RGB only)"""
        self.pipeline = dai.Pipeline()

        # RGB camera
        cam_rgb = self.pipeline.create(dai.node.ColorCamera)
        cam_rgb.setBoardSocket(dai.CameraBoardSocket.CAM_A)
        cam_rgb.setResolution(dai.ColorCameraProperties.SensorResolution.THE_1080_P)
        cam_rgb.setVideoSize(640, 400)
        cam_rgb.setPreviewSize(640, 400)
        cam_rgb.setInterleaved(False)
        cam_rgb.setFps(30)

        # Output
        xout_video = self.pipeline.create(dai.node.XLinkOut)
        xout_video.setStreamName("video")
        cam_rgb.video.link(xout_video.input)

        # Start device
        self.device = dai.Device(self.pipeline)
        self.video_queue = self.device.getOutputQueue("video", 4, False)

        # Get intrinsics
        calib = self.device.readCalibration()
        intrinsics_matrix = calib.getCameraIntrinsics(dai.CameraBoardSocket.CAM_A, 640, 400)
        self.camera_intrinsics = {
            'fx': intrinsics_matrix[0][0],
            'fy': intrinsics_matrix[1][1],
            'cx': intrinsics_matrix[0][2],
            'cy': intrinsics_matrix[1][2]
        }

        self.frame_shape = (400, 640)

        print(f"✓ OAK-D initialized (fx={self.camera_intrinsics['fx']:.1f})")

    def get_warped_meshes(self, debug=False):
        """
        Main processing loop - returns warped body part meshes

        Returns:
            dict with combined mesh data for WebSocket, or None if no body detected
        """
        # Get RGB frame
        rgb = self.video_queue.get().getCvFrame()

        if debug:
            print(f"\n[DEBUG] Frame captured: RGB={rgb.shape}")

        # Run pose detection
        rgb_mp = cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB)
        results = self.pose.process(rgb_mp)

        if not results.pose_landmarks:
            if debug:
                print("[DEBUG] ❌ MediaPipe detected NO pose landmarks")
            return None

        if debug:
            print(f"[DEBUG] ✓ MediaPipe detected {len(results.pose_landmarks.landmark)} landmarks")

        # Get current landmarks (normalized)
        current_landmarks = np.array([[lm.x, lm.y, lm.z] for lm in results.pose_landmarks.landmark])

        # Generate body part meshes
        try:
            body_part_meshes = self.warper.create_body_part_quads(
                current_landmarks,
                self.reference_landmarks,
                self.clothing_image,
                self.mask_image,
                LANDMARK_INDICES,
                mode='reference',
                debug=debug  # Pass debug flag
            )

            if debug:
                print(f"[DEBUG] ✓ Generated {len(body_part_meshes)} body part meshes:")
                for mesh in body_part_meshes:
                    print(f"[DEBUG]   - {mesh.name}: {len(mesh.vertices)} vertices")

        except Exception as e:
            if debug:
                print(f"[DEBUG] ❌ Mesh generation failed: {e}")
                import traceback
                traceback.print_exc()
            return None

        if len(body_part_meshes) == 0:
            if debug:
                print("[DEBUG] ❌ No visible body parts")
            return None

        # Convert to camera space
        meshes_3d = self.warper.convert_to_camera_space(
            body_part_meshes,
            self.frame_shape,
            self.camera_intrinsics,
            avg_depth_m=1.5
        )

        # Combine all meshes into single data structure for WebSocket
        all_vertices = []
        all_faces = []
        all_colors = []
        vertex_offset = 0

        for mesh in meshes_3d:
            all_vertices.extend(mesh.vertices.tolist())
            # Offset face indices for combined mesh
            faces_offset = (mesh.faces + vertex_offset).tolist()
            all_faces.extend(faces_offset)
            all_colors.extend(mesh.colors.tolist())
            vertex_offset += len(mesh.vertices)

        if debug:
            print(f"[DEBUG] ✓ Combined mesh: {len(all_vertices)} vertices, {len(all_faces)} faces")

        return {
            'vertices': all_vertices,
            'faces': all_faces,
            'colors': all_colors
        }

    async def websocket_handler(self, websocket):
        """Handle WebSocket connections"""
        self.websocket_clients.add(websocket)
        print(f"✓ Client connected ({len(self.websocket_clients)} total)")

        try:
            async for message in websocket:
                pass
        except websockets.exceptions.ConnectionClosed:
            pass
        finally:
            self.websocket_clients.discard(websocket)
            print(f"Client disconnected ({len(self.websocket_clients)} remaining)")

    async def broadcast_mesh(self, mesh_data):
        """Send mesh to all connected clients"""
        if not self.websocket_clients:
            return

        message = json.dumps({
            'type': 'mesh_update',
            'data': mesh_data
        })

        disconnected = set()
        for client in self.websocket_clients:
            try:
                await client.send(message)
            except websockets.exceptions.ConnectionClosed:
                disconnected.add(client)
            except Exception as e:
                print(f"Error sending to client: {e}")
                disconnected.add(client)

        for client in disconnected:
            self.websocket_clients.discard(client)

    def run(self):
        """Main loop"""
        print("\n" + "="*70)
        print("Starting warped clothing overlay...")
        print("Open http://localhost:8000/tests/clothing_viewer.html in your browser")
        print("="*70)
        print("\nControls (press keys in Camera Preview window):")
        print("  'q' - Quit")
        print("  'r' - Change resolution (10/15/20)")
        print("  'd' - Toggle debug output")
        print("="*70 + "\n")

        # Start WebSocket server in background
        async def ws_server():
            async with websockets.serve(self.websocket_handler, "localhost", 8765):
                await asyncio.Future()

        ws_thread = threading.Thread(target=lambda: asyncio.run(ws_server()), daemon=True)
        ws_thread.start()

        frame_count = 0
        debug_mode = False
        last_debug_frame = -999
        show_preview = True

        try:
            while True:
                # Enable debug every 30 frames when debug mode is on
                enable_debug = debug_mode and (frame_count - last_debug_frame >= 30)
                if enable_debug:
                    last_debug_frame = frame_count

                mesh_data = self.get_warped_meshes(debug=enable_debug)

                if mesh_data:
                    # Broadcast to web clients
                    asyncio.run(self.broadcast_mesh(mesh_data))

                    if frame_count % 30 == 0:
                        num_verts = len(mesh_data['vertices'])
                        print(f"\rFrame {frame_count}: {num_verts} vertices, "
                              f"{len(self.websocket_clients)} clients   ", end='')
                else:
                    if frame_count % 30 == 0 and not debug_mode:
                        print(f"\rFrame {frame_count}: No mesh (press 'd' for debug)   ", end='')

                # Show preview window
                if show_preview:
                    rgb_frame = self.video_queue.tryGet()
                    if rgb_frame:
                        preview = rgb_frame.getCvFrame()
                        status_text = f"Frame: {frame_count} | Debug: {'ON' if debug_mode else 'OFF'} | Res: {self.quad_resolution}"
                        cv2.putText(preview, status_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                        cv2.imshow("Camera Preview (press keys here)", preview)

                frame_count += 1

                # Handle keyboard
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    break
                elif key == ord('d'):
                    debug_mode = not debug_mode
                    print(f"\nDebug mode: {'ON' if debug_mode else 'OFF'}")
                elif key == ord('r'):
                    # Cycle resolution: 10 → 15 → 20 → 10
                    resolutions = [10, 15, 20]
                    current_idx = resolutions.index(self.quad_resolution) if self.quad_resolution in resolutions else 0
                    self.quad_resolution = resolutions[(current_idx + 1) % len(resolutions)]
                    self.warper = BodyPartWarper(quad_resolution=self.quad_resolution)
                    print(f"\nMesh resolution: {self.quad_resolution}x{self.quad_resolution}")

        finally:
            print("\n\nShutting down...")
            self.pose.close()
            self.device.close()
            cv2.destroyAllWindows()
            print("Complete!")


def main():
    print("="*70)
    print("Day 3 FINAL: Body-Part Warped Clothing Overlay")
    print("="*70)

    clothing_dir = Path(__file__).parent.parent / "generated_meshes" / "oakd_clothing"

    if not clothing_dir.exists():
        print(f"\n❌ Clothing directory not found: {clothing_dir}")
        print("Please run Day 2 first!")
        return

    try:
        # Create system
        system = WarpedClothingOverlay(clothing_dir, quad_resolution=15)

        # Run
        system.run()

    except FileNotFoundError as e:
        print(f"\n❌ Error: {e}")
        print("\nPlease regenerate clothing with updated Day 2:")
        print("  python tests/day2_oakd_sd_integration.py")
    except ValueError as e:
        print(f"\n❌ Error: {e}")


if __name__ == "__main__":
    main()
