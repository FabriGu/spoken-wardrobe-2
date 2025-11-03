"""
Day 3 REVISED: Billboard-Style Clothing Overlay

This is the SIMPLER, WORKING approach:
1. Capture RGB + pose from OAK-D
2. Load AI-generated clothing texture
3. Create paper-like billboard mesh that follows the body
4. Apply texture mapping
5. Stream to Three.js web viewer via WebSocket

Like sticking paper to your body (see user's prototype photos)!

Controls:
- 'q' - Quit
- 't' - Toggle texture on/off
- 'r' - Adjust mesh resolution (smoother/faster)
- 'd' - Toggle debug output
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

from modules.billboard_overlay import BillboardOverlay
from modules.texture_mapper import TextureMapper


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


class BillboardClothingOverlay:
    """Billboard-style clothing overlay system"""

    def __init__(self, clothing_texture_path: str, mesh_resolution: int = 20):
        print("Initializing Billboard Clothing Overlay System...")
        print("="*70)

        # === OAK-D Setup ===
        print("\n1. Setting up OAK-D...")
        self.setup_oakd()

        # === MediaPipe Pose ===
        print("\n2. Loading MediaPipe Pose...")
        mp_pose = mp.solutions.pose
        self.pose = mp_pose.Pose(
            static_image_mode=False,
            model_complexity=0,
            smooth_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )

        # === Billboard Generator ===
        print("\n3. Initializing billboard generator...")
        self.billboard = BillboardOverlay(mesh_resolution=mesh_resolution)
        self.mesh_resolution = mesh_resolution

        # === Texture Mapper ===
        print("\n4. Loading texture mapper...")
        self.texture_mapper = TextureMapper(projection_mode='planar')
        self.clothing_texture = self.texture_mapper.load_texture_image(clothing_texture_path)
        print(f"✓ Loaded clothing texture from {clothing_texture_path}")

        # === WebSocket Server ===
        print("\n5. WebSocket server will start on port 8765")
        self.websocket_clients = set()

        # State
        self.texture_enabled = True

        print("\n" + "="*70)
        print("✓ System initialized successfully!")
        print("="*70)

    def setup_oakd(self):
        """Create OAK-D pipeline (RGB only, no depth needed!)"""
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

    def get_textured_billboard(self, debug=False):
        """
        Main processing loop - returns textured billboard mesh

        Returns:
            dict with mesh data for WebSocket, or None if no body detected
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

        # Get landmarks as numpy array
        landmarks = np.array([[lm.x, lm.y, lm.z] for lm in results.pose_landmarks.landmark])

        # Generate billboard mesh
        try:
            vertices_norm, faces, uvs = self.billboard.create_body_billboard(
                landmarks, LANDMARK_INDICES
            )

            if debug:
                print(f"[DEBUG] ✓ Billboard mesh: {len(vertices_norm)} vertices, {len(faces)} faces")

        except Exception as e:
            if debug:
                print(f"[DEBUG] ❌ Billboard generation failed: {e}")
            return None

        # Convert to camera space (3D coordinates)
        vertices = self.billboard.convert_to_camera_space(
            vertices_norm,
            self.frame_shape,
            self.camera_intrinsics,
            avg_depth_m=1.5  # Assume 1.5m from camera
        )

        # Apply texture
        if self.texture_enabled:
            # Sample texture at UV coordinates
            vertex_colors = self.sample_texture_at_uvs(uvs, self.clothing_texture)
        else:
            # Use solid color for debug
            vertex_colors = np.full((len(vertices), 3), [200, 200, 200], dtype=np.uint8)

        if debug:
            print(f"[DEBUG] ✓ Texture applied ({len(vertex_colors)} colors)")

        # Prepare data for WebSocket
        return {
            'vertices': vertices.tolist(),
            'faces': faces.tolist(),
            'colors': vertex_colors.tolist(),
            'uvs': uvs.tolist()
        }

    def sample_texture_at_uvs(self, uvs: np.ndarray, texture: np.ndarray) -> np.ndarray:
        """
        Sample texture colors at UV coordinates

        Args:
            uvs: Nx2 UV coordinates [0, 1]
            texture: Texture image (H x W x 3/4)

        Returns:
            colors: Nx3 RGB colors (0-255)
        """
        h, w = texture.shape[:2]
        colors = np.zeros((len(uvs), 3), dtype=np.uint8)

        for i, (u, v) in enumerate(uvs):
            # Convert UV to pixel coordinates
            x = int(np.clip(u, 0, 1) * (w - 1))
            y = int(np.clip(v, 0, 1) * (h - 1))

            # Sample color
            color = texture[y, x]

            # Handle RGBA
            if len(color) == 4:
                color = color[:3]

            colors[i] = color

        return colors

    async def websocket_handler(self, websocket):
        """Handle WebSocket connections"""
        self.websocket_clients.add(websocket)
        print(f"✓ Client connected ({len(self.websocket_clients)} total)")

        try:
            async for message in websocket:
                pass  # Keep connection alive
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
        print("Starting billboard overlay...")
        print("Open http://localhost:8000/tests/clothing_viewer.html in your browser")
        print("="*70)
        print("\nControls (press keys in Camera Preview window):")
        print("  'q' - Quit")
        print("  't' - Toggle texture")
        print("  'r' - Change resolution (smoother/faster)")
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

                mesh_data = self.get_textured_billboard(debug=enable_debug)

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
                        status_text = f"Frame: {frame_count} | Debug: {'ON' if debug_mode else 'OFF'} | Res: {self.mesh_resolution}"
                        cv2.putText(preview, status_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                        cv2.imshow("Camera Preview (press keys here)", preview)

                frame_count += 1

                # Handle keyboard
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    break
                elif key == ord('t'):
                    self.texture_enabled = not self.texture_enabled
                    print(f"\nTexture: {'ON' if self.texture_enabled else 'OFF'}")
                elif key == ord('d'):
                    debug_mode = not debug_mode
                    print(f"\nDebug mode: {'ON' if debug_mode else 'OFF'}")
                elif key == ord('r'):
                    # Cycle resolution: 15 → 20 → 30 → 15
                    resolutions = [15, 20, 30]
                    current_idx = resolutions.index(self.mesh_resolution) if self.mesh_resolution in resolutions else 0
                    self.mesh_resolution = resolutions[(current_idx + 1) % len(resolutions)]
                    self.billboard = BillboardOverlay(mesh_resolution=self.mesh_resolution)
                    print(f"\nMesh resolution: {self.mesh_resolution}x{self.mesh_resolution}")

        finally:
            print("\n\nShutting down...")
            self.pose.close()
            self.device.close()
            cv2.destroyAllWindows()
            print("Complete!")


def find_latest_clothing():
    """Find most recent clothing image"""
    clothing_dir = Path(__file__).parent.parent / "generated_meshes" / "oakd_clothing"

    if not clothing_dir.exists():
        return None

    clothing_files = list(clothing_dir.glob("clothing_*.png"))
    if not clothing_files:
        return None

    latest = sorted(clothing_files, key=lambda p: p.stem.split('_')[1], reverse=True)[0]
    return str(latest)


def main():
    print("="*70)
    print("Day 3 REVISED: Billboard-Style Clothing Overlay")
    print("="*70)

    # Find clothing texture
    clothing_path = find_latest_clothing()

    if not clothing_path:
        print("\n❌ No clothing found!")
        print("Please run Day 2 first: python tests/day2_oakd_bodypix_sd.py")
        return

    print(f"\n✓ Found clothing: {Path(clothing_path).name}")

    # Create system
    system = BillboardClothingOverlay(clothing_path, mesh_resolution=20)

    # Run
    system.run()


if __name__ == "__main__":
    main()
