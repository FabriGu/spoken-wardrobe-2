"""
Day 3 COMPLETE: Real-Time AI Clothing Overlay

This is the FULL pipeline:
1. Capture depth + pose from OAK-D
2. Generate 3D body mesh
3. Load AI-generated clothing from Day 2
4. Apply texture mapping (UV coordinates)
5. Stream to Three.js web viewer via WebSocket

Usage:
1. First generate clothing with: python tests/day2_oakd_bodypix_sd.py
2. Then run this script
3. Open http://localhost:8000/viewer.html in browser
4. See AI clothing overlayed on your body in real-time!

Controls:
- 'q' - Quit
- 't' - Toggle texture on/off
- 'c' - Change confidence threshold
"""

import sys
from pathlib import Path
import time
import json
import asyncio
import websockets
import threading
from http.server import HTTPServer, SimpleHTTPRequestHandler
import os

# Add paths
blazepose_path = Path(__file__).parent.parent / "external" / "depthai_blazepose"
sys.path.insert(0, str(blazepose_path))

src_path = Path(__file__).parent.parent / "src"
sys.path.insert(0, str(src_path))

import cv2
import numpy as np
import depthai as dai
import mediapipe as mp

from modules.depth_mesh_generator import DepthMeshGenerator, MeshData
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


class RealtimeClothingOverlay:
    """Main system - integrates all components"""

    def __init__(self, clothing_texture_path: str):
        print("Initializing Real-Time Clothing Overlay System...")
        print("="*70)

        # === OAK-D Setup ===
        print("\n1. Setting up OAK-D...")
        self.setup_oakd()

        # === Mesh Generator ===
        print("\n2. Initializing mesh generator...")
        self.mesh_gen = DepthMeshGenerator(camera_intrinsics=self.camera_intrinsics)

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

        # === Texture Mapper ===
        print("\n4. Loading texture mapper...")
        self.texture_mapper = TextureMapper(projection_mode='planar')

        # Load clothing texture
        self.clothing_texture = self.texture_mapper.load_texture_image(clothing_texture_path)
        print(f"✓ Loaded clothing texture from {clothing_texture_path}")

        # === WebSocket Server ===
        print("\n5. WebSocket server will start on port 8765")
        self.websocket_clients = set()

        # State
        self.confidence_threshold = 200
        self.texture_enabled = True

        print("\n" + "="*70)
        print("✓ System initialized successfully!")
        print("="*70)

    def setup_oakd(self):
        """Create OAK-D pipeline"""
        self.pipeline = dai.Pipeline()

        # RGB camera
        cam_rgb = self.pipeline.create(dai.node.ColorCamera)
        cam_rgb.setBoardSocket(dai.CameraBoardSocket.CAM_A)
        cam_rgb.setResolution(dai.ColorCameraProperties.SensorResolution.THE_1080_P)
        cam_rgb.setVideoSize(640, 400)
        cam_rgb.setPreviewSize(640, 400)
        cam_rgb.setInterleaved(False)
        cam_rgb.setFps(30)

        # Stereo depth
        mono_l = self.pipeline.create(dai.node.MonoCamera)
        mono_r = self.pipeline.create(dai.node.MonoCamera)
        stereo = self.pipeline.create(dai.node.StereoDepth)

        mono_l.setResolution(dai.MonoCameraProperties.SensorResolution.THE_400_P)
        mono_l.setBoardSocket(dai.CameraBoardSocket.CAM_B)

        mono_r.setResolution(dai.MonoCameraProperties.SensorResolution.THE_400_P)
        mono_r.setBoardSocket(dai.CameraBoardSocket.CAM_C)

        stereo.setDefaultProfilePreset(dai.node.StereoDepth.PresetMode.HIGH_DENSITY)
        stereo.setDepthAlign(dai.CameraBoardSocket.CAM_A)
        stereo.setOutputSize(640, 400)
        stereo.setLeftRightCheck(True)
        stereo.setSubpixel(True)

        mono_l.out.link(stereo.left)
        mono_r.out.link(stereo.right)

        # Outputs
        xout_video = self.pipeline.create(dai.node.XLinkOut)
        xout_video.setStreamName("video")
        cam_rgb.video.link(xout_video.input)

        xout_depth = self.pipeline.create(dai.node.XLinkOut)
        xout_depth.setStreamName("depth")
        stereo.depth.link(xout_depth.input)

        xout_conf = self.pipeline.create(dai.node.XLinkOut)
        xout_conf.setStreamName("confidence")
        stereo.confidenceMap.link(xout_conf.input)

        # Start device
        self.device = dai.Device(self.pipeline)
        self.video_queue = self.device.getOutputQueue("video", 4, False)
        self.depth_queue = self.device.getOutputQueue("depth", 4, False)
        self.conf_queue = self.device.getOutputQueue("confidence", 4, False)

        # Get intrinsics
        calib = self.device.readCalibration()
        intrinsics_matrix = calib.getCameraIntrinsics(dai.CameraBoardSocket.CAM_A, 640, 400)
        self.camera_intrinsics = {
            'fx': intrinsics_matrix[0][0],
            'fy': intrinsics_matrix[1][1],
            'cx': intrinsics_matrix[0][2],
            'cy': intrinsics_matrix[1][2]
        }

        print(f"✓ OAK-D initialized (fx={self.camera_intrinsics['fx']:.1f})")

    def get_textured_mesh(self, debug=False):
        """
        Main processing loop - returns textured mesh

        Returns:
            dict with mesh data for WebSocket, or None if no body detected
        """
        # Get frames
        rgb = self.video_queue.get().getCvFrame()
        depth = self.depth_queue.get().getFrame()
        conf = self.conf_queue.get().getFrame()

        if debug:
            print(f"\n[DEBUG] Frame captured: RGB={rgb.shape}, Depth={depth.shape}, Conf={conf.shape}")
            print(f"[DEBUG] Depth range: {depth.min()}-{depth.max()}mm")
            print(f"[DEBUG] Confidence range: {conf.min()}-{conf.max()}")

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

        # Create body mask (simple version - just torso)
        h, w = rgb.shape[:2]
        mask = np.zeros((h, w), dtype=np.uint8)

        lm_px = (landmarks[:, :2] * [w, h]).astype(np.int32)

        # Torso contour
        contour_idx = [
            LANDMARK_INDICES['LEFT_WRIST'],
            LANDMARK_INDICES['LEFT_ELBOW'],
            LANDMARK_INDICES['LEFT_SHOULDER'],
            LANDMARK_INDICES['RIGHT_SHOULDER'],
            LANDMARK_INDICES['RIGHT_ELBOW'],
            LANDMARK_INDICES['RIGHT_WRIST'],
            LANDMARK_INDICES['RIGHT_HIP'],
            LANDMARK_INDICES['LEFT_HIP'],
        ]
        contour = lm_px[contour_idx]
        cv2.fillPoly(mask, [contour], 255)

        # Dilate mask
        kernel = np.ones((15, 15), np.uint8)
        mask = cv2.dilate(mask, kernel, iterations=1)

        if debug:
            mask_coverage = (mask > 0).sum() / mask.size * 100
            print(f"[DEBUG] Body mask coverage: {mask_coverage:.1f}% of frame")

        # Filter depth
        depth_filtered = depth.copy()
        depth_filtered[conf < self.confidence_threshold] = 0

        # Combined mask
        combined = ((mask > 0) & (conf >= self.confidence_threshold)).astype(np.uint8) * 255

        if debug:
            combined_coverage = (combined > 0).sum() / combined.size * 100
            depth_valid = (depth_filtered > 0).sum()
            print(f"[DEBUG] Combined mask coverage: {combined_coverage:.1f}%")
            print(f"[DEBUG] Valid depth pixels: {depth_valid}")

        # Generate point cloud GRID (keeps structure, uses NaN for holes)
        points_grid = self.mesh_gen.depth_frame_to_pointcloud(depth_filtered, combined)

        grid_h, grid_w = points_grid.shape[:2]
        valid_points = np.sum(~np.isnan(points_grid[:, :, 0]))

        if debug:
            print(f"[DEBUG] Point cloud grid: {grid_h}x{grid_w}, {valid_points} valid points")

        if valid_points < 100:
            if debug:
                print(f"[DEBUG] ❌ Not enough valid points (need 100, got {valid_points})")
            return None

        # Triangulate using grid structure
        try:
            vertices, faces = self.mesh_gen.triangulate_pointcloud_grid(points_grid)
            if debug:
                print(f"[DEBUG] ✓ Triangulation successful: {len(vertices)} vertices, {len(faces)} faces")
        except Exception as e:
            if debug:
                print(f"[DEBUG] ❌ Triangulation failed: {e}")
            return None

        if len(vertices) == 0 or len(faces) == 0:
            if debug:
                print(f"[DEBUG] ❌ Empty mesh: {len(vertices)} vertices, {len(faces)} faces")
            return None

        # Compute UV coordinates
        uv = self.texture_mapper.compute_planar_uv(vertices, landmarks, LANDMARK_INDICES)

        # Apply texture (if enabled)
        if self.texture_enabled:
            vertex_colors = self.texture_mapper.apply_texture_to_mesh(
                vertices, faces, uv, self.clothing_texture
            )
        else:
            # Use original RGB colors
            vertex_colors = self.mesh_gen.compute_vertex_colors_from_rgb(vertices, rgb)

        if debug:
            print(f"[DEBUG] ✓ Mesh complete with {'texture' if self.texture_enabled else 'RGB colors'}")

        # Prepare data for WebSocket
        return {
            'vertices': vertices.tolist(),
            'faces': faces.tolist(),
            'colors': vertex_colors.tolist(),
            'uv': uv.tolist()
        }

    async def websocket_handler(self, websocket):
        """Handle WebSocket connections"""
        self.websocket_clients.add(websocket)
        print(f"✓ Client connected ({len(self.websocket_clients)} total)")

        try:
            # Keep connection alive and handle any incoming messages
            async for message in websocket:
                # Could handle client commands here if needed
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

        # Send to all clients and track disconnected ones
        disconnected = set()
        for client in self.websocket_clients:
            try:
                await client.send(message)
            except websockets.exceptions.ConnectionClosed:
                disconnected.add(client)
            except Exception as e:
                print(f"Error sending to client: {e}")
                disconnected.add(client)

        # Remove disconnected clients
        for client in disconnected:
            self.websocket_clients.discard(client)

    def run(self):
        """Main loop"""
        print("\n" + "="*70)
        print("Starting real-time overlay...")
        print("Open http://localhost:8000/tests/clothing_viewer.html in your browser")
        print("="*70)
        print("\nControls (press keys in Camera Preview window):")
        print("  'q' - Quit")
        print("  't' - Toggle texture")
        print("  'c' - Change confidence threshold (200/100)")
        print("  'd' - Toggle debug output")
        print("  'p' - Toggle camera preview window")
        print("="*70 + "\n")

        # Start WebSocket server in background
        async def ws_server():
            async with websockets.serve(self.websocket_handler, "localhost", 8765):
                await asyncio.Future()  # run forever

        ws_thread = threading.Thread(target=lambda: asyncio.run(ws_server()), daemon=True)
        ws_thread.start()

        frame_count = 0
        debug_mode = False
        last_debug_frame = -999
        show_preview = True  # Show RGB preview for keyboard input

        try:
            while True:
                # Enable debug every 30 frames when debug mode is on
                enable_debug = debug_mode and (frame_count - last_debug_frame >= 30)
                if enable_debug:
                    last_debug_frame = frame_count

                mesh_data = self.get_textured_mesh(debug=enable_debug)

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

                # Show preview window for keyboard input
                if show_preview:
                    rgb_frame = self.video_queue.tryGet()
                    if rgb_frame:
                        preview = rgb_frame.getCvFrame()
                        # Add status overlay
                        status_text = f"Frame: {frame_count} | Debug: {'ON' if debug_mode else 'OFF'} | Texture: {'ON' if self.texture_enabled else 'OFF'}"
                        cv2.putText(preview, status_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                        cv2.imshow("Camera Preview (press keys here)", preview)

                frame_count += 1

                # Handle keyboard (non-blocking check)
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    break
                elif key == ord('t'):
                    self.texture_enabled = not self.texture_enabled
                    print(f"\nTexture: {'ON' if self.texture_enabled else 'OFF'}")
                elif key == ord('c'):
                    self.confidence_threshold = 100 if self.confidence_threshold == 200 else 200
                    print(f"\nConfidence: {self.confidence_threshold}")
                elif key == ord('d'):
                    debug_mode = not debug_mode
                    print(f"\nDebug mode: {'ON' if debug_mode else 'OFF'}")
                elif key == ord('p'):
                    show_preview = not show_preview
                    if not show_preview:
                        cv2.destroyAllWindows()
                    print(f"\nPreview: {'ON' if show_preview else 'OFF'}")

        finally:
            print("\n\nShutting down...")
            self.pose.close()
            self.device.close()
            print("Complete!")


def find_latest_clothing():
    """Find most recent clothing image from Day 2"""
    clothing_dir = Path(__file__).parent.parent / "generated_meshes" / "oakd_clothing"

    if not clothing_dir.exists():
        return None

    clothing_files = list(clothing_dir.glob("clothing_*.png"))
    if not clothing_files:
        return None

    # Sort by timestamp (filename contains timestamp)
    latest = sorted(clothing_files, key=lambda p: p.stem.split('_')[1], reverse=True)[0]
    return str(latest)


def main():
    print("="*70)
    print("Day 3 COMPLETE: Real-Time AI Clothing Overlay")
    print("="*70)

    # Find clothing texture
    clothing_path = find_latest_clothing()

    if not clothing_path:
        print("\n❌ No clothing found!")
        print("Please run Day 2 first: python tests/day2_oakd_bodypix_sd.py")
        return

    print(f"\n✓ Found clothing: {Path(clothing_path).name}")

    # Create system
    system = RealtimeClothingOverlay(clothing_path)

    # Run
    system.run()


if __name__ == "__main__":
    main()
