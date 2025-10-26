def normalize_mesh(vertices):
    verts = np.array(vertices)
    center = verts.mean(axis=0)
    verts -= center
    scale = np.max(np.linalg.norm(verts, axis=1))
    if scale > 0:
        verts /= scale
    return verts
# test_with_threejs.py
# Test script that streams mesh to Three.js web viewer

import cv2
import numpy as np
import mediapipe as mp
import time
import trimesh
import asyncio
import threading
from cage_utils import SimpleCageGenerator, MeanValueCoordinates
from keypoint_mapper import KeypointToCageMapper
from websocket_server import MeshStreamServer


class CageDeformationWaithThreeJS:
    """
    Cage deformation that streams results to Three.js for rendering.
    """
    
    def __init__(self, mesh_path=None):
        # Initialize MediaPipe
        self.mp_pose = mp.solutions.pose
        self.pose = self.mp_pose.Pose(
            static_image_mode=False,
            model_complexity=1,
            smooth_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        
        # Load mesh
        if mesh_path:
            self.mesh = trimesh.load(mesh_path)
        else:
            self.mesh = self.create_simple_mesh()
        
        # Setup cage and MVC
        cage_gen = SimpleCageGenerator(self.mesh)
        self.cage = cage_gen.generate_simple_box_cage(subdivisions=3)
        self.mvc = MeanValueCoordinates(self.mesh.vertices, self.cage)
        self.mvc.compute_weights()
        
        self.keypoint_mapper = KeypointToCageMapper()
        
        # WebSocket server
        self.ws_server = None
        self.loop = None
        
    def create_simple_mesh(self):
        """Create simple test mesh."""
        # Create a box
        mesh = trimesh.creation.box(extents=[0.5, 0.7, 0.2])
        return mesh
    
    
    def start_websocket_server(self):
        """Start WebSocket server in separate thread."""
        self.server_ready = threading.Event()
        def run_server():
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            self.loop = loop
            self.ws_server = MeshStreamServer()
            try:
                loop.run_until_complete(self.ws_server.start())
            except Exception as e:
                print("WebSocket server error:", e)
            finally:
                self.server_ready.set()

        thread = threading.Thread(target=run_server, daemon=True)
        thread.start()
        self.server_ready.wait(timeout=5)
        print("WebSocket server started!")
    
    def send_mesh_to_web(self, vertices, faces, keypoints=None):
        if self.ws_server and self.loop:
            try:
                fut = asyncio.run_coroutine_threadsafe(
                    self.ws_server.send_mesh_data(vertices, faces, keypoints),
                    self.loop
                )
                fut.result(timeout=1)
            except Exception as e:
                print(f"Error sending mesh data: {e}")
        else:
            print("WebSocket server not ready, cannot send mesh.")

    def run(self):
        # Start the websocket server
        self.start_websocket_server()
        cap = cv2.VideoCapture(0)
        frame_count = 0
        send_interval = 5  # Send every 5 frames
        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    print("Failed to grab frame from camera.")
                    break
                # Convert frame to RGB for MediaPipe
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                results = self.pose.process(rgb_frame)
                if results.pose_landmarks:
                    keypoints = []
                    for lm in results.pose_landmarks.landmark:
                        keypoints.append([lm.x - 0.5, lm.y - 0.5, lm.z])  # Center and scale
                    keypoints = np.array(keypoints) * 2.0  # Scale to [-1, 1]
                    deformed_cage = self.keypoint_mapper.simple_anatomical_mapping(
                        results.pose_landmarks,
                        self.cage,
                        frame.shape
                    )
                    frame_count += 1
                    if frame_count % send_interval == 0:
                        deformed_verts = self.mvc.deform_mesh(deformed_cage)
                        deformed_verts = normalize_mesh(deformed_verts)
                        self.send_mesh_to_web(deformed_verts, self.mesh.faces, keypoints.tolist())
                # No OpenCV viewer, just stream to web
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
        finally:
            cap.release()


def main():
    mesh_path = "generated_meshes/3dMesh_1_clothing.obj"
    app = CageDeformationWaithThreeJS(mesh_path)
    app.run()


if __name__ == "__main__":
    main()
