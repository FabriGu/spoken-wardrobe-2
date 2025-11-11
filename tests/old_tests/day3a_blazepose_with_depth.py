"""
Day 3A+ (Fixed): BlazePose + Depth in Single Pipeline

This modifies the BlazePose pipeline to also output depth + confidence.
Single device, one pipeline, all features.

Press 'q' to quit
Press 's' to save current mesh
Press 'v' to visualize in Open3D
Press SPACE to toggle skeleton overlay
"""

import sys
from pathlib import Path
import time

# Add paths
blazepose_path = Path(__file__).parent.parent / "external" / "depthai_blazepose"
sys.path.insert(0, str(blazepose_path))

src_path = Path(__file__).parent.parent / "src"
sys.path.insert(0, str(src_path))

import cv2
import numpy as np
import depthai as dai
from modules.depth_mesh_generator import DepthMeshGenerator, MeshData

try:
    import open3d as o3d
    HAS_OPEN3D = True
except ImportError:
    HAS_OPEN3D = False


class LandmarkIndex:
    LEFT_SHOULDER = 11
    RIGHT_SHOULDER = 12
    LEFT_ELBOW = 13
    RIGHT_ELBOW = 14
    LEFT_WRIST = 15
    RIGHT_WRIST = 16
    LEFT_HIP = 23
    RIGHT_HIP = 24
    LEFT_KNEE = 25
    RIGHT_KNEE = 26


class BlazePoseWithDepth:
    """
    Custom BlazePose implementation with depth streams added

    Based on BlazeposeDepthaiEdge but with depth + confidence outputs
    """

    def __init__(self):
        print("Creating unified BlazePose + Depth pipeline...")

        self.pipeline = dai.Pipeline()

        # === RGB Camera ===
        cam_rgb = self.pipeline.create(dai.node.ColorCamera)
        cam_rgb.setBoardSocket(dai.CameraBoardSocket.CAM_A)
        cam_rgb.setResolution(dai.ColorCameraProperties.SensorResolution.THE_1080_P)
        cam_rgb.setVideoSize(640, 400)
        cam_rgb.setPreviewSize(640, 400)
        cam_rgb.setInterleaved(False)
        cam_rgb.setFps(30)

        # === Stereo Depth ===
        mono_left = self.pipeline.create(dai.node.MonoCamera)
        mono_right = self.pipeline.create(dai.node.MonoCamera)
        stereo = self.pipeline.create(dai.node.StereoDepth)

        mono_left.setResolution(dai.MonoCameraProperties.SensorResolution.THE_400_P)
        mono_left.setBoardSocket(dai.CameraBoardSocket.CAM_B)

        mono_right.setResolution(dai.MonoCameraProperties.SensorResolution.THE_400_P)
        mono_right.setBoardSocket(dai.CameraBoardSocket.CAM_C)

        stereo.setDefaultProfilePreset(dai.node.StereoDepth.PresetMode.HIGH_DENSITY)
        stereo.setDepthAlign(dai.CameraBoardSocket.CAM_A)
        stereo.setOutputSize(640, 400)
        stereo.setLeftRightCheck(True)
        stereo.setSubpixel(True)

        mono_left.out.link(stereo.left)
        mono_right.out.link(stereo.right)

        # === BlazePose Networks ===
        # Note: We'll use a simplified approach - just get RGB and landmarks
        # Full BlazePose edge mode is complex, so we'll use host mode for now

        # === Outputs ===
        xout_video = self.pipeline.create(dai.node.XLinkOut)
        xout_video.setStreamName("video")
        cam_rgb.video.link(xout_video.input)

        xout_depth = self.pipeline.create(dai.node.XLinkOut)
        xout_depth.setStreamName("depth")
        stereo.depth.link(xout_depth.input)

        xout_confidence = self.pipeline.create(dai.node.XLinkOut)
        xout_confidence.setStreamName("confidence")
        stereo.confidenceMap.link(xout_confidence.input)

        # Start device
        print("Starting device...")
        self.device = dai.Device(self.pipeline)

        # Queues
        self.video_queue = self.device.getOutputQueue(name="video", maxSize=4, blocking=False)
        self.depth_queue = self.device.getOutputQueue(name="depth", maxSize=4, blocking=False)
        self.confidence_queue = self.device.getOutputQueue(name="confidence", maxSize=4, blocking=False)

        # Get intrinsics
        calib = self.device.readCalibration()
        intrinsics_matrix = calib.getCameraIntrinsics(dai.CameraBoardSocket.CAM_A, 640, 400)

        self.camera_intrinsics = {
            'fx': intrinsics_matrix[0][0],
            'fy': intrinsics_matrix[1][1],
            'cx': intrinsics_matrix[0][2],
            'cy': intrinsics_matrix[1][2]
        }

        # Import MediaPipe for CPU-based pose detection
        print("Loading MediaPipe pose detector (CPU)...")
        import mediapipe as mp
        self.mp_pose = mp.solutions.pose
        self.pose = self.mp_pose.Pose(
            static_image_mode=False,
            model_complexity=0,  # 0=lite, 1=full, 2=heavy
            smooth_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )

        print("✓ Pipeline initialized")
        print(f"  Intrinsics: fx={self.camera_intrinsics['fx']:.1f}, fy={self.camera_intrinsics['fy']:.1f}")

    def get_frames_and_pose(self):
        """
        Get RGB, Depth, Confidence, and Pose landmarks

        Returns:
            rgb_frame: BGR image
            depth_frame: Depth in mm
            confidence_frame: Confidence 0-255
            landmarks: MediaPipe pose landmarks (or None)
        """
        # Get frames
        video_msg = self.video_queue.get()
        rgb_frame = video_msg.getCvFrame()

        depth_msg = self.depth_queue.get()
        depth_frame = depth_msg.getFrame()

        confidence_msg = self.confidence_queue.get()
        confidence_frame = confidence_msg.getFrame()

        # Run MediaPipe pose detection
        rgb_for_mp = cv2.cvtColor(rgb_frame, cv2.COLOR_BGR2RGB)
        results = self.pose.process(rgb_for_mp)

        landmarks = None
        if results.pose_landmarks:
            # Convert to numpy array (33 landmarks x 3 coordinates)
            landmarks = np.array([
                [lm.x, lm.y, lm.z]
                for lm in results.pose_landmarks.landmark
            ])

        return rgb_frame, depth_frame, confidence_frame, landmarks

    def close(self):
        """Clean up"""
        self.pose.close()
        self.device.close()


def create_body_mask_from_landmarks(frame_shape, landmarks_norm):
    """Create body mask from MediaPipe landmarks"""
    h, w = frame_shape
    mask = np.zeros((h, w), dtype=np.uint8)

    # Convert to pixels
    landmarks_px = []
    for lm in landmarks_norm:
        x = int(lm[0] * w)
        y = int(lm[1] * h)
        landmarks_px.append([x, y])

    landmarks_px = np.array(landmarks_px, dtype=np.int32)

    # Body contour
    contour_indices = [
        LandmarkIndex.LEFT_WRIST,
        LandmarkIndex.LEFT_ELBOW,
        LandmarkIndex.LEFT_SHOULDER,
        LandmarkIndex.RIGHT_SHOULDER,
        LandmarkIndex.RIGHT_ELBOW,
        LandmarkIndex.RIGHT_WRIST,
        LandmarkIndex.RIGHT_HIP,
        LandmarkIndex.LEFT_HIP,
    ]

    contour = landmarks_px[contour_indices]
    cv2.fillPoly(mask, [contour], 255)

    # Dilate
    kernel = np.ones((15, 15), np.uint8)
    mask = cv2.dilate(mask, kernel, iterations=1)

    return mask


def draw_skeleton(frame, landmarks_norm):
    """Draw skeleton on frame"""
    h, w = frame.shape[:2]

    landmarks_px = []
    for lm in landmarks_norm:
        x = int(lm[0] * w)
        y = int(lm[1] * h)
        landmarks_px.append((x, y))

    connections = [
        (LandmarkIndex.LEFT_SHOULDER, LandmarkIndex.RIGHT_SHOULDER),
        (LandmarkIndex.LEFT_SHOULDER, LandmarkIndex.LEFT_HIP),
        (LandmarkIndex.RIGHT_SHOULDER, LandmarkIndex.RIGHT_HIP),
        (LandmarkIndex.LEFT_HIP, LandmarkIndex.RIGHT_HIP),
        (LandmarkIndex.LEFT_SHOULDER, LandmarkIndex.LEFT_ELBOW),
        (LandmarkIndex.LEFT_ELBOW, LandmarkIndex.LEFT_WRIST),
        (LandmarkIndex.RIGHT_SHOULDER, LandmarkIndex.RIGHT_ELBOW),
        (LandmarkIndex.RIGHT_ELBOW, LandmarkIndex.RIGHT_WRIST),
        (LandmarkIndex.LEFT_HIP, LandmarkIndex.LEFT_KNEE),
        (LandmarkIndex.RIGHT_HIP, LandmarkIndex.RIGHT_KNEE),
    ]

    for idx1, idx2 in connections:
        pt1 = landmarks_px[idx1]
        pt2 = landmarks_px[idx2]
        cv2.line(frame, pt1, pt2, (0, 255, 0), 2)

    for pt in landmarks_px:
        cv2.circle(frame, pt, 3, (0, 0, 255), -1)

    return frame


def visualize_mesh_open3d(mesh: MeshData):
    """Visualize in Open3D"""
    if not HAS_OPEN3D:
        return

    o3d_mesh = o3d.geometry.TriangleMesh()
    o3d_mesh.vertices = o3d.utility.Vector3dVector(mesh.vertices)
    o3d_mesh.triangles = o3d.utility.Vector3iVector(mesh.faces)

    if mesh.colors is not None:
        colors = mesh.colors.astype(np.float64) / 255.0
        o3d_mesh.vertex_colors = o3d.utility.Vector3dVector(colors)

    o3d_mesh.compute_vertex_normals()
    o3d.visualization.draw_geometries([o3d_mesh], window_name="Body Mesh")


def main():
    print("="*70)
    print("Day 3A+ (Fixed): BlazePose + Depth → 3D Body Mesh")
    print("="*70)
    print("\nControls:")
    print("  'q' - Quit")
    print("  's' - Save mesh")
    print("  'v' - Visualize in Open3D")
    print("  SPACE - Toggle skeleton")
    print("  'c' - Toggle confidence threshold")
    print("="*70)

    # Initialize
    system = BlazePoseWithDepth()
    mesh_gen = DepthMeshGenerator(camera_intrinsics=system.camera_intrinsics)

    frame_count = 0
    last_mesh = None
    show_skeleton = True
    confidence_threshold = 200

    try:
        while True:
            rgb_frame, depth_frame, confidence_frame, landmarks = system.get_frames_and_pose()

            if landmarks is not None:
                # Create body mask
                person_mask = create_body_mask_from_landmarks(rgb_frame.shape[:2], landmarks)

                # Filter depth by confidence
                depth_filtered = depth_frame.copy()
                depth_filtered[confidence_frame < confidence_threshold] = 0

                # Combined mask
                combined_mask = (person_mask > 0) & (confidence_frame >= confidence_threshold)
                combined_mask = combined_mask.astype(np.uint8) * 255

                # Generate point cloud
                points = mesh_gen.depth_frame_to_pointcloud(depth_filtered, combined_mask)

                if len(points) > 100:
                    h, w = depth_frame.shape
                    grid_h = h // mesh_gen.depth_downsample
                    grid_w = w // mesh_gen.depth_downsample
                    expected = grid_h * grid_w

                    if len(points) >= expected * 0.3:
                        try:
                            vertices, faces = mesh_gen.triangulate_pointcloud_grid(points, (grid_h, grid_w))
                            colors = mesh_gen.compute_vertex_colors_from_rgb(vertices, rgb_frame)
                            last_mesh = MeshData(vertices, faces, colors=colors)

                            if frame_count % 30 == 0:
                                print(f"\rFrame {frame_count}: {last_mesh}   ", end='')
                        except Exception as e:
                            if frame_count % 30 == 0:
                                print(f"\r⚠️  {e}   ", end='')

                # Draw skeleton
                if show_skeleton:
                    rgb_frame = draw_skeleton(rgb_frame, landmarks)

            frame_count += 1

            # Visualize
            depth_viz = cv2.applyColorMap(cv2.convertScaleAbs(depth_frame, alpha=0.03), cv2.COLORMAP_JET)
            conf_viz = cv2.applyColorMap(confidence_frame, cv2.COLORMAP_VIRIDIS)

            cv2.imshow("RGB + Skeleton", rgb_frame)
            cv2.imshow("Depth", depth_viz)
            cv2.imshow("Confidence", conf_viz)

            # Keys
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord(' '):
                show_skeleton = not show_skeleton
                print(f"\nSkeleton: {'ON' if show_skeleton else 'OFF'}")
            elif key == ord('c'):
                confidence_threshold = 100 if confidence_threshold == 200 else 200
                print(f"\nConfidence threshold: {confidence_threshold}")
            elif key == ord('s') and last_mesh:
                output_dir = Path(__file__).parent.parent / "generated_meshes" / "oakd_depth"
                output_dir.mkdir(parents=True, exist_ok=True)
                mesh_path = output_dir / f"body_mesh_{int(time.time())}.npz"
                np.savez(str(mesh_path), vertices=last_mesh.vertices, faces=last_mesh.faces, colors=last_mesh.colors)
                print(f"\n✓ Saved: {mesh_path}")
            elif key == ord('v') and last_mesh and HAS_OPEN3D:
                visualize_mesh_open3d(last_mesh)

    finally:
        print("\n\nCleaning up...")
        system.close()
        cv2.destroyAllWindows()
        print("Complete!")


if __name__ == "__main__":
    main()
