"""
Day 3A: OAK-D Depth → 3D Body Mesh
Convert stereo depth from OAK-D to textured 3D body surface mesh

This script:
1. Captures RGB + Depth from OAK-D Pro
2. Runs BlazePose for landmarks
3. Converts depth to 3D point cloud
4. Triangulates into mesh
5. Segments torso region
6. Visualizes with Open3D

Press 'q' to quit
Press 's' to save current mesh
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

# Try to import Open3D for visualization
try:
    import open3d as o3d
    HAS_OPEN3D = True
except ImportError:
    print("⚠️  Open3D not available - visualization disabled")
    HAS_OPEN3D = False


class OAKDDepthCapture:
    """
    Captures aligned RGB + Depth from OAK-D Pro

    This class handles the DepthAI pipeline for synchronized depth and RGB frames
    """

    def __init__(self):
        print("Initializing OAK-D pipeline...")

        # Create pipeline
        self.pipeline = dai.Pipeline()

        # Define sources
        mono_left = self.pipeline.create(dai.node.MonoCamera)
        mono_right = self.pipeline.create(dai.node.MonoCamera)
        stereo = self.pipeline.create(dai.node.StereoDepth)
        rgb = self.pipeline.create(dai.node.ColorCamera)

        # Output queues
        xout_depth = self.pipeline.create(dai.node.XLinkOut)
        xout_rgb = self.pipeline.create(dai.node.XLinkOut)

        xout_depth.setStreamName("depth")
        xout_rgb.setStreamName("rgb")

        # Configure mono cameras (for stereo depth)
        mono_left.setResolution(dai.MonoCameraProperties.SensorResolution.THE_400_P)
        mono_left.setBoardSocket(dai.CameraBoardSocket.CAM_B)  # Left camera

        mono_right.setResolution(dai.MonoCameraProperties.SensorResolution.THE_400_P)
        mono_right.setBoardSocket(dai.CameraBoardSocket.CAM_C)  # Right camera

        # Configure stereo depth
        stereo.setDefaultProfilePreset(dai.node.StereoDepth.PresetMode.HIGH_DENSITY)
        stereo.setDepthAlign(dai.CameraBoardSocket.CAM_A)  # Align to RGB camera
        stereo.setOutputSize(640, 400)  # Match mono resolution

        # Stereo settings for better quality
        stereo.setLeftRightCheck(True)  # LR-check for better accuracy
        stereo.setExtendedDisparity(False)  # Don't need extreme close range
        stereo.setSubpixel(True)  # Subpixel for smoother depth

        # Configure RGB camera
        rgb.setBoardSocket(dai.CameraBoardSocket.CAM_A)  # RGB camera
        rgb.setResolution(dai.ColorCameraProperties.SensorResolution.THE_1080_P)
        rgb.setInterleaved(False)
        rgb.setColorOrder(dai.ColorCameraProperties.ColorOrder.BGR)
        rgb.setFps(30)

        # Link mono cameras to stereo
        mono_left.out.link(stereo.left)
        mono_right.out.link(stereo.right)

        # Link outputs
        stereo.depth.link(xout_depth.input)
        rgb.video.link(xout_rgb.input)

        # Start device
        print("Starting OAK-D device...")
        self.device = dai.Device(self.pipeline)

        # Create output queues
        self.depth_queue = self.device.getOutputQueue(name="depth", maxSize=4, blocking=False)
        self.rgb_queue = self.device.getOutputQueue(name="rgb", maxSize=4, blocking=False)

        # Get camera intrinsics (for point cloud projection)
        calib = self.device.readCalibration()
        intrinsics_matrix = calib.getCameraIntrinsics(dai.CameraBoardSocket.CAM_A, 640, 400)

        self.camera_intrinsics = {
            'fx': intrinsics_matrix[0][0],
            'fy': intrinsics_matrix[1][1],
            'cx': intrinsics_matrix[0][2],
            'cy': intrinsics_matrix[1][2]
        }

        print(f"✓ OAK-D initialized")
        print(f"  Camera intrinsics: fx={self.camera_intrinsics['fx']:.1f}, "
              f"fy={self.camera_intrinsics['fy']:.1f}, "
              f"cx={self.camera_intrinsics['cx']:.1f}, "
              f"cy={self.camera_intrinsics['cy']:.1f}")

    def get_frames(self):
        """
        Get synchronized RGB + Depth frames

        Returns:
            rgb_frame: BGR image (H x W x 3)
            depth_frame: Depth map in millimeters (H x W)
        """
        # Get latest depth frame
        depth_msg = self.depth_queue.get()
        depth_frame = depth_msg.getFrame()

        # Get latest RGB frame
        rgb_msg = self.rgb_queue.get()
        rgb_frame = rgb_msg.getCvFrame()

        # Resize RGB to match depth resolution (640x400)
        rgb_frame = cv2.resize(rgb_frame, (depth_frame.shape[1], depth_frame.shape[0]))

        return rgb_frame, depth_frame

    def close(self):
        """Clean up resources"""
        if hasattr(self, 'device'):
            self.device.close()
        print("OAK-D device closed")


def visualize_mesh_open3d(mesh: MeshData, window_name: str = "Mesh Viewer"):
    """
    Visualize mesh using Open3D

    Args:
        mesh: MeshData object
        window_name: Window title
    """
    if not HAS_OPEN3D:
        print("⚠️  Open3D not available, skipping visualization")
        return

    # Create Open3D mesh
    o3d_mesh = o3d.geometry.TriangleMesh()
    o3d_mesh.vertices = o3d.utility.Vector3dVector(mesh.vertices)
    o3d_mesh.triangles = o3d.utility.Vector3iVector(mesh.faces)

    # Add colors if available
    if mesh.colors is not None:
        # Normalize to [0, 1]
        colors_normalized = mesh.colors.astype(np.float64) / 255.0
        o3d_mesh.vertex_colors = o3d.utility.Vector3dVector(colors_normalized)

    # Compute normals for better visualization
    o3d_mesh.compute_vertex_normals()

    # Visualize
    print(f"\nVisualizing mesh in Open3D...")
    print("  Controls:")
    print("    - Mouse drag: Rotate view")
    print("    - Mouse scroll: Zoom")
    print("    - Close window to continue")

    o3d.visualization.draw_geometries([o3d_mesh],
                                      window_name=window_name,
                                      width=800,
                                      height=600)


def main():
    print("="*70)
    print("Day 3A: OAK-D Depth → 3D Body Mesh")
    print("="*70)
    print("\nThis captures real depth from OAK-D and creates 3D body mesh")
    print("\nControls:")
    print("  'q' - Quit")
    print("  's' - Save current mesh")
    print("  'v' - Visualize mesh in Open3D (freezes capture)")
    print("="*70)

    # Initialize OAK-D depth capture
    oak_d = OAKDDepthCapture()

    # Initialize depth mesh generator with OAK-D intrinsics
    mesh_generator = DepthMeshGenerator(camera_intrinsics=oak_d.camera_intrinsics)

    # For testing, we'll skip BlazePose for now and use full depth frame
    # TODO: Integrate BlazePose for body segmentation

    frame_count = 0
    last_mesh = None

    try:
        while True:
            # Get RGB + Depth
            rgb_frame, depth_frame = oak_d.get_frames()

            # Create simple person mask (center region for testing)
            h, w = depth_frame.shape
            person_mask = mesh_generator.create_person_mask_from_bbox(
                (h, w),
                bbox=(w//4, h//4, 3*w//4, 3*h//4),
                padding=50
            )

            # Convert depth to point cloud
            points = mesh_generator.depth_frame_to_pointcloud(depth_frame, person_mask)

            if len(points) > 100:  # Only proceed if we have enough points
                # Triangulate into mesh
                grid_h = h // mesh_generator.depth_downsample
                grid_w = w // mesh_generator.depth_downsample

                # Only triangulate if we have a full grid
                expected_points = grid_h * grid_w
                if len(points) >= expected_points * 0.5:  # At least 50% coverage
                    try:
                        vertices, faces = mesh_generator.triangulate_pointcloud_grid(
                            points, (grid_h, grid_w)
                        )

                        # Get vertex colors from RGB
                        colors = mesh_generator.compute_vertex_colors_from_rgb(
                            vertices, rgb_frame
                        )

                        # Create mesh
                        last_mesh = MeshData(vertices, faces, colors=colors)

                        # Display info
                        if frame_count % 30 == 0:
                            print(f"\rFrame {frame_count}: {last_mesh}   ", end='')

                    except Exception as e:
                        print(f"\n⚠️  Mesh generation error: {e}")

            frame_count += 1

            # Visualize depth (colorized)
            depth_colormap = cv2.applyColorMap(
                cv2.convertScaleAbs(depth_frame, alpha=0.03),
                cv2.COLORMAP_JET
            )

            # Overlay mask (simple green tint)
            mask_overlay = depth_colormap.copy()
            mask_overlay[person_mask > 0] = cv2.addWeighted(
                mask_overlay[person_mask > 0], 0.7,
                np.full_like(mask_overlay[person_mask > 0], [0, 255, 0], dtype=np.uint8), 0.3, 0
            )
            depth_colormap = mask_overlay

            # Show frames
            cv2.imshow("RGB", rgb_frame)
            cv2.imshow("Depth", depth_colormap)

            # Handle keys
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('s') and last_mesh is not None:
                # Save mesh
                output_dir = Path(__file__).parent.parent / "generated_meshes" / "oakd_depth"
                output_dir.mkdir(parents=True, exist_ok=True)

                timestamp = int(time.time())
                mesh_path = output_dir / f"depth_mesh_{timestamp}.npz"

                np.savez(str(mesh_path),
                        vertices=last_mesh.vertices,
                        faces=last_mesh.faces,
                        colors=last_mesh.colors)

                print(f"\n✓ Saved mesh: {mesh_path}")

            elif key == ord('v') and last_mesh is not None and HAS_OPEN3D:
                # Visualize in Open3D
                visualize_mesh_open3d(last_mesh, "OAK-D Depth Mesh")

    finally:
        print("\n\nCleaning up...")
        oak_d.close()
        cv2.destroyAllWindows()
        print("Day 3A complete!")


if __name__ == "__main__":
    main()
