#!/usr/bin/env python3
"""
Basic RealSense Camera Test
============================

Tests Intel RealSense D435/D455 camera functionality:
1. RGB stream
2. Depth stream
3. Depth-to-color alignment
4. Frame synchronization

Press 'q' to quit
"""

import pyrealsense2 as rs
import numpy as np
import cv2
import sys


def main():
    print("=" * 70)
    print("Intel RealSense Basic Test")
    print("=" * 70)
    print("This will test your RealSense camera setup.")
    print("You should see RGB and depth streams side by side.")
    print("\nPress 'q' to quit")
    print("=" * 70)

    # Configure RealSense pipeline
    pipeline = rs.pipeline()
    config = rs.config()

    # Enable streams
    # RGB: 1280x720 @ 30fps (can also use 1920x1080, 640x480)
    # Depth: 1280x720 @ 30fps (must match RGB for alignment)
    config.enable_stream(rs.stream.color, 1280, 720, rs.format.bgr8, 30)
    config.enable_stream(rs.stream.depth, 1280, 720, rs.format.z16, 30)

    print("\nStarting RealSense pipeline...")
    try:
        # Start streaming
        profile = pipeline.start(config)

        # Get device information
        device = profile.get_device()
        print(f"✓ Connected to: {device.get_info(rs.camera_info.name)}")
        print(f"  Serial: {device.get_info(rs.camera_info.serial_number)}")
        print(f"  Firmware: {device.get_info(rs.camera_info.firmware_version)}")

        # Create alignment object (align depth to color frame)
        align = rs.align(rs.stream.color)

        # Create colorizer for depth visualization
        colorizer = rs.colorizer()
        colorizer.set_option(rs.option.color_scheme, 2)  # White to black

        print("\n✓ Pipeline started successfully!")
        print("\nShowing RGB (left) and Depth (right) streams")
        print("Press 'q' to quit\n")

        frame_count = 0

        while True:
            # Wait for frames
            frames = pipeline.wait_for_frames()

            # Align depth to color
            aligned_frames = align.process(frames)

            # Get aligned frames
            color_frame = aligned_frames.get_color_frame()
            depth_frame = aligned_frames.get_depth_frame()

            if not color_frame or not depth_frame:
                continue

            # Convert to numpy arrays
            color_image = np.asanyarray(color_frame.get_data())
            depth_image = np.asanyarray(depth_frame.get_data())

            # Colorize depth for visualization
            depth_colormap = np.asanyarray(colorizer.colorize(depth_frame).get_data())

            # Get depth statistics
            if frame_count % 30 == 0:  # Every second at 30 FPS
                # Get depth at center pixel
                height, width = depth_image.shape
                center_depth = depth_frame.get_distance(width // 2, height // 2)
                print(f"Center depth: {center_depth:.2f}m | "
                      f"Min: {depth_image.min()} | "
                      f"Max: {depth_image.max()}", end='\r')

            # Stack images horizontally
            combined = np.hstack((color_image, depth_colormap))

            # Add labels
            cv2.putText(combined, "RGB", (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            cv2.putText(combined, "Depth", (color_image.shape[1] + 10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

            # Show combined image
            cv2.imshow("RealSense RGB + Depth", combined)

            frame_count += 1

            # Check for quit
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break

    except Exception as e:
        print(f"\n✗ Error: {e}")
        print("\nTROUBLESHOOTING:")
        print("1. Is RealSense camera connected?")
        print("2. Is RealSense SDK installed?")
        print("3. Run: rs-enumerate-devices (to check camera)")
        print("4. Try: pip install --upgrade pyrealsense2")
        return 1

    finally:
        # Cleanup
        print("\n\nStopping pipeline...")
        pipeline.stop()
        cv2.destroyAllWindows()
        print("✓ Test complete!")

    return 0


if __name__ == "__main__":
    sys.exit(main())
