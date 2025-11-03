"""
Test OAK-D Pro with BlazePose edge mode + measured 3D depth
This is Day 1 - Basic connectivity and world_landmarks verification
"""

import sys
from pathlib import Path

# Add depthai_blazepose to path
blazepose_path = Path(__file__).parent.parent / "external" / "depthai_blazepose"
sys.path.insert(0, str(blazepose_path))

import cv2
import numpy as np
from BlazeposeDepthaiEdge import BlazeposeDepthai
from BlazeposeRenderer import BlazeposeRenderer

def main():
    print("="*60)
    print("OAK-D Pro BlazePose Test - Edge Mode with Measured 3D")
    print("="*60)
    print("\nInitializing OAK-D Pro...")
    print("This will:")
    print("  1. Run BlazePose on the OAK-D VPU (not CPU)")
    print("  2. Get world_landmarks (metric 3D in meters)")
    print("  3. Measure real depth of mid-hips reference point")
    print("\nPress 'q' or ESC to exit")
    print("Press 'SPACE' to pause/unpause")
    print("Press 'x' to toggle (x,y,z) coordinate display")
    print("="*60)

    try:
        # Initialize BlazePose tracker in Edge mode
        tracker = BlazeposeDepthai(
            input_src='rgb',              # Use OAK-D camera
            lm_model='lite',              # Lite model = faster (26 FPS vs 20 for full)
            xyz=True,                     # Enable measured 3D depth
            smoothing=True,               # Temporal filtering (reduce jitter)
            internal_fps=30,              # Target FPS
            internal_frame_height=640,    # Resolution (lower = faster)
            stats=True,                   # Print stats at exit
            trace=False                   # Disable debug messages
        )

        # Initialize renderer
        renderer = BlazeposeRenderer(
            tracker,
            show_3d=None,  # We'll do custom 3D visualization later
            output=None    # No video output for now
        )

        print("\n✓ OAK-D Pro initialized successfully!")
        print("✓ BlazePose running in Edge mode (on-device)")
        print("✓ Measured 3D depth enabled\n")

        frame_count = 0

        while True:
            # Get next frame and body pose
            frame, body = tracker.next_frame()
            if frame is None:
                break

            # Draw 2D skeleton on frame
            frame = renderer.draw(frame, body)

            # If body detected, print landmark info
            if body and frame_count % 30 == 0:  # Print every 30 frames (~1 sec)
                print("\n" + "="*60)
                print(f"Frame {frame_count}: Body detected!")

                # Print world_landmarks info
                if hasattr(body, 'world_landmarks') and body.world_landmarks is not None:
                    wl = body.world_landmarks
                    print(f"\n✓ World Landmarks: {len(wl)} points (metric 3D in meters)")
                    print(f"  Sample - Left Shoulder (landmark 11):")
                    print(f"    X: {wl[11][0]:.3f}m, Y: {wl[11][1]:.3f}m, Z: {wl[11][2]:.3f}m")
                    print(f"  Sample - Right Shoulder (landmark 12):")
                    print(f"    X: {wl[12][0]:.3f}m, Y: {wl[12][1]:.3f}m, Z: {wl[12][2]:.3f}m")

                    # Calculate shoulder width
                    shoulder_width = np.linalg.norm(wl[11] - wl[12])
                    print(f"  Shoulder width: {shoulder_width:.3f}m")

                # Print measured 3D depth (mid-hips reference point)
                if hasattr(body, 'xyz') and body.xyz is not None:
                    print(f"\n✓ Measured 3D Depth (mid-hips):")
                    print(f"    X: {body.xyz[0]:.0f}mm")
                    print(f"    Y: {body.xyz[1]:.0f}mm")
                    print(f"    Z: {body.xyz[2]:.0f}mm (distance from camera)")
                    print(f"    = {body.xyz[2]/1000:.2f} meters from camera")

                # Print 2D landmarks info
                if hasattr(body, 'landmarks') and body.landmarks is not None:
                    print(f"\n✓ 2D Landmarks: {len(body.landmarks)} points (pixel coordinates)")

                print("="*60)

            frame_count += 1

            # Show frame
            cv2.imshow("OAK-D BlazePose Test", frame)

            # Handle key presses
            key = cv2.waitKey(1)
            if key == 27 or key == ord('q'):  # ESC or 'q'
                break
            elif key == ord(' '):  # SPACE - handled by renderer
                pass

        print("\n" + "="*60)
        print("Test completed successfully!")
        print("="*60)

        # Cleanup
        renderer.exit()
        tracker.exit()
        cv2.destroyAllWindows()

    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return False

    return True

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
