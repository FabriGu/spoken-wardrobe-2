"""
Day 1: OAK-D Body Segment Quad Generator
Creates 3D body segment meshes from BlazePose world_landmarks for texture mapping

This script:
1. Runs BlazePose in edge mode with measured 3D depth
2. Generates quad meshes for body segments (torso, arms, legs)
3. Outputs world landmark 3D coordinates (ready for texture mapping)
4. Saves reference pose data for clothing generation

Press SPACE to capture reference pose (with 10s countdown)
Press 'q' to quit
"""

import sys
from pathlib import Path
import time

# Add depthai_blazepose to path
blazepose_path = Path(__file__).parent.parent / "external" / "depthai_blazepose"
sys.path.insert(0, str(blazepose_path))

import cv2
import numpy as np
from BlazeposeDepthaiEdge import BlazeposeDepthai
from BlazeposeRenderer import BlazeposeRenderer
import json

# MediaPipe landmark indices
class LandmarkIndex:
    # Face/Head
    NOSE = 0
    LEFT_EYE_INNER = 1
    LEFT_EYE = 2
    LEFT_EYE_OUTER = 3
    RIGHT_EYE_INNER = 4
    RIGHT_EYE = 5
    RIGHT_EYE_OUTER = 6
    LEFT_EAR = 7
    RIGHT_EAR = 8
    MOUTH_LEFT = 9
    MOUTH_RIGHT = 10

    # Upper body
    LEFT_SHOULDER = 11
    RIGHT_SHOULDER = 12
    LEFT_ELBOW = 13
    RIGHT_ELBOW = 14
    LEFT_WRIST = 15
    RIGHT_WRIST = 16

    # Hands
    LEFT_PINKY = 17
    RIGHT_PINKY = 18
    LEFT_INDEX = 19
    RIGHT_INDEX = 20
    LEFT_THUMB = 21
    RIGHT_THUMB = 22

    # Lower body
    LEFT_HIP = 23
    RIGHT_HIP = 24
    LEFT_KNEE = 25
    RIGHT_KNEE = 26
    LEFT_ANKLE = 27
    RIGHT_ANKLE = 28

    # Feet
    LEFT_HEEL = 29
    RIGHT_HEEL = 30
    LEFT_FOOT_INDEX = 31
    RIGHT_FOOT_INDEX = 32


class BodySegmentGenerator:
    """Generates 3D quad meshes for body segments from BlazePose landmarks"""

    @staticmethod
    def get_torso_quad(world_landmarks):
        """
        Generate torso quad mesh from shoulders and hips
        Returns: vertices (4x3), uv_coords (4x2), indices (2 triangles)
        """
        if world_landmarks is None or len(world_landmarks) < 33:
            return None

        # Torso corners: left_shoulder, right_shoulder, right_hip, left_hip
        vertices = np.array([
            world_landmarks[LandmarkIndex.LEFT_SHOULDER],   # Top-left
            world_landmarks[LandmarkIndex.RIGHT_SHOULDER],  # Top-right
            world_landmarks[LandmarkIndex.RIGHT_HIP],       # Bottom-right
            world_landmarks[LandmarkIndex.LEFT_HIP],        # Bottom-left
        ], dtype=np.float32)

        # UV coordinates (standard quad mapping)
        uv_coords = np.array([
            [0.0, 0.0],  # Top-left
            [1.0, 0.0],  # Top-right
            [1.0, 1.0],  # Bottom-right
            [0.0, 1.0],  # Bottom-left
        ], dtype=np.float32)

        # Triangle indices (2 triangles form the quad)
        indices = np.array([
            [0, 1, 2],  # First triangle
            [0, 2, 3],  # Second triangle
        ], dtype=np.int32)

        return {
            'vertices': vertices,
            'uv_coords': uv_coords,
            'indices': indices,
            'name': 'torso'
        }

    @staticmethod
    def get_left_arm_quad(world_landmarks):
        """Generate left arm quad (shoulder to wrist)"""
        if world_landmarks is None or len(world_landmarks) < 33:
            return None

        shoulder = world_landmarks[LandmarkIndex.LEFT_SHOULDER]
        elbow = world_landmarks[LandmarkIndex.LEFT_ELBOW]
        wrist = world_landmarks[LandmarkIndex.LEFT_WRIST]

        # Create quad by extruding perpendicular to arm direction
        arm_vec = wrist - shoulder
        arm_length = np.linalg.norm(arm_vec)
        if arm_length < 0.01:
            return None

        # Perpendicular vector (approximate width)
        width = arm_length * 0.15  # Arm width ~15% of length
        perp = np.array([-arm_vec[1], arm_vec[0], 0])
        perp_norm = np.linalg.norm(perp)
        if perp_norm > 0:
            perp = (perp / perp_norm) * width

        vertices = np.array([
            shoulder + perp,  # Top-left
            shoulder - perp,  # Top-right
            wrist - perp,     # Bottom-right
            wrist + perp,     # Bottom-left
        ], dtype=np.float32)

        uv_coords = np.array([
            [0.0, 0.0],
            [1.0, 0.0],
            [1.0, 1.0],
            [0.0, 1.0],
        ], dtype=np.float32)

        indices = np.array([[0, 1, 2], [0, 2, 3]], dtype=np.int32)

        return {
            'vertices': vertices,
            'uv_coords': uv_coords,
            'indices': indices,
            'name': 'left_arm'
        }

    @staticmethod
    def get_right_arm_quad(world_landmarks):
        """Generate right arm quad (shoulder to wrist)"""
        if world_landmarks is None or len(world_landmarks) < 33:
            return None

        shoulder = world_landmarks[LandmarkIndex.RIGHT_SHOULDER]
        elbow = world_landmarks[LandmarkIndex.RIGHT_ELBOW]
        wrist = world_landmarks[LandmarkIndex.RIGHT_WRIST]

        arm_vec = wrist - shoulder
        arm_length = np.linalg.norm(arm_vec)
        if arm_length < 0.01:
            return None

        width = arm_length * 0.15
        perp = np.array([-arm_vec[1], arm_vec[0], 0])
        perp_norm = np.linalg.norm(perp)
        if perp_norm > 0:
            perp = (perp / perp_norm) * width

        vertices = np.array([
            shoulder + perp,
            shoulder - perp,
            wrist - perp,
            wrist + perp,
        ], dtype=np.float32)

        uv_coords = np.array([
            [0.0, 0.0],
            [1.0, 0.0],
            [1.0, 1.0],
            [0.0, 1.0],
        ], dtype=np.float32)

        indices = np.array([[0, 1, 2], [0, 2, 3]], dtype=np.int32)

        return {
            'vertices': vertices,
            'uv_coords': uv_coords,
            'indices': indices,
            'name': 'right_arm'
        }

    @staticmethod
    def get_left_leg_quad(world_landmarks):
        """Generate left leg quad (hip to ankle)"""
        if world_landmarks is None or len(world_landmarks) < 33:
            return None

        hip = world_landmarks[LandmarkIndex.LEFT_HIP]
        knee = world_landmarks[LandmarkIndex.LEFT_KNEE]
        ankle = world_landmarks[LandmarkIndex.LEFT_ANKLE]

        leg_vec = ankle - hip
        leg_length = np.linalg.norm(leg_vec)
        if leg_length < 0.01:
            return None

        width = leg_length * 0.12
        perp = np.array([-leg_vec[1], leg_vec[0], 0])
        perp_norm = np.linalg.norm(perp)
        if perp_norm > 0:
            perp = (perp / perp_norm) * width

        vertices = np.array([
            hip + perp,
            hip - perp,
            ankle - perp,
            ankle + perp,
        ], dtype=np.float32)

        uv_coords = np.array([
            [0.0, 0.0],
            [1.0, 0.0],
            [1.0, 1.0],
            [0.0, 1.0],
        ], dtype=np.float32)

        indices = np.array([[0, 1, 2], [0, 2, 3]], dtype=np.int32)

        return {
            'vertices': vertices,
            'uv_coords': uv_coords,
            'indices': indices,
            'name': 'left_leg'
        }

    @staticmethod
    def get_right_leg_quad(world_landmarks):
        """Generate right leg quad (hip to ankle)"""
        if world_landmarks is None or len(world_landmarks) < 33:
            return None

        hip = world_landmarks[LandmarkIndex.RIGHT_HIP]
        knee = world_landmarks[LandmarkIndex.RIGHT_KNEE]
        ankle = world_landmarks[LandmarkIndex.RIGHT_ANKLE]

        leg_vec = ankle - hip
        leg_length = np.linalg.norm(leg_vec)
        if leg_length < 0.01:
            return None

        width = leg_length * 0.12
        perp = np.array([-leg_vec[1], leg_vec[0], 0])
        perp_norm = np.linalg.norm(perp)
        if perp_norm > 0:
            perp = (perp / perp_norm) * width

        vertices = np.array([
            hip + perp,
            hip - perp,
            ankle - perp,
            ankle + perp,
        ], dtype=np.float32)

        uv_coords = np.array([
            [0.0, 0.0],
            [1.0, 0.0],
            [1.0, 1.0],
            [0.0, 1.0],
        ], dtype=np.float32)

        indices = np.array([[0, 1, 2], [0, 2, 3]], dtype=np.int32)

        return {
            'vertices': vertices,
            'uv_coords': uv_coords,
            'indices': indices,
            'name': 'right_leg'
        }

    @classmethod
    def get_clothing_segments(cls, world_landmarks, clothing_type='tshirt'):
        """
        Get body segments based on clothing type

        Args:
            world_landmarks: BlazePose world landmarks (33x3 array in meters)
            clothing_type: 'tshirt', 'dress', 'shorts', etc.

        Returns:
            List of segment dictionaries
        """
        segments = []

        if clothing_type == 'tshirt':
            # T-shirt: torso + upper arms
            torso = cls.get_torso_quad(world_landmarks)
            left_arm = cls.get_left_arm_quad(world_landmarks)
            right_arm = cls.get_right_arm_quad(world_landmarks)

            if torso: segments.append(torso)
            if left_arm: segments.append(left_arm)
            if right_arm: segments.append(right_arm)

        elif clothing_type == 'dress':
            # Dress: torso + upper arms + upper legs
            torso = cls.get_torso_quad(world_landmarks)
            left_arm = cls.get_left_arm_quad(world_landmarks)
            right_arm = cls.get_right_arm_quad(world_landmarks)
            left_leg = cls.get_left_leg_quad(world_landmarks)
            right_leg = cls.get_right_leg_quad(world_landmarks)

            if torso: segments.append(torso)
            if left_arm: segments.append(left_arm)
            if right_arm: segments.append(right_arm)
            if left_leg: segments.append(left_leg)
            if right_leg: segments.append(right_leg)

        return segments


def draw_countdown(frame, seconds_left):
    """Draw countdown overlay on frame"""
    h, w = frame.shape[:2]

    # Semi-transparent overlay
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (w, h), (0, 0, 0), -1)
    frame = cv2.addWeighted(frame, 0.3, overlay, 0.7, 0)

    # Countdown text
    text = f"GET IN POSITION: {seconds_left}" if seconds_left > 0 else "CAPTURING!"
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 3
    thickness = 5

    # Get text size for centering
    (text_width, text_height), _ = cv2.getTextSize(text, font, font_scale, thickness)
    x = (w - text_width) // 2
    y = (h + text_height) // 2

    # Draw text with outline
    cv2.putText(frame, text, (x, y), font, font_scale, (0, 0, 0), thickness + 4)
    cv2.putText(frame, text, (x, y), font, font_scale, (0, 255, 255), thickness)

    return frame


def main():
    print("="*70)
    print("Day 1: OAK-D Body Segment Quad Generator")
    print("="*70)
    print("\nThis script generates 3D body segment meshes for clothing overlay")
    print("\nControls:")
    print("  SPACE - Start 10-second countdown and capture reference pose")
    print("  't'   - Switch to t-shirt mode (torso + arms)")
    print("  'd'   - Switch to dress mode (torso + arms + legs)")
    print("  'q'   - Quit")
    print("="*70)

    # Initialize BlazePose tracker
    tracker = BlazeposeDepthai(
        input_src='rgb',
        lm_model='lite',  # Lite = 26 FPS
        xyz=True,         # Measured 3D depth
        smoothing=True,
        internal_fps=30,
        internal_frame_height=640,
        stats=False,
        trace=False
    )

    renderer = BlazeposeRenderer(tracker, show_3d=None, output=None)

    print("\n✓ OAK-D Pro initialized (Edge mode + measured 3D)")
    print("✓ Ready to capture body segments\n")

    segment_generator = BodySegmentGenerator()
    clothing_type = 'tshirt'
    countdown_active = False
    countdown_start_time = None
    COUNTDOWN_DURATION = 5  # seconds

    captured_data = None

    while True:
        frame, body = tracker.next_frame()
        # print(f"frame: {frame}")
        # print(f"body: {body}")
        if frame is None:
            break

        # Draw skeleton
        frame = renderer.draw(frame, body)

        # Handle countdown
        if countdown_active:
            elapsed = time.time() - countdown_start_time
            seconds_left = max(0, COUNTDOWN_DURATION - int(elapsed))

            frame = draw_countdown(frame, seconds_left)

            if elapsed >= COUNTDOWN_DURATION:
                # Capture!
                countdown_active = False
                # print(f"body has attributes: {hasattr(body, 'world_landmarks')}")
                # print(f"body.world_landmarks: {body.world_landmarks}")
                #if body 'Body' object has no attribute 'world_landmarks' check what attributes it has
                print(f"body attributes: {body.__dict__}")
                if body and hasattr(body, 'landmarks_world'):
                    segments = segment_generator.get_clothing_segments(
                        body.landmarks_world,
                        clothing_type=clothing_type
                    )

                    captured_data = {
                        'world_landmarks': body.landmarks_world.tolist(),
                        'measured_depth': body.xyz.tolist() if hasattr(body, 'xyz') and body.xyz is not None else None,
                        'segments': [
                            {
                                'name': seg['name'],
                                'vertices': seg['vertices'].tolist(),
                                'uv_coords': seg['uv_coords'].tolist(),
                                'indices': seg['indices'].tolist()
                            }
                            for seg in segments
                        ],
                        'clothing_type': clothing_type
                    }

                    # Save to file
                    output_dir = Path(__file__).parent.parent / "generated_meshes" / "oakd_reference"
                    output_dir.mkdir(parents=True, exist_ok=True)
                    output_file = output_dir / f"body_segments_{int(time.time())}.json"

                    with open(output_file, 'w') as f:
                        json.dump(captured_data, f, indent=2)

                    print(f"\n{'='*70}")
                    print(f"✓ Captured body segments ({clothing_type})!")
                    print(f"✓ {len(segments)} segments generated")
                    print(f"✓ Saved to: {output_file}")
                    print(f"{'='*70}\n")
                else:
                    print("\n❌ No body detected! Try again.\n")

        # Display info
        cv2.putText(frame, f"Mode: {clothing_type.upper()}", (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        if captured_data:
            cv2.putText(frame, "Captured! (Press SPACE for new capture)", (10, 60),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

        cv2.imshow("OAK-D Body Segments", frame)

        # Handle keys
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord(' ') and not countdown_active:
            # Start countdown
            countdown_active = True
            countdown_start_time = time.time()
            print(f"\n⏱ Countdown started! Get in position ({COUNTDOWN_DURATION}s)...")
        elif key == ord('t'):
            clothing_type = 'tshirt'
            print(f"→ Switched to {clothing_type} mode")
        elif key == ord('d'):
            clothing_type = 'dress'
            print(f"→ Switched to {clothing_type} mode")

    print("\n" + "="*70)
    print("Day 1 complete!")
    print("="*70)

    renderer.exit()
    tracker.exit()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
