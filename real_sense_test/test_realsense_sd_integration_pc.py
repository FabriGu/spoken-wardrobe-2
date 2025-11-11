#!/usr/bin/env python3
"""
RealSense + Stable Diffusion Integration (PC Version)
======================================================

Generates AI clothing for body segments captured from RealSense camera.

Pipeline:
1. Capture body segments from RealSense (with countdown)
2. Create mask from torso quad landmarks
3. Generate clothing with Stable Diffusion
4. Save clothing image + reference data

Controls:
- SPACE: Start countdown and generate clothing
- T: T-shirt mode
- D: Dress mode
- Q: Quit
"""

import sys
from pathlib import Path
import time
import json

# RealSense
import pyrealsense2 as rs

# MediaPipe
import mediapipe as mp

# Add paths
parent_path = Path(__file__).parent.parent
sys.path.insert(0, str(parent_path / "src"))

import cv2
import numpy as np
from modules.ai_generation import ClothingGenerator


# Landmark indices (MediaPipe Pose has 33 landmarks)
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
    LEFT_ANKLE = 27
    RIGHT_ANKLE = 28


class MaskGenerator:
    """Creates 2D masks from MediaPipe landmarks for SD inpainting"""

    @staticmethod
    def landmarks_to_pixels(landmarks_normalized, frame_shape):
        """
        Convert normalized landmarks [0,1] to pixel coordinates

        Args:
            landmarks_normalized: List of landmarks with .x, .y attributes (normalized 0-1)
            frame_shape: (height, width, channels)

        Returns:
            landmarks_2d: Nx2 array (x, y in pixels)
        """
        h, w = frame_shape[:2]
        landmarks_2d = np.zeros((len(landmarks_normalized), 2), dtype=np.int32)

        for i, lm in enumerate(landmarks_normalized):
            # Convert normalized [0,1] to pixel coordinates
            px = int(lm.x * w)
            py = int(lm.y * h)

            # Clamp to frame bounds
            px = max(0, min(w - 1, px))
            py = max(0, min(h - 1, py))

            landmarks_2d[i] = [px, py]

        return landmarks_2d

    @staticmethod
    def create_torso_mask(landmarks_2d, frame_shape, padding=20):
        """Create binary mask for torso region"""
        h, w = frame_shape[:2]
        mask = np.zeros((h, w), dtype=np.uint8)

        # Get torso corners
        left_shoulder = landmarks_2d[LandmarkIndex.LEFT_SHOULDER]
        right_shoulder = landmarks_2d[LandmarkIndex.RIGHT_SHOULDER]
        left_hip = landmarks_2d[LandmarkIndex.LEFT_HIP]
        right_hip = landmarks_2d[LandmarkIndex.RIGHT_HIP]

        # Create polygon (torso quad)
        pts = np.array([
            left_shoulder,
            right_shoulder,
            right_hip,
            left_hip
        ], dtype=np.int32)

        # Expand polygon by padding
        center = pts.mean(axis=0)
        pts_expanded = center + (pts - center) * (1 + padding / 100)
        pts_expanded = pts_expanded.astype(np.int32)

        # Fill polygon
        cv2.fillPoly(mask, [pts_expanded], 255)

        return mask

    @staticmethod
    def create_tshirt_mask(landmarks_2d, frame_shape, padding=20):
        """Create mask for t-shirt (torso + upper arms)"""
        h, w = frame_shape[:2]
        mask = np.zeros((h, w), dtype=np.uint8)

        # Torso + Arms
        left_shoulder = landmarks_2d[LandmarkIndex.LEFT_SHOULDER]
        right_shoulder = landmarks_2d[LandmarkIndex.RIGHT_SHOULDER]
        left_hip = landmarks_2d[LandmarkIndex.LEFT_HIP]
        right_hip = landmarks_2d[LandmarkIndex.RIGHT_HIP]
        left_elbow = landmarks_2d[LandmarkIndex.LEFT_ELBOW]
        right_elbow = landmarks_2d[LandmarkIndex.RIGHT_ELBOW]

        pts = np.array([
            left_elbow,
            left_shoulder,
            right_shoulder,
            right_elbow,
            right_hip,
            left_hip
        ], dtype=np.int32)

        center = pts.mean(axis=0)
        pts_expanded = center + (pts - center) * (1 + padding / 100)
        pts_expanded = pts_expanded.astype(np.int32)

        cv2.fillPoly(mask, [pts_expanded], 255)

        return mask

    @staticmethod
    def create_dress_mask(landmarks_2d, frame_shape, padding=20):
        """Create mask for dress (torso + arms + upper legs)"""
        h, w = frame_shape[:2]
        mask = np.zeros((h, w), dtype=np.uint8)

        left_elbow = landmarks_2d[LandmarkIndex.LEFT_ELBOW]
        left_shoulder = landmarks_2d[LandmarkIndex.LEFT_SHOULDER]
        right_shoulder = landmarks_2d[LandmarkIndex.RIGHT_SHOULDER]
        right_elbow = landmarks_2d[LandmarkIndex.RIGHT_ELBOW]
        right_knee = landmarks_2d[LandmarkIndex.RIGHT_KNEE]
        left_knee = landmarks_2d[LandmarkIndex.LEFT_KNEE]

        pts = np.array([
            left_elbow,
            left_shoulder,
            right_shoulder,
            right_elbow,
            right_knee,
            left_knee
        ], dtype=np.int32)

        center = pts.mean(axis=0)
        pts_expanded = center + (pts - center) * (1 + padding / 100)
        pts_expanded = pts_expanded.astype(np.int32)

        cv2.fillPoly(mask, [pts_expanded], 255)

        return mask


def draw_countdown(frame, seconds_left):
    """Draw countdown overlay"""
    h, w = frame.shape[:2]
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (w, h), (0, 0, 0), -1)
    frame = cv2.addWeighted(frame, 0.3, overlay, 0.7, 0)

    text = f"GET IN POSITION: {seconds_left}" if seconds_left > 0 else "CAPTURING!"
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 3
    thickness = 5

    (text_width, text_height), _ = cv2.getTextSize(text, font, font_scale, thickness)
    x = (w - text_width) // 2
    y = (h + text_height) // 2

    cv2.putText(frame, text, (x, y), font, font_scale, (0, 0, 0), thickness + 4)
    cv2.putText(frame, text, (x, y), font, font_scale, (0, 255, 255), thickness)

    return frame


def main():
    print("="*70)
    print("RealSense + Stable Diffusion Integration (PC Version)")
    print("="*70)
    print("\nThis generates AI clothing using RealSense depth + SD")
    print("\nControls:")
    print("  SPACE - Start 5-second countdown and generate clothing")
    print("  't'   - T-shirt mode")
    print("  'd'   - Dress mode")
    print("  'q'   - Quit")
    print("\nDefault prompt: 'colorful flames pattern'")
    print("="*70)

    # Initialize RealSense
    print("\nInitializing RealSense...")
    pipeline = rs.pipeline()
    config = rs.config()

    # Configure streams
    config.enable_stream(rs.stream.color, 1280, 720, rs.format.bgr8, 30)
    config.enable_stream(rs.stream.depth, 1280, 720, rs.format.z16, 30)

    # Start pipeline
    profile = pipeline.start(config)
    device = profile.get_device()
    print(f"✓ Connected to: {device.get_info(rs.camera_info.name)}")

    # Align depth to color
    align = rs.align(rs.stream.color)

    # Initialize MediaPipe Pose
    print("Initializing MediaPipe Pose...")
    mp_pose = mp.solutions.pose
    mp_drawing = mp.solutions.drawing_utils
    pose = mp_pose.Pose(
        static_image_mode=False,
        model_complexity=1,
        smooth_landmarks=True,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
    )
    print("✓ MediaPipe Pose initialized")

    # Initialize SD (lazy load on first use)
    print("\nStable Diffusion will load on first generation")
    generator = None

    mask_gen = MaskGenerator()
    clothing_type = 'tshirt'
    countdown_active = False
    countdown_start_time = None
    COUNTDOWN_DURATION = 5

    # Clothing prompt
    CLOTHING_PROMPT = "colorful flames pattern"

    try:
        while True:
            # Get frames
            frames = pipeline.wait_for_frames()
            aligned_frames = align.process(frames)

            color_frame = aligned_frames.get_color_frame()
            if not color_frame:
                continue

            # Convert to numpy
            frame = np.asanyarray(color_frame.get_data())

            # CRITICAL: Save clean frame BEFORE any drawing
            frame_clean = frame.copy()

            # Process with MediaPipe
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = pose.process(frame_rgb)

            # Draw skeleton for display
            if results.pose_landmarks:
                mp_drawing.draw_landmarks(
                    frame,
                    results.pose_landmarks,
                    mp_pose.POSE_CONNECTIONS,
                    mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=2),
                    mp_drawing.DrawingSpec(color=(0, 255, 255), thickness=2)
                )

            # Handle countdown
            if countdown_active:
                elapsed = time.time() - countdown_start_time
                seconds_left = max(0, COUNTDOWN_DURATION - int(elapsed))
                frame = draw_countdown(frame, seconds_left)

                if elapsed >= COUNTDOWN_DURATION:
                    countdown_active = False

                    if results.pose_landmarks:
                        print("\n" + "="*70)
                        print("Captured body! Generating clothing...")
                        print("="*70)

                        # Convert normalized landmarks to 2D pixels (use CLEAN frame)
                        frame_shape = frame_clean.shape
                        landmarks_2d = mask_gen.landmarks_to_pixels(
                            results.pose_landmarks.landmark,
                            frame_shape
                        )

                        # Create mask based on clothing type
                        if clothing_type == 'tshirt':
                            mask = mask_gen.create_tshirt_mask(landmarks_2d, frame_shape)
                        elif clothing_type == 'dress':
                            mask = mask_gen.create_dress_mask(landmarks_2d, frame_shape)
                        else:
                            mask = mask_gen.create_torso_mask(landmarks_2d, frame_shape)

                        # Show mask preview
                        mask_preview = cv2.resize(mask, (320, 240))
                        cv2.imshow("Generated Mask", mask_preview)

                        # Lazy load SD generator
                        if generator is None:
                            print("\nLoading Stable Diffusion (one-time, ~30s)...")
                            generator = ClothingGenerator()
                            generator.load_model()

                        # Generate clothing (use CLEAN frame without skeleton!)
                        print(f"\nGenerating '{CLOTHING_PROMPT}' on {clothing_type}...")
                        print("This takes ~10-20 seconds...")

                        inpainted_full, clothing_png = generator.generate_clothing_from_text(
                            frame_clean, mask, CLOTHING_PROMPT
                        )

                        if inpainted_full and clothing_png:
                            # Save outputs
                            timestamp = int(time.time())
                            output_dir = Path(__file__).parent.parent / "generated_meshes" / "realsense_clothing"
                            output_dir.mkdir(parents=True, exist_ok=True)

                            # Save clothing image
                            clothing_path = output_dir / f"clothing_{timestamp}.png"
                            clothing_png.save(str(clothing_path))

                            # Save reference frame (CLEAN, no overlays!)
                            frame_path = output_dir / f"frame_{timestamp}.png"
                            cv2.imwrite(str(frame_path), frame_clean)

                            # Save mask
                            mask_path = output_dir / f"mask_{timestamp}.png"
                            cv2.imwrite(str(mask_path), mask)

                            # Save metadata
                            metadata = {
                                'timestamp': timestamp,
                                'clothing_type': clothing_type,
                                'prompt': CLOTHING_PROMPT,
                                'camera': 'RealSense',
                                'frame_shape': list(frame_shape),
                                'files': {
                                    'clothing': str(clothing_path.name),
                                    'frame': str(frame_path.name),
                                    'mask': str(mask_path.name)
                                }
                            }

                            metadata_path = output_dir / f"metadata_{timestamp}.json"
                            with open(metadata_path, 'w') as f:
                                json.dump(metadata, f, indent=2)

                            print("\n" + "="*70)
                            print("✓ Clothing generated successfully!")
                            print(f"✓ Saved to: {output_dir}")
                            print(f"  - {clothing_path.name}")
                            print(f"  - {frame_path.name}")
                            print(f"  - {mask_path.name}")
                            print(f"  - {metadata_path.name}")
                            print("="*70)

                            # Show result
                            result_preview = cv2.resize(np.array(inpainted_full), (640, 480))
                            result_preview = cv2.cvtColor(result_preview, cv2.COLOR_RGB2BGR)
                            cv2.imshow("Generated Clothing", result_preview)

                        else:
                            print("\n❌ Failed to generate clothing")

                    else:
                        print("\n❌ No body detected")

            # Display info
            cv2.putText(frame, f"Mode: {clothing_type.upper()}", (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.putText(frame, f"Prompt: {CLOTHING_PROMPT}", (10, 60),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

            cv2.imshow("RealSense + SD Integration", frame)

            # Handle keys
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord(' ') and not countdown_active:
                countdown_active = True
                countdown_start_time = time.time()
                print(f"\n⏱ Countdown started ({COUNTDOWN_DURATION}s)...")
            elif key == ord('t'):
                clothing_type = 'tshirt'
                print(f"→ Switched to {clothing_type} mode")
            elif key == ord('d'):
                clothing_type = 'dress'
                print(f"→ Switched to {clothing_type} mode")

    finally:
        print("\n" + "="*70)
        print("Cleaning up...")
        print("="*70)

        pose.close()
        pipeline.stop()
        if generator:
            generator.cleanup()
        cv2.destroyAllWindows()

        print("✓ Test complete!")


if __name__ == "__main__":
    main()
