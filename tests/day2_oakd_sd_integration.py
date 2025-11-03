"""
Day 2: OAK-D + Stable Diffusion Integration
Generates AI clothing for body segments captured from OAK-D Pro

Pipeline:
1. Capture body segments from OAK-D (with countdown)
2. Create mask from torso quad landmarks
3. Generate clothing with Stable Diffusion
4. Save clothing image + reference data for real-time overlay

Press SPACE to start countdown and generate clothing
Press 't' for t-shirt, 'd' for dress
Press 'q' to quit
"""

import sys
from pathlib import Path
import time
import json

# Add paths
blazepose_path = Path(__file__).parent.parent / "external" / "depthai_blazepose"
sys.path.insert(0, str(blazepose_path))

# Add src to path for AI generation
src_path = Path(__file__).parent.parent / "src"
sys.path.insert(0, str(src_path))

import cv2
import numpy as np
from BlazeposeDepthaiEdge import BlazeposeDepthai
from BlazeposeRenderer import BlazeposeRenderer
from modules.ai_generation import ClothingGenerator

# Landmark indices (from day1)
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
    """Creates 2D masks from 3D body landmarks for SD inpainting"""

    @staticmethod
    def landmarks_world_to_2d(landmarks_world, frame_shape, measured_depth=None):
        """
        Convert world landmarks (metric 3D) to 2D pixel coordinates

        Args:
            landmarks_world: 33x3 array (x, y, z in meters, origin at mid-hips)
            frame_shape: (height, width, channels)
            measured_depth: Optional measured depth from OAK-D (x, y, z in mm)

        Returns:
            landmarks_2d: 33x2 array (x, y in pixels)
        """
        h, w = frame_shape[:2]

        # Convert world coordinates to screen space
        # World coords: x=left/right, y=up/down, z=forward/back
        # We need to project 3D → 2D for the camera view

        landmarks_2d = np.zeros((len(landmarks_world), 2), dtype=np.int32)

        for i, lm_world in enumerate(landmarks_world):
            # Simple orthographic projection (ignore Z for now)
            # Center the coordinates and scale to frame
            x_world, y_world, z_world = lm_world

            # Scale factor (approximate - adjust based on your camera setup)
            # World coords are in meters relative to mid-hips
            # Typical person width ~0.5m, height ~1.7m
            scale_x = w / 1.5  # Assume 1.5m width captures full body
            scale_y = h / 2.0  # Assume 2m height captures full body

            # Convert to pixel coordinates (origin at top-left)
            px = int(w / 2 + x_world * scale_x)
            py = int(h / 2 - y_world * scale_y)  # Flip Y (screen coords go down)

            # Clamp to frame bounds
            px = max(0, min(w - 1, px))
            py = max(0, min(h - 1, py))

            landmarks_2d[i] = [px, py]

        return landmarks_2d

    @staticmethod
    def create_torso_mask(landmarks_2d, frame_shape, padding=20):
        """
        Create binary mask for torso region

        Args:
            landmarks_2d: 33x2 array (pixel coordinates)
            frame_shape: (height, width, channels)
            padding: Pixels to expand mask beyond landmarks

        Returns:
            mask: Binary mask (0 or 255)
        """
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

        # Torso
        left_shoulder = landmarks_2d[LandmarkIndex.LEFT_SHOULDER]
        right_shoulder = landmarks_2d[LandmarkIndex.RIGHT_SHOULDER]
        left_hip = landmarks_2d[LandmarkIndex.LEFT_HIP]
        right_hip = landmarks_2d[LandmarkIndex.RIGHT_HIP]

        # Arms
        left_elbow = landmarks_2d[LandmarkIndex.LEFT_ELBOW]
        right_elbow = landmarks_2d[LandmarkIndex.RIGHT_ELBOW]

        # Create extended polygon including arms
        # Order: left_elbow → left_shoulder → right_shoulder → right_elbow → right_hip → left_hip
        pts = np.array([
            left_elbow,
            left_shoulder,
            right_shoulder,
            right_elbow,
            right_hip,
            left_hip
        ], dtype=np.int32)

        # Expand
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

        # All points
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
    print("Day 2: OAK-D + Stable Diffusion Integration")
    print("="*70)
    print("\nThis generates AI clothing for your body using OAK-D depth + SD")
    print("\nControls:")
    print("  SPACE - Start 5-second countdown and generate clothing")
    print("  't'   - T-shirt mode")
    print("  'd'   - Dress mode")
    print("  'q'   - Quit")
    print("\nDefault prompt: 'colorful flames pattern'")
    print("(You can modify this in the code)")
    print("="*70)

    # Initialize OAK-D
    print("\nInitializing OAK-D Pro...")
    tracker = BlazeposeDepthai(
        input_src='rgb',
        lm_model='lite',
        xyz=True,
        smoothing=True,
        internal_fps=30,
        internal_frame_height=640,
        stats=False,
        trace=False
    )
    renderer = BlazeposeRenderer(tracker, show_3d=None, output=None)
    print("✓ OAK-D initialized")

    # Initialize SD (lazy load on first use)
    print("\nStable Diffusion will load on first generation")
    generator = None

    mask_gen = MaskGenerator()
    clothing_type = 'tshirt'
    countdown_active = False
    countdown_start_time = None
    COUNTDOWN_DURATION = 5

    # Clothing prompt (you can change this)
    CLOTHING_PROMPT = "colorful flames pattern"

    while True:
        frame, body = tracker.next_frame()
        if frame is None:
            break

        # CRITICAL: Save clean frame BEFORE any drawing
        frame_clean = frame.copy()

        # Draw skeleton for display only
        frame = renderer.draw(frame, body)

        # Handle countdown
        if countdown_active:
            elapsed = time.time() - countdown_start_time
            seconds_left = max(0, COUNTDOWN_DURATION - int(elapsed))
            frame = draw_countdown(frame, seconds_left)

            if elapsed >= COUNTDOWN_DURATION:
                countdown_active = False

                if body and hasattr(body, 'landmarks_world'):
                    print("\n" + "="*70)
                    print("Captured body! Generating clothing...")
                    print("="*70)

                    # Convert world landmarks to 2D pixels (use CLEAN frame shape)
                    frame_shape = frame_clean.shape
                    measured_depth = body.xyz if hasattr(body, 'xyz') else None

                    landmarks_2d = mask_gen.landmarks_world_to_2d(
                        body.landmarks_world,
                        frame_shape,
                        measured_depth
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

                    # Generate clothing (use CLEAN frame without skeleton/countdown!)
                    print(f"\nGenerating '{CLOTHING_PROMPT}' on {clothing_type}...")
                    print("This takes ~10-20 seconds...")

                    inpainted_full, clothing_png = generator.generate_clothing_from_text(
                        frame_clean, mask, CLOTHING_PROMPT
                    )

                    if inpainted_full and clothing_png:
                        # Save outputs
                        timestamp = int(time.time())
                        output_dir = Path(__file__).parent.parent / "generated_meshes" / "oakd_clothing"
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
                            'landmarks_world': body.landmarks_world.tolist(),
                            'landmarks_normalized': body.landmarks.tolist(),  # CRITICAL: normalized [0,1] coords
                            'measured_depth': measured_depth.tolist() if measured_depth is not None else None,
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

        cv2.imshow("OAK-D + SD Integration", frame)

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

    print("\n" + "="*70)
    print("Day 2 complete!")
    print("="*70)

    renderer.exit()
    tracker.exit()
    if generator:
        generator.cleanup()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
