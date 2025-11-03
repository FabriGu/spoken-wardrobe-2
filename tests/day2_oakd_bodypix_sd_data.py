"""
Day 2 FINAL: OAK-D + BodyPix + Stable Diffusion with Reference Data

This is the CORRECT Day 2 implementation that:
1. Uses CLEAN frames (no skeleton/text overlays)
2. Uses BodyPix for precise 24-part body segmentation
3. Saves BOTH world AND normalized landmarks for Day 3 warping
4. Saves body parts list for reference

Pipeline:
1. OAK-D captures RGB frame + BlazePose landmarks
2. BodyPix segments body parts (CPU, one-time during countdown)
3. Create mask from selected body parts
4. Stable Diffusion generates clothing
5. Save clothing + reference data for Day 3 warping overlay

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

src_path = Path(__file__).parent.parent / "src"
sys.path.insert(0, str(src_path))

import cv2
import numpy as np
from BlazeposeDepthaiEdge import BlazeposeDepthai
from BlazeposeRenderer import BlazeposeRenderer
from modules.ai_generation import ClothingGenerator

# Import BodyPix
from tf_bodypix.api import download_model, load_model, BodyPixModelPaths


class BodyPartSelector:
    """Selects body parts for different clothing types"""

    # BodyPix body part names (24 parts)
    TSHIRT_PARTS = [
        'torso_front', 'torso_back',
        'left_upper_arm_front', 'left_upper_arm_back',
        'left_lower_arm_front', 'left_lower_arm_back',
        'right_upper_arm_front', 'right_upper_arm_back',
        'right_lower_arm_front', 'right_lower_arm_back',
    ]

    DRESS_PARTS = TSHIRT_PARTS + [
        'left_upper_leg_front', 'left_upper_leg_back',
        'left_lower_leg_front', 'left_lower_leg_back',
        'right_upper_leg_front', 'right_upper_leg_back',
        'right_lower_leg_front', 'right_lower_leg_back',
    ]

    TORSO_ONLY_PARTS = [
        'torso_front', 'torso_back'
    ]

    @classmethod
    def get_parts_for_clothing(cls, clothing_type):
        """Get body part names for clothing type"""
        if clothing_type == 'tshirt':
            return cls.TSHIRT_PARTS
        elif clothing_type == 'dress':
            return cls.DRESS_PARTS
        elif clothing_type == 'torso':
            return cls.TORSO_ONLY_PARTS
        else:
            return cls.TSHIRT_PARTS  # Default


def draw_countdown(frame, seconds_left):
    """Draw countdown overlay on frame"""
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
    print("Day 2 FINAL: OAK-D + BodyPix + SD with Reference Data")
    print("="*70)
    print("\nFeatures:")
    print("  ✓ Uses CLEAN frame (no skeleton/text overlay)")
    print("  ✓ Uses BodyPix for precise 24-part body segmentation")
    print("  ✓ Saves normalized landmarks for Day 3 warping")
    print("\nControls:")
    print("  SPACE - Start 5-second countdown and generate clothing")
    print("  't'   - T-shirt mode (torso + arms)")
    print("  'd'   - Dress mode (torso + arms + legs)")
    print("  'q'   - Quit")
    print("\nDefault prompt: 'colorful flames pattern'")
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

    # Initialize BodyPix (load once)
    print("\nLoading BodyPix model...")
    bodypix_model = load_model(download_model(
        BodyPixModelPaths.MOBILENET_FLOAT_75_STRIDE_16
    ))
    print("✓ BodyPix loaded")

    # Initialize SD (lazy load)
    print("\nStable Diffusion will load on first generation")
    generator = None

    clothing_type = 'tshirt'
    countdown_active = False
    countdown_start_time = None
    COUNTDOWN_DURATION = 5

    # Clothing prompt (modify as needed)
    CLOTHING_PROMPT = "colorful flames pattern"

    while True:
        # Get CLEAN frame from OAK-D (before any drawing)
        frame, body = tracker.next_frame()
        if frame is None:
            break

        # CRITICAL: Save clean frame BEFORE drawing anything
        clean_frame = frame.copy()

        # NOW draw skeleton and UI (only for display, not for SD)
        display_frame = renderer.draw(frame, body)

        # Handle countdown
        if countdown_active:
            elapsed = time.time() - countdown_start_time
            seconds_left = max(0, COUNTDOWN_DURATION - int(elapsed))
            display_frame = draw_countdown(display_frame, seconds_left)

            if elapsed >= COUNTDOWN_DURATION:
                countdown_active = False

                if body and hasattr(body, 'landmarks_world') and hasattr(body, 'landmarks'):
                    print("\n" + "="*70)
                    print("Captured! Running BodyPix segmentation...")
                    print("="*70)

                    # Run BodyPix on CLEAN frame
                    print("Running BodyPix (this may take ~500ms)...")
                    bodypix_start = time.time()

                    # BodyPix expects RGB (OAK-D gives BGR)
                    clean_frame_rgb = cv2.cvtColor(clean_frame, cv2.COLOR_BGR2RGB)
                    result = bodypix_model.predict_single(clean_frame_rgb)

                    # Get binary person mask
                    person_mask = result.get_mask(threshold=0.75)

                    # Get body part names for this clothing type
                    part_names = BodyPartSelector.get_parts_for_clothing(clothing_type)

                    # Get mask for selected body parts
                    body_part_mask = result.get_part_mask(person_mask, part_names=part_names)

                    # Convert to numpy and ensure 2D
                    if hasattr(body_part_mask, 'numpy'):
                        body_part_mask = body_part_mask.numpy()
                    body_part_mask = np.squeeze(body_part_mask)

                    # Convert boolean to uint8 (0 or 255)
                    mask = (body_part_mask > 0).astype(np.uint8) * 255

                    bodypix_time = time.time() - bodypix_start
                    print(f"✓ BodyPix complete in {bodypix_time:.2f}s")

                    # Show mask preview
                    mask_preview = cv2.resize(mask, (320, 240))
                    cv2.imshow("Generated Mask (BodyPix)", mask_preview)

                    # Show colored body parts visualization
                    colored_mask = result.get_colored_part_mask(person_mask)
                    if hasattr(colored_mask, 'numpy'):
                        colored_mask = colored_mask.numpy()
                    colored_mask = colored_mask.astype(np.uint8)
                    colored_preview = cv2.resize(colored_mask, (320, 240))
                    cv2.imshow("Body Parts (BodyPix)", colored_preview)

                    # Lazy load SD generator
                    if generator is None:
                        print("\nLoading Stable Diffusion (one-time, ~30s)...")
                        generator = ClothingGenerator()
                        generator.load_model()

                    # Generate clothing
                    print(f"\nGenerating '{CLOTHING_PROMPT}' on {clothing_type}...")
                    print("This takes ~10-20 seconds...")

                    # CRITICAL: Use CLEAN frame (no skeleton/text)
                    inpainted_full, clothing_png = generator.generate_clothing_from_text(
                        clean_frame,  # Use clean_frame, not display_frame!
                        mask,
                        CLOTHING_PROMPT
                    )

                    if inpainted_full and clothing_png:
                        # Save outputs
                        timestamp = int(time.time())
                        output_dir = Path(__file__).parent.parent / "generated_meshes" / "oakd_clothing"
                        output_dir.mkdir(parents=True, exist_ok=True)

                        # Save clothing image
                        clothing_path = output_dir / f"clothing_{timestamp}.png"
                        clothing_png.save(str(clothing_path))

                        # Save CLEAN reference frame
                        frame_path = output_dir / f"frame_{timestamp}.png"
                        cv2.imwrite(str(frame_path), clean_frame)

                        # Save mask
                        mask_path = output_dir / f"mask_{timestamp}.png"
                        cv2.imwrite(str(mask_path), mask)

                        # Save metadata with BOTH world and normalized landmarks
                        metadata = {
                            'timestamp': timestamp,
                            'clothing_type': clothing_type,
                            'prompt': CLOTHING_PROMPT,
                            'body_parts': part_names,  # Which BodyPix parts were used
                            'landmarks_world': body.landmarks_world.tolist(),  # 3D world coords
                            'landmarks_normalized': body.landmarks.tolist(),   # CRITICAL: normalized [0,1] for warping!
                            'measured_depth': body.xyz.tolist() if hasattr(body, 'xyz') and body.xyz is not None else None,
                            'frame_shape': list(clean_frame.shape),
                            'bodypix_processing_time': bodypix_time,
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
                        print(f"\nMetadata includes:")
                        print(f"  ✓ Normalized landmarks for warping")
                        print(f"  ✓ Body parts: {', '.join(part_names[:5])}...")
                        print("="*70)

                        # Show result
                        result_preview = cv2.resize(np.array(inpainted_full), (640, 480))
                        result_preview = cv2.cvtColor(result_preview, cv2.COLOR_RGB2BGR)
                        cv2.imshow("Generated Clothing", result_preview)

                    else:
                        print("\n❌ Failed to generate clothing")

                else:
                    print("\n❌ No body detected or missing landmark data")

        # Display info on display_frame (not clean_frame)
        cv2.putText(display_frame, f"Mode: {clothing_type.upper()}", (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.putText(display_frame, f"Prompt: {CLOTHING_PROMPT}", (10, 60),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        cv2.putText(display_frame, "SPACE: Generate | T: T-shirt | D: Dress | Q: Quit", (10, 90),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)

        cv2.imshow("OAK-D + BlazePose + BodyPix", display_frame)

        # Handle keyboard
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord(' '):
            if not countdown_active:
                countdown_active = True
                countdown_start_time = time.time()
                print("\n🎬 Countdown started! Get in position...")
        elif key == ord('t'):
            clothing_type = 'tshirt'
            print(f"\n👕 Mode: T-SHIRT (torso + arms)")
        elif key == ord('d'):
            clothing_type = 'dress'
            print(f"\n👗 Mode: DRESS (torso + arms + legs)")

    # Cleanup
    tracker.exit()
    cv2.destroyAllWindows()
    print("\n✓ Exited")


if __name__ == "__main__":
    main()
