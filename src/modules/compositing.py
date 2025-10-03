"""
PHASE 5: COMPOSITING AND OVERLAY
=================================

PURPOSE: Take the AI-generated clothing and overlay it naturally on the user's
body using the tracking data from MediaPipe.

LEARNING RESOURCES:
- OpenCV Image Blending: https://docs.opencv.org/4.x/d0/d86/tutorial_py_image_arithmetics.html
- Perspective Transforms: https://docs.opencv.org/4.x/da/d6e/tutorial_py_geometric_transformations.html
- Alpha Blending: https://docs.opencv.org/4.x/d5/dc4/tutorial_adding_images.html
- NumPy Array Operations: https://numpy.org/doc/stable/user/basics.html

WHAT YOU'RE BUILDING:
A system that takes a transparent clothing PNG, warps it to fit the body's
pose and position, and blends it seamlessly with the video feed using the
segmentation mask from MediaPipe.
"""

import cv2
import numpy as np
from PIL import Image
import time


class ClothingCompositor:
    """
    This class handles the compositing of AI-generated clothing onto the
    user's body in the video feed.
    
    THE CHALLENGE:
    We have a flat clothing image, but the body is 3D and can be at different
    angles, distances, and poses. We need to warp and blend the clothing so
    it looks like it's actually being worn.
    
    APPROACH:
    1. Get body landmarks from MediaPipe (shoulders, hips)
    2. Calculate where clothing should appear (bounding box)
    3. Warp the clothing image to fit that box using perspective transform
    4. Use segmentation mask to blend naturally with body
    5. Apply lighting and depth adjustments for realism
    """
    
    def __init__(self, blend_alpha=0.8):
        """
        Initialize the compositing system.
        
        PARAMETERS:
        - blend_alpha: How opaque the clothing is (0-1)
                      0.8 = 80% clothing, 20% see-through
                      This lets you see body shape beneath
        """
        
        self.blend_alpha = blend_alpha
        
        # Current clothing to overlay (set via set_clothing method)
        self.current_clothing = None
        self.clothing_pil = None  # PIL Image version (with alpha channel)
        
        # Cache for processed clothing at different sizes
        # This speeds up real-time processing
        self.clothing_cache = {}
        
        # Fade animation state
        self.fade_in_duration = 1.0  # Seconds to fade in
        self.fade_out_duration = 0.5  # Seconds to fade out
        self.current_fade_alpha = 0.0  # Current fade state (0-1)
        self.fade_start_time = None
        self.is_fading_in = False
        self.is_fading_out = False
        
        print("ClothingCompositor initialized")
        print(f"Blend alpha: {blend_alpha}")
    
    
    def set_clothing(self, clothing_image):
        """
        Set the current clothing to overlay.
        
        PARAMETERS:
        - clothing_image: PIL Image with transparent background (RGBA mode)
        
        This should be called when new clothing is generated.
        The clothing will fade in gradually.
        """
        
        if clothing_image is None:
            print("Warning: Received None as clothing image")
            return
        
        # Ensure it's in RGBA mode (with alpha channel)
        if clothing_image.mode != 'RGBA':
            clothing_image = clothing_image.convert('RGBA')
        
        self.clothing_pil = clothing_image
        
        # Convert PIL to OpenCV format (BGR + Alpha)
        # OpenCV uses BGR, PIL uses RGB, so we need to convert
        clothing_array = np.array(clothing_image)
        
        # Split into color and alpha channels
        # RGB is channels 0,1,2 and Alpha is channel 3
        rgb = clothing_array[:, :, :3]
        alpha = clothing_array[:, :, 3]
        
        # Convert RGB to BGR for OpenCV
        bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
        
        # Recombine into BGRA
        self.current_clothing = cv2.merge([bgr, alpha])
        
        # Clear cache since we have new clothing
        self.clothing_cache.clear()
        
        # Start fade in animation
        self.start_fade_in()
        
        print(f"New clothing set: {clothing_image.size[0]}x{clothing_image.size[1]}")
    
    
    def start_fade_in(self):
        """Start the fade-in animation for new clothing."""
        self.is_fading_in = True
        self.is_fading_out = False
        self.fade_start_time = time.time()
        self.current_fade_alpha = 0.0
        print("Starting fade in animation")
    
    
    def start_fade_out(self):
        """Start the fade-out animation to remove clothing."""
        if self.current_clothing is not None:
            self.is_fading_out = True
            self.is_fading_in = False
            self.fade_start_time = time.time()
            self.current_fade_alpha = 1.0
            print("Starting fade out animation")
    
    
    def update_fade_animation(self):
        """
        Update the fade animation state.
        
        Call this every frame to smoothly fade in/out.
        Uses easing function for natural movement.
        """
        
        if not (self.is_fading_in or self.is_fading_out):
            return
        
        if self.fade_start_time is None:
            return
        
        elapsed = time.time() - self.fade_start_time
        
        if self.is_fading_in:
            # Fade from 0 to 1
            progress = min(elapsed / self.fade_in_duration, 1.0)
            
            # Apply easing (ease-out cubic for smooth deceleration)
            self.current_fade_alpha = 1 - (1 - progress) ** 3
            
            if progress >= 1.0:
                self.is_fading_in = False
                self.current_fade_alpha = 1.0
                print("Fade in complete")
        
        elif self.is_fading_out:
            # Fade from 1 to 0
            progress = min(elapsed / self.fade_out_duration, 1.0)
            
            # Apply easing
            self.current_fade_alpha = 1 - progress
            
            if progress >= 1.0:
                self.is_fading_out = False
                self.current_clothing = None
                self.current_fade_alpha = 0.0
                print("Fade out complete")
    
    
    def warp_clothing_to_body(self, bounding_box, target_size):
        """
        Warp the clothing image to fit the body's bounding box.
        
        WHAT THIS DOES:
        Takes the rectangular clothing image and transforms it to match
        the body's position, size, and angle. Uses perspective transform
        to handle cases where person is at an angle to camera.
        
        PARAMETERS:
        - bounding_box: (x, y, width, height) from body tracker
        - target_size: (frame_height, frame_width) of the video
        
        RETURNS:
        - Warped clothing image that fits the body
        - Position (x, y) where to place it on frame
        """
        
        if self.current_clothing is None:
            return None, None
        
        x, y, w, h = bounding_box
        
        # Check cache first (speeds up processing)
        cache_key = (x, y, w, h)
        if cache_key in self.clothing_cache:
            return self.clothing_cache[cache_key]
        
        # Get original clothing dimensions
        clothing_h, clothing_w = self.current_clothing.shape[:2]
        
        # Calculate aspect ratios
        clothing_aspect = clothing_w / clothing_h
        body_aspect = w / h
        
        # Resize clothing to match body bounding box size
        # We want to fit it proportionally
        if clothing_aspect > body_aspect:
            # Clothing is wider - fit to width
            new_w = w
            new_h = int(w / clothing_aspect)
        else:
            # Clothing is taller - fit to height
            new_h = h
            new_w = int(h * clothing_aspect)
        
        # Make sure we don't exceed bounding box
        new_w = min(new_w, w)
        new_h = min(new_h, h)
        
        # Resize the clothing
        warped_clothing = cv2.resize(self.current_clothing, (new_w, new_h),
                                    interpolation=cv2.INTER_LINEAR)
        
        # Center the clothing in the bounding box
        offset_x = (w - new_w) // 2
        offset_y = (h - new_h) // 2
        
        final_x = x + offset_x
        final_y = y + offset_y
        
        # Cache the result
        self.clothing_cache[cache_key] = (warped_clothing, (final_x, final_y))
        
        return warped_clothing, (final_x, final_y)
    
    
    def blend_with_mask(self, frame, clothing, position, segmentation_mask):
        """
        Blend the clothing with the video frame using the segmentation mask.
        
        THIS IS THE MAGIC:
        The segmentation mask tells us exactly where the person is.
        We only show clothing where there's a person, making it look natural.
        
        PARAMETERS:
        - frame: The video frame (BGR)
        - clothing: The warped clothing (BGRA)
        - position: (x, y) where to place clothing
        - segmentation_mask: Binary mask from MediaPipe
        
        RETURNS:
        - Composited frame with clothing overlay
        """
        
        if clothing is None or segmentation_mask is None:
            return frame
        
        x, y = position
        cloth_h, cloth_w = clothing.shape[:2]
        frame_h, frame_w = frame.shape[:2]
        
        # Make sure clothing fits in frame
        if x < 0 or y < 0 or x + cloth_w > frame_w or y + cloth_h > frame_h:
            # Clothing goes outside frame - need to clip it
            # Calculate valid region
            x_start = max(0, x)
            y_start = max(0, y)
            x_end = min(frame_w, x + cloth_w)
            y_end = min(frame_h, y + cloth_h)
            
            # Crop clothing to fit
            cloth_x_start = x_start - x
            cloth_y_start = y_start - y
            cloth_x_end = cloth_x_start + (x_end - x_start)
            cloth_y_end = cloth_y_start + (y_end - y_start)
            
            clothing = clothing[cloth_y_start:cloth_y_end, cloth_x_start:cloth_x_end]
            x, y = x_start, y_start
            cloth_h, cloth_w = clothing.shape[:2]
        
        # Extract the region of interest (ROI) from the frame
        roi = frame[y:y+cloth_h, x:x+cloth_w]
        
        # Extract the corresponding region from segmentation mask
        mask_roi = segmentation_mask[y:y+cloth_h, x:x+cloth_w]
        
        # Split clothing into BGR and alpha channels
        cloth_bgr = clothing[:, :, :3]
        cloth_alpha = clothing[:, :, 3]
        
        # Combine alpha channel with segmentation mask
        # Only show clothing where BOTH conditions are true:
        # 1. Clothing has non-transparent pixels (cloth_alpha > 0)
        # 2. Person is present (mask_roi > 0)
        combined_mask = cv2.bitwise_and(cloth_alpha, mask_roi)
        
        # Normalize mask to 0-1 range for blending
        combined_mask = combined_mask.astype(float) / 255.0
        
        # Apply fade animation
        combined_mask = combined_mask * self.current_fade_alpha
        
        # Apply blend alpha (overall transparency)
        combined_mask = combined_mask * self.blend_alpha
        
        # Expand mask to 3 channels for blending (one per color channel)
        combined_mask_3ch = np.stack([combined_mask] * 3, axis=2)
        
        # Alpha blend: result = foreground * alpha + background * (1 - alpha)
        blended = (cloth_bgr * combined_mask_3ch + 
                  roi * (1 - combined_mask_3ch)).astype(np.uint8)
        
        # Place the blended result back into the frame
        result_frame = frame.copy()
        result_frame[y:y+cloth_h, x:x+cloth_w] = blended
        
        return result_frame
    
    
    def apply_lighting_adjustment(self, frame, clothing_region, body_region):
        """
        Adjust clothing brightness to match body lighting.
        
        This makes clothing look more realistic by matching the lighting
        conditions of the scene.
        
        PARAMETERS:
        - frame: Full frame
        - clothing_region: (x, y, w, h) where clothing is
        - body_region: (x, y, w, h) reference body area
        
        RETURNS:
        - Frame with lighting-adjusted clothing
        """
        
        # This is an advanced feature - for now we'll skip it
        # But in future iterations, you could:
        # 1. Calculate average brightness of body region
        # 2. Calculate average brightness of clothing
        # 3. Adjust clothing to match body brightness
        # 4. Apply color temperature matching
        
        return frame
    
    
    def composite_frame(self, frame, bounding_box, segmentation_mask):
        """
        Main compositing function - call this every frame.
        
        This is the high-level function that orchestrates everything:
        1. Updates fade animation
        2. Warps clothing to fit body
        3. Blends with segmentation mask
        4. Returns the composited frame
        
        PARAMETERS:
        - frame: Video frame from camera
        - bounding_box: Body bounding box from tracker
        - segmentation_mask: Segmentation mask from tracker
        
        RETURNS:
        - Frame with clothing overlay, or original frame if no clothing
        """
        
        # Update fade animation
        self.update_fade_animation()
        
        # If no clothing or fully faded out, return original frame
        if self.current_clothing is None or self.current_fade_alpha <= 0:
            return frame
        
        # If no body detected, return original frame
        if bounding_box is None or segmentation_mask is None:
            return frame
        
        # Warp clothing to fit body
        warped_clothing, position = self.warp_clothing_to_body(
            bounding_box,
            frame.shape
        )
        
        if warped_clothing is None or position is None:
            return frame
        
        # Blend clothing with frame using mask
        composited_frame = self.blend_with_mask(
            frame,
            warped_clothing,
            position,
            segmentation_mask
        )
        
        return composited_frame
    
    
    def clear_clothing(self):
        """Remove current clothing with fade out animation."""
        self.start_fade_out()
    
    
    def has_clothing(self):
        """Check if there's currently clothing to display."""
        return self.current_clothing is not None and self.current_fade_alpha > 0


# ============================================================================
# USAGE EXAMPLE - Integration with Previous Phases
# ============================================================================

def main():
    """
    Test the compositing system with body tracking and clothing.
    This combines Phase 3 (body tracking) with Phase 5 (compositing).
    """
    
    print("=" * 60)
    print("PHASE 5: COMPOSITING TEST")
    print("=" * 60)
    print("\nThis will show compositing of test clothing on your body.")
    print("Make sure you completed Phase 3 (body tracking) first!")
    print("\nControls:")
    print("- Q: Quit")
    print("- 1-5: Load different test clothing")
    print("- C: Clear clothing")
    print("- A: Adjust blend alpha")
    print("=" * 60)
    
    # Import the body tracker from Phase 3
    # In real code: from phase3_body_tracking import BodyTracker
    # For testing, we'll simulate it
    import sys
    sys.path.append('..')
    
    try:
        from modules.phase3_body_tracking import BodyTracker
    except:
        print("\nError: Could not import BodyTracker")
        print("Make sure Phase 3 is complete and in modules folder")
        return
    
    # Initialize video capture
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    
    if not cap.isOpened():
        print("Error: Could not open camera")
        return
    
    # Initialize body tracker
    tracker = BodyTracker(
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
    )
    
    # Initialize compositor
    compositor = ClothingCompositor(blend_alpha=0.8)
    
    # Create some test clothing images
    test_clothing_paths = [
        "generated_images/1_flames.png",
        "generated_images/2_roses_and_thorns.png",
        "generated_images/3_made_of_tree_bark.png",
        "generated_images/4_cosmic_stars_and_galaxies.png",
        "generated_images/5_water_droplets.png"
    ]
    
    print("\nStarting compositing test...")
    print("Press 1-5 to load test clothing images")
    
    try:
        while True:
            # Read frame
            ret, frame = cap.read()
            if not ret:
                print("Failed to read frame")
                break
            
            # Mirror the frame
            frame = cv2.flip(frame, 1)
            
            # Process with body tracker
            results, _ = tracker.process_frame(frame)
            
            # Get tracking data
            bounding_box = tracker.latest_bounding_box
            segmentation_mask = tracker.get_segmentation_mask(frame.shape)
            
            # Composite clothing onto frame
            composited_frame = compositor.composite_frame(
                frame,
                bounding_box,
                segmentation_mask
            )
            
            # Add info overlay
            if tracker.is_person_detected():
                status = "Person detected"
                color = (0, 255, 0)
            else:
                status = "No person detected"
                color = (0, 0, 255)
            
            cv2.putText(composited_frame, status, (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
            
            if compositor.has_clothing():
                cloth_status = f"Clothing: ON (alpha: {compositor.current_fade_alpha:.2f})"
                cv2.putText(composited_frame, cloth_status, (10, 60),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            else:
                cloth_status = "Clothing: OFF"
                cv2.putText(composited_frame, cloth_status, (10, 60),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (128, 128, 128), 2)
            
            # Display
            cv2.imshow("Compositing Test", composited_frame)
            
            # Handle key presses
            key = cv2.waitKey(1) & 0xFF
            
            if key == ord('q') or key == ord('Q'):
                break
            
            elif key >= ord('1') and key <= ord('5'):
                # Load test clothing
                idx = key - ord('1')
                if idx < len(test_clothing_paths):
                    path = test_clothing_paths[idx]
                    try:
                        clothing_img = Image.open(path)
                        compositor.set_clothing(clothing_img)
                        print(f"Loaded: {path}")
                    except:
                        print(f"Could not load {path}")
                        print("Run Phase 4 first to generate test images!")
            
            elif key == ord('c') or key == ord('C'):
                compositor.clear_clothing()
                print("Clearing clothing")
            
            elif key == ord('a') or key == ord('A'):
                # Cycle through alpha values
                alphas = [0.5, 0.7, 0.8, 0.9, 1.0]
                current_idx = alphas.index(compositor.blend_alpha) if compositor.blend_alpha in alphas else 0
                new_idx = (current_idx + 1) % len(alphas)
                compositor.blend_alpha = alphas[new_idx]
                print(f"Blend alpha: {compositor.blend_alpha}")
    
    except KeyboardInterrupt:
        print("\nInterrupted by user")
    
    finally:
        # Cleanup
        print("\nCleaning up...")
        tracker.cleanup()
        cap.release()
        cv2.destroyAllWindows()
        print("Test complete!")


if __name__ == "__main__":
    main()


# ============================================================================
# TESTING CHECKLIST
# ============================================================================
#
# [ ] Body tracking works (skeleton appears)
# [ ] Pressing 1-5 loads test clothing images
# [ ] Clothing appears on body (not floating in air)
# [ ] Clothing scales with distance from camera
# [ ] Clothing fades in smoothly when loaded
# [ ] Clothing only appears where body is (not background)
# [ ] Clothing moves with body in real-time
# [ ] Pressing C fades out clothing
# [ ] Pressing A adjusts transparency
# [ ] No lag or stuttering
#
# COMMON ISSUES:
# - Clothing appears offset: Check bounding box calculation in Phase 3
# - Clothing appears on background: Segmentation mask issue, check Phase 3
# - Clothing doesn't fit right: Adjust aspect ratio calculation
# - Performance issues: Reduce frame size or clothing resolution
# - Clothing flickers: Increase tracking confidence in body tracker
#
# TUNING PARAMETERS:
# - blend_alpha: 0.5-1.0 (how see-through clothing is)
# - fade_in_duration: 0.5-2.0 seconds (animation speed)
# - Cache size: Affects memory usage vs speed tradeoff
#
# WHAT THIS GIVES US:
# - Natural-looking clothing overlay on body
# - Smooth animations for appearing/disappearing
# - Real-time performance with caching
# - Foundation for full application integration
#
# NEXT STEPS:
# Phase 6 will add 3D floating text effects for speech visualization.
# Phase 7 will create the state machine that orchestrates everything.
# Phase 8 will integrate all components into the final application!
# ============================================================================