"""
Strategy 1 Test: 2.5D Depth-Based Clothing Overlay
===================================================
Tests depth estimation + warping for 3D-like clothing effect.
Uses your existing BodyPix segmentation + Stable Diffusion images.

Place this file in: tests/test_depth_3d_clothing.py
Run from project root: python tests/test_depth_3d_clothing.py
"""

import cv2
import numpy as np
import time
import sys
import os
from pathlib import Path
from PIL import Image

# Add parent directory to path to import modules
sys.path.append(str(Path(__file__).parent.parent))

from src.modules.body_tracking import BodySegmenter

# Import depth estimation
from transformers import pipeline
import torch


class DepthBasedClothingRenderer:
    """
    Renders clothing with depth-based 3D effect.
    Takes SD-generated image + estimates depth + warps based on body position.
    """
    
    def __init__(self, clothing_image_path):
        # Load clothing image from SD inpainting
        self.clothing_img = cv2.imread(clothing_image_path, cv2.IMREAD_UNCHANGED)
        
        if self.clothing_img is None:
            raise ValueError(f"Could not load image: {clothing_image_path}")
        
        # Convert BGRA to RGBA for PIL (depth model expects RGB)
        if self.clothing_img.shape[2] == 4:
            # Has alpha channel
            clothing_rgb = cv2.cvtColor(self.clothing_img[:,:,:3], cv2.COLOR_BGR2RGB)
        else:
            # No alpha, create one
            clothing_rgb = cv2.cvtColor(self.clothing_img, cv2.COLOR_BGR2RGB)
            alpha = np.ones((self.clothing_img.shape[0], self.clothing_img.shape[1]), dtype=np.uint8) * 255
            self.clothing_img = cv2.merge([self.clothing_img, alpha])
        
        self.clothing_pil = Image.fromarray(clothing_rgb)
        
        print(f"Loaded clothing image: {self.clothing_img.shape}")
        
        # Initialize depth estimator (using MiDaS - works reliably on Mac/PC)
        print("Loading depth estimation model (this takes ~10 seconds first time)...")
        device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Using device: {device}")
        
        try:
            # Try DPT model first (best quality)
            self.depth_estimator = pipeline(
                "depth-estimation",
                model="Intel/dpt-hybrid-midas",
                device=device
            )
            print("Using Intel DPT-Hybrid model (high quality)")
        except Exception as e:
            print(f"DPT model failed, trying MiDaS small: {e}")
            # Fallback to smaller model
            self.depth_estimator = pipeline(
                "depth-estimation",
                model="Intel/dpt-large",
                device=device
            )
        
        # Pre-compute depth map for clothing image (one-time cost)
        print("Computing depth map for clothing...")
        depth_start = time.time()
        depth_result = self.depth_estimator(self.clothing_pil)
        self.depth_map = np.array(depth_result["depth"])
        
        # Normalize depth to 0-1 range
        self.depth_map = (self.depth_map - self.depth_map.min()) / (self.depth_map.max() - self.depth_map.min())
        
        print(f"Depth computed in {time.time() - depth_start:.2f}s")
        
        # Depth map for visualization (colorized)
        self.depth_viz = self.create_depth_visualization(self.depth_map)
        
        # Performance tracking
        self.warp_time_ms = 0
        self.composite_time_ms = 0
    
    def create_depth_visualization(self, depth_map):
        """Convert depth map to colorized visualization (red=close, blue=far)"""
        # Apply colormap for visualization
        depth_uint8 = (depth_map * 255).astype(np.uint8)
        depth_colored = cv2.applyColorMap(depth_uint8, cv2.COLORMAP_TURBO)
        return depth_colored
    
    def render_on_body(self, video_frame, body_mask, scale=1.0):
        """
        Main rendering function: warps clothing onto body using depth.
        
        Args:
            video_frame: Live webcam frame
            body_mask: Binary mask from BodyPix
            scale: Scale factor (1.0 = same size as clothing image)
        """
        
        # Find bounding box of body mask (where to place clothing)
        warp_start = time.time()
        
        # Find contours in mask to get body region
        contours, _ = cv2.findContours(body_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if len(contours) == 0:
            # No body detected
            return video_frame
        
        # Get largest contour (main body)
        largest_contour = max(contours, key=cv2.contourArea)
        x, y, w, h = cv2.boundingRect(largest_contour)
        
        # Resize clothing to fit body bounding box
        target_w = int(w * scale)
        target_h = int(h * scale)
        
        # Resize clothing and depth map together
        clothing_resized = cv2.resize(self.clothing_img, (target_w, target_h), interpolation=cv2.INTER_LINEAR)
        depth_resized = cv2.resize(self.depth_map, (target_w, target_h), interpolation=cv2.INTER_LINEAR)
        
        # Center clothing on body
        offset_x = x + (w - target_w) // 2
        offset_y = y + (h - target_h) // 2
        
        # Apply depth-based parallax warping (this creates the 3D effect)
        warped_clothing = self.apply_depth_parallax(clothing_resized, depth_resized, parallax_strength=10)
        
        self.warp_time_ms = (time.time() - warp_start) * 1000
        
        # Composite onto video frame
        composite_start = time.time()
        result = self.composite_with_mask(video_frame, warped_clothing, offset_x, offset_y, body_mask)
        self.composite_time_ms = (time.time() - composite_start) * 1000
        
        return result
    
    def apply_depth_parallax(self, clothing, depth, parallax_strength=10):
        """
        Apply depth-based displacement for 3D parallax effect.
        Closer pixels (higher depth) shift more than far pixels.
        """
        h, w = clothing.shape[:2]
        
        # Create output array
        warped = np.zeros_like(clothing)
        
        # For each pixel, shift horizontally based on depth
        for y in range(h):
            for x in range(w):
                # Get depth value (0=far, 1=close)
                d = depth[y, x]
                
                # Calculate horizontal shift (closer = more shift)
                shift_x = int(d * parallax_strength)
                
                # Apply shift (with bounds checking)
                src_x = x - shift_x
                if 0 <= src_x < w:
                    warped[y, x] = clothing[y, src_x]
                else:
                    warped[y, x] = clothing[y, x]  # No shift if out of bounds
        
        return warped
    
    def composite_with_mask(self, video_frame, clothing, offset_x, offset_y, body_mask):
        """
        Alpha blend clothing onto video frame, respecting body mask.
        Only shows clothing where body is present.
        """
        h_cloth, w_cloth = clothing.shape[:2]
        h_frame, w_frame = video_frame.shape[:2]
        
        # Ensure clothing fits in frame
        end_x = min(offset_x + w_cloth, w_frame)
        end_y = min(offset_y + h_cloth, h_frame)
        start_x = max(0, offset_x)
        start_y = max(0, offset_y)
        
        # Crop clothing if needed
        cloth_start_x = start_x - offset_x
        cloth_start_y = start_y - offset_y
        cloth_end_x = cloth_start_x + (end_x - start_x)
        cloth_end_y = cloth_start_y + (end_y - start_y)
        
        clothing_cropped = clothing[cloth_start_y:cloth_end_y, cloth_start_x:cloth_end_x]
        
        # Extract region of interest from video
        roi = video_frame[start_y:end_y, start_x:end_x]
        mask_roi = body_mask[start_y:end_y, start_x:end_x]
        
        # Get alpha channel from clothing
        if clothing_cropped.shape[2] == 4:
            cloth_bgr = clothing_cropped[:,:,:3]
            cloth_alpha = clothing_cropped[:,:,3]
        else:
            cloth_bgr = clothing_cropped
            cloth_alpha = np.ones((clothing_cropped.shape[0], clothing_cropped.shape[1]), dtype=np.uint8) * 255
        
        # Combine alpha with body mask
        combined_mask = cv2.bitwise_and(cloth_alpha, mask_roi)
        combined_mask = combined_mask.astype(float) / 255.0
        
        # Expand to 3 channels
        combined_mask_3ch = np.stack([combined_mask] * 3, axis=2)
        
        # Alpha blend
        blended = (cloth_bgr * combined_mask_3ch + roi * (1 - combined_mask_3ch)).astype(np.uint8)
        
        # Place back into frame
        result = video_frame.copy()
        result[start_y:end_y, start_x:end_x] = blended
        
        return result


def main():
    """
    Main test loop: shows depth-based clothing overlay on live video.
    """
    
    print("="*60)
    print("STRATEGY 1 TEST: 2.5D DEPTH-BASED CLOTHING")
    print("="*60)
    print("\nThis test demonstrates:")
    print("1. Depth estimation from SD clothing images")
    print("2. Real-time warping based on depth")
    print("3. Integration with your BodyPix segmentation")
    print("\nControls:")
    print("  Q - Quit")
    print("  1-5 - Load different clothing images")
    print("  + / - - Adjust clothing scale")
    print("  D - Toggle depth visualization")
    print("="*60)
    
    # Find generated images
    generated_dir = Path("generated_images")
    if not generated_dir.exists():
        print(f"\nError: Directory '{generated_dir}' not found!")
        print("Make sure you've generated clothing images first.")
        return
    
    # Get all PNG files in generated_images/
    image_files = sorted(list(generated_dir.glob("*_clothing.png")))
    
    if len(image_files) == 0:
        print(f"\nError: No clothing images found in '{generated_dir}'")
        print("Run AI generation first to create test images.")
        return
    
    print(f"\nFound {len(image_files)} clothing images:")
    for i, img_path in enumerate(image_files[:5], 1):  # Show first 5
        print(f"  {i}. {img_path.name}")
    
    # Initialize video capture
    print("\nInitializing camera...")
    cap = cv2.VideoCapture(1)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    
    if not cap.isOpened():
        print("Error: Could not open camera")
        return
    
    # Initialize BodyPix segmenter (your existing code)
    print("\nInitializing BodyPix segmenter...")
    segmenter = BodySegmenter(model_type='mobilenet_50')
    segmenter.load_model()
    segmenter.set_preset('torso_and_arms')
    
    # Load first clothing image with depth renderer
    print(f"\nLoading clothing image: {image_files[0].name}")
    try:
        renderer = DepthBasedClothingRenderer(str(image_files[0]))
    except Exception as e:
        print(f"Error loading clothing: {e}")
        return
    
    # Settings
    current_image_idx = 0
    clothing_scale = 1.0
    show_depth = False
    
    # Performance tracking
    fps = 0
    frame_count = 0
    fps_start_time = time.time()
    
    print("\n" + "="*60)
    print("RUNNING TEST - Look at your webcam!")
    print("="*60)
    
    try:
        while True:
            # Capture frame
            ret, frame = cap.read()
            if not ret:
                print("Failed to read frame")
                continue
            
            # Mirror frame (natural for user)
            frame = cv2.flip(frame, 1)
            
            # Get body mask from BodyPix
            body_mask = segmenter.get_mask_for_inpainting(frame, preset='torso_and_arms')
            
            # Render clothing with depth effect
            result = renderer.render_on_body(frame, body_mask, scale=clothing_scale)
            
            # Add performance info overlay
            total_time = renderer.warp_time_ms + renderer.composite_time_ms + segmenter.processing_time_ms
            
            cv2.putText(result, f"FPS: {fps:.1f}", (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            
            cv2.putText(result, f"Total: {total_time:.1f}ms", (10, 70),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            
            cv2.putText(result, f"BodyPix: {segmenter.processing_time_ms:.1f}ms", (10, 100),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)
            
            cv2.putText(result, f"Warp: {renderer.warp_time_ms:.1f}ms", (10, 125),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)
            
            cv2.putText(result, f"Composite: {renderer.composite_time_ms:.1f}ms", (10, 150),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)
            
            cv2.putText(result, f"Scale: {clothing_scale:.2f}", (10, 180),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)
            
            cv2.putText(result, f"Image: {image_files[current_image_idx].name}", (10, 210),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (150, 150, 255), 1)
            
            # Display main result
            cv2.imshow("Depth-Based 3D Clothing Test", result)
            
            # Show depth visualization if enabled
            if show_depth:
                cv2.imshow("Depth Map (Red=Close, Blue=Far)", renderer.depth_viz)
            
            # Calculate FPS
            frame_count += 1
            elapsed = time.time() - fps_start_time
            if elapsed >= 1.0:
                fps = frame_count / elapsed
                frame_count = 0
                fps_start_time = time.time()
            
            # Handle keyboard input
            key = cv2.waitKey(1) & 0xFF
            
            if key == ord('q') or key == ord('Q'):
                break
            
            elif key == ord('+') or key == ord('='):
                # Increase scale
                clothing_scale = min(2.0, clothing_scale + 0.1)
                print(f"Scale: {clothing_scale:.2f}")
            
            elif key == ord('-') or key == ord('_'):
                # Decrease scale
                clothing_scale = max(0.5, clothing_scale - 0.1)
                print(f"Scale: {clothing_scale:.2f}")
            
            elif key == ord('d') or key == ord('D'):
                # Toggle depth visualization
                show_depth = not show_depth
                if not show_depth:
                    cv2.destroyWindow("Depth Map (Red=Close, Blue=Far)")
            
            elif key >= ord('1') and key <= ord('5'):
                # Load different clothing image
                idx = key - ord('1')
                if idx < len(image_files):
                    current_image_idx = idx
                    print(f"\nLoading: {image_files[idx].name}")
                    try:
                        renderer = DepthBasedClothingRenderer(str(image_files[idx]))
                        print("Loaded successfully!")
                    except Exception as e:
                        print(f"Error loading: {e}")
    
    except KeyboardInterrupt:
        print("\nInterrupted by user")
    
    finally:
        # Cleanup
        print("\nCleaning up...")
        cap.release()
        cv2.destroyAllWindows()
        print("\n" + "="*60)
        print("TEST COMPLETE!")
        print("="*60)
        print("\nWhat to look for:")
        print("- Does clothing appear to have depth/3D look?")
        print("- Is FPS acceptable (aim for 20-30)?")
        print("- Does clothing track your body movement?")
        print("- Check depth map - does it capture garment structure?")
        print("\nIf this looks promising, we proceed to full integration!")


if __name__ == "__main__":
    main()