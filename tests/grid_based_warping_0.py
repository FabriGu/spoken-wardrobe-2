#!/usr/bin/env python3
"""
Grid-Based Clothing Warping Test
=================================
Tests the recommended grid-based warping approach for real-time pose-aware
clothing overlay without VIBE/SMPL complexity.

Compatible with: NumPy 1.23.5, Python 3.11, Mac M1 CPU
Expected Performance: 30-40 FPS at 720p

Usage:
    python tests/test_grid_based_warping.py
"""

import cv2
import numpy as np
from scipy.spatial.distance import cdist
from PIL import Image
import mediapipe as mp
import time
from pathlib import Path


class ClothingWarper:
    """
    Grid-based clothing warper using inverse distance weighting.
    This is the core warping engine that makes clothing follow body movement.
    """
    
    def __init__(self, grid_size=12):
        """
        Args:
            grid_size: Grid resolution (12 = good balance, 10 = faster, 16 = better quality)
        """
        self.grid_size = grid_size
        self.cached_maps = None
        self.cache_valid = False
        
        # Performance tracking
        self.map_compute_ms = 0
        self.warp_apply_ms = 0
    
    def compute_grid_warp_maps(self, img_shape, src_keypoints, dst_keypoints):
        """
        Create cv2.remap maps using grid-based inverse distance weighting.
        
        This is where the magic happens - we create a deformation field that
        smoothly warps the clothing from reference pose to current pose.
        
        Args:
            img_shape: (height, width) of image
            src_keypoints: Nx2 array of reference pose positions [[x,y], ...]
            dst_keypoints: Nx2 array of current pose positions
            
        Returns:
            map_x, map_y: Float32 arrays for cv2.remap
        """
        start = time.time()
        
        h, w = img_shape[:2]
        
        # Create evaluation grid (coarse grid for speed)
        grid_h = self.grid_size
        grid_w = self.grid_size
        
        grid_y = np.linspace(0, h-1, grid_h)
        grid_x = np.linspace(0, w-1, grid_w)
        
        # Compute displacement at each grid node
        grid_map_x = np.zeros((grid_h, grid_w), dtype=np.float32)
        grid_map_y = np.zeros((grid_h, grid_w), dtype=np.float32)
        
        for i, y in enumerate(grid_y):
            for j, x in enumerate(grid_x):
                # Inverse distance weighted displacement
                dx, dy = self._idw_displacement(
                    x, y, src_keypoints, dst_keypoints
                )
                grid_map_x[i, j] = x + dx
                grid_map_y[i, j] = y + dy
        
        # Interpolate to full resolution (this is fast with OpenCV)
        map_x = cv2.resize(grid_map_x, (w, h), interpolation=cv2.INTER_CUBIC)
        map_y = cv2.resize(grid_map_y, (w, h), interpolation=cv2.INTER_CUBIC)
        
        self.map_compute_ms = (time.time() - start) * 1000
        
        return map_x, map_y
    
    def _idw_displacement(self, x, y, src_pts, dst_pts, power=2):
        """
        Inverse distance weighted interpolation of displacements.
        
        Closer keypoints have more influence on the local deformation.
        This creates smooth, natural-looking warping.
        """
        # Compute distances to all keypoints
        point = np.array([[x, y]])
        distances = cdist(point, src_pts[:, :2])[0]
        
        # Avoid division by zero for points exactly on keypoints
        distances = np.maximum(distances, 1.0)
        
        # Inverse distance weights (closer = more influence)
        weights = 1.0 / (distances ** power)
        weights /= weights.sum()
        
        # Weighted displacement
        displacements = dst_pts[:, :2] - src_pts[:, :2]
        weighted_disp = (weights[:, None] * displacements).sum(axis=0)
        
        return weighted_disp[0], weighted_disp[1]
    
    def warp_clothing(self, clothing_img, map_x, map_y):
        """Apply warping using precomputed maps"""
        start = time.time()
        
        warped = cv2.remap(
            clothing_img,
            map_x.astype(np.float32),
            map_y.astype(np.float32),
            cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=(0, 0, 0, 0)
        )
        
        self.warp_apply_ms = (time.time() - start) * 1000
        
        return warped


class PoseTracker:
    """
    MediaPipe pose detection with clothing keypoint selection.
    Extracts the 8-12 keypoints needed for upper body clothing.
    """
    
    # MediaPipe keypoint indices for upper body clothing
    SHIRT_KEYPOINTS = {
        'left_shoulder': 11,
        'right_shoulder': 12,
        'left_elbow': 13,
        'right_elbow': 14,
        'left_wrist': 15,
        'right_wrist': 16,
        'left_hip': 23,
        'right_hip': 24,
    }
    
    def __init__(self):
        self.mp_pose = mp.solutions.pose
        self.pose = self.mp_pose.Pose(
            static_image_mode=False,
            model_complexity=0,  # Use lite model for speed (0=lite, 1=full, 2=heavy)
            smooth_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        
        self.mp_drawing = mp.solutions.drawing_utils
        self.processing_time_ms = 0
        
        self.frame_height = 720
        self.frame_width = 1280
    
    def detect_pose(self, frame):
        """Run MediaPipe pose detection"""
        start = time.time()
        
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.pose.process(rgb)
        
        self.processing_time_ms = (time.time() - start) * 1000
        
        return results
    
    def select_clothing_keypoints(self, pose_landmarks, frame_shape, visibility_threshold=0.5):
        """
        Extract clothing-relevant keypoints from MediaPipe's 33 landmarks.
        
        Returns: Nx2 array with [x, y] in pixel coordinates
        """
        if pose_landmarks is None:
            return None
        
        h, w = frame_shape[:2]
        self.frame_height, self.frame_width = h, w
        
        clothing_kpts = []
        
        for name, idx in self.SHIRT_KEYPOINTS.items():
            landmark = pose_landmarks.landmark[idx]
            
            if landmark.visibility > visibility_threshold:
                clothing_kpts.append([
                    landmark.x * w,
                    landmark.y * h
                ])
        
        if len(clothing_kpts) < 6:
            return None  # Not enough keypoints visible
        
        # Add interpolated points for smoother warping
        clothing_kpts.extend(self._add_interpolated_points(clothing_kpts))
        
        return np.array(clothing_kpts, dtype=np.float32)
    
    def _add_interpolated_points(self, kpts):
        """Add midpoints between key landmarks for smoother warping"""
        if len(kpts) < 4:
            return []
        
        kpts_arr = np.array(kpts)
        interpolated = []
        
        # Add midpoint between shoulders (neck/collar position)
        if len(kpts_arr) >= 2:
            neck = (kpts_arr[0] + kpts_arr[1]) / 2
            interpolated.append(neck.tolist())
        
        # Add upper arm midpoints (for sleeve deformation)
        if len(kpts_arr) >= 4:
            # Left: shoulder (0) to elbow (2)
            left_upper_arm = (kpts_arr[0] + kpts_arr[2]) / 2
            # Right: shoulder (1) to elbow (3)
            right_upper_arm = (kpts_arr[1] + kpts_arr[3]) / 2
            interpolated.extend([
                left_upper_arm.tolist(),
                right_upper_arm.tolist()
            ])
        
        # Add torso midpoint (for center deformation)
        if len(kpts_arr) >= 8:
            # Between shoulders and hips
            torso_mid = (kpts_arr[0] + kpts_arr[1] + kpts_arr[6] + kpts_arr[7]) / 4
            interpolated.append(torso_mid.tolist())
        
        return interpolated
    
    def draw_keypoints(self, frame, keypoints):
        """Visualize keypoints for debugging"""
        if keypoints is None:
            return frame
        
        for i, (x, y) in enumerate(keypoints):
            # Draw circles at keypoints
            color = (0, 255, 0) if i < 8 else (255, 0, 255)  # Green=main, Magenta=interpolated
            cv2.circle(frame, (int(x), int(y)), 5, color, -1)
            cv2.putText(frame, str(i), (int(x)+10, int(y)),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
        
        return frame


class ClothingCompositor:
    """
    Handles clothing loading, warping, and compositing onto video frame.
    """
    
    def __init__(self):
        self.warper = ClothingWarper(grid_size=12)
        self.pose_tracker = PoseTracker()
        
        self.reference_keypoints = None
        self.clothing_img = None
        
        # Caching for performance
        self.frame_count = 0
        self.cached_map_x = None
        self.cached_map_y = None
        self.map_cache_interval = 2  # Recompute maps every N frames
        
        # Performance tracking
        self.composite_time_ms = 0
    
    def load_clothing(self, clothing_path, reference_frame):
        """
        Load clothing image and capture reference pose.
        
        Args:
            clothing_path: Path to PNG with transparency
            reference_frame: Frame to capture reference pose from
        """
        # Load clothing image
        clothing_pil = Image.open(clothing_path)
        
        # Ensure RGBA
        if clothing_pil.mode != 'RGBA':
            clothing_pil = clothing_pil.convert('RGBA')
        
        # Convert to OpenCV format (BGRA)
        clothing_array = np.array(clothing_pil)
        clothing_bgr = cv2.cvtColor(clothing_array[:, :, :3], cv2.COLOR_RGB2BGR)
        alpha = clothing_array[:, :, 3]
        self.clothing_img = np.dstack([clothing_bgr, alpha])
        
        print(f"✓ Loaded clothing: {clothing_path}")
        print(f"  Size: {self.clothing_img.shape}")
        
        # Capture reference pose
        results = self.pose_tracker.detect_pose(reference_frame)
        
        if results.pose_landmarks:
            self.reference_keypoints = self.pose_tracker.select_clothing_keypoints(
                results.pose_landmarks,
                reference_frame.shape
            )
            
            if self.reference_keypoints is not None:
                print(f"✓ Reference pose captured: {len(self.reference_keypoints)} keypoints")
            else:
                print("⚠ Could not extract enough keypoints from reference pose")
        else:
            print("⚠ No pose detected in reference frame")
    
    def composite_frame(self, frame, show_keypoints=False):
        """
        Main pipeline: Detect pose → Warp clothing → Composite
        
        Args:
            frame: Current video frame
            show_keypoints: If True, draw keypoints for debugging
            
        Returns:
            Composited frame with warped clothing
        """
        if self.clothing_img is None or self.reference_keypoints is None:
            return frame
        
        # 1. Detect current pose
        results = self.pose_tracker.detect_pose(frame)
        
        if not results.pose_landmarks:
            return frame  # No pose detected
        
        current_kpts = self.pose_tracker.select_clothing_keypoints(
            results.pose_landmarks,
            frame.shape
        )
        
        if current_kpts is None:
            return frame  # Not enough keypoints
        
        # 2. Compute warping maps (cached for performance)
        if self.frame_count % self.map_cache_interval == 0:
            # Resize clothing to match frame if needed
            if self.clothing_img.shape[:2] != frame.shape[:2]:
                clothing_resized = cv2.resize(self.clothing_img, 
                                             (frame.shape[1], frame.shape[0]),
                                             interpolation=cv2.INTER_LINEAR)
            else:
                clothing_resized = self.clothing_img
            
            self.cached_map_x, self.cached_map_y = self.warper.compute_grid_warp_maps(
                frame.shape,
                self.reference_keypoints,
                current_kpts
            )
            
            self.current_clothing = clothing_resized
        
        # 3. Warp clothing using cached maps
        if self.cached_map_x is not None:
            warped_clothing = self.warper.warp_clothing(
                self.current_clothing,
                self.cached_map_x,
                self.cached_map_y
            )
        else:
            return frame
        
        # 4. Composite with alpha blending
        start = time.time()
        result = self._alpha_blend(frame, warped_clothing)
        self.composite_time_ms = (time.time() - start) * 1000
        
        # 5. Optionally draw keypoints for debugging
        if show_keypoints:
            result = self.pose_tracker.draw_keypoints(result, current_kpts)
        
        self.frame_count += 1
        
        return result
    
    def _alpha_blend(self, background, foreground):
        """Alpha blend clothing onto frame"""
        # Split foreground into BGR and alpha
        fg_bgr = foreground[:, :, :3]
        fg_alpha = foreground[:, :, 3:4] / 255.0  # Normalize to 0-1
        
        # Alpha blending formula: result = fg * alpha + bg * (1 - alpha)
        blended = (fg_bgr * fg_alpha + background * (1 - fg_alpha)).astype(np.uint8)
        
        return blended


def main():
    """
    Main test loop: Real-time grid-based clothing warping.
    """
    
    print("="*60)
    print("GRID-BASED CLOTHING WARPING TEST")
    print("="*60)
    print("\nThis test demonstrates:")
    print("- Grid-based warping using inverse distance weighting")
    print("- MediaPipe pose keypoint extraction (8-12 points)")
    print("- Real-time warping at 30-40 FPS on Mac M1 CPU")
    print("\nControls:")
    print("  Q - Quit")
    print("  K - Toggle keypoint visualization")
    print("  1-5 - Load different clothing images")
    print("  C - Capture new reference pose")
    print("="*60)
    
    # Find clothing images
    gen_dir = Path("generated_images")
    if not gen_dir.exists():
        print(f"\n✗ Error: Directory '{gen_dir}' not found!")
        print("Make sure you have generated clothing images first.")
        return
    
    clothing_files = sorted(list(gen_dir.glob("*_clothing.png")))
    
    if len(clothing_files) == 0:
        print(f"\n✗ Error: No clothing images found in '{gen_dir}'")
        print("Expected files like: 1_flames_clothing.png")
        return
    
    print(f"\n✓ Found {len(clothing_files)} clothing images:")
    for i, img_path in enumerate(clothing_files[:5], 1):
        print(f"  {i}. {img_path.name}")
    
    # Initialize camera
    print("\nInitializing camera...")
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    
    if not cap.isOpened():
        print("✗ Error: Could not open camera")
        return
    
    # Get test frame
    ret, test_frame = cap.read()
    if not ret:
        print("✗ Error: Could not read from camera")
        return
    
    h, w = test_frame.shape[:2]
    print(f"✓ Camera: {w}x{h}")
    
    # Initialize compositor
    print("\nInitializing compositor...")
    compositor = ClothingCompositor()
    
    # Load first clothing image with reference pose
    print(f"\nLoading first clothing image and capturing reference pose...")
    print("Stand in front of camera in T-pose or natural standing position...")
    time.sleep(2)  # Give user time to position
    
    ret, reference_frame = cap.read()
    if ret:
        reference_frame = cv2.flip(reference_frame, 1)
        compositor.load_clothing(str(clothing_files[0]), reference_frame)
    
    # Settings
    show_keypoints = False
    current_clothing_idx = 0
    
    # Performance tracking
    fps = 0
    frame_count = 0
    fps_start = time.time()
    
    print("\n" + "="*60)
    print("RUNNING - Move around and watch clothing adapt!")
    print("="*60)
    
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                continue
            
            # Mirror for natural interaction
            frame = cv2.flip(frame, 1)
            
            # Apply warping and compositing
            result = compositor.composite_frame(frame, show_keypoints=show_keypoints)
            
            # Calculate total processing time
            total_time = (compositor.pose_tracker.processing_time_ms +
                         compositor.warper.map_compute_ms +
                         compositor.warper.warp_apply_ms +
                         compositor.composite_time_ms)
            
            # Add performance overlay
            cv2.putText(result, f"FPS: {fps:.1f}", (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            
            cv2.putText(result, f"Total: {total_time:.1f}ms", (10, 70),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            
            # Breakdown
            y_offset = 100
            cv2.putText(result, f"Pose: {compositor.pose_tracker.processing_time_ms:.1f}ms",
                       (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
            
            cv2.putText(result, f"Map: {compositor.warper.map_compute_ms:.1f}ms",
                       (10, y_offset + 25), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
            
            cv2.putText(result, f"Warp: {compositor.warper.warp_apply_ms:.1f}ms",
                       (10, y_offset + 50), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
            
            cv2.putText(result, f"Blend: {compositor.composite_time_ms:.1f}ms",
                       (10, y_offset + 75), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
            
            # Current clothing
            clothing_name = clothing_files[current_clothing_idx].stem
            cv2.putText(result, f"Clothing: {clothing_name}",
                       (10, h - 40), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (150, 150, 255), 2)
            
            # Status
            has_reference = compositor.reference_keypoints is not None
            status = "Warping active" if has_reference else "No reference pose"
            color = (0, 255, 0) if has_reference else (0, 165, 255)
            cv2.putText(result, status, (10, h - 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
            
            # Display
            cv2.imshow("Grid-Based Clothing Warping Test", result)
            
            # Update FPS
            frame_count += 1
            elapsed = time.time() - fps_start
            if elapsed >= 1.0:
                fps = frame_count / elapsed
                frame_count = 0
                fps_start = time.time()
            
            # Handle keyboard
            key = cv2.waitKey(1) & 0xFF
            
            if key == ord('q') or key == ord('Q'):
                break
            
            elif key == ord('k') or key == ord('K'):
                show_keypoints = not show_keypoints
                print(f"Keypoint visualization: {'ON' if show_keypoints else 'OFF'}")
            
            elif key >= ord('1') and key <= ord('5'):
                idx = key - ord('1')
                if idx < len(clothing_files):
                    current_clothing_idx = idx
                    print(f"\nLoading: {clothing_files[idx].name}")
                    ret, ref_frame = cap.read()
                    if ret:
                        ref_frame = cv2.flip(ref_frame, 1)
                        compositor.load_clothing(str(clothing_files[idx]), ref_frame)
            
            elif key == ord('c') or key == ord('C'):
                print("\nCapturing new reference pose...")
                ret, ref_frame = cap.read()
                if ret:
                    ref_frame = cv2.flip(ref_frame, 1)
                    if compositor.clothing_img is not None:
                        results = compositor.pose_tracker.detect_pose(ref_frame)
                        if results.pose_landmarks:
                            compositor.reference_keypoints = \
                                compositor.pose_tracker.select_clothing_keypoints(
                                    results.pose_landmarks,
                                    ref_frame.shape
                                )
                            print("✓ New reference pose captured")
                        else:
                            print("✗ No pose detected")
    
    except KeyboardInterrupt:
        print("\nInterrupted")
    
    finally:
        cap.release()
        cv2.destroyAllWindows()
        print("\n" + "="*60)
        print("TEST COMPLETE")
        print("="*60)
        print(f"\nFinal Performance:")
        print(f"  Average FPS: {fps:.1f}")
        print(f"  Pose detection: {compositor.pose_tracker.processing_time_ms:.1f}ms")
        print(f"  Warp map computation: {compositor.warper.map_compute_ms:.1f}ms")
        print(f"  Warp application: {compositor.warper.warp_apply_ms:.1f}ms")
        print(f"  Compositing: {compositor.composite_time_ms:.1f}ms")
        print(f"  Total: {total_time:.1f}ms/frame")
        
        if fps >= 25:
            print("\n✓ Performance target achieved (25+ FPS)")
        else:
            print(f"\n⚠ Performance below target ({fps:.1f} FPS)")
            print("Try: Reduce grid_size to 10, or use map_cache_interval=3")


if __name__ == "__main__":
    main()