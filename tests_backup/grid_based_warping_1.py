"""
Robust Grid-Based Clothing Warping with Keypoint Matching
==========================================================
Solves the keypoint mismatch problem by only using points detected in both frames.
Includes graceful degradation and smooth temporal filtering.

Place in: tests/robust_grid_warping.py
Run: python tests/robust_grid_warping.py
"""

import cv2
import numpy as np
import mediapipe as mp
from pathlib import Path
from scipy.spatial.distance import cdist
import time


class RobustClothingWarper:
    """
    Grid-based warper that handles variable keypoint counts gracefully.
    Key innovation: matches keypoints by name, not index.
    """
    
    # MediaPipe keypoint indices for upper body clothing
    CLOTHING_KEYPOINTS = {
        'nose': 0,
        'left_shoulder': 11,
        'right_shoulder': 12,
        'left_elbow': 13,
        'right_elbow': 14,
        'left_wrist': 15,
        'right_wrist': 16,
        'left_hip': 23,
        'right_hip': 24,
    }
    
    def __init__(self, grid_size=12):
        self.grid_size = grid_size
        # self.clothing_img = cv2.flip(self.clothing_img, 1)
        
    def extract_keypoints_dict(self, landmarks, img_shape, visibility_threshold=0.5):
        """
        Extract keypoints as dictionary with visibility filtering.
        Returns dict of {name: (x, y)} only for visible keypoints.
        """
        h, w = img_shape[:2]
        keypoints = {}

        
        
        for name, idx in self.CLOTHING_KEYPOINTS.items():
            landmark = landmarks.landmark[idx]
            
            # Only include if visibility is good
            if landmark.visibility > visibility_threshold:
                x = int(landmark.x * w)
                y = int(landmark.y * h)
                keypoints[name] = (x, y)
        
        return keypoints
    
    def match_keypoints(self, ref_keypoints_dict, curr_keypoints_dict):
        """
        Find common keypoints between reference and current frame.
        Returns matched pairs as numpy arrays.
        """
        # Find intersection of keypoint names
        common_names = set(ref_keypoints_dict.keys()) & set(curr_keypoints_dict.keys())
        
        if len(common_names) < 4:
            # Need at least 4 points for reasonable warping
            return None, None, list(common_names)
        
        # Build matched arrays
        src_points = []
        dst_points = []
        
        # Always include these in order for stability (if available)
        priority_order = ['left_shoulder', 'right_shoulder', 'left_hip', 'right_hip',
                         'left_elbow', 'right_elbow', 'left_wrist', 'right_wrist', 'nose']
        
        matched_names = []
        for name in priority_order:
            if name in common_names:
                src_points.append(ref_keypoints_dict[name])
                dst_points.append(curr_keypoints_dict[name])
                matched_names.append(name)
        
        # Add any remaining common points
        for name in common_names:
            if name not in matched_names:
                src_points.append(ref_keypoints_dict[name])
                dst_points.append(curr_keypoints_dict[name])
                matched_names.append(name)
        
        return np.array(src_points, dtype=np.float32), np.array(dst_points, dtype=np.float32), matched_names
    
    def compute_grid_warp_maps(self, img_shape, src_keypoints_dict, dst_keypoints_dict):
        """
        Create warp maps using inverse distance weighting on matched keypoints.
        """
        # Match keypoints between reference and current
        src_pts, dst_pts, matched = self.match_keypoints(src_keypoints_dict, dst_keypoints_dict)
        
        if src_pts is None or len(src_pts) < 4:
            # Not enough points - return identity mapping
            h, w = img_shape[:2]
            y, x = np.meshgrid(np.arange(h), np.arange(w), indexing='ij')
            return x.astype(np.float32), y.astype(np.float32), len(matched) if matched else 0
        
        h, w = img_shape[:2]
        
        # Create coarse grid for efficiency
        grid_h = self.grid_size
        grid_w = self.grid_size
        
        grid_y = np.linspace(0, h-1, grid_h)
        grid_x = np.linspace(0, w-1, grid_w)
        
        # Compute displacement at each grid node
        grid_map_x = np.zeros((grid_h, grid_w), dtype=np.float32)
        grid_map_y = np.zeros((grid_h, grid_w), dtype=np.float32)
        
        for i, y in enumerate(grid_y):
            for j, x in enumerate(grid_x):
                dx, dy = self._idw_displacement(x, y, src_pts, dst_pts, power=2)
                grid_map_x[i, j] = x + dx
                grid_map_y[i, j] = y + dy
        
        # Interpolate to full resolution
        map_x = cv2.resize(grid_map_x, (w, h), interpolation=cv2.INTER_CUBIC)
        map_y = cv2.resize(grid_map_y, (w, h), interpolation=cv2.INTER_CUBIC)
        
        return map_x, map_y, len(matched)
    
    def _idw_displacement(self, x, y, src_pts, dst_pts, power=2):
        """Inverse distance weighted interpolation of displacements."""
        point = np.array([[x, y]])
        distances = cdist(point, src_pts)[0]
        
        # Avoid division by zero
        distances = np.maximum(distances, 1.0)
        
        # Inverse distance weights
        weights = 1.0 / (distances ** power)
        weights /= weights.sum()
        
        # Weighted displacement - THIS IS THE FIX
        # For cv2.remap we need INVERSE mapping: where to sample FROM
        # So we compute reference - current, not current - reference
        displacements = src_pts - dst_pts  # CHANGED: was dst_pts - src_pts
        weighted_disp = (weights[:, None] * displacements).sum(axis=0)
        
        return weighted_disp[0], weighted_disp[1]
    
    def warp_clothing(self, clothing_img, map_x, map_y):
        """Apply warping using precomputed maps."""
        return cv2.remap(
            clothing_img,
            map_x,
            map_y,
            cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=(0, 0, 0, 0)
        )


class PoseTracker:
    """MediaPipe pose tracker with keypoint extraction."""
    
    def __init__(self, min_detection_confidence=0.5, min_tracking_confidence=0.5):
        self.mp_pose = mp.solutions.pose
        self.pose = self.mp_pose.Pose(
            static_image_mode=False,
            model_complexity=0,  # Fastest model
            smooth_landmarks=True,
            min_detection_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence
        )
        self.mp_drawing = mp.solutions.drawing_utils
        self.processing_time_ms = 0
    
    def detect_pose(self, frame):
        """Process frame and return pose landmarks."""
        start = time.time()
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.pose.process(rgb)
        self.processing_time_ms = (time.time() - start) * 1000
        return results
    
    def draw_keypoints(self, frame, keypoints_dict):
        """Visualize detected keypoints."""
        for name, (x, y) in keypoints_dict.items():
            cv2.circle(frame, (x, y), 5, (0, 255, 0), -1)
            # Draw label
            cv2.putText(frame, name.replace('_', ' '), (x + 10, y - 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255, 255, 255), 1)
        return frame


class TemporalSmoother:
    """
    Smooths keypoint positions over time to reduce jitter.
    Uses exponential moving average.
    """
    
    def __init__(self, alpha=0.3):
        self.alpha = alpha  # Smoothing factor (0 = no smoothing, 1 = no memory)
        self.prev_keypoints = None
    
    def smooth(self, current_keypoints_dict):
        """Apply temporal smoothing to keypoints."""
        if self.prev_keypoints is None:
            self.prev_keypoints = current_keypoints_dict.copy()
            return current_keypoints_dict
        
        smoothed = {}
        for name, (x, y) in current_keypoints_dict.items():
            if name in self.prev_keypoints:
                prev_x, prev_y = self.prev_keypoints[name]
                # Exponential moving average
                smooth_x = self.alpha * x + (1 - self.alpha) * prev_x
                smooth_y = self.alpha * y + (1 - self.alpha) * prev_y
                smoothed[name] = (int(smooth_x), int(smooth_y))
            else:
                # New keypoint - use as is
                smoothed[name] = (x, y)
        
        self.prev_keypoints = smoothed.copy()
        return smoothed

def draw_keypoints_color(frame, keypoints_dict, color=(0,255,0)):
        for name, (x, y) in keypoints_dict.items():
            cv2.circle(frame, (x, y), 5, color, -1)
            cv2.putText(frame, name.replace('_', ' '), (x + 10, y - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.3, color, 1)
        return frame

class RobustClothingCompositor:
    """
    Complete compositor with robust keypoint matching and temporal smoothing.
    """
    
    
    def __init__(self, clothing_path):
        # Load clothing image
        self.clothing_img = cv2.flip(cv2.imread(str(clothing_path), cv2.IMREAD_UNCHANGED),1)
        if self.clothing_img is None:
            raise ValueError(f"Could not load clothing: {clothing_path}")
        
        # Ensure BGRA format
        if self.clothing_img.shape[2] == 3:
            alpha = np.ones((self.clothing_img.shape[0], self.clothing_img.shape[1]), 
                           dtype=np.uint8) * 255
            self.clothing_img = np.dstack([self.clothing_img, alpha])
        
        # self.clothing_img = cv2.flip(self.clothing_img, 1)

        print(f"✓ Loaded clothing: {clothing_path}")
        print(f"  Size: {self.clothing_img.shape}")
        
        # Initialize components
        self.warper = RobustClothingWarper(grid_size=12)
        self.pose_tracker = PoseTracker()
        self.smoother = TemporalSmoother(alpha=0.3)
        
        # Reference pose (will be captured)
        self.reference_keypoints = None
        
        # Caching for performance
        self.cached_map_x = None
        self.cached_map_y = None
        self.map_cache_interval = 2  # Update maps every N frames
        self.frame_count = 0
        
        # Performance tracking
        self.warp_time_ms = 0
        self.composite_time_ms = 0
        self.matched_keypoints = 0
    
    def capture_reference_pose(self, frame):
        """Capture the reference pose from current frame."""
        print("\nCapturing reference pose...")
        pose_results = self.pose_tracker.detect_pose(frame)
        
        if pose_results.pose_landmarks:
            self.reference_keypoints = self.warper.extract_keypoints_dict(
                pose_results.pose_landmarks, frame.shape
            )
            print(f"✓ Reference pose captured: {len(self.reference_keypoints)} keypoints")
            print(f"  Detected: {', '.join(self.reference_keypoints.keys())}")
            return True
        else:
            print("✗ No pose detected - try again")
            return False
    
    def composite_frame(self, frame, show_keypoints=False):
        """
        Main compositing function for each frame.
        """
        if self.reference_keypoints is None:
            # No reference yet - show instruction
            cv2.putText(frame, "Press C to capture reference pose", (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
            return frame
        
        # Detect current pose
        pose_results = self.pose_tracker.detect_pose(frame)
        
        if not pose_results.pose_landmarks:
            # No pose detected
            cv2.putText(frame, "No pose detected", (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            return frame
        
        # Extract current keypoints
        current_keypoints = self.warper.extract_keypoints_dict(
            pose_results.pose_landmarks, frame.shape
        )

        # Apply temporal smoothing
        current_keypoints = self.smoother.smooth(current_keypoints)
        
        # Compute warp maps (with caching for performance)
        warp_start = time.time()
        if self.frame_count % self.map_cache_interval == 0:
            self.cached_map_x, self.cached_map_y, self.matched_keypoints = \
                self.warper.compute_grid_warp_maps(
                    frame.shape,
                    self.reference_keypoints,
                    current_keypoints
                )
        self.warp_time_ms = (time.time() - warp_start) * 1000
        
        if self.matched_keypoints < 4:
            # Too few matched keypoints for good warping
            cv2.putText(frame, f"Too few keypoints ({self.matched_keypoints}/4 min)", 
                       (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 165, 255), 2)
            cv2.putText(frame, "Stand in clearer view or press C to recapture", 
                       (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 165, 255), 1)
            return frame
        
        # Warp clothing
        warped_clothing = self.warper.warp_clothing(
            self.clothing_img,
            self.cached_map_x,
            self.cached_map_y
        )
        
        # Composite onto frame
        composite_start = time.time()
        result = self._alpha_blend(frame, warped_clothing)
        self.composite_time_ms = (time.time() - composite_start) * 1000
        
        # Optionally show keypoints
        if show_keypoints:
            result = self.pose_tracker.draw_keypoints(result, current_keypoints)
            if self.reference_keypoints:
                result = draw_keypoints_color(result, self.reference_keypoints, color=(0,0,255))  # red
        
        self.frame_count += 1
        return result
    
    
    
    def _alpha_blend(self, background, overlay):
        """Alpha blend overlay onto background."""
        h_over, w_over = overlay.shape[:2]
        h_bg, w_bg = background.shape[:2]
        
        # Center overlay on background
        offset_x = (w_bg - w_over) // 2
        offset_y = (h_bg - h_over) // 2
        
        # Ensure overlay fits
        if offset_x < 0 or offset_y < 0:
            # Resize overlay to fit
            scale = min(w_bg / w_over, h_bg / h_over) * 0.9
            new_w = int(w_over * scale)
            new_h = int(h_over * scale)
            overlay = cv2.resize(overlay, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
            h_over, w_over = overlay.shape[:2]
            offset_x = (w_bg - w_over) // 2
            offset_y = (h_bg - h_over) // 2
        
        # Extract ROI
        y1, y2 = offset_y, offset_y + h_over
        x1, x2 = offset_x, offset_x + w_over
        
        if y2 > h_bg or x2 > w_bg or y1 < 0 or x1 < 0:
            return background
        
        roi = background[y1:y2, x1:x2]
        
        # Split overlay into BGR and alpha
        overlay_bgr = overlay[:, :, :3]
        overlay_alpha = overlay[:, :, 3].astype(float) / 255.0
        
        # Expand alpha to 3 channels
        alpha_3ch = np.stack([overlay_alpha] * 3, axis=2)
        
        # Alpha blend
        blended = (overlay_bgr * alpha_3ch + roi * (1 - alpha_3ch)).astype(np.uint8)
        
        # Place back
        result = background.copy()
        result[y1:y2, x1:x2] = blended
        
        return result


def main():
    """Main test application."""
    
    print("="*60)
    print("ROBUST GRID-BASED CLOTHING WARPING")
    print("="*60)
    print("\nImprovements:")
    print("✓ Handles variable keypoint counts")
    print("✓ Matches keypoints by name, not index")
    print("✓ Temporal smoothing reduces jitter")
    print("✓ Graceful degradation with <4 points")
    print("\nControls:")
    print("  Q - Quit")
    print("  C - Capture new reference pose")
    print("  K - Toggle keypoint visualization")
    print("  1-5 - Load different clothing")
    print("="*60)
    
    # Find clothing images
    generated_dir = Path("generated_images")
    if not generated_dir.exists():
        print(f"\n✗ Directory '{generated_dir}' not found!")
        return
    
    image_files = sorted(list(generated_dir.glob("*_clothing.png")))
    if len(image_files) == 0:
        print(f"\n✗ No clothing images in '{generated_dir}'")
        return
    
    print(f"\n✓ Found {len(image_files)} clothing images:")
    for i, img in enumerate(image_files[:5], 1):
        print(f"  {i}. {img.name}")
    
    # Initialize camera
    print("\nInitializing camera...")
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    
    if not cap.isOpened():
        print("✗ Could not open camera")
        return
    
    ret, test_frame = cap.read()
    if not ret:
        print("✗ Could not read from camera")
        return
    
    print(f"✓ Camera: {test_frame.shape[1]}x{test_frame.shape[0]}")
    
    # Load first clothing
    print("\nInitializing compositor...")
    try:
        compositor = RobustClothingCompositor(image_files[0])
    except Exception as e:
        print(f"✗ Error: {e}")
        return
    
    # Initial reference pose capture
    print("\nCapture reference pose:")
    print("Stand in T-pose or natural position and press C")
    
    # Settings
    current_idx = 0
    show_keypoints = False
    
    # Performance tracking
    fps_counter = 0
    fps_start = time.time()
    fps = 0
    
    print("\n" + "="*60)
    print("RUNNING")
    print("="*60)

    reference_captured = False
    start_time = time.time()        
    
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                continue
            
            # Mirror for natural feel
            frame = cv2.flip(frame, 1)
            
            # Composite clothing
            result = compositor.composite_frame(frame, show_keypoints=show_keypoints)
            
            # Performance overlay
            total_time = (compositor.pose_tracker.processing_time_ms + 
                         compositor.warp_time_ms + 
                         compositor.composite_time_ms)
            
            cv2.putText(result, f"FPS: {fps:.1f}", (10, frame.shape[0] - 120),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            
            cv2.putText(result, f"Pose: {compositor.pose_tracker.processing_time_ms:.1f}ms", 
                       (10, frame.shape[0] - 90), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            
            cv2.putText(result, f"Warp: {compositor.warp_time_ms:.1f}ms", 
                       (10, frame.shape[0] - 65), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            
            cv2.putText(result, f"Composite: {compositor.composite_time_ms:.1f}ms", 
                       (10, frame.shape[0] - 40), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            
            if compositor.reference_keypoints:
                cv2.putText(result, f"Matched: {compositor.matched_keypoints} keypoints", 
                           (10, frame.shape[0] - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (150, 255, 150), 1)
            
            # Display
            cv2.imshow("Robust Clothing Warping", result)

          
          
            
            # Update FPS
            fps_counter += 1
            if time.time() - fps_start >= 1.0:
                fps = fps_counter / (time.time() - fps_start)
                fps_counter = 0
                fps_start = time.time()
            
            # Handle keys
            key = cv2.waitKey(1) & 0xFF
            
            if key == ord('q') or key == ord('Q'):
                break

            if not reference_captured and (time.time() - start_time > 5):
                print("\nAuto-capturing reference pose after 5 seconds...")
                reference_captured = compositor.capture_reference_pose(frame)
                print("Press C to recapture reference pose if needed")
            # ...existing code...
            
            elif key == ord('c') or key == ord('C'):
                compositor.capture_reference_pose(frame)
            
            elif key == ord('k') or key == ord('K'):
                show_keypoints = not show_keypoints
                print(f"Keypoints: {'ON' if show_keypoints else 'OFF'}")
            
            elif key >= ord('1') and key <= ord('5'):
                idx = key - ord('1')
                if idx < len(image_files):
                    current_idx = idx
                    try:
                        compositor = RobustClothingCompositor(image_files[idx])
                        print(f"\n✓ Loaded: {image_files[idx].name}")
                        print("Press C to recapture reference pose for new clothing")
                    except Exception as e:
                        print(f"✗ Error: {e}")

            # automatically capture reference after 5 seconds using timer



            # if compositor.reference_keypoints is None:
            if time.time() - fps_start > 5:
                print("\nNo reference pose detected for 5 seconds.")
                # Attempt to capture reference pose
                print("\nAuto-capturing reference pose...")
                compositor.capture_reference_pose(frame)
                print("Press C to recapture reference pose if needed" )
        
    
    except KeyboardInterrupt:
        print("\n\nInterrupted")
    
    finally:
        cap.release()
        cv2.destroyAllWindows()
        print("\n" + "="*60)
        print("COMPLETE")
        print("="*60)


if __name__ == "__main__":
    main()