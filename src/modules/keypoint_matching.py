"""
Strategy 1 IMPROVED: Keypoint-Based Clothing Warping
====================================================
Uses MediaPipe pose keypoints to naturally warp clothing to match body movement.
Much more realistic than bounding box approach.

Place in: tests/test_keypoint_warping.py
Run from root: python tests/test_keypoint_warping.py
"""

import cv2
import numpy as np
import time
import sys
import os
from pathlib import Path
from PIL import Image
import mediapipe as mp

# Add parent to path
sys.path.append(str(Path(__file__).parent.parent))


class KeypointClothingRenderer:
    """
    Warps clothing using keypoint correspondence between generated image and live person.
    This is how professional AR try-on works.
    """
    
    # MediaPipe pose landmark indices for clothing anchor points
    # These are the key points we'll use to warp the clothing
    CLOTHING_KEYPOINTS = {
        'left_shoulder': 11,
        'right_shoulder': 12,
        'left_elbow': 13,
        'right_elbow': 14,
        'left_wrist': 15,
        'right_wrist': 16,
        'left_hip': 23,
        'right_hip': 24,
        'nose': 0,  # For reference point
    }
    
    def __init__(self, clothing_image_path):
        """Load clothing and extract its pose keypoints"""
        
        # Load clothing image
        self.clothing_img = cv2.imread(clothing_image_path, cv2.IMREAD_UNCHANGED)
        if self.clothing_img is None:
            raise ValueError(f"Could not load: {clothing_image_path}")
        
        print(f"Loaded clothing: {self.clothing_img.shape}")
        
        # Ensure BGRA format
        if self.clothing_img.shape[2] == 3:
            # Add alpha channel if missing
            alpha = np.ones((self.clothing_img.shape[0], self.clothing_img.shape[1]), dtype=np.uint8) * 255
            self.clothing_img = np.dstack([self.clothing_img, alpha])
        
        # Initialize MediaPipe Pose for keypoint extraction
        self.mp_pose = mp.solutions.pose
        self.pose_detector = self.mp_pose.Pose(
            static_image_mode=False,  # Video mode for real-time
            model_complexity=0,  # 0=fastest, 1=balanced, 2=most accurate
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        
        # Extract keypoints from the generated clothing image
        print("Extracting pose from clothing image...")
        self.clothing_keypoints = self._extract_keypoints_from_image(self.clothing_img)
        
        if self.clothing_keypoints is None:
            print("WARNING: Could not detect pose in clothing image!")
            print("Will use fallback positioning")
            # Create default keypoints in center of image
            h, w = self.clothing_img.shape[:2]
            self.clothing_keypoints = self._create_default_keypoints(w, h)
        else:
            print(f"Detected {len(self.clothing_keypoints)} keypoints in clothing")
        
        # Performance tracking
        self.pose_time_ms = 0
        self.warp_time_ms = 0
        self.composite_time_ms = 0
    
    def _extract_keypoints_from_image(self, image):
        """Run MediaPipe on clothing image to get source keypoints"""
        
        # Convert to RGB for MediaPipe
        rgb = cv2.cvtColor(image[:,:,:3], cv2.COLOR_BGR2RGB)
        
        # Run pose detection
        results = self.pose_detector.process(rgb)
        
        if not results.pose_landmarks:
            return None
        
        # Extract keypoints we care about
        h, w = image.shape[:2]
        keypoints = {}
        
        for name, idx in self.CLOTHING_KEYPOINTS.items():
            landmark = results.pose_landmarks.landmark[idx]
            # Convert normalized coords to pixel coords
            x = int(landmark.x * w)
            y = int(landmark.y * h)
            keypoints[name] = (x, y)
        
        return keypoints
    
    def _create_default_keypoints(self, width, height):
        """Create reasonable default keypoints if pose detection fails"""
        
        # Assume person is centered and upright
        center_x = width // 2
        center_y = height // 2
        
        shoulder_width = width // 4
        torso_height = height // 3
        arm_length = height // 4
        
        return {
            'nose': (center_x, int(center_y - torso_height * 1.2)),
            'left_shoulder': (center_x - shoulder_width, center_y - torso_height),
            'right_shoulder': (center_x + shoulder_width, center_y - torso_height),
            'left_elbow': (center_x - shoulder_width - 20, center_y - torso_height // 2),
            'right_elbow': (center_x + shoulder_width + 20, center_y - torso_height // 2),
            'left_wrist': (center_x - shoulder_width - 30, center_y),
            'right_wrist': (center_x + shoulder_width + 30, center_y),
            'left_hip': (center_x - shoulder_width // 2, center_y + torso_height),
            'right_hip': (center_x + shoulder_width // 2, center_y + torso_height),
        }
    
    def render_on_body(self, video_frame, live_keypoints):
        """
        Main rendering: warp clothing to match live body keypoints.
        
        Args:
            video_frame: Current webcam frame
            live_keypoints: Dict of current body keypoint positions
            
        Returns:
            Frame with clothing overlaid
        """
        
        if live_keypoints is None:
            return video_frame
        
        warp_start = time.time()
        
        # Convert keypoint dicts to numpy arrays for warping
        src_points = []  # Points on clothing image
        dst_points = []  # Corresponding points on live video
        
        for name in self.CLOTHING_KEYPOINTS.keys():
            if name in self.clothing_keypoints and name in live_keypoints:
                src_points.append(self.clothing_keypoints[name])
                dst_points.append(live_keypoints[name])
        
        if len(src_points) < 4:
            # Need at least 4 points for good warping
            return video_frame
        
        src_points = np.float32(src_points)
        dst_points = np.float32(dst_points)
        
        # Calculate bounding box of destination points (where clothing will appear)
        x_coords = dst_points[:, 0]
        y_coords = dst_points[:, 1]
        x_min, x_max = int(x_coords.min()), int(x_coords.max())
        y_min, y_max = int(y_coords.min()), int(y_coords.max())
        
        # Add padding
        padding = 50
        x_min = max(0, x_min - padding)
        y_min = max(0, y_min - padding)
        x_max = min(video_frame.shape[1], x_max + padding)
        y_max = min(video_frame.shape[0], y_max + padding)
        
        roi_width = x_max - x_min
        roi_height = y_max - y_min
        
        if roi_width <= 0 or roi_height <= 0:
            return video_frame
        
        # Adjust destination points to ROI coordinates
        dst_points_roi = dst_points - np.array([x_min, y_min])
        
        # Compute piecewise affine transform
        # This creates smooth warping between keypoints
        warped_clothing = self._warp_image_by_keypoints(
            self.clothing_img,
            src_points,
            dst_points_roi,
            (roi_width, roi_height)
        )
        
        self.warp_time_ms = (time.time() - warp_start) * 1000
        
        # Composite onto video frame
        composite_start = time.time()
        result = self._composite_clothing(video_frame, warped_clothing, x_min, y_min)
        self.composite_time_ms = (time.time() - composite_start) * 1000
        
        return result
    
    def _warp_image_by_keypoints(self, image, src_points, dst_points, output_size):
        """
        Warp image using piecewise affine transformation between keypoints.
        This is the core AR warping logic.
        """
        
        # Use OpenCV's estimation of affine transform
        # We'll use a simplified approach: compute homography for the full set
        # For production, you'd use Delaunay triangulation + piecewise affine
        
        if len(src_points) >= 4 and len(dst_points) >= 4:
            # Perspective transform for more flexibility
            try:
                # Find homography matrix
                matrix, mask = cv2.findHomography(src_points, dst_points, cv2.RANSAC, 5.0)
                
                if matrix is not None:
                    # Warp the image
                    warped = cv2.warpPerspective(
                        image,
                        matrix,
                        output_size,
                        flags=cv2.INTER_LINEAR,
                        borderMode=cv2.BORDER_CONSTANT,
                        borderValue=(0, 0, 0, 0)
                    )
                    return warped
            except:
                pass
        
        # Fallback: simple affine transform using 3 points
        if len(src_points) >= 3:
            matrix = cv2.getAffineTransform(src_points[:3], dst_points[:3])
            warped = cv2.warpAffine(
                image,
                matrix,
                output_size,
                flags=cv2.INTER_LINEAR,
                borderMode=cv2.BORDER_CONSTANT,
                borderValue=(0, 0, 0, 0)
            )
            return warped
        
        # Last resort: just resize
        return cv2.resize(image, output_size)
    
    def _composite_clothing(self, frame, clothing, offset_x, offset_y):
        """Alpha blend clothing onto frame"""
        
        h_cloth, w_cloth = clothing.shape[:2]
        h_frame, w_frame = frame.shape[:2]
        
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
        
        if clothing_cropped.shape[0] == 0 or clothing_cropped.shape[1] == 0:
            return frame
        
        # Extract ROI from video
        roi = frame[start_y:end_y, start_x:end_x]
        
        # Get alpha channel
        cloth_bgr = clothing_cropped[:,:,:3]
        cloth_alpha = clothing_cropped[:,:,3].astype(float) / 255.0
        
        # Expand alpha to 3 channels
        cloth_alpha_3ch = np.stack([cloth_alpha] * 3, axis=2)
        
        # Alpha blend
        blended = (cloth_bgr * cloth_alpha_3ch + roi * (1 - cloth_alpha_3ch)).astype(np.uint8)
        
        # Place back into frame
        result = frame.copy()
        result[start_y:end_y, start_x:end_x] = blended
        
        return result


class LivePoseTracker:
    """Extracts keypoints from live video using MediaPipe"""
    
    def __init__(self):
        # Initialize MediaPipe Pose
        self.mp_pose = mp.solutions.pose
        self.pose = self.mp_pose.Pose(
            static_image_mode=False,
            model_complexity=0,  # Use fastest model for real-time
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        
        self.mp_drawing = mp.solutions.drawing_utils
        self.processing_time_ms = 0
        self.latest_keypoints = None
    
    def process_frame(self, frame):
        """Extract keypoints from frame"""
        
        start = time.time()
        
        # Convert BGR to RGB
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Run pose detection
        results = self.pose.process(rgb)
        
        self.processing_time_ms = (time.time() - start) * 1000
        
        if not results.pose_landmarks:
            self.latest_keypoints = None
            return None
        
        # Extract keypoints
        h, w = frame.shape[:2]
        keypoints = {}
        
        for name, idx in KeypointClothingRenderer.CLOTHING_KEYPOINTS.items():
            landmark = results.pose_landmarks.landmark[idx]
            x = int(landmark.x * w)
            y = int(landmark.y * h)
            keypoints[name] = (x, y)
        
        self.latest_keypoints = keypoints
        return keypoints
    
    def draw_keypoints(self, frame):
        """Draw skeleton visualization on frame"""
        
        if self.latest_keypoints is None:
            return frame
        
        # Draw circles at each keypoint
        for name, (x, y) in self.latest_keypoints.items():
            cv2.circle(frame, (x, y), 5, (0, 255, 0), -1)
            cv2.putText(frame, name.split('_')[0], (x + 10, y),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255, 255, 255), 1)
        
        # Draw connections (skeleton)
        connections = [
            ('left_shoulder', 'right_shoulder'),
            ('left_shoulder', 'left_elbow'),
            ('left_elbow', 'left_wrist'),
            ('right_shoulder', 'right_elbow'),
            ('right_elbow', 'right_wrist'),
            ('left_shoulder', 'left_hip'),
            ('right_shoulder', 'right_hip'),
            ('left_hip', 'right_hip'),
        ]
        
        for point1, point2 in connections:
            if point1 in self.latest_keypoints and point2 in self.latest_keypoints:
                pt1 = self.latest_keypoints[point1]
                pt2 = self.latest_keypoints[point2]
                cv2.line(frame, pt1, pt2, (0, 255, 255), 2)
        
        return frame


def main():
    """Main test loop"""
    
    print("="*60)
    print("KEYPOINT-BASED CLOTHING WARPING TEST")
    print("="*60)
    print("\nThis version:")
    print("1. Uses MediaPipe pose keypoints (not bounding boxes)")
    print("2. Warps clothing to match your exact pose")
    print("3. Much faster and more natural-looking")
    print("\nControls:")
    print("  Q - Quit")
    print("  1-5 - Load different clothing")
    print("  K - Toggle keypoint visualization")
    print("="*60)
    
    # Find clothing images
    generated_dir = Path("generated_images")
    if not generated_dir.exists():
        print(f"\nError: '{generated_dir}' not found!")
        return
    
    image_files = sorted(list(generated_dir.glob("*_clothing.png")))
    
    if len(image_files) == 0:
        print(f"\nNo clothing images in '{generated_dir}'")
        return
    
    print(f"\nFound {len(image_files)} clothing images:")
    for i, img in enumerate(image_files[:5], 1):
        print(f"  {i}. {img.name}")
    
    # Initialize camera
    print("\nInitializing camera...")
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    
    if not cap.isOpened():
        print("Error: Could not open camera")
        return
    
    # Initialize pose tracker
    print("\nInitializing MediaPipe Pose tracker...")
    tracker = LivePoseTracker()
    
    # Load first clothing
    print(f"\nLoading clothing: {image_files[0].name}")
    try:
        renderer = KeypointClothingRenderer(str(image_files[0]))
    except Exception as e:
        print(f"Error: {e}")
        return
    
    # Settings
    current_idx = 0
    show_keypoints = False
    
    # Performance tracking
    fps = 0
    frame_count = 0
    fps_start = time.time()
    
    print("\n" + "="*60)
    print("RUNNING - Move around and watch clothing adapt to your pose!")
    print("="*60)
    
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                continue
            
            # Mirror for natural feel
            frame = cv2.flip(frame, 1)
            
            # Get current pose keypoints
            keypoints = tracker.process_frame(frame)
            
            # Render clothing with keypoint-based warping
            if keypoints is not None:
                result = renderer.render_on_body(frame, keypoints)
            else:
                result = frame
            
            # Show keypoint skeleton if enabled
            if show_keypoints:
                result = tracker.draw_keypoints(result)
            
            # Performance overlay
            total_time = tracker.processing_time_ms + renderer.warp_time_ms + renderer.composite_time_ms
            
            cv2.putText(result, f"FPS: {fps:.1f}", (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            
            cv2.putText(result, f"Total: {total_time:.1f}ms", (10, 70),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            
            cv2.putText(result, f"Pose: {tracker.processing_time_ms:.1f}ms", (10, 100),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)
            
            cv2.putText(result, f"Warp: {renderer.warp_time_ms:.1f}ms", (10, 125),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)
            
            cv2.putText(result, f"Blend: {renderer.composite_time_ms:.1f}ms", (10, 150),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)
            
            status = "Pose detected" if keypoints else "No pose detected"
            color = (0, 255, 0) if keypoints else (0, 0, 255)
            cv2.putText(result, status, (10, 180),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
            
            # Display
            cv2.imshow("Keypoint-Based Clothing AR", result)
            
            # Calculate FPS
            frame_count += 1
            elapsed = time.time() - fps_start
            if elapsed >= 1.0:
                fps = frame_count / elapsed
                frame_count = 0
                fps_start = time.time()
            
            # Handle keys
            key = cv2.waitKey(1) & 0xFF
            
            if key == ord('q') or key == ord('Q'):
                break
            
            elif key == ord('k') or key == ord('K'):
                show_keypoints = not show_keypoints
            
            elif key >= ord('1') and key <= ord('5'):
                idx = key - ord('1')
                if idx < len(image_files):
                    current_idx = idx
                    print(f"\nLoading: {image_files[idx].name}")
                    try:
                        renderer = KeypointClothingRenderer(str(image_files[idx]))
                    except Exception as e:
                        print(f"Error: {e}")
    
    except KeyboardInterrupt:
        print("\nInterrupted")
    
    finally:
        cap.release()
        cv2.destroyAllWindows()
        print("\n" + "="*60)
        print("TEST COMPLETE")
        print("="*60)
        print("\nWhat to evaluate:")
        print("- Does clothing move naturally with your body?")
        print("- Are shoulders/arms warping correctly?")
        print("- Is FPS better than before? (should be 15-25 FPS)")
        print("- Press K to see keypoint skeleton tracking")


if __name__ == "__main__":
    main()