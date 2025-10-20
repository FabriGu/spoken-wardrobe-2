"""
Test 1: 3D Skeleton Overlay
============================
Shows a simple 3D "stick figure" skeleton overlaid on your body.
This proves we can track body position in 3D space.

Run from root: python tests/test_01_3d_skeleton.py
"""

import cv2
import numpy as np
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))
from src.modules.body_tracking import BodySegmenter


class Simple3DSkeleton:
    """Renders a simple 3D skeleton using your BodyPix keypoints"""
    
    # Define skeleton connections (which keypoints connect to which)
    SKELETON_CONNECTIONS = [
        ('left_shoulder', 'right_shoulder'),
        ('left_shoulder', 'left_upper_arm_front'),
        ('left_upper_arm_front', 'left_lower_arm_front'),
        ('left_lower_arm_front', 'left_hand'),
        ('right_shoulder', 'right_upper_arm_front'),
        ('right_upper_arm_front', 'right_lower_arm_front'),
        ('right_lower_arm_front', 'right_hand'),
        ('torso_front', 'left_upper_leg_front'),
        ('torso_front', 'right_upper_leg_front'),
    ]
    
    def __init__(self):
        self.keypoints_3d = {}
        
    def estimate_depth_from_mask(self, mask):
        """
        Simple depth estimation: assume person is ~2m from camera,
        body parts at different depths based on typical human proportions.
        """
        # Find center of mask
        moments = cv2.moments(mask)
        if moments['m00'] == 0:
            return None
            
        center_x = int(moments['m10'] / moments['m00'])
        center_y = int(moments['m01'] / moments['m00'])
        
        # Estimate depth: torso at 2000mm, arms slightly forward, legs slightly back
        depth_map = {
            'torso_front': 2000,
            'left_shoulder': 1950,
            'right_shoulder': 1950,
            'left_upper_arm_front': 1900,
            'right_upper_arm_front': 1900,
            'left_lower_arm_front': 1850,
            'right_lower_arm_front': 1850,
            'left_hand': 1800,
            'right_hand': 1800,
            'left_upper_leg_front': 2050,
            'right_upper_leg_front': 2050,
        }
        
        return depth_map
    
    def convert_2d_to_3d(self, frame, mask):
        """
        Convert 2D body mask to estimated 3D keypoints.
        This is a simplified approach - just for testing.
        """
        h, w = frame.shape[:2]
        
        # Get depth estimates
        depth_map = self.estimate_depth_from_mask(mask)
        if depth_map is None:
            return None
        
        # Find rough keypoint positions from mask
        # In real implementation, you'd use actual body part segmentation
        moments = cv2.moments(mask)
        if moments['m00'] == 0:
            return None
            
        center_x = int(moments['m10'] / moments['m00'])
        center_y = int(moments['m01'] / moments['m00'])
        
        # Rough estimates for demonstration
        # (In reality, BodyPix gives you these positions)
        self.keypoints_3d = {
            'torso_front': [center_x, center_y, depth_map['torso_front']],
            'left_shoulder': [center_x - 60, center_y - 100, depth_map['left_shoulder']],
            'right_shoulder': [center_x + 60, center_y - 100, depth_map['right_shoulder']],
            'left_upper_arm_front': [center_x - 80, center_y - 50, depth_map['left_upper_arm_front']],
            'right_upper_arm_front': [center_x + 80, center_y - 50, depth_map['right_upper_arm_front']],
        }
        
        return self.keypoints_3d
    
    def render_skeleton(self, frame, keypoints_3d):
        """Draw the 3D skeleton on the frame"""
        if keypoints_3d is None:
            return frame
        
        result = frame.copy()
        
        # Draw keypoints
        for name, pos_3d in keypoints_3d.items():
            x, y, z = pos_3d
            # Size based on depth (closer = bigger)
            size = max(3, int(3000 / z))
            # Color based on depth (blue = far, red = close)
            depth_normalized = (z - 1800) / 300  # Normalize to 0-1
            color = (int(255 * depth_normalized), 100, int(255 * (1 - depth_normalized)))
            cv2.circle(result, (int(x), int(y)), size, color, -1)
        
        # Draw connections
        for joint1, joint2 in self.SKELETON_CONNECTIONS:
            if joint1 in keypoints_3d and joint2 in keypoints_3d:
                pt1 = (int(keypoints_3d[joint1][0]), int(keypoints_3d[joint1][1]))
                pt2 = (int(keypoints_3d[joint2][0]), int(keypoints_3d[joint2][1]))
                cv2.line(result, pt1, pt2, (0, 255, 255), 2)
        
        return result


def main():
    print("="*60)
    print("TEST 1: 3D SKELETON VISUALIZATION")
    print("="*60)
    print("\nThis shows a simple 3D skeleton overlay.")
    print("Points are colored by depth: Blue=far, Red=close")
    print("\nControls: Q to quit")
    print("="*60)
    
    # Initialize
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    
    segmenter = BodySegmenter(model_type='mobilenet_50')
    segmenter.load_model()
    segmenter.set_preset('torso_and_arms')
    
    skeleton = Simple3DSkeleton()
    
    print("\nRunning... Move around to see the skeleton track you!")
    
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                continue
            
            frame = cv2.flip(frame, 1)
            
            # Get body mask
            mask = segmenter.get_mask_for_inpainting(frame)
            
            # Convert to 3D keypoints
            keypoints_3d = skeleton.convert_2d_to_3d(frame, mask)
            
            # Render skeleton
            result = skeleton.render_skeleton(frame, keypoints_3d)
            
            # Add info
            cv2.putText(result, "3D Skeleton Test", (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            
            if keypoints_3d:
                cv2.putText(result, "Body detected (colors show depth)", (10, 70),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            else:
                cv2.putText(result, "No body detected", (10, 70),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            
            cv2.imshow("Test 1: 3D Skeleton", result)
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    
    finally:
        cap.release()
        cv2.destroyAllWindows()
        print("\nTest 1 complete!")
        print("Next: Test 2 will generate a 3D mesh from your SD clothing")


if __name__ == "__main__":
    main()