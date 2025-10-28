"""
Create Mock Reference Data for Testing V2 Pipeline
==================================================

This script creates synthetic reference data to test the V2 pipeline
when actual generation data is not available.

Usage:
    python tests/create_mock_reference_data.py \\
        --mesh generated_meshes/0/mesh.obj \\
        --output generated_images/0_reference.pkl

Author: AI Assistant
Date: October 26, 2025
"""

import numpy as np
import cv2
import mediapipe as mp
import pickle
import argparse
import time
from pathlib import Path


def create_mock_bodypix_masks(frame_shape: tuple) -> dict:
    """
    Create synthetic BodyPix masks.
    
    For testing, we'll create simple rectangular masks for common body parts.
    
    Args:
        frame_shape: (height, width)
    
    Returns:
        Dict of {part_name: mask_array}
    """
    h, w = frame_shape
    masks = {}
    
    # Define simple rectangular regions for each body part
    # Format: (y_min, y_max, x_min, x_max) as fractions of frame size
    regions = {
        'torso_front': (0.3, 0.6, 0.35, 0.65),
        'torso_back': (0.3, 0.6, 0.35, 0.65),
        'left_upper_arm_front': (0.3, 0.5, 0.25, 0.35),
        'left_upper_arm_back': (0.3, 0.5, 0.25, 0.35),
        'right_upper_arm_front': (0.3, 0.5, 0.65, 0.75),
        'right_upper_arm_back': (0.3, 0.5, 0.65, 0.75),
        'left_lower_arm_front': (0.5, 0.7, 0.2, 0.3),
        'left_lower_arm_back': (0.5, 0.7, 0.2, 0.3),
        'right_lower_arm_front': (0.5, 0.7, 0.7, 0.8),
        'right_lower_arm_back': (0.5, 0.7, 0.7, 0.8),
    }
    
    for part_name, (y_min, y_max, x_min, x_max) in regions.items():
        mask = np.zeros((h, w), dtype=np.uint8)
        
        y_start = int(y_min * h)
        y_end = int(y_max * h)
        x_start = int(x_min * w)
        x_end = int(x_max * w)
        
        mask[y_start:y_end, x_start:x_end] = 255
        
        masks[part_name] = mask
    
    print(f"✓ Created {len(masks)} mock BodyPix masks")
    return masks


def capture_reference_pose(frame_shape: tuple) -> tuple:
    """
    Capture a reference pose from camera using MediaPipe.
    
    Args:
        frame_shape: (height, width)
    
    Returns:
        (keypoints_2d, keypoints_3d, frame)
    """
    print("\n" + "="*60)
    print("CAPTURING REFERENCE POSE")
    print("="*60)
    print("\nInitializing camera...")
    
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        raise Exception("Could not open camera")
    
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, frame_shape[1])
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, frame_shape[0])
    
    print("✓ Camera initialized")
    print("\nInitializing MediaPipe Pose...")
    
    mp_pose = mp.solutions.pose
    mp_drawing = mp.solutions.drawing_utils
    pose = mp_pose.Pose(
        static_image_mode=False,
        model_complexity=1,
        min_detection_confidence=0.5
    )
    
    print("✓ MediaPipe initialized")
    print("\n" + "="*60)
    print("STAND IN T-POSE")
    print("="*60)
    print("Stand in a T-pose (arms out to the sides)")
    print("Press SPACE to capture reference pose")
    print("Press Q to quit")
    print("="*60 + "\n")
    
    keypoint_indices = {
        'nose': 0,
        'left_shoulder': 11,
        'right_shoulder': 12,
        'left_elbow': 13,
        'right_elbow': 14,
        'left_wrist': 15,
        'right_wrist': 16,
        'left_hip': 23,
        'right_hip': 24,
        'left_knee': 25,
        'right_knee': 26,
        'left_ankle': 27,
        'right_ankle': 28,
    }
    
    captured_frame = None
    keypoints_2d = None
    keypoints_3d = None
    
    while True:
        ret, frame = cap.read()
        if not ret:
            continue
        
        # Process with MediaPipe
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = pose.process(frame_rgb)
        
        # Draw skeleton
        display_frame = frame.copy()
        if results.pose_landmarks:
            mp_drawing.draw_landmarks(
                display_frame,
                results.pose_landmarks,
                mp_pose.POSE_CONNECTIONS
            )
            
            cv2.putText(display_frame, "Pose detected! Press SPACE to capture",
                       (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        else:
            cv2.putText(display_frame, "No pose detected",
                       (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        
        cv2.imshow("Reference Pose Capture", display_frame)
        
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            print("Capture cancelled")
            cap.release()
            cv2.destroyAllWindows()
            return None, None, None
        
        elif key == ord(' ') and results.pose_landmarks:
            # Capture this frame
            captured_frame = frame.copy()
            
            # Extract keypoints
            h, w = frame.shape[:2]
            keypoints_2d = {}
            keypoints_3d = {}
            
            for name, idx in keypoint_indices.items():
                landmark = results.pose_landmarks.landmark[idx]
                
                # 2D keypoints (pixel coordinates)
                x_px = landmark.x * w
                y_px = landmark.y * h
                keypoints_2d[name] = (x_px, y_px, landmark.z)
                
                # 3D keypoints (normalized coordinates)
                keypoints_3d[name] = (landmark.x, landmark.y, landmark.z)
            
            print(f"\n✓ Captured reference pose with {len(keypoints_2d)} keypoints")
            break
    
    cap.release()
    cv2.destroyAllWindows()
    pose.close()
    
    return keypoints_2d, keypoints_3d, captured_frame


def create_mock_reference_data(
    mesh_path: str,
    output_path: str,
    use_camera: bool = True
):
    """
    Create mock reference data.
    
    Args:
        mesh_path: Path to mesh file
        output_path: Path to save reference data
        use_camera: Whether to capture real pose from camera
    """
    print("\n" + "="*60)
    print("CREATING MOCK REFERENCE DATA")
    print("="*60)
    print(f"Mesh: {mesh_path}")
    print(f"Output: {output_path}")
    print("="*60 + "\n")
    
    frame_shape = (720, 1280)  # Standard HD
    
    # Create BodyPix masks
    bodypix_masks = create_mock_bodypix_masks(frame_shape)
    
    # Selected parts (for a t-shirt mesh)
    selected_parts = ['torso', 'left_upper_arm', 'right_upper_arm']
    
    # Capture or create keypoints
    if use_camera:
        keypoints_2d, keypoints_3d, original_frame = capture_reference_pose(frame_shape)
        
        if keypoints_2d is None:
            print("No pose captured, using synthetic data")
            use_camera = False
    
    if not use_camera:
        # Create synthetic keypoints (centered T-pose)
        print("Creating synthetic reference keypoints...")
        h, w = frame_shape
        keypoints_2d = {
            'nose': (w/2, h*0.2, 0.0),
            'left_shoulder': (w*0.35, h*0.3, 0.0),
            'right_shoulder': (w*0.65, h*0.3, 0.0),
            'left_elbow': (w*0.25, h*0.45, 0.0),
            'right_elbow': (w*0.75, h*0.45, 0.0),
            'left_wrist': (w*0.2, h*0.6, 0.0),
            'right_wrist': (w*0.8, h*0.6, 0.0),
            'left_hip': (w*0.4, h*0.6, 0.0),
            'right_hip': (w*0.6, h*0.6, 0.0),
            'left_knee': (w*0.4, h*0.75, 0.0),
            'right_knee': (w*0.6, h*0.75, 0.0),
            'left_ankle': (w*0.4, h*0.9, 0.0),
            'right_ankle': (w*0.6, h*0.9, 0.0),
        }
        keypoints_3d = keypoints_2d.copy()
        original_frame = np.zeros((h, w, 3), dtype=np.uint8) + 128
        print(f"✓ Created {len(keypoints_2d)} synthetic keypoints")
    
    # Create reference data dict
    reference_data = {
        'original_frame': original_frame,
        'bodypix_masks': bodypix_masks,
        'selected_parts': selected_parts,
        'mediapipe_keypoints_2d': keypoints_2d,
        'mediapipe_keypoints_3d': keypoints_3d,
        'frame_shape': frame_shape,
        'mesh_path': mesh_path,
        'timestamp': time.time()
    }
    
    # Save
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'wb') as f:
        pickle.dump(reference_data, f)
    
    # Also save the frame as PNG
    frame_path = output_path.with_name(output_path.stem + '_frame.png')
    cv2.imwrite(str(frame_path), original_frame)
    
    print(f"\n✓ Reference data saved: {output_path}")
    print(f"✓ Reference frame saved: {frame_path}")
    
    print("\n" + "="*60)
    print("MOCK REFERENCE DATA CREATED")
    print("="*60)
    print(f"Selected body parts: {selected_parts}")
    print(f"Keypoints: {len(keypoints_2d)}")
    print(f"BodyPix masks: {len(bodypix_masks)}")
    print("="*60 + "\n")


def main():
    parser = argparse.ArgumentParser(
        description="Create mock reference data for testing V2 pipeline"
    )
    
    parser.add_argument(
        '--mesh',
        type=str,
        required=True,
        help='Path to mesh file'
    )
    
    parser.add_argument(
        '--output',
        type=str,
        required=True,
        help='Path to save reference data (.pkl)'
    )
    
    parser.add_argument(
        '--no-camera',
        action='store_true',
        help='Use synthetic keypoints instead of capturing from camera'
    )
    
    args = parser.parse_args()
    
    if not Path(args.mesh).exists():
        print(f"Error: Mesh file not found: {args.mesh}")
        return
    
    create_mock_reference_data(
        args.mesh,
        args.output,
        use_camera=not args.no_camera
    )


if __name__ == "__main__":
    main()

