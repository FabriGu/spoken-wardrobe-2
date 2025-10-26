# keypoint_to_cage_mapper.py
# Maps MediaPipe 2D keypoints to 3D cage vertex positions

import numpy as np
import cv2


class KeypointToCageMapper:
    """
    Maps MediaPipe pose keypoints to cage vertex positions.
    Uses simple anatomical landmark correspondence.
    """
    
    # MediaPipe pose landmark indices
    MEDIAPIPE_LANDMARKS = {
        'left_shoulder': 11,
        'right_shoulder': 12,
        'left_elbow': 13,
        'right_elbow': 14,
        'left_wrist': 15,
        'right_wrist': 16,
        'left_hip': 23,
        'right_hip': 24,
    }
    
    def __init__(self, camera_params=None):
        """
        Initialize mapper.
        
        Args:
            camera_params: Dict with 'fx', 'fy', 'cx', 'cy' for camera intrinsics
        """
        # Default camera params (adjust for your camera)
        if camera_params is None:
            self.camera_params = {
                'fx': 800,  # focal length x
                'fy': 800,  # focal length y
                'cx': 640,  # principal point x
                'cy': 360,  # principal point y
            }
        else:
            self.camera_params = camera_params
        
        self.previous_cage_positions = None
    
    def unproject_2d_to_3d(self, keypoint_2d, depth_value):
        """
        Unproject a 2D keypoint to 3D using depth.
        
        Args:
            keypoint_2d: [x, y] pixel coordinates
            depth_value: depth at that pixel (in meters or arbitrary units)
            
        Returns:
            point_3d: [x, y, z] 3D coordinates
        """
        x_2d, y_2d = keypoint_2d
        
        # Unproject using pinhole camera model
        x_3d = (x_2d - self.camera_params['cx']) * depth_value / self.camera_params['fx']
        y_3d = (y_2d - self.camera_params['cy']) * depth_value / self.camera_params['fy']
        z_3d = depth_value
        
        return np.array([x_3d, y_3d, z_3d])
    
    def map_keypoints_to_cage(self, mediapipe_landmarks, depth_map, cage_mesh, frame_shape):
        """
        Map MediaPipe keypoints to cage vertex positions.
        
        Args:
            mediapipe_landmarks: MediaPipe pose landmarks
            depth_map: HxW depth map (same size as video frame)
            cage_mesh: Original cage mesh (trimesh object)
            frame_shape: (height, width) of the video frame
            
        Returns:
            deformed_cage_vertices: New positions for cage vertices
        """
        height, width = frame_shape[:2]
        
        # Convert MediaPipe landmarks to 3D positions
        keypoints_3d = {}
        
        for landmark_name, landmark_idx in self.MEDIAPIPE_LANDMARKS.items():
            if landmark_idx < len(mediapipe_landmarks.landmark):
                landmark = mediapipe_landmarks.landmark[landmark_idx]
                
                # MediaPipe gives normalized coordinates [0, 1]
                x_2d = int(landmark.x * width)
                y_2d = int(landmark.y * height)
                
                # Clamp to valid range
                x_2d = np.clip(x_2d, 0, width - 1)
                y_2d = np.clip(y_2d, 0, height - 1)
                
                # Get depth at this keypoint
                depth = depth_map[y_2d, x_2d]
                
                # Unproject to 3D
                point_3d = self.unproject_2d_to_3d([x_2d, y_2d], depth)
                keypoints_3d[landmark_name] = point_3d
        
        # Now map these keypoints to cage vertices
        # For simple prototype: find nearest cage vertex to each keypoint
        cage_vertices = np.array(cage_mesh.vertices).copy()
        
        # Simple mapping: assign keypoint positions to nearest cage vertices
        # This is a simplified approach for the prototype
        from scipy.spatial import cKDTree
        tree = cKDTree(cage_vertices)
        
        for landmark_name, keypoint_3d in keypoints_3d.items():
            # Find nearest cage vertex
            dist, idx = tree.query(keypoint_3d, k=1)
            
            # Update cage vertex position
            # Use a blend to avoid snapping
            blend_factor = 0.5
            cage_vertices[idx] = (1 - blend_factor) * cage_vertices[idx] + blend_factor * keypoint_3d
        
        # Smooth temporally
        if self.previous_cage_positions is not None:
            alpha = 0.3  # Smoothing factor
            cage_vertices = alpha * cage_vertices + (1 - alpha) * self.previous_cage_positions
        
        self.previous_cage_positions = cage_vertices
        
        return cage_vertices
    
    def simple_anatomical_mapping(self, mediapipe_landmarks, cage_mesh, frame_shape):
        """
        Simplified version that doesn't require depth map.
        Uses only 2D keypoint positions and estimates depth.
        
        Args:
            mediapipe_landmarks: MediaPipe pose landmarks
            cage_mesh: Original cage mesh
            frame_shape: (height, width) of the video frame
            
        Returns:
            deformed_cage_vertices: New positions for cage vertices
        """
        height, width = frame_shape[:2]
        cage_vertices = np.array(cage_mesh.vertices).copy()
        
        # Get bounding box center of cage
        cage_center = cage_vertices.mean(axis=0)
        cage_size = cage_vertices.max(axis=0) - cage_vertices.min(axis=0)
        
        # Extract key landmarks
        landmarks_2d = {}
        for landmark_name, landmark_idx in self.MEDIAPIPE_LANDMARKS.items():
            if landmark_idx < len(mediapipe_landmarks.landmark):
                landmark = mediapipe_landmarks.landmark[landmark_idx]
                x = landmark.x * width
                y = landmark.y * height
                landmarks_2d[landmark_name] = np.array([x, y])
        
        # Calculate scale factor from MediaPipe pose
        if 'left_shoulder' in landmarks_2d and 'right_shoulder' in landmarks_2d:
            shoulder_width_pixels = np.linalg.norm(
                landmarks_2d['right_shoulder'] - landmarks_2d['left_shoulder']
            )
            
            # Estimate scale (this is simplified - assumes person is ~50cm shoulder width)
            # Adjust this based on your setup
            estimated_shoulder_width_3d = 0.5  # meters
            scale_factor = estimated_shoulder_width_3d / (shoulder_width_pixels / width)
        else:
            scale_factor = 1.0
        
        # Transform cage vertices based on body pose
        # This is a very simplified version - just translate and scale
        
        # Calculate torso center from MediaPipe
        if 'left_shoulder' in landmarks_2d and 'right_shoulder' in landmarks_2d:
            torso_center_2d = (landmarks_2d['left_shoulder'] + landmarks_2d['right_shoulder']) / 2
            
            # Convert to normalized coordinates [-1, 1]
            torso_x = (torso_center_2d[0] / width - 0.5) * 2
            torso_y = -(torso_center_2d[1] / height - 0.5) * 2  # Flip y
            
            # Apply translation
            translation = np.array([torso_x * scale_factor, torso_y * scale_factor, 0])
            cage_vertices += translation - cage_center
        
        # Apply arm-specific deformations
        for arm_name in ['left_upper_arm', 'right_upper_arm', 'left_lower_arm', 'right_lower_arm']:
            if arm_name in landmarks_2d:
                arm_center_2d = landmarks_2d[arm_name]
                
                # Convert to normalized coordinates
                arm_x = (arm_center_2d[0] / width - 0.5) * 2
                arm_y = -(arm_center_2d[1] / height - 0.5) * 2
                
                # Find cage vertices that should move with this arm
                arm_position = np.array([arm_x * scale_factor, arm_y * scale_factor, 0])
                
                # Move cage vertices that are closest to arm position
                distances = np.linalg.norm(cage_vertices - arm_position, axis=1)
                closest_indices = np.argsort(distances)[:len(cage_vertices)//8]  # Move top 12.5%
                
                # Blend arm position with current cage position
                blend_factor = 0.2
                for idx in closest_indices:
                    cage_vertices[idx] = (1 - blend_factor) * cage_vertices[idx] + blend_factor * arm_position
        
        # Smooth temporally
        if self.previous_cage_positions is not None:
            alpha = 0.3
            cage_vertices = alpha * cage_vertices + (1 - alpha) * self.previous_cage_positions
        
        self.previous_cage_positions = cage_vertices
        
        return cage_vertices
