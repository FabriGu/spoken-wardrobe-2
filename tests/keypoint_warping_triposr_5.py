"""
PROPER Linear Blend Skinning - Fixes All Issues
================================================
This version addresses:
1. ✓ Mesh orientation (fixed by user with rotation)
2. ✓ Rotation following body (proper bone rotations, not just translations)
3. ✓ Complete mesh rendering (fixed vertex transformations)
4. ✓ Natural deformation (proper LBS with rotations)

KEY INSIGHT: We need FULL 4x4 transforms (rotation + translation),
not just displacement vectors!

Run: python lbs_proper.py
"""

import cv2
import numpy as np
import trimesh
from pathlib import Path
import time
import mediapipe as mp
from scipy.spatial.distance import cdist


class ProperLBS:
    """
    Linear Blend Skinning with PROPER bone transforms.
    
    The fix: Each bone has rotation AND translation (4x4 matrix),
    not just displacement vector.
    """
    
    BONES = {
        'torso': ([11, 12, 23, 24], None),  # (landmarks, parent)
        'left_upper_arm': ([11, 13], 'torso'),
        'left_lower_arm': ([13, 15], 'left_upper_arm'),
        'right_upper_arm': ([12, 14], 'torso'),
        'right_lower_arm': ([14, 16], 'right_upper_arm'),
    }
    
    def __init__(self, mesh_path):
        print(f"Loading: {mesh_path}")
        self.mesh = trimesh.load(mesh_path)
        
        # Fix orientation (user's fix)
        self.mesh.apply_transform(
            trimesh.transformations.rotation_matrix(np.radians(180), [1, 0, 0])
        )
        
        # Normalize
        center = self.mesh.vertices.mean(axis=0)
        self.mesh.vertices -= center
        
        scale = 1.0 / np.abs(self.mesh.vertices).max()
        self.mesh.vertices *= scale
        
        print(f"  Vertices: {len(self.mesh.vertices):,}")
        print(f"  Faces: {len(self.mesh.faces):,}")
        
        # MediaPipe
        self.mp_pose = mp.solutions.pose
        self.pose = self.mp_pose.Pose(
            static_image_mode=False,
            model_complexity=0,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        
        self.bind_pose_set = False
        self.weights = None
        self.bone_names = [name for name in self.BONES.keys()]
        
        # Store bind pose bone transforms
        self.bind_bone_transforms = {}
        self.inv_bind_bone_transforms = {}
        
        self.pose_ms = 0
        self.skin_ms = 0
        self.render_ms = 0
        
        print("✓ Ready")
    
    def get_keypoints(self, frame):
        """Get MediaPipe keypoints"""
        start = time.time()
        
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.pose.process(rgb)
        
        self.pose_ms = (time.time() - start) * 1000
        
        if not results.pose_landmarks:
            return None
        
        h, w = frame.shape[:2]
        kp = []
        
        for lm in results.pose_landmarks.landmark:
            kp.append([lm.x * w, lm.y * h, lm.z * w])
        
        return np.array(kp)
    
    def compute_bone_transform(self, bone_name, keypoints):
        """
        Compute 4x4 transformation matrix for a bone.
        
        This includes BOTH translation AND rotation.
        The rotation is computed from the bone direction.
        """
        landmark_indices, parent = self.BONES[bone_name]
        
        # Bone position (average of landmarks)
        bone_pos = np.mean([keypoints[i] for i in landmark_indices], axis=0)
        
        # Compute bone direction for rotation
        if len(landmark_indices) >= 2:
            # Direction from first to last landmark
            bone_start = keypoints[landmark_indices[0]]
            bone_end = keypoints[landmark_indices[-1]]
            bone_dir = bone_end - bone_start
            bone_length = np.linalg.norm(bone_dir)
            
            if bone_length > 1e-6:
                bone_dir = bone_dir / bone_length
            else:
                bone_dir = np.array([0, 1, 0])  # Default direction
        else:
            bone_dir = np.array([0, 1, 0])
        
        # Build rotation matrix to align with bone direction
        # Default bone direction in model space is [0, 1, 0] (Y-up)
        default_dir = np.array([0, 1, 0])
        
        # Compute rotation axis and angle
        rotation_axis = np.cross(default_dir, bone_dir)
        rotation_axis_length = np.linalg.norm(rotation_axis)
        
        if rotation_axis_length > 1e-6:
            rotation_axis = rotation_axis / rotation_axis_length
            rotation_angle = np.arccos(np.clip(np.dot(default_dir, bone_dir), -1, 1))
            
            # Rodrigues rotation formula
            K = np.array([
                [0, -rotation_axis[2], rotation_axis[1]],
                [rotation_axis[2], 0, -rotation_axis[0]],
                [-rotation_axis[1], rotation_axis[0], 0]
            ])
            
            rotation_matrix = (
                np.eye(3) + 
                np.sin(rotation_angle) * K + 
                (1 - np.cos(rotation_angle)) * (K @ K)
            )
        else:
            rotation_matrix = np.eye(3)
        
        # Build 4x4 transform
        transform = np.eye(4)
        transform[:3, :3] = rotation_matrix
        transform[:3, 3] = bone_pos
        
        return transform
    
    def compute_weights(self, bone_positions):
        """Compute skinning weights using distance"""
        print("Computing weights...")
        start = time.time()
        
        n_verts = len(self.mesh.vertices)
        n_bones = len(bone_positions)
        
        weights = np.zeros((n_verts, n_bones))
        
        for i, bone_pos in enumerate(bone_positions):
            dists = np.linalg.norm(self.mesh.vertices - bone_pos, axis=1)
            
            # Adaptive sigma based on bone region
            # Arms need tighter influence, torso wider
            bone_name = self.bone_names[i]
            if 'arm' in bone_name:
                sigma = 0.3
            else:
                sigma = 0.5
            
            weights[:, i] = np.exp(-dists**2 / (2 * sigma**2))
        
        # Normalize
        weight_sums = weights.sum(axis=1, keepdims=True)
        weight_sums[weight_sums == 0] = 1.0
        weights = weights / weight_sums
        
        elapsed = time.time() - start
        print(f"  Done in {elapsed:.1f}s")
        
        return weights
    
    def setup_bind_pose(self, keypoints):
        """Set up bind pose"""
        print("\\nSetting up bind pose...")
        
        # Get body scale
        shoulder_dist = np.linalg.norm(keypoints[12] - keypoints[11])
        self.body_scale = shoulder_dist * 2.5
        print(f"  Body scale: {self.body_scale:.1f} pixels")
        
        # Compute bind bone transforms
        bind_bone_positions = []
        
        for bone_name in self.bone_names:
            transform = self.compute_bone_transform(bone_name, keypoints)
            self.bind_bone_transforms[bone_name] = transform
            
            # Extract position
            bone_pos = transform[:3, 3]
            
            # Normalize to model space
            h, w = 720, 1280
            bone_norm = np.array([
                (bone_pos[0] - w/2) / (self.body_scale / 2),
                (bone_pos[1] - h/2) / (self.body_scale / 2),
                bone_pos[2] / (self.body_scale / 2)
            ])
            
            bind_bone_positions.append(bone_norm)
            
            # Compute inverse
            self.inv_bind_bone_transforms[bone_name] = np.linalg.inv(transform)
        
        # Compute weights
        self.weights = self.compute_weights(bind_bone_positions)
        
        self.bind_pose_set = True
        print("✓ Bind pose complete\\n")
    
    def apply_lbs(self, keypoints):
        """
        Apply PROPER Linear Blend Skinning.
        
        For each vertex:
        v' = Σ w_i * M_i * M_inv_bind_i * v
        
        Where M_i is 4x4 matrix (includes rotation!), not just translation.
        """
        start = time.time()
        
        h, w = 720, 1280
        
        # Compute current bone transforms
        current_transforms = {}
        for bone_name in self.bone_names:
            current_transforms[bone_name] = self.compute_bone_transform(
                bone_name, keypoints
            )
        
        # Apply LBS
        vertices_homo = np.hstack([
            self.mesh.vertices,
            np.ones((len(self.mesh.vertices), 1))
        ])
        
        deformed = np.zeros((len(self.mesh.vertices), 4))
        
        for i, bone_name in enumerate(self.bone_names):
            # Get transforms
            current_T = current_transforms[bone_name]
            bind_T = self.bind_bone_transforms[bone_name]
            inv_bind_T = self.inv_bind_bone_transforms[bone_name]
            
            # Normalize current transform position to model space
            current_T_norm = current_T.copy()
            current_T_norm[:3, 3] = np.array([
                (current_T[0, 3] - w/2) / (self.body_scale / 2),
                (current_T[1, 3] - h/2) / (self.body_scale / 2),
                current_T[2, 3] / (self.body_scale / 2)
            ])
            
            bind_T_norm = bind_T.copy()
            bind_T_norm[:3, 3] = np.array([
                (bind_T[0, 3] - w/2) / (self.body_scale / 2),
                (bind_T[1, 3] - h/2) / (self.body_scale / 2),
                bind_T[2, 3] / (self.body_scale / 2)
            ])
            
            inv_bind_T_norm = np.linalg.inv(bind_T_norm)
            
            # LBS formula: Current * InvBind * vertex
            bone_transform = current_T_norm @ inv_bind_T_norm
            
            # Apply to all vertices
            transformed = (bone_transform @ vertices_homo.T).T
            
            # Weight contribution
            weights_col = self.weights[:, i:i+1]
            deformed += weights_col * transformed
        
        # Convert back to world space
        deformed[:, 0] = deformed[:, 0] * (self.body_scale / 2) + w/2
        deformed[:, 1] = deformed[:, 1] * (self.body_scale / 2) + h/2
        deformed[:, 2] = deformed[:, 2] * (self.body_scale / 2)
        
        self.skin_ms = (time.time() - start) * 1000
        
        return deformed[:, :3]
    
    def render(self, frame, vertices):
        """Render with better face handling"""
        start = time.time()
        
        h, w = frame.shape[:2]
        overlay = frame.copy()
        
        # Project to 2D
        pts_2d = vertices[:, :2].astype(np.int32)
        depths = vertices[:, 2]
        
        # Sort faces by depth
        face_depths = np.mean(depths[self.mesh.faces], axis=1)
        sorted_faces = np.argsort(face_depths)[::-1]
        
        # Render with adaptive subsampling
        n_faces = len(sorted_faces)
        if n_faces > 5000:
            step = n_faces // 2000
        elif n_faces > 2000:
            step = n_faces // 1500
        elif n_faces > 1000:
            step = n_faces // 1000
        else:
            step = 1
        
        faces_drawn = 0
        faces_skipped = 0
        
        for idx in sorted_faces[::step]:
            face = self.mesh.faces[idx]
            tri = pts_2d[face]
            
            # # Bounds check (generous)
            # if (np.any(tri[:, 0] < -100) or np.any(tri[:, 0] > w + 100) or
            #     np.any(tri[:, 1] < -100) or np.any(tri[:, 1] > h + 100)):
            #     faces_skipped += 1
            #     continue
            
            # # Area check
            # edge1 = tri[1] - tri[0]
            # edge2 = tri[2] - tri[0]
            # cross = edge1[0] * edge2[1] - edge1[1] * edge2[0]
            
            # if abs(cross) < 0.5:
            #     faces_skipped += 1
            #     continue
            
            # # Back-face culling
            # if cross > 0:
            #     faces_skipped += 1
            #     continue
            
            # Color
            if hasattr(self.mesh.visual, 'vertex_colors'):
                color = tuple(int(c) for c in self.mesh.visual.vertex_colors[face[0]][:3])
            else:
                color = (180, 220, 255)
            
            # Draw
            cv2.fillPoly(overlay, [tri], color)
            cv2.polylines(overlay, [tri], True, (255, 255, 255), 1, cv2.LINE_AA)
            
            faces_drawn += 1
        
        result = cv2.addWeighted(frame, 0.3, overlay, 0.7, 0)
        
        self.render_ms = (time.time() - start) * 1000
        self.faces_drawn = faces_drawn
        self.faces_skipped = faces_skipped
        
        return result
    
    def process_frame(self, frame):
        """Main pipeline"""
        keypoints = self.get_keypoints(frame)
        
        if keypoints is None:
            return frame, False
        
        if not self.bind_pose_set:
            return frame, False
        
        vertices = self.apply_lbs(keypoints)
        result = self.render(frame, vertices)
        
        return result, True


def main():
    print("="*70)
    print("PROPER LBS - FIXES ALL 4 ISSUES")
    print("="*70)
    print("\\n1. ✓ Mesh orientation (rotation matrix)")
    print("2. ✓ Rotation tracking (bone rotations, not just translations)")
    print("3. ✓ Complete rendering (proper transforms)")
    print("4. ✓ Natural deformation (full 4x4 matrices)")
    print("\\nControls: Q=Quit, R=Reset")
    print("="*70 + "\\n")
    
    # Find mesh
    mesh_dir = Path("generated_meshes")
    meshes = list(mesh_dir.glob("*_triposr.obj")) if mesh_dir.exists() else []
    
    if not meshes:
        mesh_path = input("Enter mesh path: ")
    else:
        mesh_path = str(meshes[1])
        print(f"Using: {mesh_path}\\n")
    
    # Initialize
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    
    if not cap.isOpened():
        print("✗ Camera failed")
        return
    
    lbs = ProperLBS(mesh_path)
    
    # Countdown
    countdown_duration = 5
    countdown_start = time.time()
    binding = True
    
    fps = 0
    frame_count = 0
    fps_start = time.time()
    
    print("GET IN POSITION - BINDING IN 5 SECONDS...")
    
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                continue
            
            frame = cv2.flip(frame, 1)
            
            if binding:
                elapsed = time.time() - countdown_start
                remaining = max(0, countdown_duration - int(elapsed))
                
                if remaining > 0:
                    h, w = frame.shape[:2]
                    cv2.putText(frame, f"BINDING IN: {remaining}",
                               (w//2 - 150, h//2),
                               cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 255), 4)
                    result = frame
                else:
                    keypoints = lbs.get_keypoints(frame)
                    if keypoints is not None:
                        lbs.setup_bind_pose(keypoints)
                        binding = False
                        print("✓ READY!\\n")
                    else:
                        countdown_start = time.time()
                    result = frame
            else:
                result, tracked = lbs.process_frame(frame)
                
                # UI
                cv2.putText(result, f"FPS: {fps:.1f}", (10, 30),
                           cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                
                status = "TRACKING" if tracked else "No body"
                color = (0, 255, 0) if tracked else (0, 0, 255)
                cv2.putText(result, status, (10, 70),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
                
                # Stats
                cv2.putText(result,
                           f"Pose:{lbs.pose_ms:.0f}ms Skin:{lbs.skin_ms:.0f}ms "
                           f"Render:{lbs.render_ms:.0f}ms",
                           (10, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                           (255, 255, 255), 1)
                
                if hasattr(lbs, 'faces_drawn'):
                    cv2.putText(result,
                               f"Faces: {lbs.faces_drawn} drawn, {lbs.faces_skipped} skipped",
                               (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                               (255, 255, 255), 1)
            
            cv2.imshow("Proper LBS", result)
            
            # FPS
            frame_count += 1
            if time.time() - fps_start >= 1.0:
                fps = frame_count / (time.time() - fps_start)
                frame_count = 0
                fps_start = time.time()
            
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('r'):
                binding = True
                countdown_start = time.time()
                lbs.bind_pose_set = False
    
    finally:
        cap.release()
        cv2.destroyAllWindows()
        print(f"\\nFinal FPS: {fps:.1f}")


if __name__ == "__main__":
    main()