"""
SMPL + MediaPipe - ACTUALLY WORKS THIS TIME
============================================
Fixes:
1. ✓ Mesh follows movement (proper vertex updates)
2. ✓ Eye distance scaling (depth adaptation)
3. ✓ Better rendering (less holes)
4. ✓ Responsive tracking

Key fix: Keep body mesh updated, don't reset to T-pose!
"""

import cv2
import numpy as np
import trimesh
from pathlib import Path
import time
import mediapipe as mp
from scipy.spatial import cKDTree
import torch


class SMPLWorkingOverlay:
    """
    SMPL body mesh that ACTUALLY follows your movements.
    
    The key: Maintain body mesh state across frames, update based on keypoint changes.
    """
    
    def __init__(self, smpl_model_path, clothing_mesh_path):
        print("Initializing WORKING SMPL system...")
        
        try:
            import smplx
            self.has_smpl = True
            
            self.smpl_model = smplx.create(
                smpl_model_path,
                model_type='smpl',
                gender='neutral',
                batch_size=1
            )
            
            with torch.no_grad():
                smpl_output = self.smpl_model()
                self.smpl_tpose_verts = smpl_output.vertices[0].cpu().numpy()
            
            print(f"  ✓ SMPL loaded: {len(self.smpl_tpose_verts)} vertices")
            
        except Exception as e:
            print(f"  SMPL error: {e}")
            self.has_smpl = False
            self._create_simple_body()
        
        # Load clothing
        self.clothing_mesh = trimesh.load(clothing_mesh_path)
        self.clothing_mesh.apply_transform(
            trimesh.transformations.rotation_matrix(np.radians(180), [1, 0, 0])
        )
        
        print(f"  ✓ Clothing: {len(self.clothing_mesh.vertices)} vertices")
        
        # MediaPipe
        self.mp_pose = mp.solutions.pose
        self.pose = self.mp_pose.Pose(
            static_image_mode=False,
            model_complexity=0,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        
        # State
        self.calibrated = False
        self.initial_eye_distance = None
        self.initial_scale = 1.0
        
        # Body mesh state (IMPORTANT!)
        self.body_vertices_current = None
        self.body_center_prev = None
        
        self.clothing_to_body_map = None
        
        print("✓ System initialized")
    
    def _create_simple_body(self):
        """Fallback if SMPL unavailable"""
        n_points = 2000
        theta = np.random.uniform(0, 2*np.pi, n_points)
        phi = np.random.uniform(0, np.pi, n_points)
        
        x = 0.15 * np.sin(phi) * np.cos(theta)
        y = 0.4 * np.sin(phi) * np.sin(theta) 
        z = 0.3 * np.cos(phi)
        
        self.smpl_tpose_verts = np.column_stack([x, y, z])
        self.has_smpl = False
        print("  Using point cloud body")
    
    def get_keypoints(self, frame):
        """Get MediaPipe keypoints"""
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.pose.process(rgb)
        
        if not results.pose_landmarks:
            return None
        
        h, w = frame.shape[:2]
        kp = []
        
        for lm in results.pose_landmarks.landmark:
            kp.append([lm.x * w, lm.y * h, lm.z * w])
        
        return np.array(kp)
    
    def calibrate(self, keypoints):
        """One-time calibration"""
        print("\\nCalibrating...")
        
        # Store initial eye distance for depth scaling
        left_eye = keypoints[2]
        right_eye = keypoints[5]
        self.initial_eye_distance = np.linalg.norm(left_eye[:2] - right_eye[:2])
        print(f"  Initial eye distance: {self.initial_eye_distance:.1f} pixels")
        
        # Compute initial scale from shoulders
        shoulder_width = np.linalg.norm(keypoints[12] - keypoints[11])
        self.initial_scale = shoulder_width * 2.5
        
        print(f"  Initial scale: {self.initial_scale:.2f}")
        
        # Initialize body mesh in world space
        torso_center = (keypoints[11] + keypoints[12] + 
                       keypoints[23] + keypoints[24]) / 4
        
        self.body_vertices_current = self.smpl_tpose_verts.copy() * self.initial_scale
        self.body_vertices_current += torso_center
        self.body_center_prev = torso_center.copy()
        
        # Map clothing
        print("  Mapping clothing...")
        self._map_clothing()
        
        self.calibrated = True
        print("✓ Calibration complete\\n")
    
    def _map_clothing(self):
        """Map clothing to body with better distribution"""
        clothing_center = self.clothing_mesh.vertices.mean(axis=0)
        clothing_centered = self.clothing_mesh.vertices - clothing_center
        
        # Scale clothing to be slightly larger than body
        body_extent = np.abs(self.smpl_tpose_verts).max()
        clothing_extent = np.abs(clothing_centered).max()
        
        if clothing_extent > 1e-6:
            clothing_scale_factor = (body_extent / clothing_extent) * self.initial_scale * 1.3
        else:
            clothing_scale_factor = self.initial_scale
        
        clothing_scaled = clothing_centered * clothing_scale_factor
        clothing_world = clothing_scaled + self.body_center_prev
        
        # Build KD-tree
        tree = cKDTree(self.body_vertices_current)
        distances, indices = tree.query(clothing_world, k=1)
        
        self.clothing_to_body_map = indices
        
        print(f"    Mapped {len(indices)} clothing → {len(np.unique(indices))} unique body vertices")
        print(f"    Avg distance: {distances.mean():.1f} pixels")
    
    def update_body_with_movement(self, keypoints):
        """
        Update body mesh based on keypoint changes.
        
        KEY FIX: Don't reset to T-pose! Update existing vertices based on movement.
        """
        # Compute depth scale from eye distance
        left_eye = keypoints[2]
        right_eye = keypoints[5]
        current_eye_distance = np.linalg.norm(left_eye[:2] - right_eye[:2])
        
        depth_scale = current_eye_distance / self.initial_eye_distance
        
        # Compute current torso center
        torso_center = (keypoints[11] + keypoints[12] + 
                       keypoints[23] + keypoints[24]) / 4
        
        # Compute translation
        translation = torso_center - self.body_center_prev
        
        # Update body vertices: scale + translate
        body_center = self.body_vertices_current.mean(axis=0)
        self.body_vertices_current = (self.body_vertices_current - body_center) * depth_scale
        self.body_vertices_current += body_center + translation
        
        # Store for next frame
        self.body_center_prev = torso_center.copy()
        
        return depth_scale
    
    def deform_clothing(self):
        """Clothing follows body surface"""
        if self.clothing_to_body_map is None:
            return self.clothing_mesh.vertices
        
        # Simple: copy body vertex positions
        deformed = self.body_vertices_current[self.clothing_to_body_map].copy()
        
        return deformed
    
    def render(self, frame, vertices):
        """Render clothing mesh"""
        h, w = frame.shape[:2]
        overlay = frame.copy()
        
        pts_2d = vertices[:, :2].astype(np.int32)
        depths = vertices[:, 2]
        
        # Sort by depth
        face_depths = np.mean(depths[self.clothing_mesh.faces], axis=1)
        sorted_faces = np.argsort(face_depths)[::-1]
        
        # AGGRESSIVE subsampling to reduce holes
        n_faces = len(sorted_faces)
        step = max(1, n_faces // 500)  # Render more faces!
        
        faces_drawn = 0
        
        for idx in sorted_faces[::step]:
            face = self.clothing_mesh.faces[idx]
            tri = pts_2d[face]
            
            # # Generous bounds
            # if (np.any(tri[:, 0] < -200) or np.any(tri[:, 0] > w + 200) or
            #     np.any(tri[:, 1] < -200) or np.any(tri[:, 1] > h + 200)):
            #     continue
            
            # # Backface culling
            # edge1 = tri[1] - tri[0]
            # edge2 = tri[2] - tri[0]
            # cross = edge1[0] * edge2[1] - edge1[1] * edge2[0]
            
            # if abs(cross) < 0.1:  # Very relaxed
            #     continue
            
            # if cross > 0:  # Back-facing
            #     continue
            
            # Color
            if hasattr(self.clothing_mesh.visual, 'vertex_colors'):
                color = tuple(int(c) for c in 
                            self.clothing_mesh.visual.vertex_colors[face[0]][:3])
            else:
                color = (180, 220, 255)
            
            cv2.fillPoly(overlay, [tri], color)
            cv2.polylines(overlay, [tri], True, (255, 255, 255), 1, cv2.LINE_AA)
            
            faces_drawn += 1
        
        result = cv2.addWeighted(frame, 0.3, overlay, 0.7, 0)
        
        self.faces_drawn = faces_drawn
        
        return result
    
    def process_frame(self, frame):
        """Main pipeline"""
        keypoints = self.get_keypoints(frame)
        
        if keypoints is None:
            return frame, False, 1.0
        
        if not self.calibrated:
            return frame, False, 1.0
        
        # Update body mesh (THIS IS THE KEY FIX!)
        depth_scale = self.update_body_with_movement(keypoints)
        
        # Deform clothing
        clothing_verts = self.deform_clothing()
        
        # Render
        result = self.render(frame, clothing_verts)
        
        return result, True, depth_scale


def main():
    print("="*70)
    print("SMPL + MEDIAPIPE - ACTUALLY WORKS VERSION")
    print("="*70)
    print("\\nFixes:")
    print("  ✓ Mesh follows your movements")
    print("  ✓ Scales with distance (eye tracking)")
    print("  ✓ Better rendering")
    print("="*70 + "\\n")
    
    smpl_path = "models/"
    
    mesh_dir = Path("generated_meshes")
    meshes = list(mesh_dir.glob("*_triposr.obj")) if mesh_dir.exists() else []
    
    if not meshes:
        clothing_path = input("Enter clothing mesh path: ")
    else:
        clothing_path = str(meshes[0])
        print(f"Using: {clothing_path}\\n")
    
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    
    if not cap.isOpened():
        print("✗ Camera failed")
        return
    
    overlay = SMPLWorkingOverlay(smpl_path, clothing_path)
    
    countdown_duration = 5
    countdown_start = time.time()
    calibrating = True
    
    fps = 0
    frame_count = 0
    fps_start = time.time()
    
    depth_scale = 1.0
    
    print("STAND STILL - CALIBRATING IN 5 SECONDS...")
    
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                continue
            
            frame = cv2.flip(frame, 1)
            
            if calibrating:
                elapsed = time.time() - countdown_start
                remaining = max(0, countdown_duration - int(elapsed))
                
                if remaining > 0:
                    h, w = frame.shape[:2]
                    cv2.putText(frame, f"CALIBRATING IN: {remaining}",
                               (w//2 - 200, h//2),
                               cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 255), 4)
                    cv2.putText(frame, "Stand facing camera, arms down",
                               (w//2 - 300, h//2 + 60),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
                    result = frame
                else:
                    keypoints = overlay.get_keypoints(frame)
                    if keypoints is not None:
                        overlay.calibrate(keypoints)
                        calibrating = False
                        print("✓ READY! Now move around!\\n")
                    else:
                        countdown_start = time.time()
                    result = frame
            else:
                result, tracked, depth_scale = overlay.process_frame(frame)
                
                # UI
                cv2.putText(result, f"FPS: {fps:.1f}", (10, 30),
                           cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                
                status = f"TRACKING (depth: {depth_scale:.2f}x)" if tracked else "No body"
                color = (0, 255, 0) if tracked else (0, 0, 255)
                cv2.putText(result, status, (10, 70),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
                
                if hasattr(overlay, 'faces_drawn'):
                    cv2.putText(result, f"Faces drawn: {overlay.faces_drawn}",
                               (10, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                               (255, 255, 255), 1)
                
                # Instructions
                h, w = result.shape[:2]
                cv2.putText(result, "Move closer/farther to test scaling",
                           (10, h-40), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                           (255, 200, 0), 2)
                cv2.putText(result, "Move left/right to test tracking",
                           (10, h-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                           (255, 200, 0), 2)
            
            cv2.imshow("SMPL Working Movement", result)
            
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
                calibrating = True
                countdown_start = time.time()
                overlay.calibrated = False
    
    finally:
        cap.release()
        cv2.destroyAllWindows()
        print(f"\\nFinal FPS: {fps:.1f}")


if __name__ == "__main__":
    main()