"""
SMPL + MediaPipe - FIXED VERSION
=================================
Fixed the SMPL parameter error.

Key fixes:
- Correct body_pose dimensions
- Simplified SMPL initialization
- Better error handling
"""

import cv2
import numpy as np
import trimesh
from pathlib import Path
import time
import mediapipe as mp
from scipy.spatial import cKDTree
import torch


class SMPLMediaPipeOverlay:
    """Body mesh intermediate layer approach"""
    
    def __init__(self, smpl_model_path, clothing_mesh_path):
        print("Initializing SMPL + MediaPipe system...")
        
        try:
            import smplx
            self.has_smpl = True
            
            # Load SMPL model - SIMPLIFIED
            self.smpl_model = smplx.create(
                smpl_model_path,
                model_type='smpl',
                gender='neutral',
                batch_size=1
            )
            
            # Get T-pose vertices for reference
            with torch.no_grad():
                # Use SMPL's default parameters (T-pose)
                smpl_output = self.smpl_model()
                self.smpl_tpose_verts = smpl_output.vertices[0].cpu().numpy()
            
            print(f"  ✓ SMPL loaded: {len(self.smpl_tpose_verts)} vertices")
            
        except ImportError:
            print("  ✗ smplx not found. Using simple body.")
            self.has_smpl = False
            self._create_simple_body()
        except Exception as e:
            print(f"  ✗ SMPL error: {e}")
            print("  Using simple body mesh instead.")
            self.has_smpl = False
            self._create_simple_body()
        
        # Load clothing
        self.clothing_mesh = trimesh.load(clothing_mesh_path)
        
        # Fix orientation
        self.clothing_mesh.apply_transform(
            trimesh.transformations.rotation_matrix(np.radians(180), [1, 0, 0])
        )
        
        print(f"  ✓ Clothing loaded: {len(self.clothing_mesh.vertices)} verts")
        
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
        self.body_scale = 1.0
        self.body_offset = np.zeros(3)
        self.clothing_to_body_map = None
        self.body_vertices = None
        
        # Performance tracking
        self.pose_ms = 0
        self.deform_ms = 0
        self.render_ms = 0
        
        print("✓ System initialized")
    
    def _create_simple_body(self):
        """Fallback if SMPL not available"""
        # Create a simple humanoid shape with cylinders
        torso = trimesh.creation.cylinder(radius=0.15, height=0.6)
        
        # Translate to create basic humanoid
        torso.apply_translation([0, 0, 0])
        
        self.smpl_tpose_verts = torso.vertices
        self.has_smpl = False
        
        print("  Using simple capsule body")
    
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
    
    def calibrate(self, keypoints):
        """One-time calibration"""
        print("\\nCalibrating...")
        
        # Compute scale from shoulder width
        shoulder_width = np.linalg.norm(keypoints[12] - keypoints[11])
        
        # SMPL reference (approximate shoulder width in model space)
        smpl_shoulders = self.smpl_tpose_verts[[6890-1000, 6890-2000]]  # Approximate
        smpl_ref_width = np.linalg.norm(smpl_shoulders[1] - smpl_shoulders[0])
        
        if smpl_ref_width < 1e-6:
            smpl_ref_width = 0.4  # Fallback: 40cm in meters
        
        self.body_scale = shoulder_width / smpl_ref_width
        print(f"  Body scale: {self.body_scale:.2f}")
        
        # Compute offset
        torso_center = (keypoints[11] + keypoints[12] + 
                       keypoints[23] + keypoints[24]) / 4
        self.body_offset = torso_center
        print(f"  Body offset: {self.body_offset}")
        
        # Transform body mesh to world space
        self.body_vertices = self.smpl_tpose_verts * self.body_scale
        self.body_vertices += self.body_offset
        
        # Map clothing to body
        print("  Mapping clothing to body...")
        self._map_clothing_to_body()
        
        self.calibrated = True
        print("✓ Calibration complete\\n")
    
    def _map_clothing_to_body(self):
        """Map clothing vertices to nearest body vertices"""
        start = time.time()
        
        # Normalize clothing to match body size
        clothing_center = self.clothing_mesh.vertices.mean(axis=0)
        clothing_centered = self.clothing_mesh.vertices - clothing_center
        
        # Scale clothing
        clothing_size = np.abs(clothing_centered).max()
        body_size = np.abs(self.smpl_tpose_verts).max()
        
        clothing_scale = (body_size / clothing_size) * self.body_scale * 1.2
        clothing_scaled = clothing_centered * clothing_scale
        clothing_world = clothing_scaled + self.body_offset
        
        # Build KD-tree
        body_tree = cKDTree(self.body_vertices)
        
        # Find nearest body vertex for each clothing vertex
        distances, indices = body_tree.query(clothing_world, k=1)
        
        self.clothing_to_body_map = indices
        
        elapsed = time.time() - start
        print(f"    Mapped {len(indices)} vertices in {elapsed:.2f}s")
        print(f"    Avg distance: {distances.mean():.1f} pixels")
    
    def update_body(self, keypoints):
        """
        Update body mesh position.
        
        For now: simple translation (SMPL pose estimation is complex)
        TODO: Implement proper MediaPipe → SMPL pose conversion
        """
        # Update torso position
        torso_center = (keypoints[11] + keypoints[12] + 
                       keypoints[23] + keypoints[24]) / 4
        
        # Translate body vertices
        offset_change = torso_center - self.body_offset
        self.body_vertices = self.smpl_tpose_verts * self.body_scale
        self.body_vertices += torso_center
        
        # TODO: Add rotation based on shoulder/hip orientation
        # This would make it follow body rotation better
    
    def deform_clothing(self):
        """Deform clothing to follow body surface"""
        start = time.time()
        
        if self.clothing_to_body_map is None:
            return self.clothing_mesh.vertices
        
        # Simple: clothing vertices copy body vertex positions
        deformed = self.body_vertices[self.clothing_to_body_map].copy()
        
        # Optional: Add normal offset to prevent z-fighting
        # (Skip for now to keep simple)
        
        self.deform_ms = (time.time() - start) * 1000
        
        return deformed
    
    def render(self, frame, vertices):
        """Render clothing"""
        start = time.time()
        
        h, w = frame.shape[:2]
        overlay = frame.copy()
        
        # Project to 2D
        pts_2d = vertices[:, :2].astype(np.int32)
        depths = vertices[:, 2]
        
        # Sort faces by depth
        face_depths = np.mean(depths[self.clothing_mesh.faces], axis=1)
        sorted_faces = np.argsort(face_depths)[::-1]
        
        # Adaptive subsampling
        n_faces = len(sorted_faces)
        if n_faces > 5000:
            step = n_faces // 2000
        elif n_faces > 2000:
            step = n_faces // 1500
        else:
            step = max(1, n_faces // 1000)
        
        faces_drawn = 0
        
        for idx in sorted_faces[::step]:
            face = self.clothing_mesh.faces[idx]
            tri = pts_2d[face]
            
            # # Bounds check
            # if (np.any(tri[:, 0] < -100) or np.any(tri[:, 0] > w + 100) or
            #     np.any(tri[:, 1] < -100) or np.any(tri[:, 1] > h + 100)):
            #     continue
            
            # # Backface culling
            # edge1 = tri[1] - tri[0]
            # edge2 = tri[2] - tri[0]
            # cross = edge1[0] * edge2[1] - edge1[1] * edge2[0]
            
            # if abs(cross) < 0.5 or cross > 0:
            #     continue
            
            # Color
            if hasattr(self.clothing_mesh.visual, 'vertex_colors'):
                color = tuple(int(c) for c in 
                            self.clothing_mesh.visual.vertex_colors[face[0]][:3])
            else:
                color = (180, 220, 255)
            
            # Draw
            cv2.fillPoly(overlay, [tri], color)
            cv2.polylines(overlay, [tri], True, (255, 255, 255), 1, cv2.LINE_AA)
            
            faces_drawn += 1
        
        result = cv2.addWeighted(frame, 0.3, overlay, 0.7, 0)
        
        self.render_ms = (time.time() - start) * 1000
        self.faces_drawn = faces_drawn
        
        return result
    
    def process_frame(self, frame):
        """Main pipeline"""
        keypoints = self.get_keypoints(frame)
        
        if keypoints is None:
            return frame, False
        
        if not self.calibrated:
            return frame, False
        
        # Update body position
        self.update_body(keypoints)
        
        # Deform clothing
        clothing_verts = self.deform_clothing()
        
        # Render
        result = self.render(frame, clothing_verts)
        
        return result, True


def main():
    print("="*70)
    print("SMPL + MEDIAPIPE - FIXED VERSION")
    print("="*70)
    print("\\nBody mesh as intermediate layer")
    print("MediaPipe → Body Mesh → Clothing")
    print("="*70 + "\\n")
    
    # Paths
    smpl_path = "models/"
    
    # Find clothing mesh
    mesh_dir = Path("generated_meshes")
    meshes = list(mesh_dir.glob("*_triposr.obj")) if mesh_dir.exists() else []
    
    if not meshes:
        clothing_path = input("Enter clothing mesh path: ")
    else:
        clothing_path = str(meshes[1])
        print(f"Using: {clothing_path}\\n")
    
    # Initialize
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    
    if not cap.isOpened():
        print("✗ Camera failed")
        return
    
    overlay = SMPLMediaPipeOverlay(smpl_path, clothing_path)
    
    # Countdown
    countdown_duration = 5
    countdown_start = time.time()
    calibrating = True
    
    fps = 0
    frame_count = 0
    fps_start = time.time()
    
    print("STAND IN T-POSE - CALIBRATING IN 5 SECONDS...")
    
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
                    cv2.putText(frame, f"T-POSE IN: {remaining}",
                               (w//2 - 150, h//2),
                               cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 255), 4)
                    cv2.putText(frame, "Stand facing camera, arms out",
                               (w//2 - 270, h//2 + 60),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
                    result = frame
                else:
                    keypoints = overlay.get_keypoints(frame)
                    if keypoints is not None:
                        overlay.calibrate(keypoints)
                        calibrating = False
                        print("✓ READY!\\n")
                    else:
                        print("No body detected, restarting...")
                        countdown_start = time.time()
                    result = frame
            else:
                result, tracked = overlay.process_frame(frame)
                
                # UI
                cv2.putText(result, f"FPS: {fps:.1f}", (10, 30),
                           cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                
                status = "TRACKING" if tracked else "No body"
                color = (0, 255, 0) if tracked else (0, 0, 255)
                cv2.putText(result, status, (10, 70),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
                
                # Stats
                if hasattr(overlay, 'faces_drawn'):
                    cv2.putText(result,
                               f"Pose:{overlay.pose_ms:.0f}ms "
                               f"Deform:{overlay.deform_ms:.0f}ms "
                               f"Render:{overlay.render_ms:.0f}ms",
                               (10, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                               (255, 255, 255), 1)
                    cv2.putText(result,
                               f"Faces drawn: {overlay.faces_drawn}",
                               (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                               (255, 255, 255), 1)
            
            cv2.imshow("SMPL + MediaPipe", result)
            
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
                print("\\nResetting calibration...")
    
    finally:
        cap.release()
        cv2.destroyAllWindows()
        print(f"\\nFinal FPS: {fps:.1f}")


if __name__ == "__main__":
    main()