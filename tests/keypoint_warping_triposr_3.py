"""
Cage-Based Clothing Overlay - ACTUALLY WORKING VERSION
=======================================================
Uses a simpler, more direct approach: deform mesh based on body cage movement.

Why this works better than LBS for clothing:
- No complex bone hierarchies needed
- Direct vertex-to-body-part mapping
- Meshes actually stay together and deform naturally
- Much simpler to debug and understand

Controls:
- Q: Quit
- K: Toggle keypoint visualization  
- V: Toggle video feed
- S: Toggle stats
- D: Toggle debug (show vertex regions)
- R: Reset bind pose

Run: python tests/test_04_lbs_clothing_overlay.py
"""

import cv2
import numpy as np
import trimesh
from pathlib import Path
import sys
import time
import mediapipe as mp

sys.path.append(str(Path(__file__).parent.parent))


class CageBasedClothingOverlay:
    """
    Cage-based deformation: mesh deforms based on movement of body 'cage' points.
    
    Key insight: Instead of complex skeletal animation, we:
    1. Define a cage (control points) from body keypoints
    2. Bind each mesh vertex to nearby cage points
    3. Move vertices when cage points move
    4. Much simpler, actually works!
    """
    
    # Body keypoints we care about
    KEYPOINT_INDICES = {
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
    }
    
    def __init__(self, mesh_path):
        print(f"\n{'='*60}")
        print("CAGE-BASED CLOTHING DEFORMATION")
        print(f"{'='*60}")
        print(f"Loading: {mesh_path}")
        
        # Load mesh
        self.mesh = trimesh.load(mesh_path)
        print(f"  Vertices: {len(self.mesh.vertices):,}")
        print(f"  Faces: {len(self.mesh.faces):,}")
        
        # Center and normalize in model space
        center = self.mesh.vertices.mean(axis=0)
        self.mesh.vertices -= center
        
        bounds = self.mesh.vertices.max(axis=0) - self.mesh.vertices.min(axis=0)
        self.mesh.vertices /= max(bounds)  # Normalize to unit cube
        
        self.original_vertices = self.mesh.vertices.copy()
        
        # MediaPipe
        self.mp_pose = mp.solutions.pose
        self.mp_drawing = mp.solutions.drawing_utils
        self.pose = self.mp_pose.Pose(
            static_image_mode=False,
            model_complexity=0,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
            smooth_landmarks=True
        )
        
        # Cage binding (computed during bind pose)
        self.bind_pose_set = False
        self.cage_weights = None  # (n_vertices, n_cage_points)
        self.cage_bind = None  # Bind pose cage positions
        self.scale_factor = 1.0
        self.translation = np.zeros(3)
        
        # Stats
        self.pose_time_ms = 0
        self.deform_time_ms = 0
        self.render_time_ms = 0
        self.faces_drawn = 0
        
        print("✓ System ready")
        print(f"{'='*60}\n")
    
    def get_body_keypoints(self, frame):
        """Get MediaPipe keypoints"""
        start = time.time()
        
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.pose.process(rgb)
        
        self.pose_time_ms = (time.time() - start) * 1000
        
        if not results.pose_landmarks:
            return None, None
        
        h, w = frame.shape[:2]
        keypoints = {}
        
        for name, idx in self.KEYPOINT_INDICES.items():
            lm = results.pose_landmarks.landmark[idx]
            keypoints[name] = np.array([
                (lm.x - 0.5) * w,
                (lm.y - 0.5) * h,
                lm.z * 1000
            ])
        
        return keypoints, results.pose_landmarks
    
    def setup_bind_pose(self, keypoints):
        """
        Set up cage binding from first frame.
        
        This is the CRITICAL function - it determines how mesh maps to body.
        """
        print(f"\n{'='*60}")
        print("SETTING UP CAGE BINDING")
        print(f"{'='*60}")
        
        # Build cage from keypoints (these are our control points)
        cage_points = []
        cage_names = []
        
        for name in ['left_shoulder', 'right_shoulder', 'left_elbow', 
                     'right_elbow', 'left_wrist', 'right_wrist',
                     'left_hip', 'right_hip']:
            if name in keypoints:
                cage_points.append(keypoints[name])
                cage_names.append(name)
        
        # Add torso center
        torso = (keypoints['left_shoulder'] + keypoints['right_shoulder'] +
                 keypoints['left_hip'] + keypoints['right_hip']) / 4
        cage_points.append(torso)
        cage_names.append('torso_center')
        
        self.cage_bind = np.array(cage_points)
        self.cage_names = cage_names
        
        print(f"  Cage has {len(cage_points)} control points")
        
        # Compute scale to match body
        shoulder_width = np.linalg.norm(
            keypoints['left_shoulder'] - keypoints['right_shoulder']
        )
        torso_height = np.linalg.norm(
            (keypoints['left_shoulder'] + keypoints['right_shoulder']) / 2 -
            (keypoints['left_hip'] + keypoints['right_hip']) / 2
        )
        
        # Mesh is 1.0 units, body is ~shoulder_width pixels
        self.scale_factor = max(shoulder_width, torso_height) * 1.5
        self.translation = torso.copy()
        
        print(f"  Scale: {self.scale_factor:.1f}")
        print(f"  Translation: {self.translation}")
        
        # Transform mesh to world space for weight computation
        world_vertices = self.original_vertices * self.scale_factor + self.translation
        
        # Compute cage weights using inverse distance weighting
        print("  Computing cage weights...")
        n_vertices = len(world_vertices)
        n_cage = len(self.cage_bind)
        
        weights = np.zeros((n_vertices, n_cage))
        
        for i, vertex in enumerate(world_vertices):
            # Distance to each cage point
            distances = np.linalg.norm(self.cage_bind - vertex, axis=1)
            
            # Inverse distance weighting with power 2
            # Closer points have more influence
            inv_dist = 1.0 / (distances + 1e-6) ** 2
            
            # Normalize to sum to 1
            weights[i] = inv_dist / inv_dist.sum()
        
        self.cage_weights = weights
        
        # Show weight distribution
        dominant_cage = np.argmax(weights, axis=1)
        for i, name in enumerate(cage_names):
            count = np.sum(dominant_cage == i)
            print(f"    {name}: {count} vertices")
        
        self.bind_pose_set = True
        
        print(f"{'='*60}")
        print("✓ BIND POSE COMPLETE")
        print(f"{'='*60}\n")
    
    def deform_mesh(self, keypoints):
        """
        Deform mesh based on current cage positions.
        
        This is where the magic happens - move vertices based on
        how their cage control points moved.
        """
        start = time.time()
        
        # Build current cage
        cage_current = []
        for name in self.cage_names:
            if name == 'torso_center':
                torso = (keypoints['left_shoulder'] + keypoints['right_shoulder'] +
                        keypoints['left_hip'] + keypoints['right_hip']) / 4
                cage_current.append(torso)
            else:
                cage_current.append(keypoints[name])
        
        cage_current = np.array(cage_current)
        
        # Compute cage displacement
        cage_delta = cage_current - self.cage_bind
        
        # Deform vertices: weighted sum of cage displacements
        vertex_delta = self.cage_weights @ cage_delta
        
        # Apply to bind pose (in world space)
        world_bind = self.original_vertices * self.scale_factor + self.translation
        deformed = world_bind + vertex_delta
        
        self.deform_time_ms = (time.time() - start) * 1000
        
        return deformed
    
    def render_mesh(self, frame, vertices, show_video=True, debug_mode=False):
        """Render deformed mesh"""
        start = time.time()
        
        h, w = frame.shape[:2]
        
        # Project to 2D
        points_2d = vertices[:, :2].copy()
        points_2d[:, 0] += w / 2
        points_2d[:, 1] += h / 2
        points_2d = points_2d.astype(np.int32)
        
        depths = vertices[:, 2]
        
        # Base image
        if show_video:
            overlay = frame.copy()
        else:
            overlay = np.zeros_like(frame)
        
        # Sort by depth
        face_depths = np.mean(depths[self.mesh.faces], axis=1)
        sorted_indices = np.argsort(face_depths)[::-1]
        
        # Render with reasonable sampling
        n_faces = len(sorted_indices)
        step = max(1, n_faces // 1000)
        
        faces_drawn = 0
        
        for idx in sorted_indices[::step]:
            face = self.mesh.faces[idx]
            pts = points_2d[face]
            
            # # Simple bounds check
            # if not (np.all(pts[:, 0] >= 0) and np.all(pts[:, 0] < w) and
            #         np.all(pts[:, 1] >= 0) and np.all(pts[:, 1] < h)):
            #     continue
            
            # # Back-face culling
            # edge1 = pts[1] - pts[0]
            # edge2 = pts[2] - pts[0]
            # cross = edge1[0] * edge2[1] - edge1[1] * edge2[0]
            
            # if cross <= 0:
            #     continue
            
            # Color
            if debug_mode and self.cage_weights is not None:
                # Color by dominant cage point
                dominant = np.argmax(self.cage_weights[face[0]])
                colors = [
                    (255, 100, 100), (100, 255, 100), (100, 100, 255),
                    (255, 255, 100), (255, 100, 255), (100, 255, 255),
                    (255, 150, 100), (150, 100, 255), (200, 200, 100)
                ]
                color = colors[dominant % len(colors)]
            elif hasattr(self.mesh.visual, 'vertex_colors'):
                color = tuple(int(c) for c in self.mesh.visual.vertex_colors[face[0]][:3])
            else:
                color = (180, 220, 255)
            
            # Draw
            cv2.fillPoly(overlay, [pts], color)
            cv2.polylines(overlay, [pts], True, (255, 255, 255), 1, cv2.LINE_AA)
            
            faces_drawn += 1
        
        # Blend
        if show_video:
            result = cv2.addWeighted(frame, 0.3, overlay, 0.7, 0)
        else:
            result = overlay
        
        self.render_time_ms = (time.time() - start) * 1000
        self.faces_drawn = faces_drawn
        
        return result
    
    def draw_keypoints(self, frame, pose_landmarks):
        """Draw skeleton"""
        if pose_landmarks:
            self.mp_drawing.draw_landmarks(
                frame, pose_landmarks, self.mp_pose.POSE_CONNECTIONS,
                self.mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=3),
                self.mp_drawing.DrawingSpec(color=(0, 255, 255), thickness=2)
            )
    
    def process_frame(self, frame, show_keypoints=False, show_video=True, debug_mode=False):
        """Main pipeline"""
        keypoints, landmarks = self.get_body_keypoints(frame)
        
        if keypoints is None:
            return frame, None
        
        if show_keypoints:
            self.draw_keypoints(frame, landmarks)
        
        if not self.bind_pose_set:
            return frame, keypoints
        
        deformed = self.deform_mesh(keypoints)
        result = self.render_mesh(frame, deformed, show_video, debug_mode)
        
        if show_keypoints and show_video:
            self.draw_keypoints(result, landmarks)
        
        return result, keypoints


def countdown_display(frame, seconds_remaining):
    """Countdown overlay"""
    h, w = frame.shape[:2]
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (w, h), (0, 0, 0), -1)
    result = cv2.addWeighted(frame, 0.3, overlay, 0.7, 0)
    
    text = f"POSITION YOURSELF: {seconds_remaining}"
    (text_w, text_h), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 2.5, 4)
    
    cv2.putText(result, text, ((w - text_w) // 2, (h + text_h) // 2),
               cv2.FONT_HERSHEY_SIMPLEX, 2.5, (0, 255, 255), 4)
    
    instructions = [
        "Stand facing camera, arms slightly out",
        "Full body visible including hips",
        "Stay still when countdown hits 0"
    ]
    
    y = (h + text_h) // 2 + 80
    for inst in instructions:
        (w_inst, h_inst), _ = cv2.getTextSize(inst, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)
        cv2.putText(result, inst, ((w - w_inst) // 2, y),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        y += 40
    
    return result


def main():
    print("\n" + "="*70)
    print("CAGE-BASED CLOTHING OVERLAY - WORKING VERSION")
    print("="*70)
    print("\nSimpler approach that actually works!")
    print("Uses body cage deformation instead of complex bone math")
    print("\nControls:")
    print("  Q - Quit")
    print("  K - Toggle keypoints")
    print("  V - Toggle video")
    print("  S - Toggle stats")
    print("  D - Debug (show vertex regions)")
    print("  R - Reset bind pose")
    print("="*70 + "\n")
    
    # Find mesh
    mesh_dir = Path("generated_meshes")
    if not mesh_dir.exists():
        print("✗ No generated_meshes folder")
        return
    
    meshes = sorted(list(mesh_dir.glob("*_triposr.obj")))
    if len(meshes) == 0:
        print("✗ No meshes found")
        return
    
    print(f"Found {len(meshes)} mesh(es)")
    for i, m in enumerate(meshes[:5], 1):
        print(f"  {i}. {m.name}")
    
    mesh_path = meshes[0]
    print(f"\nUsing: {mesh_path.name}\n")
    
    # Initialize
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    
    if not cap.isOpened():
        print("✗ Camera failed")
        return
    
    system = CageBasedClothingOverlay(str(mesh_path))
    
    # State
    show_keypoints = True
    show_video = True
    show_stats = True
    debug_mode = False
    
    countdown = 6
    countdown_start = time.time()
    countdown_active = True
    
    fps = 0
    frame_count = 0
    fps_start = time.time()
    
    print("="*70)
    print("STARTING - 6 SECOND COUNTDOWN")
    print("="*70 + "\n")
    
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                continue
            
            frame = cv2.flip(frame, 1)
            
            if countdown_active:
                elapsed = time.time() - countdown_start
                remaining = max(0, countdown - int(elapsed))
                
                if remaining > 0:
                    result = countdown_display(frame, remaining)
                else:
                    countdown_active = False
                    print("CAPTURING BIND POSE...")
                    
                    keypoints, _ = system.get_body_keypoints(frame)
                    if keypoints:
                        system.setup_bind_pose(keypoints)
                        print("✓ Ready! Move your arms!")
                    else:
                        print("✗ No body detected - press R to retry")
                        countdown_active = True
                        countdown_start = time.time()
                    
                    result = frame
            else:
                result, keypoints = system.process_frame(
                    frame, show_keypoints, show_video, debug_mode
                )
                
                h, w = result.shape[:2]
                
                # FPS
                cv2.putText(result, f"FPS: {fps:.1f}", (10, 30),
                           cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                
                # Status
                if keypoints:
                    if system.bind_pose_set:
                        cv2.putText(result, "Active - Move Your Arms!", (10, 70),
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                    else:
                        cv2.putText(result, "Tracking...", (10, 70),
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 200, 0), 2)
                else:
                    cv2.putText(result, "No Body Detected", (10, 70),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                
                # Stats
                if show_stats and system.bind_pose_set:
                    total = system.pose_time_ms + system.deform_time_ms + system.render_time_ms
                    y = 110
                    cv2.putText(result, f"Pose: {system.pose_time_ms:.1f}ms", (10, y),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                    cv2.putText(result, f"Deform: {system.deform_time_ms:.1f}ms", (10, y+25),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                    cv2.putText(result, f"Render: {system.render_time_ms:.1f}ms", (10, y+50),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                    cv2.putText(result, f"Total: {total:.1f}ms", (10, y+75),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)
                    cv2.putText(result, f"Faces: {system.faces_drawn}", (10, y+100),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
                
                # Modes
                mode_y = h - 40
                if not show_video:
                    cv2.putText(result, "MESH ONLY", (10, mode_y),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 255), 2)
                if debug_mode:
                    cv2.putText(result, "DEBUG", (200, mode_y),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
            
            cv2.imshow("Cage-Based Clothing Overlay", result)
            
            # FPS
            frame_count += 1
            if time.time() - fps_start >= 1.0:
                fps = frame_count / (time.time() - fps_start)
                frame_count = 0
                fps_start = time.time()
            
            # Controls
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('k'):
                show_keypoints = not show_keypoints
            elif key == ord('v'):
                show_video = not show_video
            elif key == ord('s'):
                show_stats = not show_stats
            elif key == ord('d'):
                debug_mode = not debug_mode
            elif key == ord('r'):
                if not countdown_active:
                    print("\nResetting bind pose...")
                    system.bind_pose_set = False
                    countdown_active = True
                    countdown_start = time.time()
    
    finally:
        cap.release()
        cv2.destroyAllWindows()
        
        if system.bind_pose_set:
            print(f"\nFinal FPS: {fps:.1f}")
            print(f"Faces rendered: {system.faces_drawn}")
            print("\nYou should have seen a solid, deforming mesh!")


if __name__ == "__main__":
    main()