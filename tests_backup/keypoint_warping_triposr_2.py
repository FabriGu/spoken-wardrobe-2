"""
Linear Blend Skinning (LBS) Clothing Overlay - DIAGNOSTIC VERSION
===================================================================
Industry-standard skeletal animation with comprehensive debugging.

CRITICAL FIX: Keep mesh in model space, apply alignment after skinning.

NEW DIAGNOSTICS:
- Detailed face culling breakdown (bounds/backface/degenerate)
- Vertex position statistics (3D world space, 2D screen space)
- Winding order toggle (W key) to fix inside-out meshes
- Better scaling using both width and height

Controls:
- Q: Quit
- K: Toggle keypoint visualization
- V: Toggle video feed (mesh only mode)
- S: Toggle performance stats
- D: Toggle debug mode (color by bone influence)
- W: Flip winding order (if mesh appears inside-out)
- R: Reset bind pose (reposition yourself)

Run from root: python tests/test_04_lbs_clothing_overlay.py
"""

import cv2
import numpy as np
import trimesh
from pathlib import Path
import sys
import time
import mediapipe as mp
from scipy.spatial.distance import cdist
from scipy.sparse import lil_matrix
from scipy.sparse.csgraph import dijkstra

sys.path.append(str(Path(__file__).parent.parent))


class LinearBlendSkinningOverlay:
    """
    Linear Blend Skinning system for realistic clothing deformation.
    
    ARCHITECTURE:
    1. Mesh stays in MODEL space (original coordinates)
    2. Bones are computed in WORLD space (relative to body keypoints)
    3. Alignment transform maps model space → world space
    4. LBS deforms in model space, then alignment moves to world space
    """
    
    # Skeleton structure using MediaPipe landmark indices
    SKELETON_BONES = {
        # Core torso
        'spine': [11, 12, 23, 24],  # shoulders + hips
        'chest': [11, 12],  # shoulders
        
        # Left arm chain
        'left_shoulder': [11, 13],
        'left_elbow': [13, 15],
        'left_wrist': [15, 17],
        
        # Right arm chain
        'right_shoulder': [12, 14],
        'right_elbow': [14, 16],
        'right_wrist': [16, 18],
        
        # Lower body
        'left_hip': [23, 25],
        'right_hip': [24, 26],
    }
    
    def __init__(self, mesh_path):
        """Initialize LBS system with clothing mesh"""
        print(f"\n{'='*60}")
        print("INITIALIZING LINEAR BLEND SKINNING SYSTEM")
        print(f"{'='*60}")
        print(f"Loading mesh: {mesh_path}")
        
        # Load mesh - KEEP IN MODEL SPACE
        self.mesh = trimesh.load(mesh_path)
       
        self.mesh = self.mesh.apply_transform(trimesh.transformations.rotation_matrix(
            np.radians(-90), [0, 1, 0]
        ))
        self.mesh = self.mesh.apply_transform(trimesh.transformations.rotation_matrix(
            np.radians(90), [0, 0, 1]
        ))

        self.original_vertices = self.mesh.vertices.copy()
        
        print(f"  Vertices: {len(self.mesh.vertices):,}")
        print(f"  Faces: {len(self.mesh.faces):,}")
        
        # Center and normalize mesh in model space
        self._normalize_mesh()
        
        # Store normalized vertices (this is our MODEL SPACE)
        self.model_vertices = self.mesh.vertices.copy()
        
        # Alignment transform (computed during bind pose)
        self.alignment_transform = np.eye(4)
        
        # Initialize MediaPipe Pose
        self.mp_pose = mp.solutions.pose
        self.mp_drawing = mp.solutions.drawing_utils
        self.pose = self.mp_pose.Pose(
            static_image_mode=False,
            model_complexity=0,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
            smooth_landmarks=True
        )
        
        # LBS components
        self.bind_pose_computed = False
        self.skinning_weights = None
        self.bone_positions_bind = {}
        self.inverse_bind_matrices = {}
        
        # Performance tracking
        self.pose_time_ms = 0
        self.skinning_time_ms = 0
        self.render_time_ms = 0
        self.setup_time_s = 0
        self.faces_drawn = 0
        self.faces_culled = 0
        self.faces_culled_bounds = 0
        self.faces_culled_backface = 0
        self.faces_culled_degenerate = 0
        self.vertex_stats = {}
        
        print("✓ System initialized")
        print(f"{'='*60}\n")
    
    def _normalize_mesh(self):
        """Center and scale mesh to standard MODEL SPACE"""
        # Center at origin
        center = self.mesh.vertices.mean(axis=0)
        self.mesh.vertices -= center
        
        # Scale to reasonable size
        bounds = self.mesh.vertices.max(axis=0) - self.mesh.vertices.min(axis=0)
        target_size = 1.0  # Unit scale in model space
        scale_factor = target_size / max(bounds)
        
        self.mesh.vertices *= scale_factor
        self.mesh_scale = scale_factor
        
        # Store bounds
        self.mesh_bounds = {
            'min': self.mesh.vertices.min(axis=0),
            'max': self.mesh.vertices.max(axis=0),
            'size': self.mesh.vertices.max(axis=0) - self.mesh.vertices.min(axis=0)
        }
        
        print(f"  Normalized to unit scale: {scale_factor:.4f}")
        print(f"  Model space size: {self.mesh_bounds['size']}")
    
    def _compute_alignment_transform(self, keypoints_3d):
        """
        Compute transformation to align model space mesh to world space body.
        
        Key insight: Match mesh size to body torso dimensions, not just shoulder width.
        """
        print("  Computing alignment transform...")
        
        # Get body measurements
        left_shoulder = keypoints_3d[11]
        right_shoulder = keypoints_3d[12]
        left_hip = keypoints_3d[23]
        right_hip = keypoints_3d[24]
        
        shoulder_center = (left_shoulder + right_shoulder) / 2
        hip_center = (left_hip + right_hip) / 2
        torso_center = (shoulder_center + hip_center) / 2
        
        # Body dimensions
        shoulder_width = np.linalg.norm(left_shoulder - right_shoulder)
        torso_height = np.linalg.norm(shoulder_center - hip_center)
        
        print(f"    Body shoulder width: {shoulder_width:.1f}")
        print(f"    Body torso height: {torso_height:.1f}")
        print(f"    Body torso center: {torso_center}")
        
        # Mesh dimensions in model space
        mesh_width = self.mesh_bounds['size'][0]
        mesh_height = self.mesh_bounds['size'][1]
        
        print(f"    Mesh model width: {mesh_width:.3f}")
        print(f"    Mesh model height: {mesh_height:.3f}")
        
        # Scale to match body
        # Use the average of width and height scaling for more balanced fit
        scale_x = shoulder_width / mesh_width
        scale_y = torso_height / mesh_height
        scale = (scale_x + scale_y) / 2.0 * 1.3  # Average + 30% larger
        
        print(f"    Scale factors: X={scale_x:.2f}, Y={scale_y:.2f}")
        print(f"    Final scale: {scale:.2f}")
        
        # Build transform
        alignment = np.eye(4)
        alignment[0, 0] = scale
        alignment[1, 1] = scale
        alignment[2, 2] = scale
        alignment[:3, 3] = torso_center
        
        return alignment
    
    def _build_mesh_adjacency(self):
        """Build adjacency matrix for geodesic distance computation"""
        n_vertices = len(self.model_vertices)
        adjacency = lil_matrix((n_vertices, n_vertices))
        
        for edge in self.mesh.edges_unique:
            i, j = edge
            dist = np.linalg.norm(
                self.model_vertices[i] - self.model_vertices[j]
            )
            adjacency[i, j] = dist
            adjacency[j, i] = dist
        
        return adjacency
    
    def _compute_geodesic_skinning_weights(self, bone_positions_world):
        """
        Compute skinning weights using geodesic distances.
        
        Bones are in world space, mesh is in model space.
        We need to transform bone positions to model space for distance calculation.
        """
        print("\nComputing skinning weights...")
        print("  This may take 10-30 seconds (one-time setup)")
        start_time = time.time()
        
        n_vertices = len(self.model_vertices)
        bone_names = list(bone_positions_world.keys())
        n_bones = len(bone_names)
        
        # Transform bones from world space to model space
        alignment_inv = np.linalg.inv(self.alignment_transform)
        bone_positions_model = {}
        
        for bone_name, world_pos in bone_positions_world.items():
            # Transform to homogeneous coordinates
            world_pos_h = np.append(world_pos, 1.0)
            model_pos_h = alignment_inv @ world_pos_h
            bone_positions_model[bone_name] = model_pos_h[:3]
        
        print("  Building mesh graph...")
        adjacency = self._build_mesh_adjacency()
        
        # Find anchor vertex for each bone in model space
        print("  Finding bone anchors...")
        bone_anchors = []
        for bone_name in bone_names:
            bone_pos = bone_positions_model[bone_name]
            distances = np.linalg.norm(
                self.model_vertices - bone_pos, axis=1
            )
            anchor_idx = np.argmin(distances)
            bone_anchors.append(anchor_idx)
            print(f"    {bone_name}: vertex {anchor_idx}")
        
        # Compute geodesic distances
        print("  Computing geodesic distances (this is the slow part)...")
        geodesic_distances = dijkstra(
            adjacency,
            directed=False,
            indices=bone_anchors,
            limit=np.inf
        )
        
        # Convert to weights with adaptive sigma
        mesh_size = np.mean(self.mesh_bounds['size'])
        sigma = mesh_size * 2.0  # Wider influence in model space
        print(f"  Converting to weights (sigma={sigma:.4f})...")
        
        weights = np.exp(-geodesic_distances**2 / (2 * sigma**2))
        weights[np.isinf(geodesic_distances)] = 0.0
        
        # Normalize
        weight_sums = weights.sum(axis=0, keepdims=True)
        weight_sums[weight_sums == 0] = 1.0
        weights = weights / weight_sums
        
        elapsed = time.time() - start_time
        self.setup_time_s = elapsed
        
        print(f"  ✓ Weights computed in {elapsed:.1f}s")
        print(f"  Weight matrix shape: {weights.T.shape}")
        
        return weights.T, bone_names
    
    def _get_bone_position(self, bone_name, keypoints_3d):
        """Calculate 3D position of a bone from MediaPipe keypoints"""
        landmark_indices = self.SKELETON_BONES[bone_name]
        positions = [keypoints_3d[idx] for idx in landmark_indices]
        bone_pos = np.mean(positions, axis=0)
        return bone_pos
    
    def _compute_bone_transform_matrix(self, bone_name, keypoints_3d):
        """Compute 4x4 transformation matrix for a bone in world space"""
        bone_pos = self._get_bone_position(bone_name, keypoints_3d)
        
        transform = np.eye(4)
        transform[:3, 3] = bone_pos
        
        return transform, bone_pos
    
    def setup_bind_pose(self, keypoints_3d):
        """
        Set up bind pose and compute skinning weights.
        
        Key insight: Mesh stays in model space, we compute alignment transform
        to map it to world space.
        """
        print(f"\n{'='*60}")
        print("SETTING UP BIND POSE")
        print(f"{'='*60}")
        
        # STEP 1: Compute alignment transform (model → world)
        self.alignment_transform = self._compute_alignment_transform(keypoints_3d)
        
        # STEP 2: Compute bone positions in world space
        bone_positions = {}
        bind_transforms = {}
        
        for bone_name in self.SKELETON_BONES.keys():
            transform, bone_pos = self._compute_bone_transform_matrix(
                bone_name, keypoints_3d
            )
            bone_positions[bone_name] = bone_pos
            bind_transforms[bone_name] = transform
        
        # STEP 3: Compute skinning weights
        # (Bones in world space, mesh in model space)
        self.skinning_weights, self.bone_names = \
            self._compute_geodesic_skinning_weights(bone_positions)
        
        # STEP 4: Compute inverse bind pose matrices
        # These are in WORLD space (where bones are)
        self.inverse_bind_matrices = {}
        for bone_name in self.bone_names:
            self.inverse_bind_matrices[bone_name] = \
                np.linalg.inv(bind_transforms[bone_name])
        
        self.bind_pose_computed = True
        
        print(f"{'='*60}")
        print("✓ BIND POSE SETUP COMPLETE")
        print(f"{'='*60}\n")
    
    def get_body_keypoints_3d(self, frame):
        """Extract 3D keypoints from frame using MediaPipe"""
        start_time = time.time()
        
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.pose.process(rgb)
        
        self.pose_time_ms = (time.time() - start_time) * 1000
        
        if not results.pose_landmarks:
            return None, None
        
        h, w = frame.shape[:2]
        keypoints_3d = []
        
        # Convert to world space (centered at screen center)
        for lm in results.pose_landmarks.landmark:
            keypoints_3d.append([
                (lm.x - 0.5) * w,
                (lm.y - 0.5) * h,
                lm.z * 1000
            ])
        
        return np.array(keypoints_3d), results.pose_landmarks
    
    def apply_linear_blend_skinning(self, keypoints_3d):
        """
        Apply LBS to deform mesh.
        
        Process:
        1. Start with model space vertices
        2. Transform to bone-local space (using inverse bind pose)
        3. Transform to current world space (using current bone pose)
        4. Blend all bone influences
        5. Result is in world space, ready to render
        """
        start_time = time.time()
        
        # Compute current bone transforms (world space)
        current_transforms = {}
        for bone_name in self.bone_names:
            transform, _ = self._compute_bone_transform_matrix(
                bone_name, keypoints_3d
            )
            current_transforms[bone_name] = transform
        
        # Apply LBS to model space vertices
        vertices_homogeneous = np.hstack([
            self.model_vertices,
            np.ones((len(self.model_vertices), 1))
        ])
        
        deformed_vertices = np.zeros((len(self.model_vertices), 3))
        
        for i, bone_name in enumerate(self.bone_names):
            # Composite transform: 
            # 1. Model → World (alignment)
            # 2. World → Bone-local (inverse bind)
            # 3. Bone-local → World (current transform)
            bone_transform = (
                current_transforms[bone_name] @
                self.inverse_bind_matrices[bone_name] @
                self.alignment_transform
            )
            
            # Apply to vertices
            transformed = (bone_transform @ vertices_homogeneous.T).T[:, :3]
            
            # Weight contribution
            weights = self.skinning_weights[:, i:i+1]
            deformed_vertices += weights * transformed
        
        self.skinning_time_ms = (time.time() - start_time) * 1000
        
        return deformed_vertices
    
    def render_mesh(self, frame, vertices, show_video=True, debug_mode=False, flip_winding=False):
        """Render deformed mesh onto frame with diagnostic info"""
        start_time = time.time()
        
        h, w = frame.shape[:2]
        
        # Diagnostic: Check vertex positions
        v_min = vertices.min(axis=0)
        v_max = vertices.max(axis=0)
        v_mean = vertices.mean(axis=0)
        
        # Project to 2D
        points_2d = vertices[:, :2].copy()
        points_2d[:, 0] += w / 2
        points_2d[:, 1] += h / 2
        points_2d = points_2d.astype(np.int32)
        
        # Diagnostic: Check 2D projections
        p2d_min = points_2d.min(axis=0)
        p2d_max = points_2d.max(axis=0)
        
        # Store for display
        self.vertex_stats = {
            '3d_min': v_min,
            '3d_max': v_max,
            '3d_mean': v_mean,
            '2d_min': p2d_min,
            '2d_max': p2d_max
        }
        
        depths = vertices[:, 2]
        
        # Create base
        if show_video:
            overlay = frame.copy()
        else:
            overlay = np.zeros_like(frame)
        
        # Sort faces by depth
        face_depths = np.mean(depths[self.mesh.faces], axis=1)
        sorted_indices = np.argsort(face_depths)[::-1]
        
        # Less aggressive sampling for debugging
        n_faces = len(sorted_indices)
        if n_faces > 5000:
            step = n_faces // 2000  # Try more faces
        elif n_faces > 2000:
            step = n_faces // 1500
        else:
            step = max(1, n_faces // 1000)
        
        faces_drawn = 0
        faces_culled_bounds = 0
        faces_culled_backface = 0
        faces_culled_degenerate = 0
        
        # Render faces
        for idx in sorted_indices[::step]:
            face = self.mesh.faces[idx]
            pts = points_2d[face]
            
            # Bounds check (relaxed)
            if not (np.all(pts[:, 0] >= -100) and np.all(pts[:, 0] < w + 100) and
                    np.all(pts[:, 1] >= -100) and np.all(pts[:, 1] < h + 100)):
                faces_culled_bounds += 1
                continue
            
            # Compute area (cross product)
            edge1 = pts[1] - pts[0]
            edge2 = pts[2] - pts[0]
            cross = edge1[0] * edge2[1] - edge1[1] * edge2[0]
            
            # Degenerate check (RELAXED)
            area = abs(cross) / 2.0
            if area < 0.5:  # Very small triangles
                faces_culled_degenerate += 1
                continue
            
            # Back-face culling (allow toggling direction)
            cull_threshold = 0 if not flip_winding else 0
            if (cross < cull_threshold) if not flip_winding else (cross > cull_threshold):
                faces_culled_backface += 1
                continue
            
            # Determine color
            if debug_mode and self.skinning_weights is not None:
                vertex_weights = self.skinning_weights[face[0]]
                dominant_bone_idx = np.argmax(vertex_weights)
                
                colors = [
                    (255, 100, 100), (100, 255, 100), (100, 100, 255),
                    (255, 255, 100), (255, 100, 255), (100, 255, 255),
                    (255, 150, 100), (150, 100, 255), (200, 200, 100),
                    (100, 200, 200)
                ]
                color = colors[dominant_bone_idx % len(colors)]
            elif hasattr(self.mesh.visual, 'vertex_colors'):
                color = self.mesh.visual.vertex_colors[face[0]][:3]
                color = tuple(int(c) for c in color)
            else:
                color = (180, 220, 255)
            
            # Draw filled triangle
            cv2.fillPoly(overlay, [pts], color)
            cv2.polylines(overlay, [pts], True, (255, 255, 255), 1, cv2.LINE_AA)
            
            faces_drawn += 1
        
        # Blend
        if show_video:
            result = cv2.addWeighted(frame, 0.3, overlay, 0.7, 0)
        else:
            result = overlay
        
        self.render_time_ms = (time.time() - start_time) * 1000
        self.faces_drawn = faces_drawn
        self.faces_culled = faces_culled_bounds + faces_culled_backface + faces_culled_degenerate
        self.faces_culled_bounds = faces_culled_bounds
        self.faces_culled_backface = faces_culled_backface
        self.faces_culled_degenerate = faces_culled_degenerate
        
        return result
    
    def draw_keypoints(self, frame, pose_landmarks):
        """Draw MediaPipe skeleton overlay"""
        if pose_landmarks is None:
            return
        
        self.mp_drawing.draw_landmarks(
            frame,
            pose_landmarks,
            self.mp_pose.POSE_CONNECTIONS,
            self.mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=3),
            self.mp_drawing.DrawingSpec(color=(0, 255, 255), thickness=2)
        )
    
    def process_frame(self, frame, show_keypoints=False, show_video=True, debug_mode=False, flip_winding=False):
        """Main processing pipeline"""
        keypoints_3d, pose_landmarks = self.get_body_keypoints_3d(frame)
        
        if keypoints_3d is None:
            return frame, None
        
        if show_keypoints:
            self.draw_keypoints(frame, pose_landmarks)
        
        if not self.bind_pose_computed:
            return frame, keypoints_3d
        
        deformed_vertices = self.apply_linear_blend_skinning(keypoints_3d)
        result = self.render_mesh(frame, deformed_vertices, show_video, debug_mode, flip_winding)
        
        if show_keypoints and show_video:
            self.draw_keypoints(result, pose_landmarks)
        
        return result, keypoints_3d


def countdown_display(frame, seconds_remaining):
    """Display countdown overlay"""
    h, w = frame.shape[:2]
    overlay = frame.copy()
    
    cv2.rectangle(overlay, (0, 0), (w, h), (0, 0, 0), -1)
    result = cv2.addWeighted(frame, 0.3, overlay, 0.7, 0)
    
    text = f"GET IN POSITION: {seconds_remaining}"
    font_scale = 2.5
    thickness = 4
    
    (text_w, text_h), _ = cv2.getTextSize(
        text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness
    )
    
    text_x = (w - text_w) // 2
    text_y = (h + text_h) // 2
    
    cv2.putText(result, text, (text_x, text_y),
               cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 255, 255), thickness)
    
    instructions = [
        "Stand with arms slightly away from body",
        "Make sure all keypoints are visible",
        "Stay still when countdown reaches 0"
    ]
    
    y_offset = text_y + 80
    for instruction in instructions:
        (inst_w, inst_h), _ = cv2.getTextSize(
            instruction, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2
        )
        inst_x = (w - inst_w) // 2
        cv2.putText(result, instruction, (inst_x, y_offset),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        y_offset += 40
    
    return result


def main():
    print("\n" + "="*70)
    print("LINEAR BLEND SKINNING - REAL-TIME CLOTHING OVERLAY")
    print("="*70)
    print("\nFIXED VERSION: Mesh stays in model space")
    print("Alignment applied after skinning (no vertex scatter)")
    print("\nControls:")
    print("  Q - Quit")
    print("  K - Toggle keypoint visualization")
    print("  V - Toggle video feed (mesh only mode)")
    print("  S - Toggle performance stats")
    print("  D - Toggle debug mode (color by bone influence)")
    print("  W - Flip winding order (if faces look inside-out)")
    print("  R - Reset bind pose (reposition yourself)")
    print("="*70 + "\n")
    
    # Find mesh
    mesh_dir = Path("generated_meshes")
    if not mesh_dir.exists():
        print(f"✗ Directory not found: {mesh_dir}")
        return
    
    triposr_meshes = sorted(list(mesh_dir.glob("*_triposr.obj")))
    
    if len(triposr_meshes) == 0:
        print("✗ No TripoSR meshes found!")
        return
    
    print(f"Found {len(triposr_meshes)} TripoSR mesh(es):")
    for i, mesh in enumerate(triposr_meshes[:5], 1):
        print(f"  {i}. {mesh.name}")
    
    selected_mesh = triposr_meshes[0]
    print(f"\nUsing: {selected_mesh.name}")
    
    # Initialize camera
    print("\nInitializing camera...")
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    
    if not cap.isOpened():
        print("✗ Could not open camera")
        return
    
    print("✓ Camera opened")
    
    # Initialize LBS
    lbs = LinearBlendSkinningOverlay(str(selected_mesh))
    
    # UI state
    show_keypoints = True
    show_video = True
    show_stats = True
    debug_mode = False
    flip_winding = False
    
    # Countdown
    countdown_duration = 6
    countdown_start = time.time()
    countdown_active = True
    
    # FPS tracking
    fps = 0
    frame_count = 0
    fps_start = time.time()
    
    print("\n" + "="*70)
    print("STARTING IN 6 SECONDS - GET INTO POSITION!")
    print("="*70)
    
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                continue
            
            frame = cv2.flip(frame, 1)
            
            if countdown_active:
                elapsed = time.time() - countdown_start
                remaining = max(0, countdown_duration - int(elapsed))
                
                if remaining > 0:
                    result = countdown_display(frame, remaining)
                else:
                    countdown_active = False
                    print("\n" + "="*70)
                    print("CAPTURING BIND POSE...")
                    print("="*70)
                    
                    keypoints_3d, _ = lbs.get_body_keypoints_3d(frame)
                    
                    if keypoints_3d is not None:
                        lbs.setup_bind_pose(keypoints_3d)
                        print("\n✓ Ready! Move your arms to see the magic!")
                    else:
                        print("\n✗ No body detected! Press R to restart.")
                        countdown_active = True
                        countdown_start = time.time()
                    
                    result = frame
            else:
                result, keypoints = lbs.process_frame(
                    frame, show_keypoints, show_video, debug_mode, flip_winding
                )
                
                h, w = result.shape[:2]
                
                # FPS
                cv2.putText(result, f"FPS: {fps:.1f}", (10, 30),
                           cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                
                # Status
                if keypoints is not None:
                    if lbs.bind_pose_computed:
                        cv2.putText(result, "LBS Active - Move Your Arms!", (10, 70),
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                    else:
                        cv2.putText(result, "Body Tracked - Setting up...", (10, 70),
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 200, 0), 2)
                else:
                    cv2.putText(result, "No Body Detected", (10, 70),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                
                # Stats
                if show_stats and lbs.bind_pose_computed:
                    total = lbs.pose_time_ms + lbs.skinning_time_ms + lbs.render_time_ms
                    
                    stats_y = 110
                    line_height = 22
                    font_size = 0.45
                    
                    cv2.putText(result, f"Pose: {lbs.pose_time_ms:.1f}ms",
                               (10, stats_y), cv2.FONT_HERSHEY_SIMPLEX, font_size,
                               (255, 255, 255), 1)
                    cv2.putText(result, f"Skinning: {lbs.skinning_time_ms:.1f}ms",
                               (10, stats_y + line_height), cv2.FONT_HERSHEY_SIMPLEX,
                               font_size, (255, 255, 255), 1)
                    cv2.putText(result, f"Render: {lbs.render_time_ms:.1f}ms",
                               (10, stats_y + line_height*2), cv2.FONT_HERSHEY_SIMPLEX,
                               font_size, (255, 255, 255), 1)
                    cv2.putText(result, f"Total: {total:.1f}ms",
                               (10, stats_y + line_height*3), cv2.FONT_HERSHEY_SIMPLEX,
                               font_size, (255, 255, 0), 1)
                    
                    # Detailed culling info
                    cv2.putText(result, f"Faces drawn: {lbs.faces_drawn}",
                               (10, stats_y + line_height*4), cv2.FONT_HERSHEY_SIMPLEX,
                               font_size, (0, 255, 0), 1)
                    cv2.putText(result, f"Culled bounds: {lbs.faces_culled_bounds}",
                               (10, stats_y + line_height*5), cv2.FONT_HERSHEY_SIMPLEX,
                               font_size, (200, 200, 200), 1)
                    cv2.putText(result, f"Culled backface: {lbs.faces_culled_backface}",
                               (10, stats_y + line_height*6), cv2.FONT_HERSHEY_SIMPLEX,
                               font_size, (200, 200, 200), 1)
                    cv2.putText(result, f"Culled degen: {lbs.faces_culled_degenerate}",
                               (10, stats_y + line_height*7), cv2.FONT_HERSHEY_SIMPLEX,
                               font_size, (200, 200, 200), 1)
                    
                    # Vertex diagnostics
                    if lbs.vertex_stats:
                        stats_x2 = w - 350
                        cv2.putText(result, "Vertex Stats:",
                                   (stats_x2, stats_y), cv2.FONT_HERSHEY_SIMPLEX,
                                   font_size, (255, 255, 0), 1)
                        
                        mean_3d = lbs.vertex_stats['3d_mean']
                        cv2.putText(result, f"3D mean: ({mean_3d[0]:.0f}, {mean_3d[1]:.0f}, {mean_3d[2]:.0f})",
                                   (stats_x2, stats_y + line_height), cv2.FONT_HERSHEY_SIMPLEX,
                                   font_size, (200, 200, 200), 1)
                        
                        min_2d = lbs.vertex_stats['2d_min']
                        max_2d = lbs.vertex_stats['2d_max']
                        cv2.putText(result, f"2D X: [{min_2d[0]}, {max_2d[0]}]",
                                   (stats_x2, stats_y + line_height*2), cv2.FONT_HERSHEY_SIMPLEX,
                                   font_size, (200, 200, 200), 1)
                        cv2.putText(result, f"2D Y: [{min_2d[1]}, {max_2d[1]}]",
                                   (stats_x2, stats_y + line_height*3), cv2.FONT_HERSHEY_SIMPLEX,
                                   font_size, (200, 200, 200), 1)
                
                # Mode indicators
                mode_y = h - 40
                mode_x = 10
                if not show_video:
                    cv2.putText(result, "MESH ONLY", (mode_x, mode_y),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 255), 2)
                    mode_x += 150
                if show_keypoints:
                    cv2.putText(result, "KEYPOINTS", (mode_x, mode_y),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                    mode_x += 150
                if debug_mode:
                    cv2.putText(result, "DEBUG", (mode_x, mode_y),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
                    mode_x += 110
                if flip_winding:
                    cv2.putText(result, "FLIPPED", (mode_x, mode_y),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 128, 0), 2)
            
            cv2.imshow("LBS Clothing Overlay", result)
            
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
                print(f"Keypoints: {'ON' if show_keypoints else 'OFF'}")
            elif key == ord('v'):
                show_video = not show_video
                print(f"Display: {'VIDEO + MESH' if show_video else 'MESH ONLY'}")
            elif key == ord('s'):
                show_stats = not show_stats
                print(f"Stats: {'ON' if show_stats else 'OFF'}")
            elif key == ord('d'):
                debug_mode = not debug_mode
                print(f"Debug: {'ON' if debug_mode else 'OFF'}")
            elif key == ord('w'):
                flip_winding = not flip_winding
                print(f"Winding: {'FLIPPED' if flip_winding else 'NORMAL'}")
            elif key == ord('r'):
                if not countdown_active:
                    print("\n" + "="*70)
                    print("RESETTING BIND POSE")
                    print("="*70)
                    lbs.bind_pose_computed = False
                    countdown_active = True
                    countdown_start = time.time()
    
    finally:
        cap.release()
        cv2.destroyAllWindows()
        
        print("\n" + "="*70)
        print("SESSION COMPLETE")
        print("="*70)
        if lbs.bind_pose_computed:
            print(f"Final FPS: {fps:.1f}")
            print(f"Faces rendered: {lbs.faces_drawn}")
            print("\nYou should have seen:")
            print("  ✓ Solid clothing mesh (not dots)")
            print("  ✓ Mesh following body movement")
            print("  ✓ Smooth deformations at joints")


if __name__ == "__main__":
    main()