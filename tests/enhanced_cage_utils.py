# enhanced_cage_utils.py
# Enhanced cage-based deformation utilities with BodyPix integration
# Provides intelligent cage generation based on body part segmentation

import numpy as np
import trimesh
from scipy.spatial import cKDTree, ConvexHull
from scipy.ndimage import binary_dilation, binary_erosion


class BodyPixCageGenerator:
    """
    Generate cages based on BodyPix body part segmentation.
    Creates intelligent cage placement that follows anatomical structure.
    """
    
    def __init__(self, mesh):
        """
        Initialize cage generator with a clothing mesh.
        
        Args:
            mesh: trimesh object of the clothing mesh
        """
        self.mesh = mesh
        self.cage = None
        
        # BodyPix part names mapping
        self.body_part_groups = {
            'torso': ['torso_front', 'torso_back'],
            'left_upper_arm': ['left_upper_arm_front', 'left_upper_arm_back'],
            'right_upper_arm': ['right_upper_arm_front', 'right_upper_arm_back'],
            'left_lower_arm': ['left_lower_arm_front', 'left_lower_arm_back'],
            'right_lower_arm': ['right_lower_arm_front', 'right_lower_arm_back'],
            'left_hand': ['left_hand'],
            'right_hand': ['right_hand'],
        }
    
    def generate_anatomical_cage(self, segmentation_data, frame_shape, subdivisions=3):
        """
        Generate cage based on BodyPix segmentation data.
        
        Args:
            segmentation_data: Dict with body part masks from BodyPix
            frame_shape: Shape of the video frame
            subdivisions: Number of subdivisions for cage density
            
        Returns:
            cage_mesh: trimesh object of the generated cage
        """
        height, width = frame_shape[:2]
        
        # Extract body part masks
        body_parts = segmentation_data['body_parts']
        
        # Generate cage vertices for each body part
        cage_vertices = []
        cage_labels = []  # Track which vertices belong to which body part
        
        for part_name, part_group in self.body_part_groups.items():
            if part_name in body_parts:
                part_mask = body_parts[part_name]
                
                # Generate cage vertices for this body part
                part_vertices = self.generate_part_cage_vertices(
                    part_mask, part_name, subdivisions
                )
                
                cage_vertices.extend(part_vertices)
                cage_labels.extend([part_name] * len(part_vertices))
        
        if not cage_vertices:
            # Fallback to simple box cage
            return self.generate_simple_box_cage(subdivisions)
        
        cage_vertices = np.array(cage_vertices)
        
        # Create cage mesh using convex hull
        try:
            hull = ConvexHull(cage_vertices)
            cage_faces = hull.simplices
        except:
            # Fallback if convex hull fails
            cage_faces = self.create_simple_faces(len(cage_vertices))
        
        self.cage = trimesh.Trimesh(vertices=cage_vertices, faces=cage_faces)
        self.cage_labels = cage_labels
        
        print(f"Generated anatomical cage with {len(cage_vertices)} vertices")
        return self.cage
    
    def generate_part_cage_vertices(self, part_mask, part_name, subdivisions):
        """
        Generate cage vertices for a specific body part.
        
        Args:
            part_mask: Binary mask of the body part
            part_name: Name of the body part
            subdivisions: Number of subdivisions
            
        Returns:
            vertices: List of 3D vertices for this body part
        """
        vertices = []
        
        # Find contours of the body part
        contours, _ = cv2.findContours(
            part_mask.astype(np.uint8), 
            cv2.RETR_EXTERNAL, 
            cv2.CHAIN_APPROX_SIMPLE
        )
        
        if not contours:
            return vertices
        
        # Get the largest contour
        largest_contour = max(contours, key=cv2.contourArea)
        
        # Sample points along the contour
        contour_points = largest_contour.reshape(-1, 2)
        
        # Add points at regular intervals
        n_points = max(8, len(contour_points) // subdivisions)
        if len(contour_points) > n_points:
            indices = np.linspace(0, len(contour_points) - 1, n_points, dtype=int)
            contour_points = contour_points[indices]
        
        # Convert 2D contour points to 3D cage vertices
        for point in contour_points:
            x, y = point
            
            # Convert to normalized coordinates
            x_norm = (x / part_mask.shape[1] - 0.5) * 2
            y_norm = -(y / part_mask.shape[0] - 0.5) * 2  # Flip Y
            
            # Estimate depth based on body part
            z_norm = self.estimate_body_part_depth(part_name)
            
            vertices.append([x_norm, y_norm, z_norm])
        
        # Add internal points for better cage coverage
        internal_vertices = self.add_internal_cage_points(
            part_mask, part_name, subdivisions
        )
        vertices.extend(internal_vertices)
        
        return vertices
    
    def estimate_body_part_depth(self, part_name):
        """
        Estimate depth for different body parts.
        This is a simplified approach - in practice you'd use depth estimation.
        """
        depth_map = {
            'torso': 0.0,
            'left_upper_arm': -0.1,
            'right_upper_arm': 0.1,
            'left_lower_arm': -0.15,
            'right_lower_arm': 0.15,
            'left_hand': -0.2,
            'right_hand': 0.2,
        }
        return depth_map.get(part_name, 0.0)
    
    def add_internal_cage_points(self, part_mask, part_name, subdivisions):
        """
        Add internal cage points for better deformation control.
        """
        vertices = []
        
        # Find bounding box of the body part
        coords = np.where(part_mask > 0)
        if len(coords[0]) == 0:
            return vertices
        
        y_min, y_max = coords[0].min(), coords[0].max()
        x_min, x_max = coords[1].min(), coords[1].max()
        
        # Add internal grid points
        for i in range(subdivisions):
            for j in range(subdivisions):
                x = x_min + (x_max - x_min) * (i + 1) / (subdivisions + 1)
                y = y_min + (y_max - y_min) * (j + 1) / (subdivisions + 1)
                
                # Check if point is inside the mask
                if (0 <= x < part_mask.shape[1] and 
                    0 <= y < part_mask.shape[0] and 
                    part_mask[int(y), int(x)] > 0):
                    
                    # Convert to normalized coordinates
                    x_norm = (x / part_mask.shape[1] - 0.5) * 2
                    y_norm = -(y / part_mask.shape[0] - 0.5) * 2
                    z_norm = self.estimate_body_part_depth(part_name)
                    
                    vertices.append([x_norm, y_norm, z_norm])
        
        return vertices
    
    def generate_simple_box_cage(self, subdivisions=2):
        """
        Fallback: Generate a simple subdivided box cage.
        """
        # Get bounding box of mesh
        bbox_min = self.mesh.vertices.min(axis=0)
        bbox_max = self.mesh.vertices.max(axis=0)
        
        # Expand bbox slightly
        padding = 0.1 * (bbox_max - bbox_min)
        bbox_min -= padding
        bbox_max += padding
        
        # Create subdivided box
        cage_vertices = []
        
        x_vals = np.linspace(bbox_min[0], bbox_max[0], subdivisions + 1)
        y_vals = np.linspace(bbox_min[1], bbox_max[1], subdivisions + 1)
        z_vals = np.linspace(bbox_min[2], bbox_max[2], subdivisions + 1)
        
        for x in x_vals:
            for y in y_vals:
                for z in z_vals:
                    cage_vertices.append([x, y, z])
        
        cage_vertices = np.array(cage_vertices)
        
        # Create faces
        try:
            hull = ConvexHull(cage_vertices)
            cage_faces = hull.simplices
        except:
            cage_faces = self.create_simple_faces(len(cage_vertices))
        
        self.cage = trimesh.Trimesh(vertices=cage_vertices, faces=cage_faces)
        
        print(f"Generated simple box cage with {len(cage_vertices)} vertices")
        return self.cage
    
    def create_simple_faces(self, n_vertices):
        """
        Create simple faces for cage vertices.
        """
        # This is a very basic face generation
        # In practice, you'd want more sophisticated triangulation
        faces = []
        
        # Create faces by connecting nearby vertices
        # This is simplified - just create some basic triangles
        for i in range(0, n_vertices - 2, 3):
            if i + 2 < n_vertices:
                faces.append([i, i + 1, i + 2])
        
        return np.array(faces)


class EnhancedMeanValueCoordinates:
    """
    Enhanced Mean Value Coordinates with better performance and stability.
    """
    
    def __init__(self, mesh_vertices, cage_mesh):
        """
        Initialize enhanced MVC calculator.
        
        Args:
            mesh_vertices: Nx3 array of mesh vertex positions
            cage_mesh: trimesh object of the cage
        """
        self.mesh_vertices = np.array(mesh_vertices)
        self.cage_vertices = np.array(cage_mesh.vertices)
        self.cage_faces = np.array(cage_mesh.faces)
        self.mvc_weights = None
        
        # Precompute some values for efficiency
        self.mesh_center = self.mesh_vertices.mean(axis=0)
        self.cage_center = self.cage_vertices.mean(axis=0)
        
    def compute_weights(self):
        """
        Compute MVC weights with improved stability and performance.
        """
        n_mesh_verts = len(self.mesh_vertices)
        n_cage_verts = len(self.cage_vertices)
        
        weights = np.zeros((n_mesh_verts, n_cage_verts))
        
        print(f"Computing enhanced MVC weights for {n_mesh_verts} mesh vertices...")
        
        # Use vectorized operations for better performance
        for i, v in enumerate(self.mesh_vertices):
            if i % 1000 == 0:
                print(f"  Processing vertex {i}/{n_mesh_verts}...")
            
            # Compute distances to all cage vertices
            distances = np.linalg.norm(self.cage_vertices - v, axis=1)
            
            # Add small epsilon to avoid division by zero
            distances = np.maximum(distances, 1e-8)
            
            # Use inverse distance weighting with power 2
            inv_dist = 1.0 / (distances ** 2)
            
            # Normalize weights to sum to 1
            weights[i] = inv_dist / inv_dist.sum()
        
        self.mvc_weights = weights
        print("Enhanced MVC weights computed!")
        return weights
    
    def deform_mesh(self, deformed_cage_vertices):
        """
        Deform mesh using enhanced MVC with temporal smoothing.
        """
        if self.mvc_weights is None:
            raise ValueError("Must compute weights first!")
        
        # Apply MVC deformation
        deformed_vertices = self.mvc_weights @ deformed_cage_vertices
        
        # Apply temporal smoothing if previous frame exists
        if hasattr(self, 'previous_vertices'):
            alpha = 0.3  # Smoothing factor
            deformed_vertices = (alpha * deformed_vertices + 
                               (1 - alpha) * self.previous_vertices)
        
        self.previous_vertices = deformed_vertices.copy()
        
        return deformed_vertices
    
    def deform_mesh_with_constraints(self, deformed_cage_vertices, constraints=None):
        """
        Deform mesh with additional constraints for better stability.
        
        Args:
            deformed_cage_vertices: New cage vertex positions
            constraints: Dict with constraint information
        """
        # Basic MVC deformation
        deformed_vertices = self.deform_mesh(deformed_cage_vertices)
        
        if constraints is None:
            return deformed_vertices
        
        # Apply constraints
        if 'max_displacement' in constraints:
            max_disp = constraints['max_displacement']
            displacement = deformed_vertices - self.mesh_vertices
            displacement_norm = np.linalg.norm(displacement, axis=1)
            
            # Clamp displacement
            scale_factor = np.minimum(1.0, max_disp / (displacement_norm + 1e-8))
            displacement *= scale_factor[:, np.newaxis]
            deformed_vertices = self.mesh_vertices + displacement
        
        return deformed_vertices


def smooth_segmentation_mask(mask, iterations=2):
    """
    Smooth segmentation mask to reduce noise.
    
    Args:
        mask: Binary segmentation mask
        iterations: Number of smoothing iterations
        
    Returns:
        smoothed_mask: Smoothed binary mask
    """
    # Convert to uint8 if needed
    if mask.dtype != np.uint8:
        mask = (mask > 0).astype(np.uint8)
    
    # Apply morphological operations
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    
    # Erode then dilate to remove noise
    smoothed = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=iterations)
    
    # Dilate then erode to fill holes
    smoothed = cv2.morphologyEx(smoothed, cv2.MORPH_CLOSE, kernel, iterations=iterations)
    
    return smoothed


def extract_body_part_contours(segmentation_data):
    """
    Extract contours for each body part from segmentation data.
    
    Args:
        segmentation_data: Dict with body part masks
        
    Returns:
        contours: Dict mapping body part names to contours
    """
    contours = {}
    
    for part_name, part_mask in segmentation_data['body_parts'].items():
        if part_mask is not None and part_mask.size > 0:
            # Smooth the mask
            smoothed_mask = smooth_segmentation_mask(part_mask)
            
            # Find contours
            part_contours, _ = cv2.findContours(
                smoothed_mask, 
                cv2.RETR_EXTERNAL, 
                cv2.CHAIN_APPROX_SIMPLE
            )
            
            if part_contours:
                # Get the largest contour
                largest_contour = max(part_contours, key=cv2.contourArea)
                contours[part_name] = largest_contour
    
    return contours


# Import cv2 for the functions above
import cv2
