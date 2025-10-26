# cage_utils.py
# Simple cage-based deformation utilities for real-time clothing overlay
# Based on Mean Value Coordinates (MVC) algorithm

import numpy as np
import trimesh
from scipy.spatial import cKDTree


class SimpleCageGenerator:
    """
    Generate a simple cage around a mesh using BodyPix body part segmentation.
    This creates a coarse control mesh that encloses the clothing mesh.
    """
    
    def __init__(self, mesh):
        """
        Initialize cage generator with a clothing mesh.
        
        Args:
            mesh: trimesh object of the clothing mesh
        """
        self.mesh = mesh
        self.cage = None
        self.body_part_groups = {
            'torso': ['TORSO_FRONT', 'TORSO_BACK'],
            'left_upper_arm': ['LEFT_UPPER_ARM_FRONT', 'LEFT_UPPER_ARM_BACK'],
            'right_upper_arm': ['RIGHT_UPPER_ARM_FRONT', 'RIGHT_UPPER_ARM_BACK'],
            'left_lower_arm': ['LEFT_LOWER_ARM_FRONT', 'LEFT_LOWER_ARM_BACK'],
            'right_lower_arm': ['RIGHT_LOWER_ARM_FRONT', 'RIGHT_LOWER_ARM_BACK'],
        }
    
    def generate_simple_box_cage(self, subdivisions=2):
        """
        Generate a simple subdivided box cage around the mesh.
        This is the simplest approach - just a bounding box with subdivisions.
        
        Args:
            subdivisions: Number of subdivisions along each major body segment
            
        Returns:
            cage_mesh: trimesh object of the cage
        """
        # Get bounding box of mesh
        bbox_min = self.mesh.vertices.min(axis=0)
        bbox_max = self.mesh.vertices.max(axis=0)
        
        # Expand bbox slightly to ensure it encloses mesh
        padding = 0.1 * (bbox_max - bbox_min)
        bbox_min -= padding
        bbox_max += padding
        
        # Create subdivided box cage
        cage_vertices = []
        
        # Create grid points around the mesh
        # We'll create a simple cage with subdivisions along height (y-axis)
        x_vals = [bbox_min[0], bbox_max[0]]
        z_vals = [bbox_min[2], bbox_max[2]]
        
        # Subdivide along y-axis (height)
        y_vals = np.linspace(bbox_min[1], bbox_max[1], subdivisions + 1)
        
        # Create cage vertices as a grid
        for y in y_vals:
            for z in z_vals:
                for x in x_vals:
                    cage_vertices.append([x, y, z])
        
        cage_vertices = np.array(cage_vertices)
        
        # Create faces for the cage (convex hull)
        from scipy.spatial import ConvexHull
        hull = ConvexHull(cage_vertices)
        cage_faces = hull.simplices
        
        self.cage = trimesh.Trimesh(vertices=cage_vertices, faces=cage_faces)
        
        print(f"Generated simple cage with {len(cage_vertices)} vertices")
        return self.cage


class MeanValueCoordinates:
    """
    Compute Mean Value Coordinates for cage-based deformation.
    This is a lightweight, real-time friendly implementation.
    """
    
    def __init__(self, mesh_vertices, cage_mesh):
        """
        Initialize MVC calculator.
        
        Args:
            mesh_vertices: Nx3 array of mesh vertex positions
            cage_mesh: trimesh object of the cage
        """
        self.mesh_vertices = np.array(mesh_vertices)
        self.cage_vertices = np.array(cage_mesh.vertices)
        self.cage_faces = np.array(cage_mesh.faces)
        self.mvc_weights = None
        
    def compute_weights(self):
        """
        Compute MVC weights for all mesh vertices.
        This is done once during setup phase.
        
        Returns:
            weights: N x M array where N = num mesh vertices, M = num cage vertices
        """
        n_mesh_verts = len(self.mesh_vertices)
        n_cage_verts = len(self.cage_vertices)
        
        weights = np.zeros((n_mesh_verts, n_cage_verts))
        
        print(f"Computing MVC weights for {n_mesh_verts} mesh vertices and {n_cage_verts} cage vertices...")
        
        # For each mesh vertex, compute its MVC weights
        for i, v in enumerate(self.mesh_vertices):
            if i % 500 == 0:
                print(f"  Processing vertex {i}/{n_mesh_verts}...")
            
            # Simple MVC formula: weight is inversely proportional to distance
            # This is a simplified version for prototype - not the full MVC formula
            distances = np.linalg.norm(self.cage_vertices - v, axis=1)
            
            # Add small epsilon to avoid division by zero
            distances += 1e-8
            
            # Inverse distance weighting
            w = 1.0 / distances
            
            # Normalize weights to sum to 1
            w = w / w.sum()
            
            weights[i] = w
        
        self.mvc_weights = weights
        print("MVC weights computed!")
        return weights
    
    def deform_mesh(self, deformed_cage_vertices):
        """
        Deform the mesh using the deformed cage.
        This is the real-time operation - very fast.
        
        Args:
            deformed_cage_vertices: M x 3 array of deformed cage vertex positions
            
        Returns:
            deformed_mesh_vertices: N x 3 array of deformed mesh vertices
        """
        if self.mvc_weights is None:
            raise ValueError("Must compute weights first!")
        
        # Simple matrix multiplication: deformed_verts = weights @ cage_verts
        # This is the core of real-time cage deformation
        deformed_vertices = self.mvc_weights @ deformed_cage_vertices
        
        return deformed_vertices


def smooth_temporal(current_positions, previous_positions, alpha=0.3):
    """
    Smooth positions across frames to reduce jitter.
    
    Args:
        current_positions: Current frame positions
        previous_positions: Previous frame positions
        alpha: Smoothing factor (0-1), lower = more smoothing
        
    Returns:
        smoothed_positions: Temporally smoothed positions
    """
    if previous_positions is None:
        return current_positions
    
    return alpha * current_positions + (1 - alpha) * previous_positions
