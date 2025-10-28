"""
TripoSR Pipeline Module
=======================

This module extracts the core TripoSR logic from clothing_to_3d_triposr_2.py
into callable functions without changing the working logic.

This allows the pipeline to be used programmatically while keeping the
original script intact for interactive use.

Author: AI Assistant
Date: October 26, 2025
"""

import numpy as np
import torch
import trimesh
import rembg
import logging
import os
import time
import sys
from PIL import Image
from pathlib import Path

# Add TripoSR to path (same as clothing_to_3d_triposr_2.py)
triposr_path = Path(__file__).parent.parent / "external" / "TripoSR"

if not triposr_path.exists():
    raise RuntimeError(
        f"TripoSR not found at: {triposr_path}\n"
        "Expected structure:\n"
        "  spoken-wardrobe/\n"
        "  ├── external/\n"
        "  │   └── TripoSR/  ← Clone here\n"
    )

sys.path.insert(0, str(triposr_path))

# Import TripoSR
from tsr.system import TSR
from tsr.utils import remove_background, resize_foreground
try:
    from tsr.utils import save_video
except ImportError:
    save_video = None  # Optional


class Timer:
    """Timer utility from original script."""
    def __init__(self):
        self.items = {}
        self.time_scale = 1000.0  # ms
        self.time_unit = "ms"

    def start(self, name: str) -> None:
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        self.items[name] = time.time()
        logging.info(f"{name} ...")

    def end(self, name: str) -> float:
        if name not in self.items:
            return
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        start_time = self.items.pop(name)
        delta = time.time() - start_time
        t = delta * self.time_scale
        logging.info(f"{name} finished in {t:.2f}{self.time_unit}.")


def simplify_mesh_for_realtime(mesh, target_faces=5000, min_faces=1000):
    """
    Simplify mesh for real-time rendering using quadric decimation.
    
    Args:
        mesh: trimesh.Trimesh object
        target_faces: Target number of faces (5k recommended for 60fps)
        min_faces: Minimum faces to preserve mesh shape
    
    Returns:
        simplified_mesh: Decimated trimesh.Trimesh
    """
    import trimesh
    
    original_faces = len(mesh.faces)
    original_vertices = len(mesh.vertices)
    
    # Skip if already simplified enough
    if original_faces <= target_faces:
        logging.info(f"Mesh already has {original_faces} faces, skipping simplification")
        return mesh
    
    logging.info(f"Simplifying mesh for real-time rendering...")
    logging.info(f"  Original: {original_vertices} vertices, {original_faces} faces")
    
    try:
        # Use quadric error metrics (QEM) for high-quality decimation
        # This preserves visual fidelity while reducing poly count
        simplified = mesh.simplify_quadratic_decimation(target_faces)
        
        # Safety check - don't over-simplify
        if len(simplified.faces) < min_faces:
            logging.warning(f"Decimation too aggressive ({len(simplified.faces)} < {min_faces}), using {min_faces}")
            simplified = mesh.simplify_quadratic_decimation(min_faces)
        
        reduction_pct = (1 - len(simplified.faces) / original_faces) * 100
        
        logging.info(f"  Simplified: {len(simplified.vertices)} vertices, {len(simplified.faces)} faces")
        logging.info(f"  Reduction: {reduction_pct:.1f}% fewer faces")
        
        return simplified
        
    except Exception as e:
        logging.warning(f"Simplification failed: {e}, using original mesh")
        return mesh


def detect_and_correct_orientation(mesh, z_scale_factor=1.0, apply_flip=True):
    """
    Detect and correct mesh orientation to upright.
    
    This is the EXACT logic from clothing_to_3d_triposr_2.py.
    Do not modify - it works correctly with TripoSR output.
    
    Args:
        mesh: trimesh.Trimesh object
        z_scale_factor: Scale factor for Z-axis (< 1.0 reduces "fatness")
        apply_flip: Whether to apply 180° X-axis flip
    
    Returns:
        corrected_mesh: trimesh.Trimesh with corrected orientation
        transform_info: Dict with transformation details
    """
    import trimesh
    
    vertices = mesh.vertices
    faces = mesh.faces
    
    # Get vertex colors if available
    vertex_colors = None
    if hasattr(mesh.visual, 'vertex_colors'):
        vertex_colors = mesh.visual.vertex_colors.copy()
    
    # Center the mesh
    centroid = vertices.mean(axis=0)
    centered_vertices = vertices - centroid
    
    # Calculate bounding box dimensions
    bbox_size = centered_vertices.max(axis=0) - centered_vertices.min(axis=0)
    logging.info(f"Original bbox size: X={bbox_size[0]:.3f}, Y={bbox_size[1]:.3f}, Z={bbox_size[2]:.3f}")
    
    # Step 1: Identify the tallest dimension (should be vertical/Y-axis)
    tallest_axis = np.argmax(bbox_size)
    logging.info(f"Tallest axis: {['X', 'Y', 'Z'][tallest_axis]}")
    
    # Create rotation matrix to align tallest axis with Y
    rotation_matrix = np.eye(4)
    
    if tallest_axis == 0:  # X is tallest, rotate to Y
        logging.info("Rotating X → Y (90° around Z-axis)")
        angle = np.pi / 2
        c, s = np.cos(angle), np.sin(angle)
        R = np.array([
            [c, -s, 0],
            [s, c, 0],
            [0, 0, 1]
        ])
        rotation_matrix[:3, :3] = R
    elif tallest_axis == 2:  # Z is tallest, rotate to Y
        logging.info("Rotating Z → Y (90° around X-axis)")
        angle = -np.pi / 2
        c, s = np.cos(angle), np.sin(angle)
        R = np.array([
            [1, 0, 0],
            [0, c, -s],
            [0, s, c]
        ])
        rotation_matrix[:3, :3] = R
    else:
        logging.info("Y is already tallest - no initial rotation needed")
    
    # Apply rotation
    centered_vertices = (rotation_matrix[:3, :3] @ centered_vertices.T).T
    
    # Step 1.5: Flip 180° around X-axis to fix upside-down issue
    if apply_flip:
        logging.info("Applying 180° flip around X-axis (fixes upside-down orientation)")
        flip_transform = np.eye(4)
        angle = np.pi  # 180 degrees
        c, s = np.cos(angle), np.sin(angle)
        R_flip = np.array([
            [1, 0, 0],
            [0, c, -s],
            [0, s, c]
        ])
        flip_transform[:3, :3] = R_flip
        centered_vertices = (R_flip @ centered_vertices.T).T
    else:
        logging.info("Skipping 180° flip (--no-flip enabled)")
    
    # Step 1.6: Rotate to face forward (fix 90° left rotation)
    logging.info("Applying 90° rotation around Y-axis (face forward)")
    forward_transform = np.eye(4)
    angle = np.pi / 2  # 90 degrees clockwise
    c, s = np.cos(angle), np.sin(angle)
    R_forward = np.array([
        [c, 0, s],
        [0, 1, 0],
        [-s, 0, c]
    ])
    forward_transform[:3, :3] = R_forward
    centered_vertices = (R_forward @ centered_vertices.T).T
    
    # Step 2: Apply Z-axis scaling to control "fatness"
    logging.info(f"Applying Z-axis scaling: {z_scale_factor}")
    centered_vertices[:, 2] *= z_scale_factor
    
    # Recalculate bounding box after all transformations
    new_bbox_size = centered_vertices.max(axis=0) - centered_vertices.min(axis=0)
    logging.info(f"Final bbox size: X={new_bbox_size[0]:.3f}, Y={new_bbox_size[1]:.3f}, Z={new_bbox_size[2]:.3f}")
    
    # Create new mesh with corrected orientation
    corrected_mesh = trimesh.Trimesh(
        vertices=centered_vertices,
        faces=faces,
        process=False
    )
    
    # Restore vertex colors if available
    if vertex_colors is not None:
        corrected_mesh.visual.vertex_colors = vertex_colors
    
    transform_info = {
        'original_bbox': bbox_size.tolist(),
        'final_bbox': new_bbox_size.tolist(),
        'tallest_axis': ['X', 'Y', 'Z'][tallest_axis],
        'z_scale_factor': z_scale_factor,
        'flip_applied': apply_flip
    }
    
    return corrected_mesh, transform_info


def generate_mesh_from_image(
    image_path: str,
    output_dir: str,
    model=None,
    device="cuda:0",
    auto_orient=True,
    z_scale=0.8,
    apply_flip=True,
    no_remove_bg=False,
    foreground_ratio=0.75,  # Updated from Test 3 (Balanced) - more padding
    mc_resolution=196,       # Updated from Test 3 (Balanced) - much better than 110
    chunk_size=8192,
    model_save_format="obj"
):
    """
    Generate 3D mesh from image using TripoSR.
    
    This is the core logic from clothing_to_3d_triposr_2.py, adapted to be
    callable as a function without changing the working logic.
    
    Args:
        image_path: Path to input image
        output_dir: Directory to save output
        model: Pre-loaded TripoSR model (if None, will load)
        device: Device to use ('cuda:0', 'cpu', etc.)
        auto_orient: Automatically correct mesh orientation
        z_scale: Z-axis scale factor (< 1.0 reduces "fatness")
        apply_flip: Apply 180° X-axis flip
        no_remove_bg: Skip background removal
        foreground_ratio: Foreground size ratio
        mc_resolution: Marching cubes resolution
        chunk_size: Chunk size for surface extraction
        model_save_format: Output format ('obj' or 'glb')
    
    Returns:
        mesh_path: Path to generated mesh file
        corrected_mesh: trimesh.Trimesh object
    """
    timer = Timer()
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Check device availability
    if device.startswith("cuda") and not torch.cuda.is_available():
        logging.warning("CUDA not available, falling back to CPU")
        device = "cpu"
    
    # Load model if not provided
    model_loaded_here = False
    if model is None:
        timer.start("Initializing model")
        model = TSR.from_pretrained(
            "stabilityai/TripoSR",
            config_name="config.yaml",
            weight_name="model.ckpt",
        )
        model.renderer.set_chunk_size(chunk_size)
        model.to(device)
        timer.end("Initializing model")
        model_loaded_here = True
    
    # Process image (background removal, etc.)
    timer.start("Processing image")
    
    if no_remove_bg:
        image = np.array(Image.open(image_path).convert("RGB"))
        rembg_session = None
    else:
        rembg_session = rembg.new_session()
        image = remove_background(Image.open(image_path), rembg_session)
        image = resize_foreground(image, foreground_ratio)
        image = np.array(image).astype(np.float32) / 255.0
        image = image[:, :, :3] * image[:, :, 3:4] + (1 - image[:, :, 3:4]) * 0.5
        image = Image.fromarray((image * 255.0).astype(np.uint8))
        
        # Save processed input
        image.save(os.path.join(output_dir, "input.png"))
    
    timer.end("Processing image")
    
    # Run model
    timer.start("Running model")
    with torch.no_grad():
        scene_codes = model([image], device=device)
    timer.end("Running model")
    
    # Extract mesh
    timer.start("Extracting mesh")
    meshes = model.extract_mesh(scene_codes, True, resolution=mc_resolution)  # True = vertex colors
    timer.end("Extracting mesh")
    
    # Apply orientation correction and Z-scaling
    if auto_orient:
        timer.start("Correcting orientation and scaling")
        meshes[0], transform_info = detect_and_correct_orientation(
            meshes[0],
            z_scale_factor=z_scale,
            apply_flip=apply_flip
        )
        logging.info(f"Mesh corrected: {transform_info}")
        timer.end("Correcting orientation and scaling")
    
    # NOTE: Mesh simplification moved to Three.js side (SimplifyModifier)
    # Python-side decimation was too aggressive and destroyed mesh integrity
    # See enhanced_mesh_viewer_v2.html for SimplifyModifier implementation
    
    # Export mesh
    out_mesh_path = os.path.join(output_dir, f"mesh.{model_save_format}")
    
    timer.start("Exporting mesh")
    meshes[0].export(out_mesh_path)
    timer.end("Exporting mesh")
    
    logging.info(f"✓ Mesh saved: {out_mesh_path}")
    
    # Clean up model if we loaded it here
    if model_loaded_here:
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    
    return out_mesh_path, meshes[0]


# ============================================================================
# Convenience Functions
# ============================================================================

def generate_mesh_simple(image_path: str, output_dir: str, z_scale: float = 0.8):
    """
    Simplified interface for mesh generation with sensible defaults.
    
    Args:
        image_path: Path to input image
        output_dir: Directory to save output
        z_scale: Z-axis scale factor (default: 0.8)
    
    Returns:
        mesh_path: Path to generated mesh file
    """
    return generate_mesh_from_image(
        image_path=image_path,
        output_dir=output_dir,
        z_scale=z_scale,
        auto_orient=True,
        apply_flip=True
    )[0]  # Return just the path


# ============================================================================
# Testing
# ============================================================================

if __name__ == "__main__":
    print("="*70)
    print("TripoSR Pipeline Module")
    print("="*70)
    print("\nThis module should be imported, not run directly.")
    print("Use create_consistent_pipeline_v2.py for the full pipeline.")
    print("="*70)

