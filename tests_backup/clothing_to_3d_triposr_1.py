"""
Modified TripoSR Run Script with Automatic Orientation Correction
===================================================================
Based on official TripoSR example with added:
- Automatic upright orientation correction
- Z-axis scaling to reduce "fat" meshes
- Maintains all original functionality

Place in: external/TripoSR/run_with_orientation.py
Run: python run_with_orientation.py path/to/image.png --z-scale 0.7
"""

import argparse
import logging
import os
import time

import numpy as np
import rembg
import torch
import xatlas
from PIL import Image

# import torch
# import numpy as np
# from PIL import Image
import trimesh
from pathlib import Path
import sys
# import time


# Add TripoSR to path
triposr_path = Path(__file__).parent.parent / "external" / "TripoSR"

if not triposr_path.exists():
    print(f"✗ TripoSR not found at: {triposr_path}")
    print("\nExpected structure:")
    print("  spoken-wardrobe/")
    print("  ├── external/")
    print("  │   └── TripoSR/  ← Clone here")
    sys.exit(1)

sys.path.insert(0, str(triposr_path))

from tsr.system import TSR
from tsr.utils import remove_background, resize_foreground, save_video
from tsr.bake_texture import bake_texture


# ============================================================================
# ADDED: Orientation Correction Functions
# ============================================================================

def detect_and_correct_orientation(mesh, z_scale_factor=1.0):
    """
    Detect mesh orientation and correct to upright position.
    Also applies Z-axis scaling to reduce "fatness".
    
    Args:
        mesh: Mesh object with vertices and faces
        z_scale_factor: Scale factor for Z-axis (< 1.0 reduces depth)
                       **ADJUST THIS to control mesh "fatness"**
                       Recommended range: 0.6 - 0.9
                       
    Returns:
        mesh: Corrected mesh
        transform_info: Dict with transformation details
    """
    import trimesh
    
    # Convert to trimesh for easier manipulation
    if hasattr(mesh, 'vertices'):
        vertices = mesh.vertices
        faces = mesh.faces
        vertex_colors = mesh.vertex_colors if hasattr(mesh, 'vertex_colors') else None
    else:
        # Already trimesh
        vertices = mesh.vertices
        faces = mesh.faces
        vertex_colors = mesh.visual.vertex_colors[:, :3] if hasattr(mesh.visual, 'vertex_colors') else None
    
    mesh_tri = trimesh.Trimesh(
        vertices=vertices,
        faces=faces,
        vertex_colors=vertex_colors,
        process=False
    )
    
    # Get bounding box
    bounds = mesh_tri.bounds
    size = bounds[1] - bounds[0]
    
    logging.info(f"Original dimensions: X={size[0]:.3f}, Y={size[1]:.3f}, Z={size[2]:.3f}")
    
    # Find tallest dimension (should be Y for upright)
    axis_names = ['X', 'Y', 'Z']
    tallest_idx = np.argmax(size)
    tallest_axis = axis_names[tallest_idx]
    
    logging.info(f"Tallest dimension: {tallest_axis}")
    
    # Create transformation matrix
    transform = np.eye(4)
    
    # Step 1: Rotate to make Y the tallest (upright)
    if tallest_idx == 0:  # X is tallest -> rotate to Y
        logging.info("Rotating X → Y (90° around Z)")
        angle = -np.pi / 2
        c, s = np.cos(angle), np.sin(angle)
        R = np.array([
            [c, -s, 0],
            [s, c, 0],
            [0, 0, 1]
        ])
        transform[:3, :3] = R
        
    elif tallest_idx == 2:  # Z is tallest -> rotate to Y
        logging.info("Rotating Z → Y (90° around X)")
        angle = np.pi / 2
        c, s = np.cos(angle), np.sin(angle)
        R = np.array([
            [1, 0, 0],
            [0, c, -s],
            [0, s, c]
        ])
        transform[:3, :3] = R
    else:
        logging.info("Already upright (Y is tallest)")
    
    # Apply rotation
    mesh_tri.apply_transform(transform)
    
    # Step 2: Apply Z-axis scaling to reduce "fatness"
    # *** THIS IS THE KEY PARAMETER TO ADJUST ***
    if z_scale_factor != 1.0:
        logging.info(f"Applying Z-axis scale: {z_scale_factor}")
        scale_matrix = np.eye(4)
        scale_matrix[2, 2] = z_scale_factor  # Scale Z-axis only
        mesh_tri.apply_transform(scale_matrix)
    
    # Verify result
    new_bounds = mesh_tri.bounds
    new_size = new_bounds[1] - new_bounds[0]
    logging.info(f"Final dimensions: X={new_size[0]:.3f}, Y={new_size[1]:.3f}, Z={new_size[2]:.3f}")
    
    # Convert back to original mesh format
    if hasattr(mesh, 'vertices'):
        mesh.vertices = mesh_tri.vertices
    else:
        mesh = mesh_tri
    
    transform_info = {
        'original_size': size,
        'final_size': new_size,
        'tallest_axis_was': tallest_axis,
        'rotation_applied': tallest_idx != 1,
        'z_scale_applied': z_scale_factor
    }
    
    return mesh, transform_info


# ============================================================================
# Original TripoSR Code with Minimal Modifications
# ============================================================================

class Timer:
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


timer = Timer()


logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(message)s", level=logging.INFO
)
parser = argparse.ArgumentParser()
parser.add_argument("image", type=str, nargs="+", help="Path to input image(s).")
parser.add_argument(
    "--device",
    default="cuda:0",
    type=str,
    help="Device to use. If no CUDA-compatible device is found, will fallback to 'cpu'. Default: 'cuda:0'",
)
parser.add_argument(
    "--pretrained-model-name-or-path",
    default="stabilityai/TripoSR",
    type=str,
    help="Path to the pretrained model. Could be either a huggingface model id is or a local path. Default: 'stabilityai/TripoSR'",
)
parser.add_argument(
    "--chunk-size",
    default=8192,
    type=int,
    help="Evaluation chunk size for surface extraction and rendering. Smaller chunk size reduces VRAM usage but increases computation time. 0 for no chunking. Default: 8192",
)
parser.add_argument(
    "--mc-resolution",
    default=256,
    type=int,
    help="Marching cubes grid resolution. Default: 256"
)
parser.add_argument(
    "--no-remove-bg",
    action="store_true",
    help="If specified, the background will NOT be automatically removed from the input image, and the input image should be an RGB image with gray background and properly-sized foreground. Default: false",
)
parser.add_argument(
    "--foreground-ratio",
    default=0.85,
    type=float,
    help="Ratio of the foreground size to the image size. Only used when --no-remove-bg is not specified. Default: 0.85",
)
parser.add_argument(
    "--output-dir",
    default="calibration_data_1/",
    type=str,
    help="Output directory to save the results. Default: 'output/'",
)
parser.add_argument(
    "--model-save-format",
    default="obj",
    type=str,
    choices=["obj", "glb"],
    help="Format to save the extracted mesh. Default: 'obj'",
)
parser.add_argument(
    "--bake-texture",
    action="store_true",
    help="Bake a texture atlas for the extracted mesh, instead of vertex colors",
)
parser.add_argument(
    "--texture-resolution",
    default=2048,
    type=int,
    help="Texture atlas resolution, only useful with --bake-texture. Default: 2048"
)
parser.add_argument(
    "--render",
    action="store_true",
    help="If specified, save a NeRF-rendered video. Default: false",
)

# ============================================================================
# ADDED: New Arguments for Orientation & Scaling
# ============================================================================
parser.add_argument(
    "--auto-orient",
    action="store_true",
    default=True,
    help="Automatically correct mesh orientation to upright. Default: true",
)
parser.add_argument(
    "--no-auto-orient",
    dest="auto_orient",
    action="store_false",
    help="Disable automatic orientation correction.",
)
parser.add_argument(
    "--z-scale",
    default=0.8,
    type=float,
    help="*** ADJUST THIS to reduce mesh 'fatness' *** Scale factor for Z-axis depth. < 1.0 reduces depth, > 1.0 increases. Recommended: 0.6-0.9. Default: 0.8",
)

args = parser.parse_args()

output_dir = args.output_dir
os.makedirs(output_dir, exist_ok=True)

device = args.device
if not torch.cuda.is_available():
    device = "cpu"

timer.start("Initializing model")
model = TSR.from_pretrained(
    args.pretrained_model_name_or_path,
    config_name="config.yaml",
    weight_name="model.ckpt",
)
model.renderer.set_chunk_size(args.chunk_size)
model.to(device)
timer.end("Initializing model")

timer.start("Processing images")
images = []

if args.no_remove_bg:
    rembg_session = None
else:
    rembg_session = rembg.new_session()

for i, image_path in enumerate(args.image):
    if args.no_remove_bg:
        image = np.array(Image.open(image_path).convert("RGB"))
    else:
        image = remove_background(Image.open(image_path), rembg_session)
        image = resize_foreground(image, args.foreground_ratio)
        image = np.array(image).astype(np.float32) / 255.0
        image = image[:, :, :3] * image[:, :, 3:4] + (1 - image[:, :, 3:4]) * 0.5
        image = Image.fromarray((image * 255.0).astype(np.uint8))
        if not os.path.exists(os.path.join(output_dir, str(i))):
            os.makedirs(os.path.join(output_dir, str(i)))
        image.save(os.path.join(output_dir, str(i), f"input.png"))
    images.append(image)
timer.end("Processing images")

for i, image in enumerate(images):
    logging.info(f"Running image {i + 1}/{len(images)} ...")

    timer.start("Running model")
    with torch.no_grad():
        scene_codes = model([image], device=device)
    timer.end("Running model")

    if args.render:
        timer.start("Rendering")
        render_images = model.render(scene_codes, n_views=30, return_type="pil")
        for ri, render_image in enumerate(render_images[0]):
            render_image.save(os.path.join(output_dir, str(i), f"render_{ri:03d}.png"))
        save_video(
            render_images[0], os.path.join(output_dir, str(i), f"render.mp4"), fps=30
        )
        timer.end("Rendering")

    timer.start("Extracting mesh")
    meshes = model.extract_mesh(scene_codes, not args.bake_texture, resolution=args.mc_resolution)
    timer.end("Extracting mesh")
    
    # ============================================================================
    # ADDED: Apply Orientation Correction & Z-Scaling
    # ============================================================================
    if args.auto_orient:
        timer.start("Correcting orientation and scaling")
        meshes[0], transform_info = detect_and_correct_orientation(
            meshes[0], 
            z_scale_factor=args.z_scale  # *** THIS CONTROLS THE "FATNESS" ***
        )
        logging.info(f"Mesh corrected: {transform_info}")
        timer.end("Correcting orientation and scaling")

    out_mesh_path = os.path.join(output_dir, str(i), f"mesh.{args.model_save_format}")
    if args.bake_texture:
        out_texture_path = os.path.join(output_dir, str(i), "texture.png")

        timer.start("Baking texture")
        bake_output = bake_texture(meshes[0], model, scene_codes[0], args.texture_resolution)
        timer.end("Baking texture")

        timer.start("Exporting mesh and texture")
        xatlas.export(out_mesh_path, meshes[0].vertices[bake_output["vmapping"]], bake_output["indices"], bake_output["uvs"], meshes[0].vertex_normals[bake_output["vmapping"]])
        Image.fromarray((bake_output["colors"] * 255.0).astype(np.uint8)).transpose(Image.FLIP_TOP_BOTTOM).save(out_texture_path)
        timer.end("Exporting mesh and texture")
    else:
        timer.start("Exporting mesh")
        meshes[0].export(out_mesh_path)
        timer.end("Exporting mesh")