"""
Interactive TripoSR Run Script with Automatic Orientation Correction
====================================================================
Enhanced version of clothing_to_3d_triposr_1.py with:
- Interactive image selection from generated_images directory
- Easy-to-use prompts for all options
- Maintains all original functionality and syntax
- No command-line arguments needed (but still supported)

Usage:
    python tests/clothing_to_3d_triposr_2.py              # Interactive mode
    python tests/clothing_to_3d_triposr_2.py image.png    # Command-line mode (original)
"""

import argparse
import logging
import os
import time
from pathlib import Path
import sys

import numpy as np
import rembg
import torch
import xatlas
from PIL import Image

import trimesh


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
# ADDED: Interactive Mode Functions
# ============================================================================

def list_images_in_directory(directory):
    """List all valid image files in a directory."""
    valid_extensions = {'.png', '.jpg', '.jpeg', '.bmp', '.tiff', '.webp'}
    image_files = []
    
    if not directory.exists():
        return image_files
    
    for file_path in sorted(directory.iterdir()):
        if file_path.is_file() and file_path.suffix.lower() in valid_extensions:
            image_files.append(file_path)
    
    return image_files


def interactive_image_selection():
    """Interactively select image(s) from generated_images directory."""
    # Get project root and generated_images directory
    project_root = Path(__file__).parent.parent
    generated_images_dir = project_root / "generated_images"
    
    print("\n" + "="*70)
    print("INTERACTIVE IMAGE SELECTION")
    print("="*70)
    
    # Check if directory exists
    if not generated_images_dir.exists():
        print(f"✗ Directory not found: {generated_images_dir}")
        print("\nCreating directory...")
        generated_images_dir.mkdir(parents=True, exist_ok=True)
        print("✓ Directory created. Please add images and run again.")
        sys.exit(1)
    
    # List available images
    image_files = list_images_in_directory(generated_images_dir)
    
    if not image_files:
        print(f"✗ No images found in: {generated_images_dir}")
        print("\nSupported formats: PNG, JPG, JPEG, BMP, TIFF, WEBP")
        sys.exit(1)
    
    print(f"\nFound {len(image_files)} image(s) in {generated_images_dir.name}/:\n")
    
    # Display images with numbers
    for idx, img_path in enumerate(image_files, 1):
        file_size = img_path.stat().st_size / 1024  # KB
        print(f"  [{idx}] {img_path.name} ({file_size:.1f} KB)")
    
    print("\n" + "-"*70)
    
    # Get user selection
    while True:
        try:
            selection = input("\nEnter image number(s) (comma-separated for multiple, or 'all'): ").strip().lower()
            
            if selection == 'all':
                selected_images = image_files
                break
            
            # Parse comma-separated numbers
            indices = [int(x.strip()) for x in selection.split(',')]
            
            # Validate indices
            if all(1 <= idx <= len(image_files) for idx in indices):
                selected_images = [image_files[idx - 1] for idx in indices]
                break
            else:
                print(f"✗ Invalid selection. Please enter numbers between 1 and {len(image_files)}")
        
        except ValueError:
            print("✗ Invalid input. Please enter numbers separated by commas (e.g., 1,3,5) or 'all'")
        except KeyboardInterrupt:
            print("\n\nCancelled by user.")
            sys.exit(0)
    
    print(f"\n✓ Selected {len(selected_images)} image(s):")
    for img in selected_images:
        print(f"  • {img.name}")
    
    return [str(img) for img in selected_images]


def interactive_options():
    """Interactively get processing options from user."""
    print("\n" + "="*70)
    print("PROCESSING OPTIONS")
    print("="*70)
    
    options = {}
    
    # Auto-orient
    print("\n1. Automatic Orientation Correction")
    print("   Automatically correct mesh to upright position")
    response = input("   Enable? [Y/n]: ").strip().lower()
    options['auto_orient'] = response != 'n'
    
    # Z-scale
    if options['auto_orient']:
        print("\n2. Z-Axis Scale (Depth/Fatness Control)")
        print("   < 1.0 = thinner/flatter mesh")
        print("   = 1.0 = no scaling")
        print("   > 1.0 = thicker mesh")
        print("   Recommended range: 0.6 - 0.9")
        while True:
            try:
                response = input("   Enter Z-scale [default: 0.8]: ").strip()
                options['z_scale'] = float(response) if response else 0.8
                if 0.1 <= options['z_scale'] <= 2.0:
                    break
                print("   ✗ Please enter a value between 0.1 and 2.0")
            except ValueError:
                print("   ✗ Invalid input. Please enter a number.")
    else:
        options['z_scale'] = 1.0
    
    # Remove background
    print("\n3. Background Removal")
    print("   Automatically remove background from input image")
    response = input("   Enable? [Y/n]: ").strip().lower()
    options['no_remove_bg'] = response == 'n'
    
    # Foreground ratio
    if not options['no_remove_bg']:
        print("\n4. Foreground Ratio")
        print("   Ratio of foreground size to image size")
        print("   Recommended: 0.75 - 0.95")
        while True:
            try:
                response = input("   Enter ratio [default: 0.85]: ").strip()
                options['foreground_ratio'] = float(response) if response else 0.85
                if 0.1 <= options['foreground_ratio'] <= 1.0:
                    break
                print("   ✗ Please enter a value between 0.1 and 1.0")
            except ValueError:
                print("   ✗ Invalid input. Please enter a number.")
    else:
        options['foreground_ratio'] = 0.85
    
    # Marching cubes resolution
    print("\n5. Mesh Resolution (Marching Cubes)")
    print("   Higher = more detailed but slower")
    print("   Recommended: 110 (default), 150 (high), 200 (very high)")
    while True:
        try:
            response = input("   Enter resolution [default: 110]: ").strip()
            options['mc_resolution'] = int(response) if response else 110
            if 50 <= options['mc_resolution'] <= 300:
                break
            print("   ✗ Please enter a value between 50 and 300")
        except ValueError:
            print("   ✗ Invalid input. Please enter a number.")
    
    # Texture baking
    print("\n6. Texture Baking")
    print("   Bake texture atlas (vs. vertex colors)")
    response = input("   Enable? [y/N]: ").strip().lower()
    options['bake_texture'] = response == 'y'
    
    if options['bake_texture']:
        print("\n7. Texture Resolution")
        print("   Higher = better quality but larger file")
        print("   Common: 1024, 2048 (default), 4096")
        while True:
            try:
                response = input("   Enter resolution [default: 2048]: ").strip()
                options['texture_resolution'] = int(response) if response else 2048
                if 512 <= options['texture_resolution'] <= 8192:
                    break
                print("   ✗ Please enter a value between 512 and 8192")
            except ValueError:
                print("   ✗ Invalid input. Please enter a number.")
    else:
        options['texture_resolution'] = 2048
    
    # Output format
    print("\n8. Output Format")
    print("   [1] OBJ (default)")
    print("   [2] GLB")
    response = input("   Select format [1]: ").strip()
    options['model_save_format'] = 'glb' if response == '2' else 'obj'
    
    # Output directory
    print("\n9. Output Directory")
    default_output = "generated_meshes"
    response = input(f"   Enter directory [default: {default_output}]: ").strip()
    options['output_dir'] = response if response else default_output
    
    # Render video
    print("\n10. Render Video")
    print("    Generate NeRF-rendered 360° video (slower)")
    response = input("    Enable? [y/N]: ").strip().lower()
    options['render'] = response == 'y'
    
    print("\n" + "="*70)
    print("OPTIONS SUMMARY")
    print("="*70)
    print(f"  Auto-orient: {options['auto_orient']}")
    if options['auto_orient']:
        print(f"  Z-scale: {options['z_scale']}")
    print(f"  Remove background: {not options['no_remove_bg']}")
    if not options['no_remove_bg']:
        print(f"  Foreground ratio: {options['foreground_ratio']}")
    print(f"  Mesh resolution: {options['mc_resolution']}")
    print(f"  Bake texture: {options['bake_texture']}")
    if options['bake_texture']:
        print(f"  Texture resolution: {options['texture_resolution']}")
    print(f"  Output format: {options['model_save_format'].upper()}")
    print(f"  Output directory: {options['output_dir']}")
    print(f"  Render video: {options['render']}")
    print("="*70)
    
    response = input("\nProceed with these options? [Y/n]: ").strip().lower()
    if response == 'n':
        print("\nCancelled by user.")
        sys.exit(0)
    
    return options


# ============================================================================
# ADDED: Orientation Correction Functions (from original)
# ============================================================================

def detect_and_correct_orientation(mesh, z_scale_factor=1.0, apply_flip=True):
    """
    Detect mesh orientation and correct to upright position.
    Also applies Z-axis scaling to reduce "fatness".
    
    Args:
        mesh: Mesh object with vertices and faces
        z_scale_factor: Scale factor for Z-axis (< 1.0 reduces depth)
                       **ADJUST THIS to control mesh "fatness"**
                       Recommended range: 0.6 - 0.9
        apply_flip: Whether to apply 180° X-axis flip to fix upside-down orientation
                       
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
    
    # Apply initial rotation to make Y tallest
    mesh_tri.apply_transform(transform)
    
    # Step 1.5: Flip 180° around X-axis to fix upside-down issue
    # TripoSR meshes are often upside-down after making them upright
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
        mesh_tri.apply_transform(flip_transform)
    else:
        logging.info("Skipping 180° flip (--no-flip enabled)")
    
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
parser.add_argument("image", type=str, nargs="*", help="Path to input image(s). If not specified, enters interactive mode.")
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
    default=110,
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
    default="generated_meshes",
    type=str,
    help="Output directory to save the results. Default: 'generated_meshes'",
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
# ADDED: New Arguments for Orientation & Scaling (from original)
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
parser.add_argument(
    "--no-flip",
    action="store_true",
    help="Disable the 180° X-axis flip that fixes upside-down meshes. Use if your mesh comes out upside-down WITH the flip enabled.",
)

args = parser.parse_args()

# ============================================================================
# INTERACTIVE MODE: If no images specified, enter interactive mode
# ============================================================================
if not args.image:
    print("\n" + "="*70)
    print("TRIPOSR INTERACTIVE MODE")
    print("Convert 2D images to 3D meshes with ease!")
    print("="*70)
    
    # Interactive image selection
    selected_images = interactive_image_selection()
    args.image = selected_images
    
    # Interactive options
    options = interactive_options()
    
    # Apply options to args
    args.auto_orient = options['auto_orient']
    args.z_scale = options['z_scale']
    args.no_remove_bg = options['no_remove_bg']
    args.foreground_ratio = options['foreground_ratio']
    args.mc_resolution = options['mc_resolution']
    args.bake_texture = options['bake_texture']
    args.texture_resolution = options['texture_resolution']
    args.model_save_format = options['model_save_format']
    args.output_dir = options['output_dir']
    args.render = options['render']

# ============================================================================
# Continue with original processing logic
# ============================================================================

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
    # ADDED: Apply Orientation Correction & Z-Scaling (from original)
    # ============================================================================
    if args.auto_orient:
        timer.start("Correcting orientation and scaling")
        meshes[0], transform_info = detect_and_correct_orientation(
            meshes[0], 
            z_scale_factor=args.z_scale,  # *** THIS CONTROLS THE "FATNESS" ***
            apply_flip=not args.no_flip  # Apply flip unless --no-flip is specified
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

print("\n" + "="*70)
print("✓ PROCESSING COMPLETE!")
print("="*70)
print(f"Output saved to: {output_dir}")
print("="*70)

