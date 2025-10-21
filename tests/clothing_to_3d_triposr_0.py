"""
TripoSR Mesh Generation with Automatic Upright Correction
==========================================================
Generates meshes and automatically orients them upright.

Run from root: python tests/test_triposr_upright_generation.py
"""

import torch
import numpy as np
from PIL import Image
import trimesh
from pathlib import Path
import sys
import time

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

# Import TripoSR
# from tsr.system import TSR
from tsr.system import TSR
from tsr.utils import remove_background, resize_foreground, save_video
from tsr.bake_texture import bake_texture


def detect_and_correct_orientation(mesh):
    """
    Detect mesh orientation and correct to upright position.
    
    Args:
        mesh: trimesh.Trimesh object
        
    Returns:
        corrected_mesh: Upright mesh
        transform: Transformation applied
    """
    print("\n  Detecting orientation...")
    
    bounds = mesh.bounds
    size = bounds[1] - bounds[0]
    
    print(f"    Dimensions: X={size[0]:.3f}, Y={size[1]:.3f}, Z={size[2]:.3f}")
    
    # Find tallest dimension
    axis_sizes = {'X': (size[0], 0), 'Y': (size[1], 1), 'Z': (size[2], 2)}
    tallest_axis, (tallest_size, tallest_idx) = max(axis_sizes.items(), key=lambda x: x[1][0])
    
    print(f"    Tallest: {tallest_axis}")
    
    # Target: Y should be tallest
    transform = np.eye(4)
    
    if tallest_idx == 1:  # Already Y
        print("    ✓ Already upright")
    elif tallest_idx == 0:  # X is tallest
        print("    Rotating X → Y...")
        # Rotate -90° around Z
        angle = -np.pi / 2
        transform = trimesh.transformations.rotation_matrix(angle, [0, 0, 1])
    elif tallest_idx == 2:  # Z is tallest
        print("    Rotating Z → Y...")
        # Rotate 90° around X
        angle = np.pi / 2
        transform = trimesh.transformations.rotation_matrix(angle, [1, 0, 0])
    
    # Apply transform
    if not np.allclose(transform, np.eye(4)):
        mesh_corrected = mesh.copy()
        mesh_corrected.apply_transform(transform)
        print("    ✓ Orientation corrected")
        
        # Verify
        new_size = mesh_corrected.bounds[1] - mesh_corrected.bounds[0]
        print(f"    New dimensions: X={new_size[0]:.3f}, Y={new_size[1]:.3f}, Z={new_size[2]:.3f}")
        
        return mesh_corrected, transform
    else:
        return mesh, transform


class TripoSRGenerator:
    """
    TripoSR with automatic upright correction.
    """
    
    def __init__(self, device='auto'):
        print("="*60)
        print("Initializing TripoSR with Upright Correction")
        print("="*60)
        
        # Auto-detect device
        if device == 'auto':
            if torch.cuda.is_available():
                device = 'cuda'
            elif torch.backends.mps.is_available():
                device = 'cpu'
            else:
                device = 'cpu'
        
        self.device = device
        print(f"Using device: {device}")
        
        try:
            print("Loading TripoSR model...")
            self.model = TSR.from_pretrained(
                "stabilityai/TripoSR",
                config_name="config.yaml",
                weight_name="model.ckpt",
            )
            
            self.model.renderer.set_chunk_size(8192)
            self.model.to(device)
            
            print("✓ TripoSR loaded\n")
            
        except Exception as e:
            print(f"✗ Error: {e}")
            raise
    
    def preprocess_image(self, image_path, target_size=512):
        """Convert RGBA to RGB and resize"""
        print(f"\nPreprocessing: {image_path.name}")
        
        image = Image.open(image_path)

        image = remove_background(image)
        
        # Convert to RGB
        if image.mode == 'RGBA':
            rgb_image = Image.new('RGB', image.size, (255, 255, 255))
            rgb_image.paste(image, mask=image.split()[3])
            image = rgb_image
        elif image.mode != 'RGB':
            image = image.convert('RGB')
        
        # Resize and center
        image = self._resize_and_center(image, target_size)
        
        print(f"  ✓ {image.size}, mode: {image.mode}")
        
        return image
    
    def _resize_and_center(self, image, target_size=512, foreground_ratio=0.85):
        """Resize preserving aspect ratio"""
        image_np = np.array(image)
        
        # Find foreground
        is_foreground = np.any(image_np < 250, axis=2)
        coords = np.argwhere(is_foreground)
        
        if len(coords) == 0:
            return image.resize((target_size, target_size), Image.LANCZOS)
        
        # Get bounding box
        y_min, x_min = coords.min(axis=0)
        y_max, x_max = coords.max(axis=0)
        
        fg_width = x_max - x_min
        fg_height = y_max - y_min
        
        # Calculate scale
        scale = target_size * foreground_ratio / max(fg_width, fg_height)
        
        # Resize
        new_width = int(image.width * scale)
        new_height = int(image.height * scale)
        resized = image.resize((new_width, new_height), Image.LANCZOS)
        
        # Center on canvas
        canvas = Image.new('RGB', (target_size, target_size), (255, 255, 255))
        offset_x = (target_size - new_width) // 2
        offset_y = (target_size - new_height) // 2
        canvas.paste(resized, (offset_x, offset_y))
        
        return canvas
    
    def generate_mesh(self, image_pil, mc_resolution=256, auto_correct_orientation=True):
        """
        Generate 3D mesh and optionally correct orientation.
        
        Args:
            image_pil: PIL Image (RGB)
            mc_resolution: Marching cubes resolution
            auto_correct_orientation: Auto-correct to upright
            
        Returns:
            mesh: trimesh.Trimesh object
            transform: Orientation correction applied
        """
        print("\nGenerating 3D mesh...")
        
        if image_pil.mode != 'RGB':
            image_pil = image_pil.convert('RGB')
        
        start_time = time.time()

        # Remove background and resize using imported functions from tsr.utils
        '''
        try:
            # remove background -> expect RGBA PIL image (foreground on alpha)
            print("  Removing background and resizing foreground...")
            image_pil = remove_background(image_pil)

            # resize foreground to occupy a reasonable portion of the canvas
            # processed = resize_foreground(processed, 0.85)
            # Composite RGBA over neutral grey as expected by TripoSR preprocessing in example
            img_np = np.array(image_pil).astype(np.float32) / 255.0
            #If image has alpha channel, composite onto 0.5 grey, otherwise keep RGB
            # if img_np.shape[2] == 4:
            #     img_np = img_np[:, :, :3] * img_np[:, :, 3:4] + (1 - img_np[:, :, 3:4]) * 0.5
            #     img_pil_for_model = Image.fromarray((img_np * 255.0).astype(np.uint8))
            #     image_pil = img_pil_for_model
            #     print("  ✓ Background removed and image prepared")
            if image_pil.mode != 'RGB':
                print(f"⚠ Warning: Converting {image_pil.mode} to RGB")
                image_pil = image_pil.convert('RGB')
        except Exception as e:
            # Fallback to original image if background removal fails
            print(f"  ⚠ Background removal/resize failed, using original image: {e}")
            image_pil = image_pil
        '''


       
        
        try:
            with torch.no_grad():
                print("  Running TripoSR...")
                
                scene_codes = self.model([image_pil], device=self.device)
                

                print(f"  Extracting mesh (res={mc_resolution})...")
                meshes = self.model.extract_mesh(
                    scene_codes,
                    True,
                    resolution=mc_resolution
                )
                # meshes = self.model.extract_mesh(scene_codes, bake_texture, resolution=mc_resolution)

                
                mesh_data = meshes[0]
                
                # Convert to trimesh
                vertices = mesh_data.vertices
                faces = mesh_data.faces
                
                # Get colors
                if hasattr(mesh_data, 'vertex_colors') and mesh_data.vertex_colors is not None:
                    vertex_colors = mesh_data.vertex_colors
                else:
                    vertex_colors = None
                
                mesh = trimesh.Trimesh(
                    vertices=vertices,
                    faces=faces,
                    vertex_colors=vertex_colors,
                    process=False
                )
            
            elapsed = time.time() - start_time
            print(f"  ✓ Generated in {elapsed:.1f}s")
            print(f"    Vertices: {len(vertices):,}, Faces: {len(faces):,}")
            
            # Auto-correct orientation
            transform = np.eye(4)
            if auto_correct_orientation:
                mesh, transform = detect_and_correct_orientation(mesh)
            
            return mesh, transform
            
        except Exception as e:
            print(f"✗ Error: {e}")
            import traceback
            traceback.print_exc()
            return None, None
    
    def save_mesh(self, mesh, output_path):
        """Save mesh to file"""
        if mesh is None:
            return False
        
        try:
            output_path = Path(output_path)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            mesh.export(str(output_path))
            print(f"✓ Saved: {output_path}")
            return True
        except Exception as e:
            print(f"✗ Error: {e}")
            return False


def main():
    print("="*60)
    print("TRIPOSR WITH AUTOMATIC UPRIGHT CORRECTION")
    print("="*60)
    print("\nThis version:")
    print("✓ Automatically detects mesh orientation")
    print("✓ Corrects to upright position")
    print("✓ Saves ready-to-use meshes")
    print("="*60)
    
    # Find images
    generated_dir = Path("generated_images")
    if not generated_dir.exists():
        print(f"\n✗ {generated_dir} not found!")
        return
    
    image_files = sorted(list(generated_dir.glob("*full.png")))
    
    if len(image_files) == 0:
        print(f"\n✗ No images in {generated_dir}")
        return
    
    print(f"\nFound {len(image_files)} images:")
    for i, img in enumerate(image_files[:5], 1):
        print(f"  {i}. {img.name}")
    
    # Choose
    choice = input("\nProcess which? (1-5) or 'all': ").strip().lower()
    
    if choice == 'all':
        selected = image_files
    else:
        try:
            idx = int(choice) - 1
            selected = [image_files[idx]] if 0 <= idx < len(image_files) else [image_files[0]]
        except:
            selected = [image_files[0]]
    
    print(f"\nProcessing {len(selected)} image(s)")
    
    # Initialize
    try:
        generator = TripoSRGenerator(device='auto')
    except Exception as e:
        print(f"\n✗ Failed: {e}")
        return
    
    # Output dir
    output_dir = Path("generated_meshes")
    output_dir.mkdir(exist_ok=True)
    
    # Process
    success = 0
    
    for i, img_path in enumerate(selected, 1):
        print(f"\n{'='*60}")
        print(f"[{i}/{len(selected)}] {img_path.name}")
        print('='*60)
        
        output_path = output_dir / f"{img_path.stem}_upright.obj"
        
        try:
            # Preprocess
            image_pil = generator.preprocess_image(img_path)
            
            # Generate with auto-correction
            mesh, transform = generator.generate_mesh(
                image_pil, 
                mc_resolution=256,
                auto_correct_orientation=True
            )
            
            # Save
            if mesh is not None:
                if generator.save_mesh(mesh, output_path):
                    success += 1
                    
                    # Also save transform info
                    import pickle
                    transform_path = output_dir / f"{img_path.stem}_transform.pkl"
                    with open(transform_path, 'wb') as f:
                        pickle.dump({'transform': transform}, f)
                    print(f"✓ Saved transform: {transform_path.name}")
        
        except Exception as e:
            print(f"✗ Error: {e}")
            import traceback
            traceback.print_exc()
    
    # Summary
    print("\n" + "="*60)
    print("GENERATION COMPLETE!")
    print("="*60)
    print(f"Success: {success}/{len(selected)}")
    
    if success > 0:
        print(f"\nMeshes saved to: {output_dir}/")
        print("\nAll meshes are now upright and ready for calibration!")
        print("\nNext step:")
        print("  python tests/test_01_calibration_keypoints.py")
        
        # Preview
        try:
            first = next(output_dir.glob("*_upright.obj"))
            mesh = trimesh.load(first)
            print(f"\nShowing preview of {first.name}...")
            mesh.show()
        except:
            print("(Preview not available)")


if __name__ == "__main__":
    main()