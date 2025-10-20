"""
TripoSR Mesh Generation (Fixed RGBA Issue)
===========================================
Works with cloned GitHub repo.
Properly handles RGBA → RGB conversion.

Run from root: python tests/test_02_triposr_fixed.py
"""

import torch
import numpy as np
from PIL import Image
import trimesh
from pathlib import Path
import sys
import time

# Add TripoSR to path (assuming you cloned to external/TripoSR)
triposr_path = Path(__file__).parent.parent / "external" / "TripoSR"

if not triposr_path.exists():
    print(f"✗ TripoSR not found at: {triposr_path}")
    print("\nExpected structure:")
    print("  spoken-wardrobe/")
    print("  ├── external/")
    print("  │   └── TripoSR/  ← Clone here")
    print("\nRun:")
    print("  cd spoken-wardrobe")
    print("  mkdir -p external")
    print("  cd external")
    print("  git clone https://github.com/VAST-AI-Research/TripoSR.git")
    sys.exit(1)



sys.path.insert(0, str(triposr_path))

# Now import TripoSR
from tsr.system import TSR
from tsr.utils import remove_background, resize_foreground, save_video
from tsr.bake_texture import bake_texture


class TripoSRGenerator:
    """
    TripoSR mesh generator with proper RGBA handling.
    """
    
    def __init__(self, device='auto'):
        """Initialize TripoSR from local clone"""
        print("="*60)
        print("Initializing TripoSR from local installation...")
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
            # Load model
            print("Loading TripoSR model...")
            self.model = TSR.from_pretrained(
                "stabilityai/TripoSR",
                config_name="config.yaml",
                weight_name="model.ckpt",
            )
            
            self.model.renderer.set_chunk_size(8192)
            self.model.to(device)
            
            print("✓ TripoSR loaded successfully!")
            
        except Exception as e:
            print(f"✗ Error loading TripoSR: {e}")
            print("\nTroubleshooting:")
            print("1. Make sure you ran: pip install -e . in TripoSR directory")
            print("2. Check that all TripoSR dependencies are installed")
            raise
    
    def preprocess_image(self, image_path, target_size=512):
        """
        Preprocess image for TripoSR.
        
        KEY FIX: Properly converts RGBA → RGB before passing to model.
        """
        print(f"\nPreprocessing: {image_path}")
        
        # Load image
        image = Image.open(image_path)
        print(f"  Loaded: {image.size}, mode: {image.mode}")
        
        # CRITICAL: Convert to RGB (TripoSR expects 3 channels, not 4)
        if image.mode == 'RGBA':
            print("  Converting RGBA → RGB (compositing on white)")
            # Create white background
            rgb_image = Image.new('RGB', image.size, (255, 255, 255))
            # Paste image using alpha as mask
            rgb_image.paste(image, mask=image.split()[3])
            image = rgb_image
        elif image.mode != 'RGB':
            print(f"  Converting {image.mode} → RGB")
            image = image.convert('RGB')
        
        # Verify it's now RGB
        assert image.mode == 'RGB', f"Image must be RGB, got {image.mode}"
        print(f"  ✓ Converted to RGB: {image.size}")
        
        # Resize and center (similar to TripoSR's resize_foreground)
        image = self._resize_and_center(image, target_size)
        
        print(f"  ✓ Preprocessed: {image.size}, mode: {image.mode}")
        
        return image
    
    def _resize_and_center(self, image, target_size=512, foreground_ratio=0.85):
        """
        Resize and center foreground.
        Handles both transparent and opaque images.
        """
        image_np = np.array(image)
        
        # Find non-white pixels (foreground)
        is_foreground = np.any(image_np < 250, axis=2)
        coords = np.argwhere(is_foreground)
        
        if len(coords) == 0:
            # No foreground, just resize
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
        
        # Create canvas and paste centered
        canvas = Image.new('RGB', (target_size, target_size), (255, 255, 255))
        offset_x = (target_size - new_width) // 2
        offset_y = (target_size - new_height) // 2
        canvas.paste(resized, (offset_x, offset_y))
        
        return canvas
    
    def generate_mesh(self, image_pil, mc_resolution=256):
        """
        Generate 3D mesh from RGB image.
        
        Args:
            image_pil: PIL Image in RGB mode (NOT RGBA!)
            mc_resolution: Marching cubes resolution
            
        Returns:
            trimesh.Trimesh object
        """
        print("\nGenerating 3D mesh with TripoSR...")
        
        # Double-check image is RGB
        if image_pil.mode != 'RGB':
            print(f"⚠ Warning: Converting {image_pil.mode} to RGB")
            image_pil = image_pil.convert('RGB')
        
        start_time = time.time()
        
        try:
            with torch.no_grad():
                # Generate scene codes
                print("  Running TripoSR model...")
                scene_codes = self.model([image_pil], device=self.device)
                
                # Extract mesh
                print(f"  Extracting mesh (resolution={mc_resolution})...")
                meshes = self.model.extract_mesh(
                    scene_codes,
                    True,
                    resolution=mc_resolution
                )
                # meshes = self.model.extract_mesh(scene_codes, not tsr.utils.bake_texture, resolution=mc_resolution)

                
                # Get mesh data
                mesh_data = meshes[0]
                
                # Convert to trimesh
                vertices = mesh_data.vertices
                faces = mesh_data.faces
                
                # Get vertex colors
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
            print(f"✓ Mesh generated in {elapsed:.1f} seconds")
            print(f"  Vertices: {len(vertices):,}")
            print(f"  Faces: {len(faces):,}")
            
            return mesh
            
        except Exception as e:
            print(f"✗ Error generating mesh: {e}")
            import traceback
            traceback.print_exc()
            return None
    
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
            print(f"✗ Error saving: {e}")
            return False


def main():
    print("="*60)
    print("TRIPOSR MESH GENERATION (RGBA FIXED)")
    print("="*60)
    print("\nThis version:")
    print("✓ Works with cloned GitHub repo")
    print("✓ Properly handles RGBA → RGB conversion")
    print("✓ No tensor size mismatch errors")
    print("\nRequirements:")
    print("  - TripoSR cloned to external/TripoSR/")
    print("  - Installed: cd external/TripoSR && pip install -e .")
    print("="*60)
    
    # Find images
    generated_dir = Path("generated_images")
    if not generated_dir.exists():
        print(f"\n✗ {generated_dir} not found!")
        print("Generate clothing images first")
        return
    
    image_files = sorted(list(generated_dir.glob("*_clothing.png")))
    
    if len(image_files) == 0:
        print(f"\n✗ No clothing images in {generated_dir}")
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
    
    print(f"\nWill process {len(selected)} image(s)")
    
    # Initialize
    try:
        generator = TripoSRGenerator(device='auto')
    except Exception as e:
        print(f"\n✗ Failed to initialize: {e}")
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
        
        output_path = output_dir / f"{img_path.stem}_triposr.obj"
        
        try:
            # Preprocess (RGBA → RGB conversion happens here)
            image_pil = generator.preprocess_image(img_path)
            
            # Save preprocessed for debugging
            preprocessed_path = output_dir / f"{img_path.stem}_preprocessed.png"
            image_pil.save(preprocessed_path)
            print(f"  Saved preprocessed: {preprocessed_path.name}")
            
            # Verify it's RGB before passing to model
            print(f"  Image mode before generation: {image_pil.mode}")
            assert image_pil.mode == 'RGB', "Must be RGB!"
            
            # Generate mesh
            mesh = generator.generate_mesh(image_pil, mc_resolution=256)
            
            # Save
            if mesh is not None:
                if generator.save_mesh(mesh, output_path):
                    success += 1
        
        except Exception as e:
            print(f"✗ Error: {e}")
            import traceback
            traceback.print_exc()
    
    # Summary
    print("\n" + "="*60)
    print("CONVERSION COMPLETE!")
    print("="*60)
    print(f"Successfully converted: {success}/{len(selected)} images")
    
    if success > 0:
        print(f"\nGenerated meshes in: {output_dir}/")
        for obj in sorted(output_dir.glob("*_triposr.obj")):
            print(f"  - {obj.name}")
        
        print("\nNext:")
        print("  python tests/test_03_triposr_keypoint_warping.py")
        
        # Show first mesh
        try:
            first = next(output_dir.glob("*_triposr.obj"))
            mesh = trimesh.load(first)
            print(f"\nShowing 3D preview...")
            mesh.show()
        except:
            print("(3D preview not available)")


if __name__ == "__main__":
    main()