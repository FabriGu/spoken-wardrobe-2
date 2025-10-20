"""
Alternative: Improved Depth-Based Conversion (No TripoSR)
==========================================================
Uses depth estimation BUT properly handles transparency.
No complex installations needed - uses your existing dependencies.

This is a WORKING SOLUTION while we figure out TripoSR.

Run from root: python tests/test_02_simple_depth_improved.py
"""

import cv2
import numpy as np
from PIL import Image
import trimesh
from pathlib import Path
import sys

sys.path.append(str(Path(__file__).parent.parent))


class ImprovedDepthMeshGenerator:
    """
    Depth-based mesh generation that respects transparency.
    Key fix: Mask out transparent regions BEFORE depth estimation.
    """
    
    def __init__(self):
        self.depth_estimator = None
    
    def load_depth_model(self):
        """Load depth model (lazy loading)"""
        if self.depth_estimator is not None:
            return
        
        print("Loading depth model...")
        from transformers import pipeline
        
        self.depth_estimator = pipeline(
            "depth-estimation",
            model="Intel/dpt-large"
        )
        print("✓ Depth model loaded")
    
    def preprocess_transparent_image(self, image_pil):
        """
        KEY FIX: Handle transparent backgrounds properly.
        
        Instead of letting depth model see black, we:
        1. Extract alpha channel
        2. Create white background
        3. Composite image on white
        4. Keep alpha for later masking
        """
        print("Preprocessing transparent image...")
        
        if image_pil.mode != 'RGBA':
            print("⚠ Image has no alpha channel, adding white background")
            # Convert to RGBA
            img_rgb = image_pil.convert('RGB')
            alpha = Image.new('L', img_rgb.size, 255)
            image_pil = Image.merge('RGBA', (*img_rgb.split(), alpha))
        
        # Extract alpha channel
        alpha_channel = image_pil.split()[3]
        alpha_array = np.array(alpha_channel)
        
        # Create white background
        background = Image.new('RGB', image_pil.size, (255, 255, 255))
        
        # Composite image on white (for depth estimation)
        rgb_on_white = Image.alpha_composite(
            Image.new('RGBA', image_pil.size, (255, 255, 255, 255)),
            image_pil
        ).convert('RGB')
        
        print(f"✓ Preprocessed: {image_pil.size}, alpha preserved")
        
        return rgb_on_white, alpha_array
    
    def estimate_depth(self, image_pil):
        """Estimate depth from preprocessed image"""
        self.load_depth_model()
        
        print("Estimating depth...")
        result = self.depth_estimator(image_pil)
        depth = np.array(result["depth"])
        
        # Normalize
        depth = (depth - depth.min()) / (depth.max() - depth.min() + 1e-8)
        
        print(f"✓ Depth map computed: {depth.shape}")
        return depth
    
    def create_mesh_with_alpha_masking(self, rgb_image, depth_map, alpha_mask, 
                                       depth_scale=50, alpha_threshold=128):
        """
        Create mesh but ONLY for non-transparent regions.
        
        KEY: Use alpha_mask to determine which vertices to include.
        """
        print(f"Creating mesh with alpha masking (threshold={alpha_threshold})...")
        
        # Convert to numpy if needed
        if isinstance(rgb_image, Image.Image):
            rgb_array = np.array(rgb_image)
        else:
            rgb_array = rgb_image
        
        h, w = alpha_mask.shape
        
        # Resize depth to match if needed
        if depth_map.shape != alpha_mask.shape:
            depth_map = cv2.resize(depth_map, (w, h))
        
        # Create vertices only for visible pixels
        vertices = []
        colors = []
        vertex_map = {}  # Maps (y, x) to vertex index
        
        step = 4  # Downsample for performance
        
        vertex_idx = 0
        for y in range(0, h, step):
            for x in range(0, w, step):
                # Only create vertex if pixel is NOT transparent
                if alpha_mask[y, x] > alpha_threshold:
                    z = depth_map[y, x] * depth_scale
                    vertices.append([x, y, z])
                    colors.append(rgb_array[y, x])
                    vertex_map[(y, x)] = vertex_idx
                    vertex_idx += 1
        
        if len(vertices) == 0:
            print("✗ No visible vertices (all transparent)!")
            return None
        
        vertices = np.array(vertices, dtype=np.float32)
        colors = np.array(colors, dtype=np.uint8)
        
        print(f"  Created {len(vertices):,} vertices (transparent areas excluded)")
        
        # Create faces connecting neighboring visible vertices
        faces = []
        
        for y in range(0, h - step, step):
            for x in range(0, w - step, step):
                # Get indices of quad corners (if they exist)
                tl = vertex_map.get((y, x))
                tr = vertex_map.get((y, x + step))
                bl = vertex_map.get((y + step, x))
                br = vertex_map.get((y + step, x + step))
                
                # Only create faces if all corners are visible
                if tl is not None and tr is not None and bl is not None and br is not None:
                    # Triangle 1
                    faces.append([tl, tr, bl])
                    # Triangle 2
                    faces.append([tr, br, bl])
        
        if len(faces) == 0:
            print("⚠ No faces created (vertices too sparse)")
            return None
        
        faces = np.array(faces)
        
        print(f"  Created {len(faces):,} faces")
        
        # Create mesh
        mesh = trimesh.Trimesh(
            vertices=vertices,
            faces=faces,
            vertex_colors=colors,
            process=False
        )
        
        print(f"✓ Mesh created: {len(vertices):,} vertices, {len(faces):,} faces")
        
        return mesh
    
    def save_mesh(self, mesh, output_path):
        """Save mesh to file"""
        if mesh is None:
            return False
        
        try:
            mesh.export(output_path)
            print(f"✓ Mesh saved: {output_path}")
            return True
        except Exception as e:
            print(f"✗ Error saving: {e}")
            return False


def main():
    print("="*60)
    print("IMPROVED DEPTH-BASED MESH GENERATION")
    print("="*60)
    print("\nThis version:")
    print("✓ Properly handles transparent backgrounds")
    print("✓ No TripoSR installation needed")
    print("✓ Uses your existing dependencies")
    print("✓ Creates clean meshes (no background artifacts)")
    print("\nHow it works:")
    print("1. Extract alpha channel from PNG")
    print("2. Composite on white for depth estimation")
    print("3. Create mesh ONLY where alpha > threshold")
    print("4. Result: Clean mesh, no background geometry")
    print("="*60)
    
    # Find images
    generated_dir = Path("generated_images")
    if not generated_dir.exists():
        print(f"\n✗ {generated_dir} not found!")
        return
    
    image_files = sorted(list(generated_dir.glob("*_clothing.png")))
    
    if len(image_files) == 0:
        print(f"\n✗ No images found in {generated_dir}")
        return
    
    print(f"\nFound {len(image_files)} images:")
    for i, img in enumerate(image_files[:5], 1):
        print(f"  {i}. {img.name}")
    
    # Choose image
    choice = input("\nWhich image? (1-5) or 'all': ").strip().lower()
    
    if choice == 'all':
        selected_images = image_files
    else:
        try:
            idx = int(choice) - 1
            selected_images = [image_files[idx]] if 0 <= idx < len(image_files) else [image_files[0]]
        except:
            selected_images = [image_files[0]]
    
    # Initialize generator
    generator = ImprovedDepthMeshGenerator()
    
    # Output directory
    output_dir = Path("generated_meshes")
    output_dir.mkdir(exist_ok=True)
    
    # Process each image
    success_count = 0
    
    for i, img_path in enumerate(selected_images, 1):
        print(f"\n{'='*60}")
        print(f"[{i}/{len(selected_images)}] Processing: {img_path.name}")
        print('='*60)
        
        output_path = output_dir / f"{img_path.stem}_clean.obj"
        
        try:
            # Load image
            image_pil = Image.open(img_path)
            print(f"Loaded: {image_pil.size}, mode: {image_pil.mode}")
            
            # Preprocess (handle transparency)
            rgb_for_depth, alpha_mask = generator.preprocess_transparent_image(image_pil)
            
            # Estimate depth
            depth_map = generator.estimate_depth(rgb_for_depth)
            
            # Save depth visualization
            depth_viz = (depth_map * 255).astype(np.uint8)
            depth_colored = cv2.applyColorMap(depth_viz, cv2.COLORMAP_TURBO)
            depth_viz_path = output_dir / f"{img_path.stem}_depth.png"
            cv2.imwrite(str(depth_viz_path), depth_colored)
            print(f"✓ Depth visualization: {depth_viz_path.name}")
            
            # Create mesh with alpha masking
            mesh = generator.create_mesh_with_alpha_masking(
                rgb_for_depth, depth_map, alpha_mask,
                depth_scale=50,
                alpha_threshold=128  # Pixels with alpha < 128 are excluded
            )
            
            # Save mesh
            if mesh is not None:
                if generator.save_mesh(mesh, str(output_path)):
                    success_count += 1
        
        except Exception as e:
            print(f"✗ Error: {e}")
            import traceback
            traceback.print_exc()
    
    # Summary
    print("\n" + "="*60)
    print("CONVERSION COMPLETE")
    print("="*60)
    print(f"Success: {success_count}/{len(selected_images)}")
    print(f"\nOutput: {output_dir}")
    
    if success_count > 0:
        print("\nGenerated files:")
        for obj_file in sorted(output_dir.glob("*_clean.obj")):
            print(f"  - {obj_file.name}")
        
        print("\nNext: Run Test 3 to see overlay")
        print("  python tests/test_03_3d_clothing_overlay.py")
        
        # Show first mesh
        try:
            first_mesh = next(output_dir.glob("*_clean.obj"))
            mesh = trimesh.load(first_mesh)
            print(f"\nShowing 3D preview...")
            mesh.show()
        except:
            print("(3D preview not available)")


if __name__ == "__main__":
    main()