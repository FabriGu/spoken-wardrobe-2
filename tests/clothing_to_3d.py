"""
Test 2: Convert 2D Clothing to 3D Mesh
=======================================
Takes your SD-generated clothing image and creates a simple 3D mesh.
Uses depth estimation to add the third dimension.

Run from root: python tests/test_02_clothing_to_3d.py
"""

import cv2
import numpy as np
from PIL import Image
import trimesh
from pathlib import Path
import sys

sys.path.append(str(Path(__file__).parent.parent))


class ClothingMeshGenerator:
    """
    Converts 2D clothing image to 3D mesh using depth estimation.
    This is a simplified approach - good enough for overlay without physics.
    """
    
    def __init__(self):
        self.depth_estimator = None
        print("ClothingMeshGenerator initialized")
    
    def load_depth_model(self):
        """Load depth estimation model (lazy loading for speed)"""
        if self.depth_estimator is not None:
            return
        
        print("Loading depth estimation model...")
        from transformers import pipeline
        import torch
        
        device = "cuda" if torch.cuda.is_available() else "cpu"
        
        # Use a small, fast depth model for testing
        self.depth_estimator = pipeline(
            "depth-estimation",
            model="Intel/dpt-large",  # or "LiheYoung/depth-anything-small-hf" for faster
            device=device
        )
        print("Depth model loaded!")
    
    def estimate_depth(self, image_pil):
        """
        Estimate depth map from RGB image.
        Returns normalized depth map (0=far, 1=close).
        """
        self.load_depth_model()
        
        print("Estimating depth...")
        result = self.depth_estimator(image_pil)
        depth = np.array(result["depth"])
        
        # Normalize to 0-1
        depth = (depth - depth.min()) / (depth.max() - depth.min() + 1e-8)
        
        return depth
    
    def create_mesh_from_depth(self, rgb_image, depth_map, depth_scale=50):
        """
        Create 3D mesh from RGB image + depth map.
        
        Args:
            rgb_image: PIL Image (RGBA)
            depth_map: Numpy array (H, W) with normalized depth
            depth_scale: How much to extrude depth (in pixels)
            
        Returns:
            trimesh.Trimesh object
        """
        print("Creating 3D mesh from depth...")
        
        # Convert to numpy
        if isinstance(rgb_image, Image.Image):
            rgb_array = np.array(rgb_image.convert('RGB'))
        else:
            rgb_array = rgb_image
        
        h, w = depth_map.shape
        
        # Resize depth to match image if needed
        if depth_map.shape != rgb_array.shape[:2]:
            depth_map = cv2.resize(depth_map, (w, h))
        
        # Create vertex grid
        # Each pixel becomes a 3D point
        vertices = []
        colors = []
        
        # Downsample for speed (every 4th pixel)
        step = 4
        
        for y in range(0, h, step):
            for x in range(0, w, step):
                # Z coordinate from depth (scaled)
                z = depth_map[y, x] * depth_scale
                
                # Only include non-transparent pixels
                # (Assuming you have alpha channel)
                vertices.append([x, y, z])
                colors.append(rgb_array[y, x])
        
        vertices = np.array(vertices, dtype=np.float32)
        colors = np.array(colors, dtype=np.uint8)
        
        # Create faces by connecting neighboring vertices
        faces = []
        cols = w // step
        
        for y in range(0, h // step - 1):
            for x in range(0, w // step - 1):
                # Current vertex index
                idx = y * cols + x
                
                # Create two triangles for this quad
                # Triangle 1
                faces.append([idx, idx + 1, idx + cols])
                # Triangle 2
                faces.append([idx + 1, idx + cols + 1, idx + cols])
        
        faces = np.array(faces)
        
        # Create trimesh
        mesh = trimesh.Trimesh(
            vertices=vertices,
            faces=faces,
            vertex_colors=colors,
            process=False  # Don't auto-clean for speed
        )
        
        print(f"Mesh created: {len(vertices)} vertices, {len(faces)} faces")
        
        return mesh
    
    def save_mesh(self, mesh, output_path):
        """Save mesh to file"""
        mesh.export(output_path)
        print(f"Mesh saved to: {output_path}")


def main():
    print("="*60)
    print("TEST 2: CONVERT 2D CLOTHING TO 3D MESH")
    print("="*60)
    print("\nThis will:")
    print("1. Load a clothing image from generated_images/")
    print("2. Estimate depth map")
    print("3. Create a 3D mesh")
    print("4. Save it and show preview")
    print("="*60)
    
    # Find generated clothing images
    generated_dir = Path("generated_images/test")
    if not generated_dir.exists():
        print("\nError: generated_images/ folder not found!")
        print("Run your SD generation first to create test images.")
        return
    
    image_files = sorted(list(generated_dir.glob("*_clothing.png")))
    print(image_files)
    # open and show images in separate tab to check that they are valid
    # Image.open(image_files[0]).show()  # just to trigger any loading issues


    if len(image_files) == 0:
        print("\nNo clothing images found!")
        return
    
    print(f"\nFound {len(image_files)} clothing images:")
    for i, img in enumerate(image_files[:5], 1):
        print(f"  {i}. {img.name}")
    
    # Let user choose
    choice = input("\nWhich image to convert? (1-5): ").strip()
    try:
        idx = int(choice) - 1
        if idx < 0 or idx >= len(image_files):
            print("Invalid choice, using first image")
            idx = 0
    except:
        idx = 0
    

    selected_image = image_files[idx]
    # selected_image = image_files[-1]
    print(selected_image)
    
    print(f"\nUsing image: {selected_image.name}")
    # selected_image.show()

    print(f"\nProcessing: {selected_image.name}")
    
    # Load image
    clothing_img = Image.open(selected_image)
    print(f"Image size: {clothing_img.size}")
    # Image.open(image_files[0]).show()

    # clothing_img = Image.open(image_files[2])
    # print(clothing_img)
    # Create mesh generator
    generator = ClothingMeshGenerator()
    
    # Estimate depth
    depth_map = generator.estimate_depth(clothing_img)
    
    # Save depth visualization
    depth_viz = (depth_map * 255).astype(np.uint8)
    depth_colored = cv2.applyColorMap(depth_viz, cv2.COLORMAP_TURBO)
    cv2.imwrite("depth_map_visualization.png", depth_colored)
    print("\nDepth map saved to: depth_map_visualization.png")
    
    # Create 3D mesh
    mesh = generator.create_mesh_from_depth(
        clothing_img,
        depth_map,
        depth_scale=50  # Adjust this for more/less depth
    )
    
    # Save mesh
    output_dir = Path("generated_meshes")
    output_dir.mkdir(exist_ok=True)
    
    output_path = output_dir / f"{selected_image.stem}_mesh.obj"
    generator.save_mesh(mesh, str(output_path))
    
    print("\n" + "="*60)
    print("TEST 2 COMPLETE!")
    print("="*60)
    print(f"\nGenerated files:")
    print(f"  - 3D mesh: {output_path}")
    print(f"  - Depth viz: depth_map_visualization.png")
    print("\nYou can:")
    print("  - Open the .obj file in Blender to see the 3D mesh")
    print("  - View depth_map_visualization.png to see depth estimation")
    print("\nNext: Test 3 will overlay this mesh on live video!")
    
    # Optional: Show mesh in 3D viewer
    try:
        print("\nAttempting to show 3D preview...")
        mesh.show()
    except:
        print("(3D preview not available - install pyglet for interactive viewing)")


if __name__ == "__main__":
    main()