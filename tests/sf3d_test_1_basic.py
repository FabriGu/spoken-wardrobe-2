#!/usr/bin/env python3
"""
SF3D Test 1: Basic Mesh Generation
===================================

Tests Stable Fast 3D for generating clothing meshes with UV-unwrapped textures.

Key SF3D Features:
- 0.5 second generation (much faster than TripoSR)
- UV-unwrapped textures (not vertex colors)
- Quad/Triangle remeshing options for animation
- Better topology than TripoSR's marching cubes
- Outputs GLB with baked textures

Settings:
- remesh_option: "none" (default, fast)
- texture_resolution: 1024 (balance quality/speed)
- foreground_ratio: 0.75 (more padding for clothing)
"""

import sys
import os
from pathlib import Path
import argparse

# Add SF3D to path
project_root = Path(__file__).parent.parent
sf3d_path = project_root / "external" / "stable-fast-3d"
sys.path.insert(0, str(sf3d_path))

import torch
from PIL import Image
import rembg

from sf3d.system import SF3D
from sf3d.utils import get_device, remove_background, resize_foreground


def main():
    parser = argparse.ArgumentParser(description='SF3D Test 1: Basic mesh generation')
    parser.add_argument(
        'image',
        type=str,
        help='Path to input image'
    )
    parser.add_argument(
        '--output-dir',
        default='generated_meshes/sf3d_test_1',
        type=str,
        help='Output directory. Default: generated_meshes/sf3d_test_1'
    )
    parser.add_argument(
        '--texture-resolution',
        default=1024,
        type=int,
        help='Texture resolution in pixels. Default: 1024'
    )
    parser.add_argument(
        '--remesh-option',
        choices=['none', 'triangle', 'quad'],
        default='none',
        help='Remeshing option. "quad" recommended for animation. Default: none'
    )
    parser.add_argument(
        '--foreground-ratio',
        default=0.75,
        type=float,
        help='Foreground size ratio. Default: 0.75'
    )
    parser.add_argument(
        '--target-vertex-count',
        default=-1,
        type=int,
        help='Target vertex count for remeshing. -1 = no reduction. Default: -1'
    )
    args = parser.parse_args()

    # Validate paths
    image_path = Path(args.image)
    if not image_path.exists():
        print(f"❌ Image not found: {image_path}")
        sys.exit(1)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("SF3D Test 1: Basic Mesh Generation")
    print("=" * 70)
    print(f"Input: {image_path}")
    print(f"Output: {output_dir}")
    print(f"\nSettings:")
    print(f"  - texture_resolution: {args.texture_resolution}")
    print(f"  - remesh_option: {args.remesh_option}")
    print(f"  - foreground_ratio: {args.foreground_ratio}")
    print(f"  - target_vertex_count: {args.target_vertex_count}")
    print("=" * 70)

    # Detect device
    device = get_device()
    print(f"\n🖥️  Device: {device}")

    if device == "cpu":
        print("⚠️  Running on CPU (slower). For faster generation, use a GPU.")

    # Load SF3D model
    print("\n📦 Loading SF3D model...")
    print("   (First run will download ~2GB model from Hugging Face)")

    try:
        model = SF3D.from_pretrained(
            "stabilityai/stable-fast-3d",
            config_name="config.yaml",
            weight_name="model.safetensors",
        )
        model.to(device)
        model.eval()
        print("   ✓ Model loaded successfully")
    except Exception as e:
        print(f"\n❌ Failed to load SF3D model: {e}")
        print("\n💡 If you see an authentication error:")
        print("   1. Visit https://huggingface.co/stabilityai/stable-fast-3d")
        print("   2. Click 'Request Access' and wait for approval")
        print("   3. Run: huggingface-cli login")
        print("   4. Enter your Hugging Face token")
        sys.exit(1)

    # Process image
    print(f"\n🖼️  Processing image...")
    rembg_session = rembg.new_session()

    image = remove_background(
        Image.open(image_path).convert("RGBA"),
        rembg_session
    )
    image = resize_foreground(image, args.foreground_ratio)

    # Save preprocessed image
    preprocessed_path = output_dir / "input_preprocessed.png"
    image.save(preprocessed_path)
    print(f"   ✓ Saved preprocessed image: {preprocessed_path}")

    # Generate mesh
    print(f"\n🚀 Generating 3D mesh...")
    print(f"   This should take ~0.5 seconds on GPU, ~10 seconds on CPU")

    import time
    start_time = time.time()

    with torch.no_grad():
        # Use autocast only for CUDA
        if device == "cuda":
            with torch.autocast(device_type=device, dtype=torch.bfloat16):
                mesh, glob_dict = model.run_image(
                    [image],
                    bake_resolution=args.texture_resolution,
                    remesh=args.remesh_option,
                    vertex_count=args.target_vertex_count,
                )
        else:
            # MPS/CPU - no autocast
            mesh, glob_dict = model.run_image(
                [image],
                bake_resolution=args.texture_resolution,
                remesh=args.remesh_option,
                vertex_count=args.target_vertex_count,
            )

    elapsed = time.time() - start_time
    print(f"   ✓ Generation completed in {elapsed:.2f} seconds")

    # Save mesh
    output_mesh_path = output_dir / "mesh.glb"
    mesh.export(str(output_mesh_path), include_normals=True)

    print(f"\n✅ Success!")
    print(f"   Mesh saved to: {output_mesh_path}")
    print(f"   Preprocessed image: {preprocessed_path}")

    # Print mesh info
    print(f"\n📊 Mesh Information:")
    print(f"   Vertices: {len(mesh.vertices)}")
    print(f"   Faces: {len(mesh.faces)}")

    # Memory usage
    if device == "cuda":
        peak_memory = torch.cuda.max_memory_allocated() / 1024 / 1024
        print(f"   Peak GPU Memory: {peak_memory:.1f} MB")
    elif device == "mps":
        peak_memory = torch.mps.driver_allocated_memory() / 1024 / 1024
        print(f"   Peak GPU Memory: {peak_memory:.1f} MB")

    print("\n" + "=" * 70)
    print("Next Steps:")
    print("=" * 70)
    print(f"1. View in Blender or online GLB viewer:")
    print(f"   https://gltf-viewer.donmccurdy.com/")
    print(f"\n2. Use with rigged clothing test:")
    print(f"   python tests/test_a_rigged_clothing.py")
    print(f"   (Update line 117 to point to {output_mesh_path})")
    print(f"\n3. View in textured Three.js viewer:")
    print(f"   python tests/sf3d_test_2_viewer.py {output_mesh_path}")
    print("=" * 70)


if __name__ == "__main__":
    main()
