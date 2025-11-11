#!/usr/bin/env python3
"""
SF3D API Client - Connect to JupyterHub FastAPI Server
======================================================

Tests SF3D mesh generation through the FastAPI server running in JupyterHub.

This script:
1. Takes a cropped clothing image
2. Uploads it to the FastAPI server on school GPU
3. Receives the generated GLB file
4. Saves it locally

Usage:
    python tests/sf3d_api_client.py <path_to_image>

Example:
    python tests/sf3d_api_client.py comfyui_generated_images/1762650549/generated_clothing.png

With custom server:
    python tests/sf3d_api_client.py image.png --server http://itp-ml.itp.tsoa.nyu.edu:8765
"""

import sys
from pathlib import Path
import argparse
import time
import requests

def main():
    parser = argparse.ArgumentParser(description='Generate 3D mesh via SF3D API')
    parser.add_argument(
        'image',
        type=str,
        help='Path to input image (cropped clothing)'
    )
    parser.add_argument(
        '--server',
        default='http://itp-ml.itp.tsoa.nyu.edu:8765',
        type=str,
        help='SF3D API server URL. Default: http://itp-ml.itp.tsoa.nyu.edu:8765'
    )
    parser.add_argument(
        '--texture-resolution',
        default=1024,
        type=int,
        choices=[512, 1024, 2048],
        help='Texture resolution. Default: 1024'
    )
    parser.add_argument(
        '--remesh',
        choices=['none', 'triangle', 'quad'],
        default='none',
        help='Remesh option. Default: none'
    )
    parser.add_argument(
        '--foreground-ratio',
        default=0.85,
        type=float,
        help='Foreground ratio (0.5-1.0). Default: 0.85'
    )
    parser.add_argument(
        '--output',
        default='sf3d_api_output.glb',
        type=str,
        help='Output mesh filename. Default: sf3d_api_output.glb'
    )
    args = parser.parse_args()

    # Validate image path
    image_path = Path(args.image)
    if not image_path.exists():
        print(f"❌ Image not found: {image_path}")
        sys.exit(1)

    print("="*70)
    print("SF3D API Client - Image to 3D Mesh")
    print("="*70)
    print(f"Image: {image_path}")
    print(f"Server: {args.server}")
    print(f"Output: {args.output}")
    print(f"\nSettings:")
    print(f"  Texture Resolution: {args.texture_resolution}")
    print(f"  Remesh Option: {args.remesh}")
    print(f"  Foreground Ratio: {args.foreground_ratio}")
    print("="*70)

    # Test connection
    print("\n🌐 Testing server connection...")
    try:
        response = requests.get(f"{args.server}/health", timeout=5)
        if response.status_code == 200:
            health = response.json()
            print(f"   ✓ Server is healthy")
            print(f"   Device: {health.get('device', 'unknown')}")
            print(f"   CUDA available: {health.get('cuda_available', False)}")
        else:
            print(f"   ⚠️  Server responded with status {response.status_code}")
    except requests.exceptions.RequestException as e:
        print(f"\n❌ Cannot connect to server: {e}")
        print(f"   Make sure:")
        print(f"   1. You're on NYU network/VPN")
        print(f"   2. JupyterHub notebook is running")
        print(f"   3. Server URL is correct: {args.server}")
        sys.exit(1)

    # Load and upload image
    print(f"\n📤 Uploading image...")
    try:
        with open(image_path, 'rb') as f:
            files = {'file': (image_path.name, f, 'image/png')}
            data = {
                'texture_resolution': args.texture_resolution,
                'remesh_option': args.remesh,
                'foreground_ratio': args.foreground_ratio
            }

            print(f"\n🚀 Generating 3D mesh...")
            print(f"   (This may take 5-30 seconds on GPU)")
            start_time = time.time()

            response = requests.post(
                f"{args.server}/generate",
                files=files,
                data=data,
                timeout=120  # 2 minutes max
            )

            if response.status_code == 200:
                elapsed = time.time() - start_time

                # Get metadata from headers
                generation_time = response.headers.get('X-Generation-Time', 'unknown')
                file_size = response.headers.get('X-File-Size', 'unknown')

                print(f"\n✅ Generation completed!")
                print(f"   Server processing time: {generation_time}s")
                print(f"   Total time (with upload): {elapsed:.1f}s")

                # Save mesh
                output_path = Path(args.output)
                with open(output_path, 'wb') as f:
                    f.write(response.content)

                actual_size = len(response.content)
                print(f"\n📥 Mesh saved!")
                print(f"   Path: {output_path.absolute()}")
                print(f"   Size: {actual_size / 1024:.1f} KB")

                print(f"\n🎯 Next Steps:")
                print(f"   1. View in Blender")
                print(f"   2. View online: https://gltf-viewer.donmccurdy.com/")
                print(f"   3. Use with rigged clothing: python tests/test_a_rigged_clothing.py")

            else:
                print(f"\n❌ Server error: {response.status_code}")
                print(f"   Response: {response.text}")
                sys.exit(1)

    except requests.exceptions.Timeout:
        print(f"\n❌ Request timed out")
        print(f"   The server may be busy or the image is too complex")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

    print("="*70)


if __name__ == "__main__":
    main()
