#!/usr/bin/env python3
"""
ComfyUI SF3D Test - Image to 3D Mesh
====================================

Tests SF3D mesh generation through ComfyUI using the ComfyUI-3D-Pack.

This script:
1. Takes a cropped clothing image
2. Uploads it to ComfyUI server
3. Runs SF3D workflow to generate 3D mesh
4. Downloads the resulting GLB file

Usage:
    python tests/comfyui_sf3d_test0.py <path_to_image>

Example:
    python tests/comfyui_sf3d_test0.py comfyui_generated_images/1762650549/generated_clothing.png
"""

import sys
from pathlib import Path
import argparse
import time

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / "src"))

from modules.comfyui_client import ComfyUIClient
import cv2
import numpy as np


def main():
    parser = argparse.ArgumentParser(description='Test SF3D mesh generation via ComfyUI')
    parser.add_argument(
        'image',
        type=str,
        help='Path to input image (cropped clothing)'
    )
    parser.add_argument(
        '--server',
        default='http://itp-ml.itp.tsoa.nyu.edu:9199',
        type=str,
        help='ComfyUI server URL. Default: http://itp-ml.itp.tsoa.nyu.edu:9199'
    )
    parser.add_argument(
        '--workflow',
        default='workflows/sf3d_generation_api_correct.json',
        type=str,
        help='Workflow JSON file. Default: workflows/sf3d_generation_api_correct.json'
    )
    parser.add_argument(
        '--texture-resolution',
        default=1024,
        type=int,
        help='Texture resolution (512, 1024, 2048). Default: 1024'
    )
    parser.add_argument(
        '--remesh',
        choices=['None', 'triangle', 'quad'],
        default='None',
        help='Remesh option. Default: None'
    )
    parser.add_argument(
        '--foreground-ratio',
        default=0.85,
        type=float,
        help='Foreground ratio (0.5-1.0). Default: 0.85'
    )
    parser.add_argument(
        '--output',
        default='sf3d_test_output.glb',
        type=str,
        help='Output mesh filename. Default: sf3d_test_output.glb'
    )
    args = parser.parse_args()

    # Validate image path
    image_path = Path(args.image)
    if not image_path.exists():
        print(f"❌ Image not found: {image_path}")
        sys.exit(1)

    # Validate workflow
    workflow_path = Path(args.workflow)
    if not workflow_path.exists():
        print(f"❌ Workflow not found: {workflow_path}")
        sys.exit(1)

    print("="*70)
    print("ComfyUI SF3D Test - Image to 3D Mesh")
    print("="*70)
    print(f"Image: {image_path}")
    print(f"Server: {args.server}")
    print(f"Output: {args.output}")
    print(f"\nSettings:")
    print(f"  Texture Resolution: {args.texture_resolution}")
    print(f"  Remesh Option: {args.remesh}")
    print(f"  Foreground Ratio: {args.foreground_ratio}")
    print("="*70)

    # Initialize ComfyUI client
    print("\n🌐 Connecting to ComfyUI server...")
    client = ComfyUIClient(args.server)

    # Test connection
    if not client.test_connection():
        print("\n❌ Cannot connect to ComfyUI server")
        print(f"   Make sure the server is running at: {args.server}")
        print(f"   And that you're on the school network/VPN")
        sys.exit(1)

    # Load image
    print(f"\n📷 Loading image...")
    image = cv2.imread(str(image_path))
    if image is None:
        print(f"❌ Failed to load image: {image_path}")
        sys.exit(1)

    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    print(f"   Image shape: {image_rgb.shape}")

    # Upload image
    print(f"\n📤 Uploading image to server...")
    uploaded_filename = client.upload_image(image_rgb, f"sf3d_input_{client.client_id}.png")
    if not uploaded_filename:
        print("❌ Failed to upload image")
        sys.exit(1)

    # Load workflow template
    print(f"\n📋 Loading workflow template...")
    workflow_template = client.load_workflow_template(str(workflow_path))
    if not workflow_template:
        print(f"❌ Failed to load workflow: {workflow_path}")
        sys.exit(1)

    # Prepare workflow
    print(f"\n⚙️  Preparing SF3D workflow...")
    workflow = prepare_sf3d_workflow(
        workflow_template,
        uploaded_filename,
        args.texture_resolution,
        args.remesh,
        args.foreground_ratio,
        args.output  # Keep full filename including .glb
    )
    print(f"   ✓ Workflow prepared")

    # Queue workflow
    print(f"\n🚀 Queueing SF3D generation...")
    prompt_id = client.queue_prompt(workflow)
    if not prompt_id:
        print("❌ Failed to queue workflow")
        sys.exit(1)

    # Wait for completion
    print(f"\n⏳ Generating 3D mesh (this may take 5-30 seconds)...")
    start_time = time.time()

    history = client.wait_for_completion(prompt_id)
    if not history:
        print("❌ Generation failed or timed out")
        sys.exit(1)

    elapsed = time.time() - start_time
    print(f"\n✅ Generation completed in {elapsed:.1f} seconds!")

    # Download mesh
    print(f"\n📥 Downloading mesh file...")
    mesh_data = download_mesh_from_history(client, history)

    if mesh_data:
        # Save mesh to file
        output_path = Path(args.output)
        with open(output_path, 'wb') as f:
            f.write(mesh_data)

        print(f"\n✅ Success!")
        print(f"   Mesh saved to: {output_path.absolute()}")
        print(f"   File size: {len(mesh_data) / 1024:.1f} KB")

        print(f"\n🎯 Next Steps:")
        print(f"   1. View in Blender")
        print(f"   2. View online: https://gltf-viewer.donmccurdy.com/")
        print(f"   3. Use with rigged clothing: python tests/test_a_rigged_clothing.py")
    else:
        print("\n❌ Failed to download mesh")
        print("   Check ComfyUI server logs for errors")

    print("="*70)


def prepare_sf3d_workflow(workflow_template, image_filename, texture_resolution,
                          remesh_option, foreground_ratio, output_filename):
    """
    Prepare SF3D workflow by replacing template variables.

    Args:
        workflow_template: Base workflow dict
        image_filename: Uploaded input image filename
        texture_resolution: Texture size (512, 1024, 2048)
        remesh_option: "None", "triangle", or "quad"
        foreground_ratio: Size ratio (0.5-1.0)
        output_filename: Output filename with extension

    Returns:
        Prepared workflow dict
    """
    import copy
    import json

    workflow = copy.deepcopy(workflow_template)

    # Convert to JSON string for replacement
    workflow_str = json.dumps(workflow)

    # Replace placeholders
    replacements = {
        '"{INPUT_IMAGE}"': json.dumps(image_filename),
        '"{TEXTURE_RESOLUTION}"': str(texture_resolution),
        '"{REMESH_OPTION}"': json.dumps(remesh_option),
        '"{FOREGROUND_RATIO}"': str(foreground_ratio),
        '"{OUTPUT_FILENAME}"': json.dumps(output_filename)
    }

    for placeholder, value in replacements.items():
        workflow_str = workflow_str.replace(placeholder, value)

    workflow = json.loads(workflow_str)

    print(f"   Input: {image_filename}")
    print(f"   Texture: {texture_resolution}x{texture_resolution}")
    print(f"   Remesh: {remesh_option}")
    print(f"   Foreground Ratio: {foreground_ratio}")
    print(f"   Output: {output_filename}")

    return workflow


def download_mesh_from_history(client, history):
    """
    Download mesh file from ComfyUI history.

    ComfyUI-3D-Pack saves meshes to the output folder. We need to extract
    the filename from the history and download it.

    Args:
        client: ComfyUIClient instance
        history: History dict from wait_for_completion

    Returns:
        bytes: Mesh file data, or None if failed
    """
    try:
        outputs = history.get('outputs', {})

        # Look for Save3DMesh node output
        for node_id, node_output in outputs.items():
            # Check for mesh files
            if 'meshes' in node_output:
                meshes = node_output['meshes']
                if len(meshes) > 0:
                    mesh_info = meshes[0]
                    filename = mesh_info['filename']
                    subfolder = mesh_info.get('subfolder', '')
                    folder_type = mesh_info.get('type', 'output')

                    print(f"   Found mesh: {filename}")

                    # Download using ComfyUI's view endpoint
                    import requests
                    params = {
                        'filename': filename,
                        'subfolder': subfolder,
                        'type': folder_type
                    }

                    response = requests.get(
                        f"{client.base_url}/view",
                        params=params,
                        timeout=30
                    )

                    if response.status_code == 200:
                        return response.content
                    else:
                        print(f"   ✗ Download failed: HTTP {response.status_code}")
                        return None

            # Fallback: check for files output (some nodes use this)
            if 'files' in node_output:
                files = node_output['files']
                if len(files) > 0:
                    file_info = files[0]
                    filename = file_info.get('filename', file_info.get('name', ''))

                    if filename.endswith('.glb') or filename.endswith('.obj'):
                        print(f"   Found file: {filename}")

                        import requests
                        response = requests.get(
                            f"{client.base_url}/view",
                            params={'filename': filename, 'type': 'output'},
                            timeout=30
                        )

                        if response.status_code == 200:
                            return response.content

        print("   ✗ No mesh file found in output")
        print("   History structure:", outputs.keys())
        return None

    except Exception as e:
        print(f"   ✗ Error downloading mesh: {e}")
        import traceback
        traceback.print_exc()
        return None


if __name__ == "__main__":
    main()
