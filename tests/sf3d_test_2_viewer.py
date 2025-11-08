#!/usr/bin/env python3
"""
SF3D Test 2: View Textured GLB Mesh
====================================

Opens the Three.js viewer to display SF3D-generated textured meshes.

Usage:
    python tests/sf3d_test_2_viewer.py <path_to_mesh.glb>
    python tests/sf3d_test_2_viewer.py  # Opens viewer without loading a file

The viewer supports:
- Drag and drop GLB files
- Textured meshes with UV mapping
- Rotation, zoom, pan controls
- Wireframe toggle (T key)
- Lighting toggle (L key)
"""

import sys
import os
import webbrowser
from pathlib import Path
import argparse


def main():
    parser = argparse.ArgumentParser(description='Open SF3D textured mesh viewer')
    parser.add_argument(
        'mesh',
        type=str,
        nargs='?',
        help='Optional: Path to GLB file to view'
    )
    args = parser.parse_args()

    project_root = Path(__file__).parent.parent
    viewer_path = project_root / "tests" / "sf3d_viewer_textured.html"

    if not viewer_path.exists():
        print(f"❌ Viewer not found: {viewer_path}")
        sys.exit(1)

    # Check if mesh file exists
    mesh_path = None
    if args.mesh:
        mesh_path = Path(args.mesh)
        if not mesh_path.exists():
            print(f"❌ Mesh file not found: {mesh_path}")
            sys.exit(1)

        if not mesh_path.suffix.lower() == '.glb':
            print(f"⚠️  Warning: File doesn't have .glb extension: {mesh_path}")
            print(f"   Viewer may not load it correctly.")

    print("=" * 70)
    print("SF3D Test 2: Textured Mesh Viewer")
    print("=" * 70)

    if mesh_path:
        print(f"📁 Mesh: {mesh_path.absolute()}")
        print(f"\n💡 The viewer will open in your browser.")
        print(f"   Drag and drop the mesh file onto the viewer to load it.")
    else:
        print(f"📁 No mesh specified")
        print(f"\n💡 The viewer will open in your browser.")
        print(f"   Click 'Load GLB File' or drag and drop a .glb file.")

    print(f"\n🎮 Controls:")
    print(f"   • Left Mouse: Rotate")
    print(f"   • Right Mouse: Pan")
    print(f"   • Scroll: Zoom")
    print(f"   • R: Reset view")
    print(f"   • T: Toggle wireframe")
    print(f"   • L: Toggle lighting")

    print("=" * 70)

    # Open viewer in browser
    viewer_url = viewer_path.as_uri()
    print(f"\n🌐 Opening viewer: {viewer_url}")

    try:
        webbrowser.open(viewer_url)
        print(f"\n✅ Viewer opened in browser!")

        if mesh_path:
            print(f"\n📌 Next Step:")
            print(f"   Drag and drop this file onto the viewer:")
            print(f"   {mesh_path.absolute()}")
    except Exception as e:
        print(f"\n⚠️  Could not open browser automatically: {e}")
        print(f"\n   Please open this URL manually:")
        print(f"   {viewer_url}")

    print("\n" + "=" * 70)


if __name__ == "__main__":
    main()
