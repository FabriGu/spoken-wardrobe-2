"""
Mesh Simplification for Real-time Rendering
============================================
Reduces mesh complexity from 40k vertices to ~2k for real-time performance.

Run: python tests/simplify_mesh.py
"""

import trimesh
from pathlib import Path
import numpy as np


def simplify_mesh_for_realtime(mesh_path, target_faces=4000):
    """
    Simplify mesh for real-time rendering.
    
    Args:
        mesh_path: Path to mesh file
        target_faces: Target face count (default 4000 = ~2000 vertices)
        
    Returns:
        simplified mesh
    """
    print(f"\nLoading: {mesh_path.name}")
    
    # Load mesh
    mesh = trimesh.load(mesh_path, process=False)
    
    print(f"  Original: {len(mesh.vertices):,} vertices, {len(mesh.faces):,} faces")
    
    # Simplify using quadric decimation
    print(f"  Simplifying to ~{target_faces} faces...")
    
    try:
        # Trimesh's simplify_quadric_decimation
        simplified = mesh.simplify_quadric_decimation(target_faces)
        
        print(f"  Result: {len(simplified.vertices):,} vertices, {len(simplified.faces):,} faces")
        print(f"  Reduction: {100 * (1 - len(simplified.vertices)/len(mesh.vertices)):.1f}%")
        
        return simplified
        
    except Exception as e:
        print(f"  ✗ Error: {e}")
        print("  Trying alternative method...")
        
        # Fallback: simple vertex clustering
        simplified = mesh.simplify_vertex_clustering(voxel_size=0.01)
        
        print(f"  Result: {len(simplified.vertices):,} vertices, {len(simplified.faces):,} faces")
        
        return simplified


def main():
    print("="*60)
    print("MESH SIMPLIFICATION FOR REAL-TIME RENDERING")
    print("="*60)
    print("\nThis will create low-poly versions of your meshes")
    print("for smooth real-time rendering.")
    print("\nTarget: ~2000-4000 vertices (down from ~40,000)")
    print("="*60)
    
    # Find calibrated meshes
    calib_dir = Path("calibration_data")
    if not calib_dir.exists():
        print(f"\n✗ {calib_dir} not found")
        return
    
    mesh_files = sorted(list(calib_dir.glob("*_corrected.obj")))
    
    if len(mesh_files) == 0:
        print(f"\n✗ No meshes in {calib_dir}")
        return
    
    print(f"\nFound {len(mesh_files)} meshes:")
    for i, mesh_file in enumerate(mesh_files, 1):
        print(f"  {i}. {mesh_file.name}")
    
    # Choose target face count
    print("\nFace count options:")
    print("  1. 2000 faces (fastest, less detail)")
    print("  2. 4000 faces (balanced)")
    print("  3. 8000 faces (more detail, slower)")
    
    choice = input("\nChoose (1-3) [default: 2]: ").strip() or "2"
    
    face_targets = {'1': 2000, '2': 4000, '3': 8000}
    target_faces = face_targets.get(choice, 4000)
    
    print(f"\nTarget: {target_faces} faces")
    
    # Process all meshes
    success = 0
    
    for mesh_file in mesh_files:
        try:
            # Simplify
            simplified = simplify_mesh_for_realtime(mesh_file, target_faces)
            
            # Save with "_simple" suffix
            output_path = mesh_file.parent / f"{mesh_file.stem}_simple.obj"
            simplified.export(str(output_path))
            
            print(f"  ✓ Saved: {output_path.name}\n")
            success += 1
            
        except Exception as e:
            print(f"  ✗ Failed: {e}\n")
    
    print("="*60)
    print(f"Simplified {success}/{len(mesh_files)} meshes")
    print("="*60)
    
    if success > 0:
        print("\nSimplified meshes saved with '_simple' suffix")
        print("\nUpdate your scripts to use these:")
        print("  - Look for '*_simple.obj' instead of '*_corrected.obj'")
        print("  - Or manually specify the simplified mesh")
        print("\nExpected FPS improvement: 1-2 FPS → 15-25 FPS")


if __name__ == "__main__":
    main()