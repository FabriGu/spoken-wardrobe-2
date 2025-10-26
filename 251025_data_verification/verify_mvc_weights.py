"""
MVC Weights Verification Script
================================
Checks if MVC weights are computed and stored correctly.

Usage:
    python 251025_data_verification/verify_mvc_weights.py
"""

import sys
from pathlib import Path
import numpy as np
import trimesh
import time

# Add parent directory and tests directory to path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))
sys.path.append(str(project_root / "tests"))

# Import from main system
from enhanced_cage_utils import BodyPixCageGenerator, EnhancedMeanValueCoordinates
from tf_bodypix.api import download_model, load_model, BodyPixModelPaths


def verify_mvc_weights(mesh, cage):
    """
    Verify MVC weights computation and properties.
    
    Args:
        mesh: Trimesh mesh object
        cage: Trimesh cage object
    """
    print("\n" + "="*70)
    print("MVC WEIGHTS VERIFICATION")
    print("="*70)
    
    print(f"\n📊 Input Dimensions:")
    print(f"   Mesh vertices: {len(mesh.vertices)}")
    print(f"   Cage vertices: {len(cage.vertices)}")
    print(f"   Expected weight matrix shape: ({len(mesh.vertices)}, {len(cage.vertices)})")
    
    # Compute MVC weights
    print(f"\n⏱️  Computing MVC weights...")
    start_time = time.time()
    
    mvc = EnhancedMeanValueCoordinates(mesh.vertices, cage)
    mvc.compute_weights()
    
    elapsed = time.time() - start_time
    print(f"   ✓ Computation completed in {elapsed:.2f} seconds")
    
    # Verify weight properties
    print(f"\n🔍 Weight Matrix Properties:")
    weights = mvc.mvc_weights
    
    print(f"   Shape: {weights.shape}")
    print(f"   Data type: {weights.dtype}")
    print(f"   Memory size: {weights.nbytes / 1024 / 1024:.2f} MB")
    print(f"   Min value: {weights.min():.6f}")
    print(f"   Max value: {weights.max():.6f}")
    print(f"   Mean value: {weights.mean():.6f}")
    
    # Check if weights sum to 1 (critical property!)
    print(f"\n✅ Critical Properties Check:")
    row_sums = weights.sum(axis=1)
    print(f"   Row sums (should all be ~1.0):")
    print(f"      Mean: {row_sums.mean():.10f}")
    print(f"      Std: {row_sums.std():.10f}")
    print(f"      Min: {row_sums.min():.10f}")
    print(f"      Max: {row_sums.max():.10f}")
    
    if np.allclose(row_sums, 1.0, atol=1e-6):
        print(f"      ✓ PASS: All rows sum to 1.0")
    else:
        print(f"      ✗ FAIL: Rows don't sum to 1.0!")
        bad_rows = np.where(np.abs(row_sums - 1.0) > 1e-6)[0]
        print(f"      → {len(bad_rows)} / {len(row_sums)} rows have incorrect sums")
    
    # Check for NaN or Inf
    if np.any(np.isnan(weights)):
        print(f"      ✗ WARNING: Weights contain NaN values!")
    else:
        print(f"      ✓ No NaN values")
    
    if np.any(np.isinf(weights)):
        print(f"      ✗ WARNING: Weights contain Inf values!")
    else:
        print(f"      ✓ No Inf values")
    
    # Test deformation
    print(f"\n🧪 Deformation Test:")
    test_deformation(mvc, cage)
    
    # Performance test
    print(f"\n⚡ Performance Test:")
    test_deformation_performance(mvc, cage)
    
    return mvc


def test_deformation(mvc, cage):
    """Test if deformation works with the computed weights."""
    print(f"   Testing identity deformation (cage unchanged)...")
    
    # Deform with original cage (should return original mesh)
    deformed_verts = mvc.deform_mesh(cage.vertices)
    
    # Check if vertices are (approximately) unchanged
    diff = np.linalg.norm(deformed_verts - mvc.mesh_vertices, axis=1)
    max_diff = diff.max()
    mean_diff = diff.mean()
    
    print(f"   Max displacement: {max_diff:.6f}")
    print(f"   Mean displacement: {mean_diff:.6f}")
    
    if max_diff < 1e-4:
        print(f"   ✓ PASS: Identity deformation preserves mesh")
    else:
        print(f"   ⚠ WARNING: Identity deformation has unexpected displacement")
    
    # Test simple transformation
    print(f"\n   Testing translation deformation...")
    translated_cage = cage.vertices + np.array([0.1, 0.0, 0.0])
    deformed_verts = mvc.deform_mesh(translated_cage)
    
    # Check if mesh moved approximately by the same amount
    mesh_displacement = (deformed_verts - mvc.mesh_vertices).mean(axis=0)
    print(f"   Cage translation: [0.1, 0.0, 0.0]")
    print(f"   Mesh displacement: [{mesh_displacement[0]:.3f}, {mesh_displacement[1]:.3f}, {mesh_displacement[2]:.3f}]")
    
    if np.abs(mesh_displacement[0] - 0.1) < 0.05:
        print(f"   ✓ PASS: Mesh follows cage translation")
    else:
        print(f"   ⚠ WARNING: Mesh doesn't follow cage translation properly")


def test_deformation_performance(mvc, cage):
    """Test deformation performance (should be fast!)."""
    num_iterations = 100
    
    print(f"   Running {num_iterations} deformation iterations...")
    start_time = time.time()
    
    for i in range(num_iterations):
        # Small random perturbation
        perturbed_cage = cage.vertices + np.random.randn(*cage.vertices.shape) * 0.01
        deformed_verts = mvc.deform_mesh(perturbed_cage)
    
    elapsed = time.time() - start_time
    avg_time = (elapsed / num_iterations) * 1000  # ms
    
    print(f"   Total time: {elapsed:.2f} seconds")
    print(f"   Average time per deformation: {avg_time:.2f} ms")
    print(f"   Approximate FPS if this was only operation: {1000/avg_time:.1f} FPS")
    
    if avg_time < 5:
        print(f"   ✓ PASS: Deformation is fast enough for real-time")
    elif avg_time < 20:
        print(f"   ⚠ Acceptable: Deformation speed is okay but could be faster")
    else:
        print(f"   ✗ FAIL: Deformation is too slow for real-time!")


def create_test_setup():
    """Create test mesh and cage."""
    print("\n" + "="*70)
    print("CREATING TEST SETUP")
    print("="*70)
    
    # Create simple test mesh
    print(f"\n1. Creating test mesh...")
    vertices = []
    width, height, depth = 0.4, 0.6, 0.15
    
    for z_offset in [-depth/2, depth/2]:
        for y in np.linspace(0, height, 8):
            for x in np.linspace(-width/2, width/2, 6):
                x_curved = x * (1 + 0.1 * np.sin(y * np.pi / height))
                vertices.append([x_curved, y, z_offset])
    
    vertices = np.array(vertices)
    from scipy.spatial import ConvexHull
    hull = ConvexHull(vertices)
    mesh = trimesh.Trimesh(vertices=vertices, faces=hull.simplices)
    mesh.vertices -= mesh.vertices.mean(axis=0)
    mesh.vertices /= np.max(np.linalg.norm(mesh.vertices, axis=1))
    
    print(f"   ✓ Mesh: {len(mesh.vertices)} vertices, {len(mesh.faces)} faces")
    
    # Generate cage
    print(f"\n2. Loading BodyPix model...")
    bodypix_model = load_model(download_model(
        BodyPixModelPaths.MOBILENET_FLOAT_50_STRIDE_16
    ))
    
    print(f"\n3. Generating cage...")
    frame = np.ones((480, 640, 3), dtype=np.uint8) * 128
    result = bodypix_model.predict_single(frame)
    person_mask = result.get_mask(threshold=0.75)
    
    # Get masks (already numpy arrays)
    person_mask_np = person_mask.numpy() if hasattr(person_mask, 'numpy') else person_mask
    colored_mask = result.get_colored_part_mask(person_mask)
    colored_mask_np = colored_mask.numpy() if hasattr(colored_mask, 'numpy') else colored_mask
    torso_mask = result.get_part_mask(person_mask, part_names=['torso_front', 'torso_back'])
    torso_mask_np = torso_mask.numpy() if hasattr(torso_mask, 'numpy') else torso_mask
    
    segmentation_data = {
        'person_mask': person_mask_np.astype(np.uint8),
        'colored_mask': colored_mask_np.astype(np.uint8),
        'body_parts': {
            'torso': torso_mask_np.astype(np.uint8),
        }
    }
    
    cage_generator = BodyPixCageGenerator(mesh)
    cage = cage_generator.generate_anatomical_cage(
        segmentation_data,
        frame.shape,
        subdivisions=3
    )
    
    print(f"   ✓ Cage: {len(cage.vertices)} vertices, {len(cage.faces)} faces")
    
    return mesh, cage


def check_per_frame_recomputation():
    """
    Check if the main system is recomputing weights per frame.
    This simulates what happens in test_integration.py
    """
    print("\n" + "="*70)
    print("CHECKING FOR PER-FRAME RECOMPUTATION")
    print("="*70)
    
    print("\n⚠️  This test simulates the real-time loop to check if weights")
    print("   are being recomputed every frame (which would be VERY BAD)")
    
    mesh, cage = create_test_setup()
    
    # Initialize MVC (should happen once)
    print(f"\n1. Initial MVC computation (this should be SLOW)...")
    start_time = time.time()
    mvc = EnhancedMeanValueCoordinates(mesh.vertices, cage)
    mvc.compute_weights()
    init_time = time.time() - start_time
    print(f"   ✓ Initial computation: {init_time:.2f} seconds")
    
    # Store the initial weights
    initial_weights = mvc.mvc_weights.copy()
    
    # Simulate several frames of deformation
    print(f"\n2. Simulating 10 frames of deformation (should be FAST)...")
    frame_times = []
    
    for i in range(10):
        start_time = time.time()
        
        # Perturb cage slightly (simulates keypoint changes)
        perturbed_cage = cage.vertices + np.random.randn(*cage.vertices.shape) * 0.01
        
        # Deform mesh
        deformed_verts = mvc.deform_mesh(perturbed_cage)
        
        frame_time = (time.time() - start_time) * 1000
        frame_times.append(frame_time)
        
        if i == 0:
            print(f"   Frame {i+1}: {frame_time:.2f} ms")
    
    avg_frame_time = np.mean(frame_times)
    print(f"   ...")
    print(f"   Average frame time: {avg_frame_time:.2f} ms")
    
    # Check if weights changed
    if np.array_equal(initial_weights, mvc.mvc_weights):
        print(f"\n   ✓ GOOD: Weights did NOT change between frames")
    else:
        print(f"\n   ✗ BAD: Weights CHANGED between frames!")
        print(f"      → This means weights are being recomputed per frame!")
    
    # Verdict
    print(f"\n📊 Performance Verdict:")
    if avg_frame_time < 5:
        print(f"   ✓ EXCELLENT: {avg_frame_time:.2f} ms per frame (~{1000/avg_frame_time:.0f} FPS)")
        print(f"      → Deformation is fast enough for real-time")
    elif avg_frame_time < 20:
        print(f"   ✓ GOOD: {avg_frame_time:.2f} ms per frame (~{1000/avg_frame_time:.0f} FPS)")
        print(f"      → Acceptable for real-time with other operations")
    else:
        print(f"   ✗ TOO SLOW: {avg_frame_time:.2f} ms per frame")
        print(f"      → Likely recomputing weights every frame!")
        print(f"      → Check test_integration.py for cage regeneration")


def main():
    """Main verification function."""
    print("\n" + "="*70)
    print("MVC WEIGHTS VERIFICATION TOOL")
    print("="*70)
    print("\nThis tool verifies that MVC weights are computed correctly")
    print("and checks for common performance issues.\n")
    
    # Create test setup
    mesh, cage = create_test_setup()
    
    # Verify MVC weights
    mvc = verify_mvc_weights(mesh, cage)
    
    # Check for per-frame recomputation
    check_per_frame_recomputation()
    
    print("\n" + "="*70)
    print("VERIFICATION COMPLETE")
    print("="*70)
    print("\n📋 Key Findings:")
    print("   1. Check if weights sum to 1.0 ✓ or ✗")
    print("   2. Check deformation time (should be <5ms)")
    print("   3. Check if weights change per frame (they shouldn't!)")
    print("\n   If any checks failed, see:")
    print("   → docs/251025_steps_forward.md")
    print("="*70 + "\n")


if __name__ == "__main__":
    main()

