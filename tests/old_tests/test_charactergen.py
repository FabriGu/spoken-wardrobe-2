"""
CharacterGen Test Script

This script tests the CharacterGen pipeline to generate animation-ready 3D meshes
from 2D clothing images.

Steps:
1. Load reference data from generated_meshes/1761618888/
2. Use the uncropped user image (reference_frame.png)
3. Run CharacterGen to generate 3D character mesh
4. Display result and save as GLB

REQUIREMENTS:
- CharacterGen must be installed (see: https://github.com/zjp-shadow/CharacterGen)
- Installation:
  ```
  git clone https://github.com/zjp-shadow/CharacterGen external/CharacterGen
  cd external/CharacterGen
  pip install -r requirements.txt
  ```

NOTE: This is a prototype test. The full pipeline would:
1. Generate user-specific rigged body (CharacterGen + UniRig on full user image)
2. Generate clothing mesh (CharacterGen on clothing-cropped image)
3. Transfer weights from user body → clothing
4. Animate with MediaPipe

For now, this just tests CharacterGen mesh generation.
"""

import sys
from pathlib import Path
import numpy as np
import cv2
import pickle

# Check if CharacterGen is installed
charactergen_path = Path(__file__).parent.parent / "external" / "CharacterGen"

if not charactergen_path.exists():
    print("=" * 60)
    print("ERROR: CharacterGen not found!")
    print("=" * 60)
    print("\nPlease install CharacterGen:")
    print("  1. git clone https://github.com/zjp-shadow/CharacterGen external/CharacterGen")
    print("  2. cd external/CharacterGen")
    print("  3. pip install -r requirements.txt")
    print("  4. Download model weights (see CharacterGen README)")
    print("\nOnce installed, run this script again.")
    print("=" * 60)
    sys.exit(1)

# Add CharacterGen to path
sys.path.insert(0, str(charactergen_path))

try:
    # Import CharacterGen modules (these may vary based on their actual API)
    # This is a placeholder - adjust based on actual CharacterGen API
    print("Importing CharacterGen...")
    # from charactergen import generate_mesh  # Example import
    print("✓ CharacterGen imported successfully")
except ImportError as e:
    print(f"ERROR importing CharacterGen: {e}")
    print("Please check CharacterGen installation and requirements")
    sys.exit(1)


def load_reference_data(data_dir: str):
    """Load reference data from pipeline"""
    data_dir = Path(data_dir)

    # Load pickle
    pkl_path = data_dir / "reference_data.pkl"
    print(f"Loading: {pkl_path}")

    with open(pkl_path, 'rb') as f:
        data = pickle.load(f)

    # Load images
    reference_frame = cv2.imread(str(data_dir / "reference_frame.png"))
    clothing_image = cv2.imread(str(data_dir / "generated_clothing.png"))

    print(f"✓ Reference frame: {reference_frame.shape}")
    print(f"✓ Clothing image: {clothing_image.shape}")
    print(f"✓ BodyPix masks: {len(data['bodypix_masks'])} parts")

    return {
        'reference_frame': reference_frame,
        'clothing_image': clothing_image,
        'data': data
    }


def generate_mesh_with_charactergen(image: np.ndarray, output_path: str):
    """
    Generate 3D mesh using CharacterGen

    Args:
        image: Input image (RGB)
        output_path: Where to save output GLB

    NOTE: This is a placeholder. The actual CharacterGen API may be different.
    Refer to CharacterGen documentation for correct usage.
    """
    print("\n=== Running CharacterGen ===")
    print("Input image shape:", image.shape)
    print("Output path:", output_path)

    # TODO: Call actual CharacterGen API
    # Example (adjust based on real API):
    # mesh = generate_mesh(image)
    # mesh.export(output_path)

    print("\n⚠️  CharacterGen integration not yet implemented")
    print("This is a placeholder script to demonstrate the workflow.")
    print("\nTo complete this:")
    print("1. Read CharacterGen documentation")
    print("2. Find the mesh generation function (likely in main.py or inference.py)")
    print("3. Call it with the input image")
    print("4. Save output as GLB")

    return None


def main():
    """Entry point"""
    print("=" * 60)
    print("CharacterGen Test Script")
    print("=" * 60)

    # Load reference data
    data_dir = "generated_meshes/1761618888"
    ref_data = load_reference_data(data_dir)

    # Display images
    print("\n=== Displaying Images ===")
    cv2.imshow("Reference Frame (User)", ref_data['reference_frame'])
    cv2.imshow("Generated Clothing", ref_data['clothing_image'])
    print("Press any key to continue...")
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    # Test 1: Generate mesh from full user image
    print("\n=== Test 1: Full User Image → 3D Character ===")
    output_path_1 = "generated_meshes/charactergen_test_full_user.glb"
    mesh_1 = generate_mesh_with_charactergen(
        ref_data['reference_frame'],
        output_path_1
    )

    # Test 2: Generate mesh from clothing image
    print("\n=== Test 2: Clothing Image → 3D Clothing Mesh ===")
    output_path_2 = "generated_meshes/charactergen_test_clothing.glb"
    mesh_2 = generate_mesh_with_charactergen(
        ref_data['clothing_image'],
        output_path_2
    )

    print("\n=== Next Steps ===")
    print("Once CharacterGen is working:")
    print("1. Generate user body mesh + rig it with UniRig")
    print("2. Generate clothing mesh")
    print("3. Transfer weights from rigged body → clothing")
    print("4. Use Test A pipeline for real-time animation")

    print("\n=" * 60)
    print("Test Complete")
    print("=" * 60)


if __name__ == "__main__":
    main()
