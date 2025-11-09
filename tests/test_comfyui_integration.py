#!/usr/bin/env python3
"""
Test ComfyUI Integration

Standalone test script to verify ComfyUI remote generation works
WITHOUT modifying existing ai_generation.py code.

This script:
1. Tests connection to school's ComfyUI server
2. Uploads test images
3. Generates clothing using remote GPU
4. Compares results with local generation

Usage:
    python test_comfyui_integration.py
"""

import cv2
import numpy as np
from PIL import Image
import time
import os
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.modules.comfyui_client import ComfyUIClient


def test_connection_only():
    """Test 1: Just check if server is reachable."""
    print("\n" + "="*70)
    print("TEST 1: Connection Test")
    print("="*70)

    client = ComfyUIClient("http://itp-ml.itp.tsoa.nyu.edu:9199")

    if client.test_connection():
        print("\n✅ SUCCESS: Can connect to ComfyUI server!")
        print("\nNext steps:")
        print("  1. You can access the web interface at:")
        print("     http://itp-ml.itp.tsoa.nyu.edu:9199")
        print("  2. Create an inpainting workflow there")
        print("  3. Export as 'Save (API Format)'")
        print("  4. Save to workflows/sdxl_inpainting_api.json")
        return True
    else:
        print("\n❌ FAILED: Cannot connect to server")
        print("\nTroubleshooting:")
        print("  • Are you on NYU network or VPN?")
        print("  • Is the server running?")
        print("  • Try opening http://itp-ml.itp.tsoa.nyu.edu:9199 in browser")
        return False


def test_image_upload():
    """Test 2: Upload images to server."""
    print("\n" + "="*70)
    print("TEST 2: Image Upload Test")
    print("="*70)

    # Create test images
    print("\n📷 Creating test images...")

    # Simple test image (colored square)
    test_image = np.zeros((512, 512, 3), dtype=np.uint8)
    test_image[100:400, 100:400] = [255, 100, 50]  # Orange square

    # Simple test mask (white circle on black)
    test_mask = np.zeros((512, 512), dtype=np.uint8)
    cv2.circle(test_mask, (256, 256), 150, 255, -1)

    print(f"  Image shape: {test_image.shape}")
    print(f"  Mask shape: {test_mask.shape}")

    # Upload
    client = ComfyUIClient("http://itp-ml.itp.tsoa.nyu.edu:9199")

    print("\n📤 Uploading to server...")
    img_name = client.upload_image(test_image, "debug_image_to_sd.png")
    mask_name = client.upload_image(test_mask, "debug_mask_to_sd.png")

    if img_name and mask_name:
        print("\n✅ SUCCESS: Images uploaded!")
        print(f"  Image: {img_name}")
        print(f"  Mask: {mask_name}")
        return True
    else:
        print("\n❌ FAILED: Upload failed")
        return False


def test_workflow_preparation():
    """Test 3: Load and prepare workflow."""
    print("\n" + "="*70)
    print("TEST 3: Workflow Preparation Test")
    print("="*70)

    workflow_path = "workflows/sdxl_inpainting_api.json"

    if not os.path.exists(workflow_path):
        print(f"\n⚠️  Workflow file not found: {workflow_path}")
        print("\nThis is expected - you need to:")
        print("  1. Open ComfyUI web interface")
        print("  2. Create your inpainting workflow")
        print("  3. Export it as 'Save (API Format)'")
        print("  4. Save to workflows/sdxl_inpainting_api.json")
        print("\nSkipping this test for now.")
        return False

    client = ComfyUIClient("http://itp-ml.itp.tsoa.nyu.edu:9199")

    print("\n📋 Loading workflow template...")
    workflow = client.load_workflow_template(workflow_path)

    if not workflow:
        print("\n❌ FAILED: Could not load workflow")
        return False

    print(f"  Loaded {len(workflow)} nodes")

    print("\n⚙️  Preparing workflow with test parameters...")
    prepared = client.prepare_workflow(
        workflow_template=workflow,
        prompt="fashion clothing made of flames, fiery red, red detailed fabric texture, high quality, actual fire, flames, hot",
        negative_prompt="low quality, blurry, distorted, deformed, ugly, naked, naked body, naked person, naked man, naked woman, naked child, naked baby, slutty, vulgar, obscene, explicit, explicit content, explicit image, explicit picture, explicit photo, explicit artwork, explicit art, explicit drawing, explicit painting, explicit sketch, explicit line art, explicit digital art, explicit digital painting, explicit digital sketch, explicit digital line art, explicit digital artwork, explicit digital painting, explicit digital sketch, explicit digital line art, explicit digital artwork",
        image_filename="test.png",
        mask_filename="mask.png",
        seed=101,
        steps=30,
        cfg=7.5
    )

    if prepared:
        print("\n✅ SUCCESS: Workflow prepared!")
        return True
    else:
        print("\n❌ FAILED: Workflow preparation failed")
        return False


def test_full_generation():
    """Test 4: Complete generation pipeline."""
    print("\n" + "="*70)
    print("TEST 4: Full Generation Test")
    print("="*70)

    # Check for test images from existing pipeline
    if not os.path.exists('debug_image_to_sd.png') or not os.path.exists('debug_mask_to_sd.png'):
        print("\n⚠️  Test images not found!")
        print("\nYou need to run your existing Stable Diffusion pipeline first")
        print("to generate debug_image_to_sd.png and debug_mask_to_sd.png")
        print("\nSkipping full generation test.")
        return False

    # Check for workflow
    workflow_path = "workflows/sdxl_inpainting_api.json"
    if not os.path.exists(workflow_path):
        print(f"\n⚠️  Workflow not found: {workflow_path}")
        print("\nYou need to export your workflow from ComfyUI first.")
        print("Skipping full generation test.")
        return False

    # Check if it's actually an API format workflow
    import json
    with open(workflow_path, 'r') as f:
        workflow_check = json.load(f)

    # API format has nodes as top-level keys like "1", "2", "3"
    # UI format has a "nodes" array
    if "nodes" in workflow_check or "_meta" in workflow_check:
        print(f"\n⚠️  Workflow is in UI format, not API format!")
        print("\nThis is a TEMPLATE file. You need to:")
        print("  1. Open ComfyUI web interface tomorrow at school")
        print("  2. Create an inpainting workflow")
        print("  3. Export as 'Save (API Format)' (NOT regular save)")
        print("  4. Replace this template file")
        print("\nSkipping generation test for now.")
        return False

    # Load test images
    print("\n📷 Loading test images...")
    test_image = cv2.imread('debug_image_to_sd.png')
    test_image = cv2.cvtColor(test_image, cv2.COLOR_BGR2RGB)
    test_mask = cv2.imread('debug_mask_to_sd.png', cv2.IMREAD_GRAYSCALE)

    print(f"  Image shape: {test_image.shape}")
    print(f"  Mask shape: {test_mask.shape}")

    # Generate
    client = ComfyUIClient("http://itp-ml.itp.tsoa.nyu.edu:9199", timeout=300)

    print("\n🚀 Starting generation...")
    start_time = time.time()

    result = client.generate_inpainting(
        image=test_image,
        mask=test_mask,
        prompt="fashion clothing made of flames, fiery red, red detailed fabric texture, high quality, actual fire, flames, hot",
        negative_prompt="low quality, blurry, distorted, deformed, ugly, naked, naked body, naked person, naked man, naked woman, naked child, naked baby, slutty, vulgar, obscene, explicit, explicit content, explicit image, explicit picture, explicit photo, explicit artwork, explicit art, explicit drawing, explicit painting, explicit sketch, explicit line art, explicit digital art, explicit digital painting, explicit digital sketch, explicit digital line art, explicit digital artwork, explicit digital painting, explicit digital sketch, explicit digital line art, explicit digital artwork",
        workflow_path=workflow_path,
        seed=101,
        steps=30,
        cfg=7.5
    )

    elapsed = time.time() - start_time

    if result:
        # Save result
        output_path = "comfyui_test_result.png"
        result.save(output_path)

        print(f"\n✅ SUCCESS: Generation complete in {elapsed:.1f}s!")
        print(f"  Result saved to: {output_path}")
        print(f"  Size: {result.size}")
        print(f"  Mode: {result.mode}")

        # Compare with local generation time
        print("\n📊 Performance Comparison:")
        print(f"  ComfyUI (remote): {elapsed:.1f}s")
        print(f"  Local SD (Mac MPS): ~15-30s")
        print(f"  Speedup: {15/elapsed:.1f}x - {30/elapsed:.1f}x faster")

        return True
    else:
        print(f"\n❌ FAILED: Generation failed after {elapsed:.1f}s")
        return False


def test_speed_comparison():
    """Test 5: Speed comparison with multiple generations."""
    print("\n" + "="*70)
    print("TEST 5: Speed Comparison Test")
    print("="*70)

    print("\nThis test would generate 5 images with different prompts")
    print("and compare average speed between local and remote.")
    print("\n⚠️  Skipping for now - run test_full_generation() first")
    return False


def main():
    """Run all tests."""
    print("="*70)
    print("ComfyUI Integration Test Suite")
    print("="*70)
    print("\nThis will test the ComfyUI integration WITHOUT modifying")
    print("your existing ai_generation.py code.")
    print("\nThe tests are progressive - each one builds on the previous.")

    results = {}

    # Test 1: Connection
    results['connection'] = test_connection_only()

    if not results['connection']:
        print("\n" + "="*70)
        print("⚠️  Cannot proceed - server not reachable")
        print("="*70)
        print("\nMake sure you're on NYU network or VPN, then try again.")
        return

    # Test 2: Upload
    results['upload'] = test_image_upload()

    # Test 3: Workflow
    results['workflow'] = test_workflow_preparation()

    # Test 4: Full generation (only if workflow exists)
    if results['workflow']:
        results['generation'] = test_full_generation()
    else:
        results['generation'] = False
        print("\n⚠️  Skipping generation test (no workflow)")

    # Summary
    print("\n" + "="*70)
    print("TEST SUMMARY")
    print("="*70)

    for test_name, passed in results.items():
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"  {status}  {test_name.title()}")

    print("\n" + "="*70)

    if all(results.values()):
        print("🎉 ALL TESTS PASSED!")
        print("\nNext steps:")
        print("  1. ComfyUI integration is working")
        print("  2. You can now modify ai_generation.py to use remote mode")
        print("  3. Or use comfyui_client.py directly in your pipeline")
    else:
        print("⚠️  Some tests failed")
        print("\nNext steps:")
        if not results['workflow']:
            print("  1. Create workflow in ComfyUI web interface")
            print("  2. Export as 'Save (API Format)'")
            print("  3. Save to workflows/sdxl_inpainting_api.json")
            print("  4. Re-run this test")
        if not results['generation']:
            print("  1. Make sure workflow is correct")
            print("  2. Check server logs for errors")
            print("  3. Try generating manually in ComfyUI web interface first")


if __name__ == "__main__":
    main()
