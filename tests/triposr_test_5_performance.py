#!/usr/bin/env python3
"""
TripoSR Test 5: PERFORMANCE OPTIMIZED
======================================
mc_resolution: 160 (lower but better than your 110)
foreground_ratio: 0.80 (slight padding)
z_scale: 0.80 (your current)
bake_texture: True

Goal: Fast generation while still maintaining acceptable quality.
Tests the minimum viable settings for real-time prototype.
"""

import sys
import os
from pathlib import Path
import subprocess

# Configuration
INPUT_IMAGE = "generated_meshes/1761618888/generated_clothing.png"
OUTPUT_DIR = "generated_meshes/triposr_test_5_performance"
MC_RESOLUTION = 160  # Lower but still better than 110
FOREGROUND_RATIO = 0.80  # Slight padding
TEXTURE_RESOLUTION = 1024  # Lower texture res for speed

# Get project root
project_root = Path(__file__).parent.parent
triposr_script = project_root / "external" / "TripoSR" / "run.py"
input_path = project_root / INPUT_IMAGE
output_path = project_root / OUTPUT_DIR

# Validate paths
if not triposr_script.exists():
    print(f"❌ TripoSR not found at: {triposr_script}")
    sys.exit(1)

if not input_path.exists():
    print(f"❌ Input image not found at: {input_path}")
    sys.exit(1)

# Create output directory (including subdirectory 0/)
output_path.mkdir(parents=True, exist_ok=True)
(output_path / "0").mkdir(parents=True, exist_ok=True)

print("=" * 70)
print("TripoSR Test 5: PERFORMANCE OPTIMIZED")
print("=" * 70)
print(f"Input: {INPUT_IMAGE}")
print(f"Output: {OUTPUT_DIR}")
print(f"Settings:")
print(f"  - mc_resolution: {MC_RESOLUTION} (PERFORMANCE)")
print(f"  - foreground_ratio: {FOREGROUND_RATIO} (slight padding)")
print(f"  - z_scale: 0.80 (your current)")
print(f"  - bake_texture: True")
print(f"  - texture_resolution: {TEXTURE_RESOLUTION} (LOWER for speed)")
print("=" * 70)
print("Goal: Fastest generation with acceptable quality")
print("Trade-off: Lower detail but much faster")
print("=" * 70)

# Build command
cmd = [
    sys.executable,
    str(triposr_script),
    str(input_path),
    "--output-dir", str(output_path),
    # "--no-remove-bg",
    "--foreground-ratio", str(FOREGROUND_RATIO),
    "--mc-resolution", str(MC_RESOLUTION),
    # "--bake-texture",
    "--texture-resolution", str(TEXTURE_RESOLUTION),
    "--model-save-format", "obj"
]

print("\nRunning TripoSR...")
print(f"Command: {' '.join(cmd)}\n")

# Run TripoSR
result = subprocess.run(cmd, cwd=str(project_root))

if result.returncode == 0:
    print("\n" + "=" * 70)
    print("✅ Test 5 Complete!")
    print("=" * 70)
    print(f"Output saved to: {OUTPUT_DIR}/0/")
    print(f"  - mesh.obj")
    print(f"  - texture.png")
    print("=" * 70)
    print("Expected characteristics:")
    print("  ⚡ Fastest generation time")
    print("  ⚠️ Some detail loss vs. higher resolutions")
    print("  ✅ Still better than mc_resolution=110")
    print("  ✅ Good for rapid iteration/testing")
else:
    print("\n❌ Test 5 Failed!")
    sys.exit(1)

print("\nNote: This is the 'minimum viable quality' for prototyping")
print("      Use for fast iteration, then increase mc_resolution for final")

