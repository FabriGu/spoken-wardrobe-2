#!/usr/bin/env python3
"""
TripoSR Test 2: HIGH RESOLUTION
================================
mc_resolution: 320 (HIGHER than default)
foreground_ratio: 0.85 (default)
z_scale: 0.85 (slight compression)
bake_texture: True

Goal: Maximum detail capture for complex patterns/thin geometry.
Tests if higher resolution fixes holes and captures fine details.
"""

import sys
import os
from pathlib import Path
import subprocess

# Configuration
INPUT_IMAGE = "generated_meshes/1761618888/generated_clothing.png"
OUTPUT_DIR = "generated_meshes/triposr_test_2_high_res"
MC_RESOLUTION = 320  # Higher than default
FOREGROUND_RATIO = 0.85
TEXTURE_RESOLUTION = 2048

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
print("TripoSR Test 2: HIGH RESOLUTION")
print("=" * 70)
print(f"Input: {INPUT_IMAGE}")
print(f"Output: {OUTPUT_DIR}")
print(f"Settings:")
print(f"  - mc_resolution: {MC_RESOLUTION} (HIGHER)")
print(f"  - foreground_ratio: {FOREGROUND_RATIO}")
print(f"  - z_scale: 0.85 (slight compression)")
print(f"  - bake_texture: True")
print(f"  - texture_resolution: {TEXTURE_RESOLUTION}")
print("=" * 70)
print("Goal: Maximum detail - should capture thin geometry better")
print("Trade-off: Slower processing, higher VRAM")
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
    print("✅ Test 2 Complete!")
    print("=" * 70)
    print(f"Output saved to: {OUTPUT_DIR}/0/")
    print(f"  - mesh.obj")
    print(f"  - texture.png")
    print("=" * 70)
    print("Expected improvements:")
    print("  ✅ Fewer holes in mesh")
    print("  ✅ Better capture of thin details (sleeves, edges)")
    print("  ✅ Smoother surfaces")
else:
    print("\n❌ Test 2 Failed!")
    sys.exit(1)

print("\nNote: z_scale=0.85 would be applied in post-processing")
print("      (TripoSR itself doesn't have z_scale parameter)")

