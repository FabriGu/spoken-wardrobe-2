#!/usr/bin/env python3
"""
TripoSR Test 1: DEFAULT SETTINGS
=================================
mc_resolution: 256 (official default)
foreground_ratio: 0.85 (official default)
z_scale: 1.0 (no scaling)
bake_texture: True

This is the baseline - official TripoSR defaults with no modifications.
"""

import sys
import os
from pathlib import Path
import subprocess

# Configuration
INPUT_IMAGE = "generated_meshes/1761618888/generated_clothing.png"
OUTPUT_DIR = "generated_meshes/triposr_test_1_default"
MC_RESOLUTION = 256
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
print("TripoSR Test 1: DEFAULT SETTINGS")
print("=" * 70)
print(f"Input: {INPUT_IMAGE}")
print(f"Output: {OUTPUT_DIR}")
print(f"Settings:")
print(f"  - mc_resolution: {MC_RESOLUTION}")
print(f"  - foreground_ratio: {FOREGROUND_RATIO}")
print(f"  - z_scale: 1.0 (no scaling)")
print(f"  - bake_texture: True")
print(f"  - texture_resolution: {TEXTURE_RESOLUTION}")
print("=" * 70)

# Build command
cmd = [
    sys.executable,
    str(triposr_script),
    str(input_path),
    "--output-dir", str(output_path),
    # "--no-remove-bg",  # Skip background removal
    "--foreground-ratio", str(FOREGROUND_RATIO),
    "--mc-resolution", str(MC_RESOLUTION),
    # "--bake-texture",  # Enable texture baking
    "--texture-resolution", str(TEXTURE_RESOLUTION),
    "--model-save-format", "obj"
]

print("\nRunning TripoSR...")
print(f"Command: {' '.join(cmd)}\n")

# Run TripoSR
result = subprocess.run(cmd, cwd=str(project_root))

if result.returncode == 0:
    print("\n" + "=" * 70)
    print("✅ Test 1 Complete!")
    print("=" * 70)
    print(f"Output saved to: {OUTPUT_DIR}/0/")
    print(f"  - mesh.obj")
    print(f"  - texture.png")
    print(f"  - input.png (preprocessed)")
    print("=" * 70)
else:
    print("\n❌ Test 1 Failed!")
    sys.exit(1)

# Note: z_scale=1.0 means no post-processing scaling applied
print("\nNote: This test uses z_scale=1.0 (no Z-axis scaling)")
print("      This is the 'raw' TripoSR output for comparison.")

