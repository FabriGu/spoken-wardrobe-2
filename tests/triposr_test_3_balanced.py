#!/usr/bin/env python3
"""
TripoSR Test 3: BALANCED SETTINGS
==================================
mc_resolution: 196 (medium - between your 110 and default 256)
foreground_ratio: 0.75 (more padding than default)
z_scale: 0.80 (your current setting)
bake_texture: True

Goal: Balance between quality and performance.
Tests if more padding helps with edge artifacts.
"""

import sys
import os
from pathlib import Path
import subprocess

# Configuration
INPUT_IMAGE = "generated_meshes/1761618888/generated_clothing.png"
OUTPUT_DIR = "generated_meshes/triposr_test_3_balanced"
MC_RESOLUTION = 196  # Between 110 and 256
FOREGROUND_RATIO = 0.75  # More padding
TEXTURE_RESOLUTION = 2048

# Get project root
project_root = Path(__file__).parent.parent
# Use Mac-fixed version of TripoSR runner (fixes CUDA device detection on Mac)
triposr_script = project_root / "tests" / "triposr_run_mac_fixed.py"
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
print("TripoSR Test 3: BALANCED SETTINGS")
print("=" * 70)
print(f"Input: {INPUT_IMAGE}")
print(f"Output: {OUTPUT_DIR}")
print(f"Settings:")
print(f"  - mc_resolution: {MC_RESOLUTION} (BALANCED)")
print(f"  - foreground_ratio: {FOREGROUND_RATIO} (MORE PADDING)")
print(f"  - z_scale: 0.80 (your current)")
print(f"  - bake_texture: True")
print(f"  - texture_resolution: {TEXTURE_RESOLUTION}")
print("=" * 70)
print("Goal: Good quality/performance trade-off with better framing")
print("Trade-off: Moderate speed, moderate detail")
print("=" * 70)

# Build command
# Use venv python explicitly (not conda python)
venv_python = project_root / "venv" / "bin" / "python"
cmd = [
    str(venv_python),
    str(triposr_script),
    str(input_path),
    "--output-dir", 'generated_meshes/triposr_glb',
    # "--no-remove-bg",
    "--foreground-ratio", str(FOREGROUND_RATIO),
    "--mc-resolution", str(MC_RESOLUTION),
    # "--bake-texture",
    "--texture-resolution", str(TEXTURE_RESOLUTION),
    "--model-save-format", "glb"
]

print("\nRunning TripoSR...")
print(f"Command: {' '.join(cmd)}\n")

# Run TripoSR
result = subprocess.run(cmd, cwd=str(project_root))

if result.returncode == 0:
    print("\n" + "=" * 70)
    print("✅ Test 3 Complete!")
    print("=" * 70)
    print(f"Output saved to: {OUTPUT_DIR}/0/")
    print(f"  - mesh.obj")
    print(f"  - texture.png")
    print("=" * 70)
    print("Expected improvements:")
    print("  ✅ Better than mc_resolution=110 (fewer holes)")
    print("  ✅ Faster than mc_resolution=256 or 320")
    print("  ✅ More context around clothing (foreground_ratio=0.75)")
    print("  ✅ Good balance for real-time prototype")
else:
    print("\n❌ Test 3 Failed!")
    sys.exit(1)

print("\nNote: This is the recommended 'sweet spot' for your use case")
print("      foreground_ratio=0.75 gives clothing more breathing room")

