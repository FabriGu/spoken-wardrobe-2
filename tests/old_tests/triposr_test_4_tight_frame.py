#!/usr/bin/env python3
"""
TripoSR Test 4: TIGHT FRAMING
==============================
mc_resolution: 256 (default)
foreground_ratio: 0.70 (MUCH more padding)
z_scale: 0.75 (flatter - good for fitted clothing)
bake_texture: True

Goal: Test if extra padding helps TripoSR understand context better.
Good for clothing that might be touching frame edges.
"""

import sys
import os
from pathlib import Path
import subprocess

# Configuration
INPUT_IMAGE = "generated_meshes/1761618888/generated_clothing.png"
OUTPUT_DIR = "generated_meshes/triposr_test_4_tight_frame"
MC_RESOLUTION = 256
FOREGROUND_RATIO = 0.70  # Much more padding
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
print("TripoSR Test 4: TIGHT FRAMING")
print("=" * 70)
print(f"Input: {INPUT_IMAGE}")
print(f"Output: {OUTPUT_DIR}")
print(f"Settings:")
print(f"  - mc_resolution: {MC_RESOLUTION} (default)")
print(f"  - foreground_ratio: {FOREGROUND_RATIO} (MUCH MORE PADDING)")
print(f"  - z_scale: 0.75 (flatter)")
# print(f"  - bake_texture: True")
print(f"  - texture_resolution: {TEXTURE_RESOLUTION}")
print("=" * 70)
print("Goal: Maximum context - clothing only takes 70% of frame")
print("Trade-off: Clothing appears smaller but has more context")
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
    print("✅ Test 4 Complete!")
    print("=" * 70)
    print(f"Output saved to: {OUTPUT_DIR}/0/")
    print(f"  - mesh.obj")
    print(f"  - texture.png")
    print("=" * 70)
    print("Expected improvements:")
    print("  ✅ Better edge reconstruction (no clipping)")
    print("  ✅ More spatial context for TripoSR")
    print("  ⚠️ Clothing will appear smaller in frame")
else:
    print("\n❌ Test 4 Failed!")
    sys.exit(1)

print("\nNote: Compare input.png with other tests to see framing difference")
print("      If your clothing was touching edges, this should help!")

