#!/usr/bin/env python3
"""
Installation Verification Script
=================================

Verifies that all dependencies are correctly installed with proper versions.

Usage:
    python verify_installation.py
"""

import sys
import subprocess


def check_import(module_name, display_name=None):
    """Try to import a module and return success status"""
    if display_name is None:
        display_name = module_name

    try:
        mod = __import__(module_name)
        version = getattr(mod, '__version__', 'unknown')
        print(f"✅ {display_name}: {version}")
        return True, version
    except ImportError as e:
        print(f"❌ {display_name}: NOT INSTALLED ({e})")
        return False, None


def check_chumpy():
    """Check if chumpy is the correct patched version"""
    print("\n" + "="*70)
    print("CHECKING CHUMPY (CRITICAL)")
    print("="*70)

    try:
        import chumpy
        print(f"✅ Chumpy imported successfully")
        print(f"   Version: {chumpy.__version__}")
        print(f"   Location: {chumpy.__file__}")

        # Check if installed from git
        try:
            result = subprocess.run(
                [sys.executable, '-m', 'pip', 'show', 'chumpy'],
                capture_output=True,
                text=True
            )

            if 'github.com/mattloper/chumpy' in result.stdout:
                print("✅ Installed from GitHub (patched version)")

                # Try to check commit
                import json
                import os
                import pathlib

                chumpy_path = pathlib.Path(chumpy.__file__).parent.parent
                direct_url_file = chumpy_path / 'chumpy-0.71.dist-info' / 'direct_url.json'

                if direct_url_file.exists():
                    with open(direct_url_file, 'r') as f:
                        data = json.load(f)
                        commit = data.get('vcs_info', {}).get('commit_id', 'unknown')
                        expected = '9b045ff5d6588a24a0bab52c83f032e2ba433e17'

                        if commit.startswith(expected[:8]):  # Check first 8 chars
                            print(f"✅ Correct commit: {commit[:8]}...")
                        else:
                            print(f"⚠️  Warning: Different commit installed")
                            print(f"   Expected: {expected}")
                            print(f"   Got:      {commit}")
                            print(f"\n   To fix:")
                            print(f"   pip uninstall chumpy -y")
                            print(f"   pip install git+https://github.com/mattloper/chumpy@{expected}")
            else:
                print("⚠️  WARNING: Installed from PyPI (not patched version)")
                print("   This may cause 'getargs()' errors!")
                print("\n   To fix:")
                print("   pip uninstall chumpy -y")
                print("   pip install git+https://github.com/mattloper/chumpy@9b045ff5d6588a24a0bab52c83f032e2ba433e17")

        except Exception as e:
            print(f"⚠️  Could not verify git installation: {e}")

        return True

    except ImportError as e:
        print(f"❌ Chumpy NOT installed: {e}")
        print("\n   To install:")
        print("   pip install git+https://github.com/mattloper/chumpy@9b045ff5d6588a24a0bab52c83f032e2ba433e17")
        return False


def check_tfjs_patch():
    """Check if tfjs-graph-converter patch is applied"""
    print("\n" + "="*70)
    print("CHECKING TFJS-GRAPH-CONVERTER PATCH (CRITICAL)")
    print("="*70)

    try:
        from tfjs_graph_converter import util
        print("✅ tfjs-graph-converter imported successfully")
        print("✅ NumPy compatibility patch is applied")
        return True
    except AttributeError as e:
        if 'bool' in str(e):
            print("❌ NumPy compatibility patch NOT applied!")
            print("   Error: " + str(e))
            print("\n   To fix (Mac):")
            print("   sed -i '' 's/np\\.bool$/np.bool_/g' venv/lib/python3.11/site-packages/tfjs_graph_converter/util.py")
            print("\n   To fix (PC):")
            print("   powershell -Command \"(Get-Content venv\\Lib\\site-packages\\tfjs_graph_converter\\util.py) -replace 'np\\.bool$', 'np.bool_' | Set-Content venv\\Lib\\site-packages\\tfjs_graph_converter\\util.py\"")
            return False
        else:
            raise
    except ImportError as e:
        print(f"⚠️  tfjs-graph-converter not installed: {e}")
        return False


def check_gpu():
    """Check GPU availability"""
    print("\n" + "="*70)
    print("CHECKING GPU SUPPORT")
    print("="*70)

    try:
        import torch

        # Check CUDA (PC)
        if torch.cuda.is_available():
            print(f"✅ CUDA available")
            print(f"   Device: {torch.cuda.get_device_name(0)}")
            print(f"   VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
            return 'cuda'

        # Check MPS (Mac)
        elif torch.backends.mps.is_available():
            print(f"✅ MPS (Mac GPU) available")
            return 'mps'

        else:
            print(f"⚠️  No GPU detected - using CPU")
            print(f"   Performance will be slower")
            return 'cpu'

    except ImportError:
        print(f"❌ PyTorch not installed")
        return None


def main():
    print("="*70)
    print("SPOKEN WARDROBE 2 - INSTALLATION VERIFICATION")
    print("="*70)

    print("\n" + "="*70)
    print("CHECKING CORE DEPENDENCIES")
    print("="*70)

    # Core libraries
    core_libs = [
        ('numpy', 'NumPy'),
        ('cv2', 'OpenCV'),
        ('mediapipe', 'MediaPipe'),
        ('trimesh', 'Trimesh'),
        ('websockets', 'WebSockets'),
        ('scipy', 'SciPy'),
    ]

    all_core_ok = True
    for module, name in core_libs:
        ok, version = check_import(module, name)
        all_core_ok = all_core_ok and ok

    # Check NumPy version specifically
    try:
        import numpy as np
        if np.__version__ == '1.23.5':
            print(f"✅ NumPy version correct: 1.23.5")
        else:
            print(f"⚠️  NumPy version mismatch!")
            print(f"   Expected: 1.23.5")
            print(f"   Got:      {np.__version__}")
            print(f"   This may cause compatibility issues!")
    except:
        pass

    print("\n" + "="*70)
    print("CHECKING AI/ML FRAMEWORKS")
    print("="*70)

    ai_libs = [
        ('torch', 'PyTorch'),
        ('tensorflow', 'TensorFlow'),
        ('transformers', 'Transformers'),
        ('diffusers', 'Diffusers'),
    ]

    for module, name in ai_libs:
        check_import(module, name)

    # GPU check
    gpu_type = check_gpu()

    # Chumpy check
    chumpy_ok = check_chumpy()

    # tfjs patch check
    tfjs_ok = check_tfjs_patch()

    print("\n" + "="*70)
    print("CHECKING PLATFORM-SPECIFIC DEPENDENCIES")
    print("="*70)

    # Check OAK-D (Mac)
    try:
        import depthai
        print(f"✅ OAK-D (depthai): {depthai.__version__}")
    except ImportError:
        print(f"⚠️  OAK-D (depthai) not installed (Mac only)")

    # Check RealSense (PC)
    try:
        import pyrealsense2 as rs
        print(f"✅ RealSense (pyrealsense2): {rs.__version__}")
    except ImportError:
        print(f"⚠️  RealSense (pyrealsense2) not installed (PC only)")

    print("\n" + "="*70)
    print("SUMMARY")
    print("="*70)

    if all_core_ok and chumpy_ok and tfjs_ok and gpu_type:
        print("✅ All critical dependencies verified!")
        print(f"   GPU: {gpu_type.upper()}")
        print("\nYou're ready to run the project!")
    else:
        print("⚠️  Some issues found - see details above")
        print("\nRefer to SETUP_GUIDE.md for troubleshooting")

    print("="*70)


if __name__ == "__main__":
    main()
