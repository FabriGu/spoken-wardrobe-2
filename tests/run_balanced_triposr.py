#!/usr/bin/env python3
"""
Run All TripoSR Tests
=====================

Executes all 5 TripoSR test configurations and generates a comparison report.

Usage:
    python tests/run_all_triposr_tests.py

Tests:
    1. Default Settings (baseline)
    2. High Resolution (maximum quality)
    3. Balanced Settings (recommended)
    4. Tight Framing (more padding)
    5. Performance Optimized (fastest)
"""

import sys
import subprocess
import time
from pathlib import Path
from datetime import datetime

# Get project root
project_root = Path(__file__).parent.parent

# Test scripts
tests = [
   
    {
        "name": "Test 3: Balanced Settings",
        "script": "tests/triposr_test_3_balanced.py",
        "description": "Recommended - Good quality/speed trade-off",
        "settings": "mc_res=196, fg_ratio=0.75, z_scale=0.80"
    },
]

def print_header():
    print("\n" + "=" * 80)
    print(" " * 20 + "TripoSR SETTINGS COMPARISON TEST")
    print("=" * 80)
    print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Input Image: generated_meshes/1761618888/generated_clothing.png")
    print(f"Total Tests: {len(tests)}")
    print("=" * 80)

def print_test_list():
    print("\nTests to run:")
    for i, test in enumerate(tests, 1):
        print(f"\n{i}. {test['name']}")
        print(f"   {test['description']}")
        print(f"   Settings: {test['settings']}")
    print("\n" + "=" * 80)

def run_test(test_num, test_info):
    """Run a single test script."""
    print(f"\n{'=' * 80}")
    print(f"RUNNING: {test_info['name']} ({test_num}/{len(tests)})")
    print(f"{'=' * 80}")
    
    script_path = project_root / test_info['script']
    
    if not script_path.exists():
        print(f"❌ Script not found: {script_path}")
        return False, 0
    
    start_time = time.time()
    
    # Run the test script
    result = subprocess.run(
        [sys.executable, str(script_path)],
        cwd=str(project_root)
    )
    
    elapsed_time = time.time() - start_time
    
    if result.returncode == 0:
        print(f"\n✅ {test_info['name']} completed in {elapsed_time:.2f}s")
        return True, elapsed_time
    else:
        print(f"\n❌ {test_info['name']} FAILED!")
        return False, elapsed_time

def print_summary(results):
    """Print summary of all test results."""
    print("\n" + "=" * 80)
    print(" " * 30 + "TEST SUMMARY")
    print("=" * 80)
    
    successful = sum(1 for r in results if r['success'])
    total_time = sum(r['time'] for r in results)
    
    print(f"\nCompleted: {successful}/{len(results)} tests successful")
    print(f"Total Time: {total_time:.2f}s")
    print(f"Average Time: {total_time/len(results):.2f}s per test")
    
    print("\nIndividual Results:")
    print("-" * 80)
    print(f"{'Test':<40} {'Status':<10} {'Time':<10}")
    print("-" * 80)
    
    for result in results:
        status = "✅ PASS" if result['success'] else "❌ FAIL"
        print(f"{result['name']:<40} {status:<10} {result['time']:>6.2f}s")
    
    print("-" * 80)
    
    # Print output locations
    print("\nOutput Locations:")
    print("-" * 80)
    for i, result in enumerate(results, 1):
        if result['success']:
            output_dir = f"generated_meshes/triposr_test_{i}_*/0/"
            print(f"{i}. {output_dir}")
            print(f"   └─ mesh.obj, texture.png, input.png")
    
    print("\n" + "=" * 80)
    print("COMPARISON GUIDE:")
    print("=" * 80)
    print("\nWhat to look for in each test:")
    print("\n1. Default (mc_res=256):")
    print("   - Check if holes are present")
    print("   - Compare mesh smoothness")
    print("   - This is the baseline")
    
    print("\n2. High Resolution (mc_res=320):")
    print("   - Should have FEWER holes than default")
    print("   - Better capture of thin details (sleeves, edges)")
    print("   - Smoother surfaces overall")
    
    print("\n3. Balanced (mc_res=196, fg_ratio=0.75):")
    print("   - Better than your current mc_res=110")
    print("   - More padding around clothing")
    print("   - Best quality/speed trade-off")
    
    print("\n4. Tight Framing (fg_ratio=0.70):")
    print("   - Clothing appears smaller in frame")
    print("   - More context/padding around edges")
    print("   - Compare edge quality with others")
    
    print("\n5. Performance (mc_res=160):")
    print("   - Fastest generation")
    print("   - Some detail loss acceptable")
    print("   - Good for rapid iteration")
    
    print("\n" + "=" * 80)
    print("RECOMMENDATION:")
    print("=" * 80)
    print("\nBased on your needs (balance quality + performance):")
    print("  • If holes are the main issue: Use Test 2 (High Resolution)")
    print("  • For best overall balance: Use Test 3 (Balanced)")
    print("  • If edges are problematic: Use Test 4 (Tight Framing)")
    print("  • For rapid prototyping: Use Test 5 (Performance)")
    print("\nCompare the meshes in Blender or your 3D viewer to choose!")
    print("=" * 80 + "\n")

def main():
    print_header()
    print_test_list()
    
    # Ask for confirmation
    response = input("\nRun all 5 tests? This will take several minutes. (y/n): ")
    if response.lower() != 'y':
        print("Aborted.")
        return
    
    # Run all tests
    results = []
    for i, test in enumerate(tests, 1):
        success, elapsed = run_test(i, test)
        results.append({
            'name': test['name'],
            'success': success,
            'time': elapsed
        })
        
        # Brief pause between tests
        if i < len(tests):
            print(f"\nStarting next test in 3 seconds...")
            time.sleep(3)
    
    # Print summary
    print_summary(results)
    
    # Final timestamp
    print(f"Finished: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")

if __name__ == "__main__":
    main()

