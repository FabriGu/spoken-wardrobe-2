#!/bin/bash
# Quick Verification Script
# Runs all verification tools in sequence

echo "================================="
echo "CAGE DEFORMATION VERIFICATION"
echo "================================="
echo ""

# Check if in correct directory
if [ ! -f "tests/test_integration.py" ]; then
    echo "Error: Run this from project root!"
    echo "Usage: bash 251025_data_verification/quick_verify.sh"
    exit 1
fi

# Activate venv if exists
if [ -d "venv" ]; then
    echo "Activating virtual environment..."
    source venv/bin/activate
fi

echo ""
echo "=== Step 1: Verifying Cage Structure ==="
echo ""
python 251025_data_verification/verify_cage_structure.py

echo ""
echo ""
echo "=== Step 2: Verifying MVC Weights ==="
echo ""
python 251025_data_verification/verify_mvc_weights.py

echo ""
echo ""
echo "=== Step 3: Real-Time Verification ==="
echo ""
echo "Starting real-time verification..."
echo "Please open 251025_data_verification/verification_viewer.html in your browser"
echo ""
echo "Press Ctrl+C to stop when done"
echo ""

python 251025_data_verification/verify_deformation.py

echo ""
echo "================================="
echo "VERIFICATION COMPLETE"
echo "================================="
echo ""
echo "Check the output above and see:"
echo "  → docs/251025_steps_forward.md"
echo ""

