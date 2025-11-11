#!/bin/bash
# Fix Chumpy Installation
# Installs the correct patched version for getargs compatibility

echo "========================================"
echo "Fixing Chumpy Installation"
echo "========================================"
echo ""
echo "Current commit: 580566ea..."
echo "Required commit: 9b045ff5..."
echo ""
echo "This fixes: 'getargs() got an unexpected keyword argument' error"
echo ""

read -p "Continue? (y/n) " -n 1 -r
echo
if [[ ! $REPLY =~ ^[Yy]$ ]]
then
    echo "Cancelled."
    exit 1
fi

echo ""
echo "Uninstalling current chumpy..."
pip uninstall chumpy -y

echo ""
echo "Installing patched chumpy from git..."
pip install git+https://github.com/mattloper/chumpy@9b045ff5d6588a24a0bab52c83f032e2ba433e17

echo ""
echo "========================================"
echo "Verifying installation..."
echo "========================================"
python verify_installation.py 2>&1 | grep -A 15 "CHECKING CHUMPY"

echo ""
echo "✅ Done!"
