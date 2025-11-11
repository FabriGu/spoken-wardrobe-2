# Chumpy Installation Fix

## Issue

You need a **patched version** of chumpy to fix the `getargs()` error.

**Currently installed:** Commit `580566ea...` ❌
**Required:** Commit `9b045ff5...` ✅

## Quick Fix

### Option 1: Run the fix script

**Mac/Linux:**
```bash
./fix_chumpy.sh
```

**Windows:**
```cmd
fix_chumpy.bat
```

### Option 2: Manual installation

```bash
pip uninstall chumpy -y
pip install git+https://github.com/mattloper/chumpy@9b045ff5d6588a24a0bab52c83f032e2ba433e17
```

## Verify Installation

```bash
python verify_installation.py
```

Look for this in the output:
```
CHECKING CHUMPY (CRITICAL)
✅ Chumpy imported successfully
✅ Installed from GitHub (patched version)
✅ Correct commit: 9b045ff...
```

## Why This Patch Is Needed

The PyPI version of chumpy (0.70) has a bug where `getargs()` receives unexpected keyword arguments when used with modern NumPy versions. This specific git commit fixes that issue.

## For Fresh Installations

This is now **automatically handled** in `requirements-core.txt`, so new installations will get the correct version by default when running:

```bash
pip install -r requirements-core.txt
```

## Technical Details

**Repository:** https://github.com/mattloper/chumpy
**Patched Commit:** `9b045ff5d6588a24a0bab52c83f032e2ba433e17`
**What it fixes:** getargs() compatibility with NumPy 1.20+

---

**Last Updated:** 2025-01-06
