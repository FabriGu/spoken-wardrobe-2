# 🧪 TripoSR Settings Tests - Quick Reference

## Created: 5 Test Scripts for TripoSR Settings Comparison

All tests use your image: `generated_meshes/1761618888/generated_clothing.png`

---

## 🚀 Quick Start

### Run ALL 5 tests at once (recommended):

```bash
python tests/run_all_triposr_tests.py
```

### Or run individual tests:

```bash
python tests/triposr_test_1_default.py      # Baseline
python tests/triposr_test_2_high_res.py     # Maximum quality
python tests/triposr_test_3_balanced.py     # ⭐ Recommended
python tests/triposr_test_4_tight_frame.py  # More padding
python tests/triposr_test_5_performance.py  # Fastest
```

---

## 📊 Test Configurations Summary

| Test               | mc_res | fg_ratio | z_scale | Purpose      |
| ------------------ | ------ | -------- | ------- | ------------ |
| **1. Default**     | 256    | 0.85     | 1.0     | Baseline     |
| **2. High Res**    | 320    | 0.85     | 0.85    | Fix holes    |
| **3. Balanced** ⭐ | 196    | 0.75     | 0.80    | Best overall |
| **4. Tight Frame** | 256    | 0.70     | 0.75    | Fix edges    |
| **5. Performance** | 160    | 0.80     | 0.80    | Fastest      |

---

## 🎯 Which Test Should You Focus On?

### Your main issues: Holes + Flat geometry + Inconsistency

**Start with Test 2 (High Resolution)** - `mc_resolution=320`

- Your current setting (110) is **too low** ← Main cause of holes!
- Test 2 uses 320 (higher than default 256)
- Should fix most holes and capture thin details

**Then try Test 3 (Balanced)** - `mc_resolution=196, foreground_ratio=0.75`

- Still better than your current 110
- Faster than Test 2
- More padding (0.75 vs 0.85) for better context
- **This is my top recommendation for your prototype**

**Compare with Test 1 (Default)** - Official baseline

- See how your chosen settings compare to TripoSR defaults

---

## 📂 Output Locations

After running tests, check:

```
generated_meshes/
├── triposr_test_1_default/0/
│   ├── mesh.obj
│   ├── texture.png
│   └── input.png
├── triposr_test_2_high_res/0/
├── triposr_test_3_balanced/0/
├── triposr_test_4_tight_frame/0/
└── triposr_test_5_performance/0/
```

---

## 🔍 What to Look For

1. **Holes**: Open meshes in Blender/viewer, check for gaps

   - Test 2 should have fewest holes

2. **Thin details**: Check sleeves, edges, collars

   - Test 2 should capture best

3. **Edge quality**: Check if clothing edges are clean

   - Test 4 should have cleanest edges

4. **Processing time**: Check terminal output

   - Test 5 should be fastest
   - Test 3 should be good balance

5. **Frame padding**: Check `input.png` in each output
   - Test 4 has most padding (70%)
   - Test 1 has least padding (85%)

---

## ⚡ Expected Processing Times (rough estimates)

- Test 1 (Default): ~30-45s
- Test 2 (High Res): ~60-90s (slowest)
- Test 3 (Balanced): ~20-35s ⭐
- Test 4 (Tight Frame): ~30-45s
- Test 5 (Performance): ~15-25s (fastest)

**Total for all 5**: ~3-5 minutes

---

## 🎯 My Top Recommendation

Based on your issues (holes, flat geometry) and need for balance:

**Use Test 3 (Balanced) settings:**

```python
mc_resolution = 196        # Much better than your 110
foreground_ratio = 0.75   # More padding than default
z_scale = 0.80            # Your current (reasonable)
```

**Why?**

- ✅ Fixes holes (196 >> 110)
- ✅ Better framing (more padding)
- ✅ Fast enough for real-time prototype
- ✅ Good quality/performance balance

If you still see holes, bump up to Test 2's `mc_resolution=320`.

---

## 📚 Full Documentation

See `docs/TRIPOSR_TEST_GUIDE.md` for:

- Detailed test explanations
- Comparison guide
- How to apply chosen settings to your pipeline
- Troubleshooting
- Performance vs quality matrix

---

## ✅ Next Steps

1. Run the tests:

   ```bash
   python tests/run_all_triposr_tests.py
   ```

2. Compare the meshes visually

3. Note which test gave the best results

4. Update your pipeline (`tests/triposr_pipeline.py`) with those settings

5. Re-run your full pipeline:
   ```bash
   python tests/create_consistent_pipeline_v2.py
   ```

---

## 🐛 Quick Troubleshooting

**"TripoSR not found"**:

```bash
git clone https://github.com/VAST-AI-Research/TripoSR.git external/TripoSR
```

**"Input image not found"**:

```bash
ls -la generated_meshes/1761618888/generated_clothing.png
# If missing, update INPUT_IMAGE in test scripts
```

**Out of memory**:

- Try Test 5 first (lowest memory usage)
- Or add `--device cpu` flag (slower but works)

---

## 📝 Notes

- All tests use `--no-remove-bg` (as requested)
- All tests use `--bake-texture` (as requested)
- z_scale is noted but not applied (TripoSR doesn't have this param)
- Apply z_scale in post-processing with your pipeline

---

**Good luck! Compare the results and let me know which settings work best!** 🎉
