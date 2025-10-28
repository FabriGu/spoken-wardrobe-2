# BodyPix Official Part Names - CRITICAL BUG FOUND! 🚨

## Official BodyPix Part Names

From `/venv/lib/python3.11/site-packages/tf_bodypix/bodypix_js_utils/part_channels.py`:

```python
PART_CHANNELS = [
    'left_face',                    # 0
    'right_face',                   # 1
    'left_upper_arm_front',         # 2
    'left_upper_arm_back',          # 3
    'right_upper_arm_front',        # 4
    'right_upper_arm_back',         # 5
    'left_lower_arm_front',         # 6
    'left_lower_arm_back',          # 7
    'right_lower_arm_front',        # 8
    'right_lower_arm_back',         # 9
    'left_hand',                    # 10
    'right_hand',                   # 11
    'torso_front',                  # 12
    'torso_back',                   # 13
    'left_upper_leg_front',         # 14
    'left_upper_leg_back',          # 15
    'right_upper_leg_front',        # 16
    'right_upper_leg_back',         # 17
    'left_lower_leg_front',         # 18
    'left_lower_leg_back',          # 19
    'right_lower_leg_front',        # 20
    'right_lower_leg_back',         # 21
    'left_feet',                    # 22  ← NOTE: PLURAL "feet"
    'right_feet'                    # 23  ← NOTE: PLURAL "feet"
]
```

**Total: 24 body parts (indices 0-23)**

---

## 🚨 BUG FOUND IN YOUR CODE!

### What You're Using (WRONG):

```python
'left_foot', 'right_foot'  # ❌ INCORRECT - THESE DON'T EXIST!
```

### What You Should Use (CORRECT):

```python
'left_feet', 'right_feet'  # ✅ CORRECT - PLURAL!
```

---

## Where The Bug Exists:

### ❌ File: `tests/create_consistent_pipeline_v2.py` (Line 236)

```python
'left_foot', 'right_foot'  # WRONG!
```

### ❌ File: `tests/keypoint_mapper_v2.py` (Lines 222-223)

```python
'left_foot': 'left_lower_leg',   # WRONG!
'right_foot': 'right_lower_leg',  # WRONG!
```

### ❌ File: `tests/enhanced_cage_utils_v2.py` (Lines 46-47, 65-66, 297-298)

```python
'left_foot': ['left_foot'],      # WRONG!
'right_foot': ['right_foot'],    # WRONG!
'left_foot': 0.45,               # WRONG!
'right_foot': 0.45,              # WRONG!
'left_foot': ['left_ankle'],     # WRONG!
'right_foot': ['right_ankle'],   # WRONG!
```

### ❌ File: `src/modules/body_tracking.py` (Lines 32-33, 57, 82, 91)

```python
"LEFT_FOOT": 22,    # WRONG!
"RIGHT_FOOT": 23    # WRONG!
'left_foot', 'right_foot'  # WRONG!
```

---

## Why This Matters:

When you call:

```python
part_mask = result.get_part_mask(mask, part_names=['left_foot'])
```

BodyPix looks up `'left_foot'` in `PART_CHANNEL_INDEX_BY_NAME` dictionary, which is:

```python
{
    'left_face': 0,
    'right_face': 1,
    ...
    'left_feet': 22,  # ← Only 'left_feet' exists!
    'right_feet': 23
}
```

**Result**: `'left_foot'` is **NOT FOUND**, so it returns an **empty mask** (all zeros)!

This is why your foot/leg segmentation might be failing!

---

## All 24 Official Part Names (Copy-Paste Ready):

### Face (2 parts):

```python
'left_face'
'right_face'
```

### Arms (8 parts):

```python
'left_upper_arm_front'
'left_upper_arm_back'
'right_upper_arm_front'
'right_upper_arm_back'
'left_lower_arm_front'
'left_lower_arm_back'
'right_lower_arm_front'
'right_lower_arm_back'
```

### Hands (2 parts):

```python
'left_hand'
'right_hand'
```

### Torso (2 parts):

```python
'torso_front'
'torso_back'
```

### Legs (8 parts):

```python
'left_upper_leg_front'
'left_upper_leg_back'
'right_upper_leg_front'
'right_upper_leg_back'
'left_lower_leg_front'
'left_lower_leg_back'
'right_lower_leg_front'
'right_lower_leg_back'
```

### Feet (2 parts - PLURAL!):

```python
'left_feet'   # ← NOTE: PLURAL!
'right_feet'  # ← NOTE: PLURAL!
```

---

## Comparison: What's Different

| Body Part       | ❌ Your Code (Wrong) | ✅ Official (Correct) |
| --------------- | -------------------- | --------------------- |
| Left foot       | `'left_foot'`        | `'left_feet'`         |
| Right foot      | `'right_foot'`       | `'right_feet'`        |
| Everything else | ✅ Correct           | ✅ Correct            |

---

## Impact on Your Code:

### Before Fix (With Bug):

When selecting dress/legs:

```python
self.selected_parts = [
    'torso', 'left_upper_leg', 'right_upper_leg',
    'left_lower_leg', 'right_lower_leg'
]

# Maps to:
part_map = {
    'left_lower_leg': ['left_lower_leg_front', 'left_lower_leg_back'],
    'right_lower_leg': ['right_lower_leg_front', 'right_lower_leg_back']
}
# ✅ This works!

# But if you had foot selection:
part_map = {
    'left_foot': ['left_foot']  # ❌ Returns EMPTY mask!
}
```

So actually, **you might not be affected** if you're not explicitly selecting feet!

But the foot references in your cage/keypoint mapping code are using wrong names for future use.

---

## Pattern Insight:

BodyPix uses **PLURAL** for body parts that are naturally plural:

- ✅ `'left_feet'` / `'right_feet'` (plural - you have feet)
- ✅ `'left_hand'` / `'right_hand'` (singular - you have a hand)
- ✅ `'left_face'` / `'right_face'` (singular - you have a face)

---

## How To Verify Part Names Work:

Test script:

```python
from tf_bodypix.api import load_model, download_model, BodyPixModelPaths
from tf_bodypix.bodypix_js_utils.part_channels import PART_CHANNELS

# Load model
model = load_model(download_model(
    BodyPixModelPaths.MOBILENET_FLOAT_75_STRIDE_16
))

# Print all valid part names
print("Valid BodyPix part names:")
for i, name in enumerate(PART_CHANNELS):
    print(f"  {i:2d}. {name}")

# Test if your part name exists
test_names = ['left_foot', 'left_feet']
for name in test_names:
    if name in PART_CHANNELS:
        print(f"✅ '{name}' exists")
    else:
        print(f"❌ '{name}' DOES NOT EXIST")
```

---

## Source Reference:

Official TensorFlow.js BodyPix implementation:
https://github.com/tensorflow/tfjs-models/blob/body-pix-v2.0.4/body-pix/src/part_channels.ts

The Python library `tf-bodypix` directly ports these names from the JavaScript version.

---

## Summary:

✅ **22 out of 24 part names**: You're using correctly
❌ **2 out of 24 part names**: Wrong (`foot` instead of `feet`)

**Good news**: If you're not explicitly selecting feet in your clothing generation, this bug hasn't affected you yet.

**Bad news**: Your cage/keypoint code references `left_foot`/`right_foot` which won't work when feet are needed.

---

## Recommended Fix:

Global find-and-replace in your codebase:

- `'left_foot'` → `'left_feet'`
- `'right_foot'` → `'right_feet'`
- `"LEFT_FOOT"` → `"LEFT_FEET"`
- `"RIGHT_FOOT"` → `"RIGHT_FEET"`

**But only in BodyPix-related contexts!** Don't change MediaPipe keypoint names like `left_ankle`, etc.
