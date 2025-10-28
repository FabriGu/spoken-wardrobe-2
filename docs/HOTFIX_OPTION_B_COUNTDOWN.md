# Hotfix: Option B Calibration Countdown

**Date**: October 28, 2025  
**Feature**: 5-second countdown before T-pose calibration

---

## Problem

Option B required immediate T-pose calibration when pressing SPACE, which didn't give enough time to get into position.

---

## Solution

Added a 5-second countdown timer that starts when SPACE is pressed:

1. **Press SPACE**: Starts countdown
2. **5 seconds**: User has time to get into T-pose
3. **Automatic calibration**: After countdown completes

---

## Implementation

### Added State Variables

```python
self.calibration_countdown = 0  # 0 = not counting, >0 = counting down
self.calibration_start_time = 0
```

### Countdown Logic (Each Frame)

```python
if self.calibration_countdown > 0:
    elapsed = time.time() - self.calibration_start_time
    remaining = 5.0 - elapsed

    if remaining <= 0:
        # Countdown complete, calibrate now!
        self.calibrate_from_frame(frame)
        self.calibration_countdown = 0
    else:
        # Update countdown display
        self.calibration_countdown = int(remaining) + 1
```

### Visual Feedback

Large, bright cyan text on screen during countdown:

```
"CALIBRATING IN 5... GET INTO T-POSE!"
"CALIBRATING IN 4... GET INTO T-POSE!"
...
"CALIBRATING IN 1... GET INTO T-POSE!"
```

### Key Handler

```python
elif key == ord(' ') and not self.is_calibrated and self.calibration_countdown == 0:
    print("\n⏱️  Starting 5-second countdown... Get into T-pose!")
    self.calibration_countdown = 5
    self.calibration_start_time = time.time()
```

---

## Usage

```bash
python tests/test_integration_skinning.py --mesh generated_meshes/1761618888/mesh.obj
```

1. Wait for camera to initialize
2. Press **SPACE**
3. Console shows: "⏱️ Starting 5-second countdown... Get into T-pose!"
4. Screen shows countdown: "CALIBRATING IN 5... GET INTO T-POSE!"
5. Get into T-pose during countdown
6. After 5 seconds: Automatic calibration
7. Console shows: "🎯 Calibrating NOW!"

---

## Files Modified

- `tests/test_integration_skinning.py`:
  - Lines 395-396: Added state variables
  - Lines 518-530: Added countdown logic
  - Lines 568-572: Added countdown display
  - Lines 593-597: Modified SPACE key handler

---

## Testing

Now you can properly test Option B with enough time to get into T-pose!

```bash
python tests/test_integration_skinning.py --mesh generated_meshes/1761618888/mesh.obj
```

Good luck! 🚀
