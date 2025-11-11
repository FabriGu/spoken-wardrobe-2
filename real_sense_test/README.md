# RealSense Test Scripts

This folder contains PC-compatible versions of the main pipeline scripts that use **Intel RealSense** cameras instead of OAK-D Pro.

## Requirements

### Hardware
- Intel RealSense D435/D435i/D455 depth camera
- Windows PC with NVIDIA GPU (recommended) or powerful CPU
- USB 3.0 port

### Software
- Python 3.11
- NVIDIA GPU drivers (for CUDA support)
- Intel RealSense SDK 2.0

## Installation Guide

### Step 1: Install Intel RealSense SDK 2.0

**Windows:**
1. Download the latest Intel RealSense SDK from: https://github.com/IntelRealSense/librealsense/releases
2. Install `Intel.RealSense.SDK-WIN10-<version>.exe`
3. Reboot your PC

**Verify Installation:**
```bash
# Open RealSense Viewer to test camera
"C:\Program Files (x86)\Intel RealSense SDK 2.0\Intel RealSense Viewer.exe"
```

### Step 2: Install Python Dependencies

```bash
# Activate your virtual environment first
source venv/bin/activate  # Mac/Linux
# or
venv\Scripts\activate  # Windows

# Install RealSense Python wrapper
pip install pyrealsense2

# Install MediaPipe for pose tracking
pip install mediapipe

# Install other dependencies (if not already installed)
pip install -r ../requirements.txt
```

### Step 3: Test RealSense Camera

```bash
python test_realsense_basic.py
```

This will:
- Detect your RealSense camera
- Show RGB and depth streams
- Verify alignment is working

## Available Scripts

### 1. `test_realsense_basic.py`
Basic RealSense camera test - shows RGB + depth streams.

**Usage:**
```bash
python test_realsense_basic.py
```

### 2. `test_realsense_rigged_clothing_pc.py`
RealSense version of the main rigged clothing animation pipeline.

**Usage:**
```bash
# First, open the viewer in your browser
open ../tests/clothing_viewer.html

# Then run the test
python test_realsense_rigged_clothing_pc.py
```

**Controls:**
- SPACE: Start T-pose calibration
- H: Toggle human/clothing mesh
- Q: Quit

### 3. `test_realsense_sd_integration_pc.py`
RealSense version of Stable Diffusion clothing generation.

**Usage:**
```bash
python test_realsense_sd_integration_pc.py
```

**Controls:**
- SPACE: Start 5-second countdown and generate clothing
- T: T-shirt mode
- D: Dress mode
- Q: Quit

## Key Differences from OAK-D Version

| Feature | OAK-D Pro | RealSense |
|---------|-----------|-----------|
| **Pose Tracking** | On-device (BlazePose) | CPU/GPU (MediaPipe) |
| **Performance** | ~30 FPS | ~20-25 FPS (depends on GPU) |
| **Depth Quality** | Good | Excellent (better range) |
| **Latency** | ~30ms | ~50-70ms |
| **Integration** | Single library | RealSense SDK + MediaPipe |

## Troubleshooting

### Camera Not Detected
```bash
# Check if RealSense is connected
rs-enumerate-devices
```

### Low FPS
- **Enable CUDA:** Ensure PyTorch is using GPU (check script output)
- **Lower resolution:** Edit script to use 640x480 instead of 1280x720
- **Close other apps:** Ensure no other app is using the camera

### Depth Alignment Issues
- Clean camera lenses (especially IR projector)
- Ensure good lighting (avoid direct sunlight)
- Adjust `align_to` parameter in script

## Performance Tips

1. **Use CUDA:** Ensure PyTorch detects your GPU:
   ```python
   import torch
   print(torch.cuda.is_available())  # Should be True
   ```

2. **Optimize MediaPipe:**
   - Use `model_complexity=0` for faster tracking
   - Reduce `min_detection_confidence` if losing tracking

3. **Optimize RealSense:**
   - Use 848x480 @ 30fps instead of 1920x1080
   - Disable depth post-processing filters if not needed

## Notes

- RealSense depth data is in **millimeters** (same as OAK-D)
- Depth alignment is critical - always align depth to color frame
- MediaPipe runs on CPU by default, GPU support is experimental
- Expect ~20% lower FPS compared to OAK-D due to CPU-based pose tracking
