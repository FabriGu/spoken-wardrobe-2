# Speech-to-Clothing Pipeline - Quick Start Guide

## What This Does

An integrated pipeline that lets you:
1. **Speak** your dream dress description
2. **Transcribe** with Whisper AI
3. **Generate** clothing with ComfyUI
4. **View** results in real-time

## Files Created

### Backend
- `src/modules/speech_to_clothing.py` - Main pipeline orchestrator

### Frontend
- `tests/speechrecognition_sd_viewer.html` - Three.js visualization viewer

## How to Run

### Option 1: With Three.js Viewer (Recommended for Demo)

```bash
# 1. Open the viewer in your browser
open tests/speechrecognition_sd_viewer.html

# Press 'S' key to start simulation mode (no backend needed)
# This shows you all the UI states and transitions
```

### Option 2: Full Pipeline with Python Backend

```bash
# 1. Activate environment
source venv/bin/activate

# 2. Run the pipeline (default: no OpenCV windows)
python src/modules/speech_to_clothing.py

# OR with debug viewer (shows OpenCV windows)
python src/modules/speech_to_clothing.py --viewer

# The pipeline will:
# - Calibrate microphone (3 seconds of silence)
# - Wait for body detection (OAK-D Pro camera)
# - Listen for speech trigger
# - Record for 10 seconds
# - Transcribe with Whisper (detailed console output)
# - Prompt for A-pose
# - Capture frame + BodyPix mask
# - Generate with ComfyUI
# - Save to comfyui_generated_images/{timestamp}/
# - Display result (if --viewer flag used)
```

## Pipeline Flow

```
1. CALIBRATING (3s)
   "Please stay quiet..."
   ↓
2. WAITING_FOR_BODY
   "Stand in front of camera..."
   ↓
3. BODY_DETECTED ✓
   ↓
4. WAITING_FOR_SPEECH
   "Describe your dream dress using your imagination"
   ↓
5. RECORDING (10s)
   [Countdown shown in corner]
   ↓
6. TRANSCRIBED
   [Text displayed in bottom left]
   ↓
7. WAITING_FOR_POSE
   "Stand in A-POSE (arms slightly out)"
   [3 second countdown]
   ↓
8. CAPTURING
   "Hold still..."
   📸
   ↓
9. GENERATING
   "Creating your dream dress..."
   [May take 30-60 seconds on GPU]
   ↓
10. DONE ✓
    [Shows generated image, then comparison view]
```

## ComfyUI Settings Explained

In `src/modules/speech_to_clothing.py` (lines 320-350), you'll find different presets:

### PRESET 1: Creative & Imaginative ⭐ (ACTIVE BY DEFAULT)
```python
seed = 100
steps = 35
cfg = 9.5
```
- **Best for**: Artistic, imaginative descriptions
- **Quality**: High
- **Speed**: ~30-40s on GPU
- **Adherence to prompt**: Very close

### PRESET 2: Balanced Quality/Speed
```python
seed = 100
steps = 25
cfg = 7.5
```
- **Best for**: Quick testing
- **Quality**: Good
- **Speed**: ~20-25s on GPU
- **Adherence to prompt**: Moderate

### PRESET 3: Fast Preview
```python
seed = 100
steps = 15
cfg = 6.0
```
- **Best for**: Rapid iteration
- **Quality**: Lower but acceptable
- **Speed**: ~10-15s on GPU
- **Adherence to prompt**: Loose (more creative freedom)

### PRESET 4: Highly Creative
```python
seed = 100
steps = 40
cfg = 12.0
```
- **Best for**: When you want EXACT prompt following
- **Quality**: Very high
- **Speed**: ~40-50s on GPU
- **Adherence to prompt**: Extremely strict

### PRESET 5: Realistic/Photographic
```python
seed = 100
steps = 30
cfg = 8.0
```
- **Best for**: Photorealistic results
- **Quality**: High
- **Speed**: ~25-35s on GPU
- **Adherence to prompt**: Close

## Parameter Explanation

### Seed
- **What it is**: Random number that initializes generation
- **Range**: Any integer (0-999999)
- **Effect**: Same seed + same prompt = identical result
- **Tip**: Change seed (42, 123, 999) to get variations

### Steps
- **What it is**: Number of denoising iterations
- **Range**: 10-50 (recommended)
- **Effect**:
  - Lower (10-20): Faster but less refined
  - Medium (25-35): Balanced
  - Higher (35-50): Better quality but slower
- **Tip**: 25-35 is sweet spot for most cases

### CFG (Classifier-Free Guidance)
- **What it is**: How strictly to follow the prompt
- **Range**: 5-15 (recommended)
- **Effect**:
  - Lower (5-7): More artistic freedom, creative interpretation
  - Medium (7-10): Balanced adherence
  - Higher (10-15): Strict prompt following, less variation
- **Tip**:
  - Use 7-9 for natural, varied results
  - Use 10-12 for specific, detailed prompts
  - Avoid >15 (can cause artifacts)

## How to Experiment with Settings

1. **Open** `src/modules/speech_to_clothing.py`
2. **Find** lines 320-350 (the settings section)
3. **Comment out** current preset (add `#` at start of line)
4. **Uncomment** different preset (remove `#`)
5. **Run** pipeline again

Example:
```python
# PRESET 1: Creative & Imaginative (currently active)
# seed = 100
# steps = 35
# cfg = 9.5

# PRESET 2: Balanced Quality/Speed (let's try this one)
seed = 100
steps = 25
cfg = 7.5
```

## Example Prompts

### Good Prompts (Specific Details)
- ✅ "flowing silk dress with golden embroidery and cherry blossoms"
- ✅ "elegant ball gown with layers of tulle in pastel pink and lavender"
- ✅ "sleek cocktail dress with sequins and art deco patterns"
- ✅ "bohemian maxi dress with floral prints and lace trim"

### Less Effective Prompts (Too Vague)
- ❌ "nice dress"
- ❌ "something blue"
- ❌ "dress"

### Tips for Better Results
1. **Include materials**: silk, velvet, tulle, lace, sequins
2. **Specify colors**: specific shades work better than generic colors
3. **Add details**: embroidery, patterns, textures, trim
4. **Set the style**: elegant, bohemian, vintage, modern, couture
5. **Keep it under 20 words**: Too long can confuse the model

## Output Files

All generated images are saved to organized folders:

```
comfyui_generated_images/
└── {timestamp}/
    ├── original_frame.png    # Clean frame from OAK-D
    ├── mask.png              # BodyPix body part mask
    ├── generated_clothing.png # Final ComfyUI result
    └── metadata.json         # Prompt and settings
```

Example:
```
comfyui_generated_images/1731088234/
├── original_frame.png
├── mask.png
├── generated_clothing.png
└── metadata.json
```

## Troubleshooting

### "Microphone calibration failed"
**Solution**:
- Check System Preferences → Security & Privacy → Microphone
- Grant permission to Terminal/Python
- Make sure no other apps are using microphone

### "OAK-D camera not found"
**Solution**:
- Ensure OAK-D Pro is connected via USB
- Check cable connection
- Try different USB port
- Restart the device

### "Cannot connect to ComfyUI server"
**Solution**:
- Make sure you're on NYU network or VPN
- Check server is running: http://itp-ml.itp.tsoa.nyu.edu:9199
- Verify workflow file exists: `workflows/sdxl_inpainting_api.json`

### "Body not detected"
**Solution**:
- Stand further from camera (full body should be visible)
- Ensure good lighting
- Stand still for a moment

### "Whisper transcription is blank"
**Solution**:
- Speak louder and clearer
- Reduce background noise
- Check microphone is working: `System Preferences → Sound → Input`
- Try increasing duration from 10s to 15s in code

### "Generated image looks wrong"
**Solution**:
- Try different seed values (42, 123, 999)
- Adjust CFG:
  - If too creative/wrong: Increase CFG to 10-12
  - If too strict/boring: Decrease CFG to 6-8
- Increase steps for better quality (30-40)
- Improve your prompt with more specific details

### "Generation is very slow on Mac"
**Solution**:
- This is expected! Mac MPS can't run ComfyUI efficiently
- Use school's GPU server (2-5 seconds vs 5+ minutes on Mac)
- For testing on Mac: Reduce steps to 15-20

## Viewer Controls (Three.js)

### Keyboard Shortcuts
- **R**: Reset/Reload page
- **S**: Start simulation mode (for testing without backend)

### UI Elements

**Status Indicator (Top Left)**:
- 🟡 Calibrating
- 🔵 Waiting
- 🔴 Recording
- 🟣 Generating
- 🟢 Complete

**Body Detection (Top Left, under status)**:
- ❌ Red = No body detected
- ✓ Green = Body detected

**Instructions (Top Center)**:
- Shows current step instructions

**Recording Countdown (Top Right)**:
- Red pulsing circle with seconds remaining

**Transcription (Bottom Left)**:
- Shows what Whisper transcribed

**Generated Image**:
- Full screen display of result
- Then splits into comparison view (before/after)

## Next Steps

### For Better Integration
1. **Add WebSocket server** to Python backend for real-time viewer updates
2. **Save captured frames** for later reference
3. **Add retry button** if generation fails
4. **Export results** with metadata (prompt, settings, timestamp)

### For Better Results
1. **Try all 5 presets** to see which works best for your style
2. **Experiment with seeds** (42, 123, 999, etc.)
3. **Test different prompts** with varying detail levels
4. **Compare settings** side-by-side

### For Production Use
1. **Add error recovery** (auto-retry on failure)
2. **Implement queue system** (multiple users)
3. **Add prompt templates** (predefined styles)
4. **Cache successful prompts** for faster re-generation

---

## Files Overview

### Backend Pipeline
```
src/modules/speech_to_clothing.py
├── SpeechToClothingPipeline (main class)
├── calibrate_microphone()
├── initialize_camera()
├── is_body_detected()
├── record_speech_fixed_duration()
├── transcribe_with_whisper()
├── capture_frame_with_bodypix()
└── generate_clothing_with_comfyui()
```

### Frontend Viewer
```
tests/speechrecognition_sd_viewer.html
├── Camera feed (mirrored)
├── State management (10 states)
├── UI components (status, instructions, countdowns)
├── WebSocket client (for backend communication)
└── Simulation mode (for testing)
```

### Supporting Files
```
src/modules/comfyui_client.py (ComfyUI API client)
src/modules/speechRecognition.py (Whisper wrapper)
workflows/sdxl_inpainting_api.json (ComfyUI workflow)
```

---

**Created**: November 8, 2024
**Status**: ✅ Ready to test
**Next**: Run simulation in viewer, then test full pipeline
