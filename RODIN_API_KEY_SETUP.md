# Rodin API Key Configuration for ComfyUI

## The Situation

The Rodin nodes in ComfyUI need an API key to work. Since ComfyUI is running on the school server (as user `itp`), you have **three options** for configuring the key.

---

## Option 1: Ask IT to Configure (Recommended)

**Contact your IT admin** and ask them to set the Rodin API key for ComfyUI.

### What to tell them:

```
"Hi, I need to configure a Rodin API key for the ComfyUI instance
running on itp-ml.itp.tsoa.nyu.edu:9199.

Can you please:
1. Set the environment variable RODIN_API_KEY="<my_key>"
2. Restart ComfyUI

Alternatively, the key can be configured in:
- ComfyUI's extra_model_paths.yaml or
- The Rodin node's settings in the web interface
"
```

### Get your API key:
1. Subscribe to Rodin at: https://hyperhuman.deemos.com/
2. Go to Account → API Keys
3. Copy your key

---

## Option 2: Configure via ComfyUI Web Interface (If You Have Access)

If the IT admin gives you access to configure nodes:

1. Go to: http://itp-ml.itp.tsoa.nyu.edu:9199
2. Load the Rodin workflow: `workflows/rodin_regular_api.json`
3. Click on the **Rodin3D_Regular** node
4. Look for "API Key" or "Settings" input
5. Paste your key there
6. **Save the workflow** (this may persist the key)

**Problem**: This might not persist across server restarts, and you need access to the UI.

---

## Option 3: Test Without Configuring (Check if Already Set)

The Rodin nodes might **already have a key configured** if:
- Your IT department set up Rodin for the class
- Someone else already configured it

### Test it:

```bash
# Run with 2D only first (to make sure pipeline works)
python src/modules/speech_to_clothing_with_3d.py --skip-3d

# Then try with 3D
python src/modules/speech_to_clothing_with_3d.py
```

**If you get an error like**:
```
"Rodin API key not found"
"401 Unauthorized"
"Missing API key"
```

Then the key is not configured → Go to Option 1.

---

## How Rodin Nodes Store Keys

ComfyUI Rodin nodes can store keys in multiple ways:

### 1. Environment Variable (Best)
```bash
export RODIN_API_KEY="your_key_here"
```
Set this before starting ComfyUI.

### 2. Config File
`ComfyUI/extra_model_paths.yaml` or similar config:
```yaml
rodin:
  api_key: "your_key_here"
```

### 3. Node Settings (Per-Workflow)
The key is saved in the workflow JSON itself:
```json
{
  "class_type": "Rodin3D_Regular",
  "inputs": {
    "api_key": "your_key_here",
    ...
  }
}
```

**Security Warning**: Don't commit workflows with hardcoded keys to GitHub!

---

## Verifying the Key is Set

Once configured, test with a simple workflow:

1. Go to ComfyUI web interface: http://itp-ml.itp.tsoa.nyu.edu:9199
2. Load `workflows/rodin_regular_api.json`
3. Upload a test image
4. Click "Queue Prompt"
5. Watch the console/logs

**Success**: You'll see progress like "Rodin generation started..."
**Failure**: Error message about missing/invalid API key

---

## Updated Workflow Files

I've created:

- **`workflows/rodin_regular_api.json`** - Uses Rodin3D_Regular (cheaper, ~70s)
- **`workflows/rodin_gen2_api.json`** - Uses Rodin Gen-2 (premium, ~90s)

Your pipeline now uses **Rodin Regular** by default to save credits.

---

## Cost Comparison

| Model | Speed | Quality | Credits | Use Case |
|-------|-------|---------|---------|----------|
| **Regular** | 70s | Good | Lower | Production (recommended) |
| Gen-2 | 90s | Best | Higher | Final output only |
| Sketch | 20s | Basic | Lowest | Quick tests |

Start with **Regular** - it's a good balance.

---

## Troubleshooting

### "Node Rodin3D_Regular does not exist"

**Cause**: Rodin extension not installed in ComfyUI

**Fix**: Ask IT to install via ComfyUI Manager:
1. Open ComfyUI Manager
2. Search "Rodin"
3. Install the Rodin extension
4. Restart ComfyUI

### "API key invalid"

**Causes**:
1. Wrong key pasted
2. Rodin subscription not active
3. Key not set in right place

**Fix**:
1. Verify key works: Test at https://hyperhuman.deemos.com/
2. Check subscription is active
3. Re-configure using Option 1

### "Generation failed: timeout"

**Causes**:
1. Server is busy
2. Rodin API is down
3. Image is too complex

**Fix**:
- Try again later
- Check Rodin API status
- Use simpler image
- Increase timeout in code (currently 300s)

---

## Files Updated

```
workflows/
├── rodin_regular_api.json     # NEW - Regular tier (default)
└── rodin_gen2_api.json         # Gen-2 tier (premium)

src/modules/
└── speech_to_clothing_with_3d.py   # Updated to use Regular
```

---

## Next Steps

1. **Get API key** from Rodin
2. **Ask IT** to configure it (Option 1)
3. **Test pipeline**:
   ```bash
   # Test 2D only first
   python src/modules/speech_to_clothing_with_3d.py --skip-3d

   # Then test with 3D
   python src/modules/speech_to_clothing_with_3d.py
   ```
4. **Check output** in `comfyui_generated_mesh/<timestamp>/`

---

**Status**: Workflow ready, needs API key configured
**Cost**: Rodin Regular tier recommended (~70s per mesh)
