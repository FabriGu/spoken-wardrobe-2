# TripoSR Interactive Script Guide

## Quick Start

### Interactive Mode (Easiest!)

```bash
cd /Users/fabrizioguccione/Projects/spoken_wardrobe_2
python tests/clothing_to_3d_triposr_2.py
```

The script will:

1. **Show you all images** in `generated_images/` directory
2. **Let you select** which image(s) to process
3. **Ask for all options** with helpful prompts and defaults
4. **Process and save** the 3D mesh(es)

### Command-Line Mode (Original)

```bash
# Still works like the original script!
python tests/clothing_to_3d_triposr_2.py path/to/image.png --z-scale 0.7 --bake-texture
```

---

## Interactive Prompts Explained

### 1. Image Selection

- Lists all images in `generated_images/` directory
- Enter numbers: `1`, `1,3,5`, or `all`
- Supports: PNG, JPG, JPEG, BMP, TIFF, WEBP

### 2. Automatic Orientation Correction

- **Recommended: Y (yes)**
- Rotates mesh to upright position
- **Automatically flips mesh 180° to fix upside-down orientation**
- Fixes sideways/upside-down meshes
- Note: If your mesh still comes out upside-down, use `--no-flip` flag

### 3. Z-Axis Scale (Fatness Control)

- **< 1.0** = thinner/flatter mesh (e.g., 0.7)
- **= 1.0** = no scaling
- **> 1.0** = thicker mesh (e.g., 1.2)
- **Recommended: 0.6 - 0.9** for clothing
- **Default: 0.8**

### 4. Background Removal

- **Recommended: Y (yes)**
- Automatically removes background from input image
- Say 'n' only if your image already has clean background

### 5. Foreground Ratio

- Controls how much of the image the object fills
- **0.85** = object fills 85% of frame (default)
- Higher = larger object in frame
- **Recommended: 0.75 - 0.95**

### 6. Mesh Resolution

- Controls mesh detail/quality
- **110** = default (good balance)
- **150** = high quality
- **200+** = very high (slower)
- Lower = faster but less detailed

### 7. Texture Baking

- **N (no)** = use vertex colors (simpler, smaller file)
- **Y (yes)** = create texture atlas (better quality, larger file)

### 8. Texture Resolution (if baking)

- **1024** = low quality (fast)
- **2048** = default (good quality)
- **4096** = high quality (slower, larger file)

### 9. Output Format

- **[1] OBJ** = most compatible (default)
- **[2] GLB** = modern format (for web/AR)

### 10. Output Directory

- Where to save the mesh
- **Default: `generated_meshes`**
- Will be created if doesn't exist

### 11. Render Video

- Generate 360° rotating video of mesh
- **N (no)** = faster (recommended)
- **Y (yes)** = creates MP4 video (much slower)

---

## Example Session

```
TRIPOSR INTERACTIVE MODE
======================================================================
Convert 2D images to 3D meshes with ease!
======================================================================

INTERACTIVE IMAGE SELECTION
======================================================================

Found 3 image(s) in generated_images/:

  [1] shirt_red.png (145.2 KB)
  [2] jacket_blue.png (198.7 KB)
  [3] dress_green.png (212.4 KB)

----------------------------------------------------------------------

Enter image number(s) (comma-separated for multiple, or 'all'): 1,3

✓ Selected 2 image(s):
  • shirt_red.png
  • dress_green.png

PROCESSING OPTIONS
======================================================================

1. Automatic Orientation Correction
   Automatically correct mesh to upright position
   Enable? [Y/n]:

2. Z-Axis Scale (Depth/Fatness Control)
   < 1.0 = thinner/flatter mesh
   = 1.0 = no scaling
   > 1.0 = thicker mesh
   Recommended range: 0.6 - 0.9
   Enter Z-scale [default: 0.8]: 0.7

3. Background Removal
   Automatically remove background from input image
   Enable? [Y/n]:

... [continues with all options] ...

OPTIONS SUMMARY
======================================================================
  Auto-orient: True
  Z-scale: 0.7
  Remove background: True
  Foreground ratio: 0.85
  Mesh resolution: 110
  Bake texture: False
  Output format: OBJ
  Output directory: generated_meshes
  Render video: False
======================================================================

Proceed with these options? [Y/n]:

[Processing begins...]
```

---

## Tips & Tricks

### For Clothing Meshes

- Use **Z-scale: 0.6 - 0.8** to get thinner, more realistic fabric
- Enable **background removal**
- Use **foreground ratio: 0.85-0.9** to fill frame
- Resolution **110** is usually enough

### For Accessories (Hats, Bags)

- Use **Z-scale: 0.8 - 1.0** for more volume
- Higher resolution (**150+**) for small details

### For Quick Tests

- Use **lower resolution (80-100)**
- Skip **texture baking**
- Skip **video rendering**

### For Final High-Quality

- Use **resolution: 150-200**
- Enable **texture baking** with **2048-4096** resolution
- Export as **GLB** for web/AR use

---

## Output Structure

```
generated_meshes/
├── 0/
│   ├── input.png          # Processed input (with removed background)
│   ├── mesh.obj           # 3D mesh file
│   └── mesh.mtl           # Material file (if OBJ)
├── 1/
│   ├── input.png
│   ├── mesh.obj
│   └── mesh.mtl
...
```

With texture baking:

```
generated_meshes/
├── 0/
│   ├── input.png
│   ├── mesh.obj
│   ├── mesh.mtl
│   └── texture.png        # Texture atlas
```

---

## Comparison: Old vs New Script

| Feature               | `clothing_to_3d_triposr_1.py` | `clothing_to_3d_triposr_2.py`   |
| --------------------- | ----------------------------- | ------------------------------- |
| **Usage**             | Command-line only             | Interactive + Command-line      |
| **Image Selection**   | Manual path entry             | Browse `generated_images/`      |
| **Options**           | Flags (--z-scale, etc.)       | Interactive prompts             |
| **Defaults**          | Must specify                  | Smart defaults offered          |
| **Multiple Images**   | Manual list                   | Easy selection (1,3,5 or 'all') |
| **Beginner-Friendly** | ⭐⭐                          | ⭐⭐⭐⭐⭐                      |
| **Automation**        | ⭐⭐⭐                        | ⭐⭐⭐                          |

---

## Troubleshooting

### "No images found in generated_images/"

- Make sure you have images in the `generated_images/` directory
- Supported formats: PNG, JPG, JPEG, BMP, TIFF, WEBP

### "TripoSR not found"

- Clone TripoSR to `external/TripoSR/` directory
- See main project README for setup instructions

### Mesh is sideways/upside-down

- Enable **auto-orient** option (should be enabled by default)
- The script now automatically flips meshes 180° after making them upright
- If mesh is STILL upside-down after auto-orient, try adding `--no-flip` flag:
  ```bash
  python tests/clothing_to_3d_triposr_2.py --no-flip
  ```

### Mesh is too "fat" or "thick"

- Reduce **Z-scale** (try 0.6, 0.7)
- Lower values = thinner mesh

### Mesh is too thin

- Increase **Z-scale** (try 0.9, 1.0)
- Higher values = thicker mesh

### Process is too slow

- Lower **mesh resolution** (try 80-100)
- Disable **texture baking**
- Disable **video rendering**
- Use GPU if available (automatic)

---

## Command-Line Reference (Advanced)

All original flags still work:

```bash
python tests/clothing_to_3d_triposr_2.py IMAGE [OPTIONS]

Required:
  IMAGE                 Path to image(s). Omit for interactive mode.

Optional:
  --device DEVICE       Device: 'cuda:0' or 'cpu' [default: cuda:0]
  --z-scale FLOAT       Z-axis scale factor [default: 0.8]
  --no-auto-orient      Disable auto-orientation
  --no-flip             Disable 180° flip (if mesh comes out upside-down WITH flip)
  --no-remove-bg        Don't remove background
  --foreground-ratio F  Foreground size ratio [default: 0.85]
  --mc-resolution N     Mesh resolution [default: 110]
  --bake-texture        Bake texture atlas
  --texture-resolution N Texture size [default: 2048]
  --model-save-format F  Format: 'obj' or 'glb' [default: obj]
  --output-dir DIR      Output directory [default: generated_meshes]
  --render              Create 360° video
  --chunk-size N        VRAM chunk size [default: 8192]
```

### Examples:

```bash
# Quick test (low quality, fast)
python tests/clothing_to_3d_triposr_2.py image.png --mc-resolution 80 --z-scale 0.7

# High quality with texture
python tests/clothing_to_3d_triposr_2.py image.png --mc-resolution 150 \
    --bake-texture --texture-resolution 4096 --z-scale 0.8

# Multiple images
python tests/clothing_to_3d_triposr_2.py img1.png img2.png img3.png \
    --z-scale 0.7 --output-dir my_meshes

# Export as GLB for web
python tests/clothing_to_3d_triposr_2.py image.png --model-save-format glb
```

---

## Next Steps

After generating your mesh:

1. **View it** in Blender, MeshLab, or an online viewer
2. **Use it** in the cage deformation system (`test_integration.py`)
3. **Adjust Z-scale** if needed and re-run
4. **Export different formats** for different uses

Enjoy! 🎉
