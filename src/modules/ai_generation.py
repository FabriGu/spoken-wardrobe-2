import torch
from diffusers import StableDiffusionInpaintPipeline
from PIL import Image
import numpy as np
import cv2
import time
import os

class ClothingGenerator:
    def __init__(self, model_id="runwayml/stable-diffusion-inpainting", cache_dir="./models"):
        
        self.model_id = model_id 
        self.cache_dir = cache_dir

        #create cache directory (Claude)
        os.makedirs(cache_dir, exist_ok=True)

        #initialized when load model 
        self.pipeline = None
        self.device = None

        #generation parameters 
        self.num_inference_steps = 10 # higher = higher quality but slower
        self.guidance_scale = 7.5 #how closely to follow prompt (1-20)

        #cache for generated images 
        self.generation_cache = {}

        print("model initialized")

    def load_model(self):
        start_time = time.time()

        #device (claude)
        if torch.backends.mps.is_available():
            self.device = "mps"  # Mac GPU
            dtype = torch.float32
            print("Using Mac GPU (MPS) acceleration")
        elif torch.cuda.is_available():
            self.device = "cuda"  # Nvidia GPU
            dtype = torch.float16
            print("Using NVIDIA GPU (CUDA) acceleration")
        else:
            self.device = "cpu"
            dtype = torch.float32
            print("Using CPU (will be slow!)")

        #load impainting pipeline 
        self.pipeline = StableDiffusionInpaintPipeline.from_pretrained(
            self.model_id,
            torch_dtype= dtype,
            cache_dir=self.cache_dir,
            safety_checker=None  # Disable safety checker for now
        )

        #move to device
        self.pipeline = self.pipeline.to(self.device)

        #enable optimization
        self.pipeline.enable_attention_slicing()

        #on Mac memory iptimi
        if self.device == "mps":
            self.pipeline.enable_attention_slicing("max")

        load_time = time.time() - start_time 
        print(f"model loaded in {load_time:.1f} seconbds")

    def create_prompt(self, user_input): #CLAUDE
        """
        Transform user input into a detailed prompt for SD Inpainting.
        
        PROMPT ENGINEERING FOR INPAINTING:
        Different from regular SD! We need to describe clothing that fits
        naturally on a person, not a standalone image.
        
        PARAMETERS:
        - user_input: What the user said (e.g., "flames", "roses")
        
        RETURNS:
        - prompt: Detailed description for SD
        - negative_prompt: What to avoid
        """
        
        # Build the prompt
        # Focus on how the clothing looks ON the person
        prompt = f"""elegant fashion clothing made of {user_input},
        worn by a person, haute couture style, detailed fabric texture,
        natural lighting, photorealistic, high quality, professional fashion photography"""
        
        # Negative prompt - what we DON'T want
        # These prevent common SD inpainting problems
        negative_prompt = """low quality, blurry, distorted, deformed, ugly,
        bad anatomy, extra limbs, disconnected, floating objects,
        text, watermark, signature, face distortion, warped clothing,
        unnatural proportions, artifacts, noise"""

        # prompt = f"bright colorful {user_input} pattern on t-shirt"
        # negative_prompt = "dark, black, shadow, blurry, low quality"
        
        return prompt, negative_prompt
    
    # def generate_clothing_inpainting(self, frame, mask, prompt, negative_prompt=None, seed = None):
    #     if self.pipeline is None:
    #         raise Exception("Model not loaded")
        
    #     #for time computation
    #     start_time = time.time()

    #     #bgr to rgb 
    #     frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    #     #python image library conversion of frame
    #     image_pil = Image.fromarray(frame_rgb)

    #     #python image library conversion mask 
    #     # mask_inverted = mask_inverted = 255 - mask
    #     # mask_inverted = cv2.bitwise_not(mask) #invert mask (claude)
    #     mask_pil = Image.fromarray(mask)

    #     #ensure correct sizes (SD works best with 512x512)
    #     original_size = image_pil.size
    #     print(original_size)
        
    #     # target_size = original_size
        
    #     # target_size = (512, 512)
    #     # image_resized = image_pil.resize(target_size, Image.LANCZOS)
    #     # mask_resized = mask_pil.resize(target_size, Image.NEAREST)

    #     image_resized = image_pil
    #     mask_resized = mask_pil

    #     #set random see if provided 
    #     if seed is not None:
    #         generator = torch.Generator(device = self.device).manual_seed(seed)
    #     else:
    #         generator = None

    #     # Run SD Inpainting CLAUDE)
    #     # The model will:
    #     # 1. Look at the original image
    #     # 2. See where the mask is
    #     # 3. Generate clothing that fits naturally in that area
    #     with torch.no_grad():
    #         result = self.pipeline(
    #             prompt=prompt,
    #             negative_prompt=negative_prompt,
    #             image=image_resized,
    #             mask_image=mask_resized,
    #             num_inference_steps=self.num_inference_steps,
    #             guidance_scale=self.guidance_scale,
    #             generator=generator
    #         )
        
    #     # Get the generated image
    #     inpainted_image = result.images[0]
        
    #     # Resize back to original size
    #     inpainted_image = inpainted_image.resize(original_size, Image.LANCZOS)
        
    #     gen_time = time.time() - start_time
    #     print(f"✓ Generation complete in {gen_time:.1f} seconds!")
        
    #     return inpainted_image

    def generate_clothing_inpainting(self, frame, mask, prompt, negative_prompt=None, seed = None):
        if self.pipeline is None:
            raise Exception("Model not loaded")
        
        start_time = time.time()

        # Validate inputs before any conversion
        print("\n" + "="*60)
        print("DEBUGGING: Checking input data before PIL conversion")
        print("="*60)
        
        validate_array_for_pil(frame, "frame from OpenCV")
        validate_array_for_pil(mask, "mask from body segmentation")
        
        # BGR to RGB conversion
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        validate_array_for_pil(frame_rgb, "frame after BGR→RGB")
        
        # Convert to PIL with EXPLICIT modes
        image_pil = Image.fromarray(frame_rgb, mode='RGB')
        mask_pil = Image.fromarray(mask, mode='L')  # Explicit grayscale mode
        
        # Verify PIL conversion worked correctly
        print(f"\nPIL Image mode: {image_pil.mode}, size: {image_pil.size}")
        print(f"PIL Mask mode: {mask_pil.mode}, size: {mask_pil.size}")
        
        # Resize to the model's expected input size
        original_size = image_pil.size
        target_size = (512, 512)  # SD inpainting was trained on this size
        
        print(f"\nResizing from {original_size} to {target_size}")
        image_resized = image_pil.resize(target_size, Image.LANCZOS)
        mask_resized = mask_pil.resize(target_size, Image.NEAREST)  # NEAREST for masks to keep sharp edges
        
        # Save debug images to see exactly what SD receives
        image_resized.save("debug_image_to_sd.png")
        mask_resized.save("debug_mask_to_sd.png")
        print("Saved debug_image_to_sd.png and debug_mask_to_sd.png")
        print("Check these files to see what Stable Diffusion receives")
        
        # Set seed for reproducibility
        if seed is not None:
            generator = torch.Generator(device = self.device).manual_seed(seed)
        else:
            generator = None

        print("\nCalling Stable Diffusion pipeline...")
        print(f"Prompt: {prompt[:100]}...")  # Show first 100 chars
        print("="*60 + "\n")
        
        # Run SD Inpainting
        with torch.no_grad():
            result = self.pipeline(
                prompt=prompt,
                negative_prompt=negative_prompt,
                image=image_resized,
                mask_image=mask_resized,
                num_inference_steps=self.num_inference_steps,
                guidance_scale=self.guidance_scale,
                generator=generator
            )

        
        # Add this inspection code:
        print("\n=== INSPECTING RESULT OBJECT ===")
        print(f"Result type: {type(result)}")
        print(f"Result attributes: {dir(result)}")
        print(f"Number of images in result: {len(result.images)}")

        for i, img in enumerate(result.images):
            print(f"\nImage {i}:")
            print(f"  Type: {type(img)}")
            print(f"  Mode: {img.mode}")
            print(f"  Size: {img.size}")
            
            # Convert to numpy to check actual pixel values
            img_array = np.array(img)
            print(f"  Array shape: {img_array.shape}")
            print(f"  Array dtype: {img_array.dtype}")
            print(f"  Value range: [{img_array.min()}, {img_array.max()}]")
            print(f"  Mean value: {img_array.mean():.2f}")
            
            # Save it with a clear name
            img.save(f"result_image_{i}.png")
            print(f"  Saved as: result_image_{i}.png")

        # Check if there's a hidden black image issue
        if hasattr(result, 'nsfw_content_detected'):
            print(f"\nNSFW content detected: {result.nsfw_content_detected}")
            
        print("=== END RESULT INSPECTION ===\n")
        
        # Get the generated image
        inpainted_image = result.images[0]
        
        # Resize back to original size
        inpainted_image = inpainted_image.resize(original_size, Image.LANCZOS)
        
        gen_time = time.time() - start_time
        print(f"✓ Generation complete in {gen_time:.1f} seconds!")
        
        return inpainted_image

    def extract_clothing_with_transparency(self, inpainted_image, mask):
        #convert image to numpy
        inpainted_array = np.array(inpainted_image)

        #make sure mask is same size (claude)
        if mask.shape[:2] != inpainted_array.shape[:2]:
            mask = cv2.resize(mask, 
                            (inpainted_array.shape[1], inpainted_array.shape[0]),
                            interpolation=cv2.INTER_NEAREST)
            
        #Create RGBA image (RGB + Alpha channel)
        clothing_rgba = np.zeros(
            (inpainted_array.shape[0], inpainted_array.shape[1], 4),
            dtype=np.uint8
        )
        
        # Copy RGB channels
        clothing_rgba[:, :, :3] = inpainted_array
        
        # Set alpha channel from mask
        # Where mask is 255 (clothing area), alpha = 255 (opaque)
        # Where mask is 0 (background), alpha = 0 (transparent)
        clothing_rgba[:, :, 3] = mask
        
        # Convert to PIL Image
        clothing_png = Image.fromarray(clothing_rgba, mode='RGBA')
        
        print("✓ Clothing extracted with transparency")
        
        return clothing_png

    def generate_clothing_from_text(self, frame, mask, text, use_cache=True):
        """
        Main function: Generate clothing from text prompt.
        
        THIS IS WHAT YOU'LL CALL FROM YOUR MAIN APP!
        
        PARAMETERS:
        - frame: Video frame (numpy array, BGR)
        - mask: Body part mask from Phase 3 (numpy array, 0 or 255)
        - text: User's spoken text ("flames", "roses", etc.)
        - use_cache: Whether to check cache for this text
        
        RETURNS:
        - inpainted_full: PIL Image with full inpainted frame
        - clothing_png: PIL Image (RGBA) with just the clothing (transparent bg)
        
        RETURNS None, None if generation fails
        """
        
        # Check cache
        if use_cache and text in self.generation_cache:
            print(f"Using cached result for '{text}'")
            return self.generation_cache[text]
        
        try:
            # Create prompt
            prompt, negative_prompt = self.create_prompt(text)
            
            # Generate with inpainting
            inpainted_full = self.generate_clothing_inpainting(
                frame, mask, prompt, negative_prompt, seed=100
            )
            
            # Extract clothing with transparency
            clothing_png = self.extract_clothing_with_transparency(
                inpainted_full, mask
            )
            
            # Cache the result
            if use_cache:
                self.generation_cache[text] = (inpainted_full, clothing_png)
            
            return inpainted_full, clothing_png
        
        except Exception as e:
            print(f"✗ Error generating clothing: {e}")
            import traceback
            traceback.print_exc()
            return None, None
    
    
    def save_images(self, inpainted_full, clothing_png, base_filename="generated"):
        """
        Save the generated images to disk.
        
        PARAMETERS:
        - inpainted_full: Full inpainted image
        - clothing_png: Clothing with transparency
        - base_filename: Base name for files
        
        Saves two files:
        - {base_filename}_full.png - Full inpainted frame
        - {base_filename}_clothing.png - Just the clothing with transparency
        """
        
        os.makedirs("generated_images", exist_ok=True)
        
        full_path = os.path.join("generated_images", f"{base_filename}_full.png")
        clothing_path = os.path.join("generated_images", f"{base_filename}_clothing.png")
        
        inpainted_full.save(full_path)
        clothing_png.save(clothing_path)
        
        print(f"✓ Saved: {full_path}")
        print(f"✓ Saved: {clothing_path}")
        
        return full_path, clothing_path
    
    
    def cleanup(self):
        """Clean up resources."""
        if self.pipeline is not None:
            del self.pipeline
            self.pipeline = None
            
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            
            print("ClothingGenerator cleaned up")


def validate_array_for_pil(array, name="array"):
    """
    Debug function to check if array is safe for PIL conversion.
    This helps catch silent data type and value range issues.
    """
    print(f"\n=== Validating {name} ===")
    print(f"Shape: {array.shape}")
    print(f"Dtype: {array.dtype}")
    print(f"Value range: [{array.min()}, {array.max()}]")
    print(f"Unique values: {np.unique(array)[:10]}")  # Show first 10 unique values
    
    # Check for common problems
    if array.dtype not in [np.uint8, np.int32, np.float32]:
        print(f"⚠️  WARNING: dtype {array.dtype} may cause PIL issues!")
        print(f"   Recommended: convert to uint8 for images/masks")
    
    if len(array.shape) == 3 and array.shape[2] == 1:
        print(f"⚠️  WARNING: Shape has single channel dimension {array.shape}")
        print(f"   PIL expects 2D arrays for grayscale, not 3D with 1 channel")
    
    if array.dtype in [np.float32, np.float64]:
        if array.max() <= 1.0:
            print(f"⚠️  WARNING: Float array in [0,1] range")
            print(f"   PIL expects [0,255] for uint8 images")
            print(f"   Multiply by 255 and convert: (array * 255).astype(np.uint8)")
    
    if name.lower().find('mask') >= 0:
        # Special checks for masks
        if not (array.min() == 0 and array.max() == 255):
            print(f"⚠️  WARNING: Mask should have values 0 and 255 only")
            print(f"   Current range: [{array.min()}, {array.max()}]")
    
    print(f"=== End validation ===\n")
    return array   

def main():
    """
    Test the clothing generation with inpainting.
    This demonstrates the Phase 3 → Phase 4 pipeline.
    """
    
    print("=" * 60)
    print("PHASE 4: AI CLOTHING GENERATION TEST")
    print("=" * 60)
    print("\nThis will generate clothing using SD Inpainting.")
    print("You need Phase 3 working first!")
    print("=" * 60)
    
    # For testing, we'll create a simple test scenario
    # In the real app, you'll get frame and mask from Phase 3
    
    print("\nNOTE: This test requires you to have:")
    print("1. A test image with a person")
    print("2. A corresponding mask from Phase 3")
    print("\nFor now, we'll do a simplified test.")
    
    # Initialize generator
    generator = ClothingGenerator()
    
    # Load model
    try:
        generator.load_model()
    except Exception as e:
        print(f"\n✗ Error loading model: {e}")
        print("\nTROUBLESHOOTING:")
        print("1. Make sure you have enough disk space (~5GB)")
        print("2. Check your internet connection")
        print("3. Try again - sometimes downloads timeout")
        return
    
    # Test prompts
    test_prompts = [
        "flames",
        "roses and thorns",
        "water droplets",
        "golden scales",
        "starry night sky"
    ]
    
    print("\n" + "="*60)
    print("INTERACTIVE MODE")
    print("="*60)
    print("This test mode requires:")
    print("1. Run Phase 3 test first")
    print("2. Save a frame and mask using 'S' key in Phase 3")
    print("3. Place them in the project root as 'test_frame.png' and 'test_mask.png'")
    print("\nOnce you have those files, this will generate clothing!")
    print("="*60)
    
    # Check for test files
    if not os.path.exists('test_frame_0.png') or not os.path.exists('test_mask_0.png'):
        print("\n⚠ Test files not found!")
        print("Please run Phase 3 first and save a frame + mask.")
        print("Then rerun this test.")
        return
    
    # Load test frame and mask
    test_frame = cv2.imread('test_frame_0.png')
    test_mask = cv2.imread('test_mask_0.png', cv2.IMREAD_GRAYSCALE)
    
    print("\n✓ Test files loaded!")
    print(f"Frame size: {test_frame.shape}")
    print(f"Mask size: {test_mask.shape}")
    
    # Test with different prompts
    for i, prompt_text in enumerate(test_prompts):
        print(f"\n[{i+1}/{len(test_prompts)}] Generating: '{prompt_text}'")
        
        # Generate clothing
        inpainted_full, clothing_png = generator.generate_clothing_from_text(
            test_frame, test_mask, prompt_text
        )
        
        if inpainted_full is not None and clothing_png is not None:
            # Save the results
            generator.save_images(
                inpainted_full, clothing_png,
                base_filename=f"{i+1}_{prompt_text.replace(' ', '_')}"
            )
            
            print(f"✓ Success! Check generated_images folder")
        else:
            print(f"✗ Failed to generate")
        
        print("-" * 60)
    
    print("\n" + "="*60)
    print("TEST COMPLETE!")
    print("="*60)
    print("Check the 'generated_images' folder for results.")
    print("You should see:")
    print("- *_full.png files: Complete inpainted frames")
    print("- *_clothing.png files: Just the clothing with transparency")
    print("\nThe *_clothing.png files can be overlaid on live video!")
    
    # Cleanup
    generator.cleanup()


if __name__ == "__main__":
    main()



