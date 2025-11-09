#!/usr/bin/env python3
"""
ComfyUI API Client for Remote GPU Generation

This module provides a client for interfacing with a remote ComfyUI server
to generate clothing images using GPU-accelerated Stable Diffusion models.

Usage:
    client = ComfyUIClient("http://itp-ml.itp.tsoa.nyu.edu:9199")
    result_image = client.generate_inpainting(
        image_path="frame.png",
        mask_path="mask.png",
        prompt="fashion clothing made of flames",
        workflow_path="workflows/sdxl_inpainting_api.json"
    )
"""

import requests
import json
import time
import io
import os
from PIL import Image
import numpy as np
from typing import Optional, Dict, Any, Tuple
import uuid


class ComfyUIClient:
    """Client for interacting with ComfyUI REST API."""

    def __init__(self, base_url: str, timeout: int = 300):
        """
        Initialize ComfyUI client.

        Args:
            base_url: Base URL of ComfyUI server (e.g., "http://server:9199")
            timeout: Maximum time to wait for generation (seconds)
        """
        self.base_url = base_url.rstrip('/')
        self.timeout = timeout
        self.client_id = str(uuid.uuid4())

        print(f"ComfyUI Client initialized")
        print(f"  Base URL: {self.base_url}")
        print(f"  Client ID: {self.client_id}")
        print(f"  Timeout: {self.timeout}s")

    def test_connection(self) -> bool:
        """
        Test if the ComfyUI server is accessible.

        Returns:
            True if server is reachable, False otherwise
        """
        try:
            response = requests.get(f"{self.base_url}/system_stats", timeout=10)
            if response.status_code == 200:
                stats = response.json()
                print("✓ ComfyUI server is reachable")
                print(f"  System stats: {json.dumps(stats, indent=2)}")
                return True
            else:
                print(f"✗ Server responded with status {response.status_code}")
                return False
        except requests.exceptions.ConnectionError:
            print(f"✗ Cannot connect to {self.base_url}")
            print("  Make sure you're on the NYU network or VPN")
            return False
        except Exception as e:
            print(f"✗ Connection test failed: {e}")
            return False

    def upload_image(self, image_data: np.ndarray, filename: str) -> Optional[str]:
        """
        Upload an image to ComfyUI server.

        Args:
            image_data: Image as numpy array (RGB or grayscale)
            filename: Name for the uploaded file

        Returns:
            Filename on server if successful, None otherwise
        """
        try:
            # Convert numpy array to PIL Image
            if len(image_data.shape) == 2:
                # Grayscale
                pil_image = Image.fromarray(image_data, mode='L')
            elif image_data.shape[2] == 3:
                # RGB
                pil_image = Image.fromarray(image_data, mode='RGB')
            elif image_data.shape[2] == 4:
                # RGBA
                pil_image = Image.fromarray(image_data, mode='RGBA')
            else:
                raise ValueError(f"Unsupported image shape: {image_data.shape}")

            # Convert to bytes
            img_bytes = io.BytesIO()
            pil_image.save(img_bytes, format='PNG')
            img_bytes.seek(0)

            # Upload to server
            files = {'image': (filename, img_bytes, 'image/png')}
            response = requests.post(
                f"{self.base_url}/upload/image",
                files=files,
                timeout=30
            )

            if response.status_code == 200:
                result = response.json()
                uploaded_name = result.get('name', filename)
                print(f"✓ Uploaded: {filename} → {uploaded_name}")
                return uploaded_name
            else:
                print(f"✗ Upload failed: {response.status_code}")
                print(f"  Response: {response.text}")
                return None

        except Exception as e:
            print(f"✗ Error uploading image: {e}")
            import traceback
            traceback.print_exc()
            return None

    def load_workflow_template(self, workflow_path: str) -> Optional[Dict]:
        """
        Load workflow JSON template from file.

        Args:
            workflow_path: Path to workflow JSON file

        Returns:
            Workflow dict if successful, None otherwise
        """
        try:
            with open(workflow_path, 'r') as f:
                workflow = json.load(f)
            print(f"✓ Loaded workflow template: {workflow_path}")
            return workflow
        except Exception as e:
            print(f"✗ Error loading workflow: {e}")
            return None

    def prepare_workflow(
        self,
        workflow_template: Dict,
        prompt: str,
        negative_prompt: str,
        image_filename: str,
        mask_filename: str,
        seed: int = 100,
        steps: int = 30,
        cfg: float = 7.5
    ) -> Dict:
        """
        Prepare workflow by replacing template variables.

        Args:
            workflow_template: Base workflow dict
            prompt: Positive prompt for generation
            negative_prompt: Negative prompt
            image_filename: Name of uploaded input image
            mask_filename: Name of uploaded mask image
            seed: Random seed
            steps: Number of inference steps
            cfg: Guidance scale

        Returns:
            Prepared workflow dict
        """
        import copy
        workflow = copy.deepcopy(workflow_template)

        # Convert to JSON string for easy replacement
        workflow_str = json.dumps(workflow)

        # Replace template variables
        # For strings: replace with escaped string
        # For numbers: replace placeholder with actual number (no quotes)
        replacements = {
            '"{PROMPT}"': json.dumps(prompt),  # Properly escape and quote
            '"{NEGATIVE_PROMPT}"': json.dumps(negative_prompt),
            '"{INPUT_IMAGE}"': json.dumps(image_filename),
            '"{MASK_IMAGE}"': json.dumps(mask_filename),
            '"{SEED}"': str(seed),  # Number - no quotes
            '"{STEPS}"': str(steps),
            '"{CFG}"': str(cfg)
        }

        for placeholder, value in replacements.items():
            workflow_str = workflow_str.replace(placeholder, value)

        # Convert back to dict
        workflow = json.loads(workflow_str)

        print("✓ Prepared workflow with:")
        print(f"  Prompt: {prompt[:60]}...")
        print(f"  Negative: {negative_prompt[:60]}...")
        print(f"  Seed: {seed}, Steps: {steps}, CFG: {cfg}")

        return workflow

    def queue_prompt(self, workflow: Dict) -> Optional[str]:
        """
        Submit workflow to ComfyUI queue.

        Args:
            workflow: Prepared workflow dict

        Returns:
            Prompt ID if successful, None otherwise
        """
        try:
            payload = {
                "prompt": workflow,
                "client_id": self.client_id
            }

            response = requests.post(
                f"{self.base_url}/prompt",
                json=payload,
                timeout=30
            )

            if response.status_code == 200:
                result = response.json()
                prompt_id = result.get('prompt_id')
                print(f"✓ Queued prompt: {prompt_id}")
                return prompt_id
            else:
                print(f"✗ Failed to queue prompt: {response.status_code}")
                print(f"  Response: {response.text}")
                return None

        except Exception as e:
            print(f"✗ Error queuing prompt: {e}")
            return None

    def wait_for_completion(self, prompt_id: str) -> Optional[Dict]:
        """
        Wait for generation to complete by polling history endpoint.

        Args:
            prompt_id: ID of the queued prompt

        Returns:
            History dict with results if successful, None otherwise
        """
        print(f"⏳ Waiting for generation to complete...")
        start_time = time.time()

        while True:
            # Check timeout
            elapsed = time.time() - start_time
            if elapsed > self.timeout:
                print(f"✗ Timeout after {self.timeout}s")
                return None

            try:
                # Poll history
                response = requests.get(
                    f"{self.base_url}/history/{prompt_id}",
                    timeout=10
                )

                if response.status_code == 200:
                    history = response.json()

                    # Check if our prompt is in history
                    if prompt_id in history:
                        prompt_history = history[prompt_id]

                        # Check if generation is complete
                        if 'outputs' in prompt_history:
                            elapsed_time = time.time() - start_time
                            print(f"✓ Generation complete in {elapsed_time:.1f}s")
                            return prompt_history

                    # Still processing, wait and retry
                    time.sleep(1)
                    print(f"  Waiting... ({elapsed:.1f}s)", end='\r')
                else:
                    print(f"\n✗ History check failed: {response.status_code}")
                    return None

            except Exception as e:
                print(f"\n✗ Error checking history: {e}")
                return None

    def download_result(self, history: Dict) -> Optional[Image.Image]:
        """
        Download result image from history.

        Args:
            history: History dict from wait_for_completion

        Returns:
            PIL Image if successful, None otherwise
        """
        try:
            # Extract image info from history
            outputs = history.get('outputs', {})

            # Find the SaveImage node output
            for node_id, node_output in outputs.items():
                if 'images' in node_output:
                    images = node_output['images']
                    if len(images) > 0:
                        img_info = images[0]
                        filename = img_info['filename']
                        subfolder = img_info.get('subfolder', '')
                        folder_type = img_info.get('type', 'output')

                        # Download image
                        params = {
                            'filename': filename,
                            'subfolder': subfolder,
                            'type': folder_type
                        }

                        response = requests.get(
                            f"{self.base_url}/view",
                            params=params,
                            timeout=30
                        )

                        if response.status_code == 200:
                            img = Image.open(io.BytesIO(response.content))
                            print(f"✓ Downloaded result: {filename}")
                            print(f"  Size: {img.size}, Mode: {img.mode}")
                            return img
                        else:
                            print(f"✗ Download failed: {response.status_code}")
                            return None

            print("✗ No images found in history")
            return None

        except Exception as e:
            print(f"✗ Error downloading result: {e}")
            import traceback
            traceback.print_exc()
            return None

    def generate_inpainting(
        self,
        image: np.ndarray,
        mask: np.ndarray,
        prompt: str,
        negative_prompt: str = "low quality, blurry, distorted",
        workflow_path: str = "workflows/sdxl_inpainting_api.json",
        seed: int = 100,
        steps: int = 30,
        cfg: float = 7.5
    ) -> Optional[Image.Image]:
        """
        Generate inpainted image using ComfyUI.

        This is the main function you'll call from ClothingGenerator.

        Args:
            image: Input image as numpy array (RGB)
            mask: Mask as numpy array (grayscale, 0-255)
            prompt: Positive text prompt
            negative_prompt: Negative text prompt
            workflow_path: Path to workflow JSON template
            seed: Random seed
            steps: Number of inference steps
            cfg: Guidance scale

        Returns:
            PIL Image result if successful, None otherwise
        """
        print("\n" + "="*70)
        print("ComfyUI Remote Generation")
        print("="*70)

        try:
            # 1. Test connection
            if not self.test_connection():
                raise ConnectionError("Cannot connect to ComfyUI server")

            # 2. Combine image and mask into RGBA (ComfyUI LoadImage expects this)
            print("\n🖼️  Combining image and mask into RGBA...")

            # Ensure image is RGB
            if len(image.shape) == 2:
                # Grayscale to RGB
                image_rgb = np.stack([image, image, image], axis=2)
            else:
                image_rgb = image

            # INVERT THE MASK!
            # BodyPix: 255 = body (where we want clothing)
            # Inpainting: 255 = areas to inpaint
            # So we need to INVERT: inpaint the body area, keep the background
            inverted_mask = 255 - mask

            # Create RGBA image (RGB + Alpha channel from inverted mask)
            h, w = image_rgb.shape[:2]
            rgba_image = np.zeros((h, w, 4), dtype=np.uint8)
            rgba_image[:, :, :3] = image_rgb  # RGB channels
            rgba_image[:, :, 3] = inverted_mask  # Alpha channel (INVERTED mask)

            print(f"  RGBA shape: {rgba_image.shape}")
            print(f"  Original mask range: [{mask.min()}, {mask.max()}]")
            print(f"  Inverted mask range: [{inverted_mask.min()}, {inverted_mask.max()}]")
            print(f"  ✓ Mask inverted for inpainting (white = inpaint area)")

            # 3. Upload combined RGBA image
            print("\n📤 Uploading combined image...")
            combined_filename = self.upload_image(rgba_image, f"input_{self.client_id}.png")

            if not combined_filename:
                raise Exception("Failed to upload image")

            # 4. Load workflow template
            print("\n📋 Loading workflow...")
            workflow_template = self.load_workflow_template(workflow_path)
            if not workflow_template:
                raise Exception(f"Failed to load workflow: {workflow_path}")

            # 5. Prepare workflow
            print("\n⚙️  Preparing workflow...")
            workflow = self.prepare_workflow(
                workflow_template,
                prompt,
                negative_prompt,
                combined_filename,
                combined_filename,  # Same file for both (has mask in alpha channel)
                seed,
                steps,
                cfg
            )

            # 5. Queue prompt
            print("\n🚀 Queueing generation...")
            prompt_id = self.queue_prompt(workflow)
            if not prompt_id:
                raise Exception("Failed to queue prompt")

            # 6. Wait for completion
            print("\n⏳ Generating (this may take 30-60s on first run)...")
            history = self.wait_for_completion(prompt_id)
            if not history:
                raise Exception("Generation timed out or failed")

            # 7. Download result
            print("\n📥 Downloading result...")
            result_image = self.download_result(history)
            if not result_image:
                raise Exception("Failed to download result")

            print("\n" + "="*70)
            print("✅ ComfyUI Generation Complete!")
            print("="*70)

            return result_image

        except Exception as e:
            print(f"\n✗ ComfyUI generation failed: {e}")
            import traceback
            traceback.print_exc()
            return None


# Example usage / test
def main():
    """Test the ComfyUI client."""

    print("="*70)
    print("ComfyUI Client Test")
    print("="*70)

    # Initialize client
    client = ComfyUIClient("http://itp-ml.itp.tsoa.nyu.edu:9199")

    # Test connection
    print("\n1. Testing connection...")
    if not client.test_connection():
        print("\n✗ Cannot connect to server. Exiting.")
        print("\nTroubleshooting:")
        print("  - Are you on the NYU network or VPN?")
        print("  - Is the server running?")
        print("  - Is the URL correct?")
        return

    print("\n2. Checking for test images...")
    if not os.path.exists('debug_image_to_sd.png') or not os.path.exists('debug_mask_to_sd.png'):
        print("✗ Test images not found!")
        print("  Run your Stable Diffusion pipeline first to generate:")
        print("  - debug_image_to_sd.png")
        print("  - debug_mask_to_sd.png")
        return

    # Load test images
    print("\n3. Loading test images...")
    import cv2
    test_image = cv2.imread('debug_image_to_sd.png')
    test_image = cv2.cvtColor(test_image, cv2.COLOR_BGR2RGB)
    test_mask = cv2.imread('debug_mask_to_sd.png', cv2.IMREAD_GRAYSCALE)

    print(f"  Image shape: {test_image.shape}")
    print(f"  Mask shape: {test_mask.shape}")

    # Check for workflow
    workflow_path = "workflows/sdxl_inpainting_api.json"
    print(f"\n4. Checking for workflow: {workflow_path}")
    if not os.path.exists(workflow_path):
        print(f"✗ Workflow not found!")
        print(f"\nYou need to create the workflow JSON file first:")
        print(f"  1. Open ComfyUI web interface")
        print(f"  2. Create an inpainting workflow")
        print(f"  3. Export as 'Save (API Format)'")
        print(f"  4. Save to {workflow_path}")
        print(f"\nFor now, the test will skip workflow generation.")
        return

    # Test generation
    print("\n5. Testing generation...")
    result = client.generate_inpainting(
        image=test_image,
        mask=test_mask,
        prompt="fashion clothing made of flames, detailed fabric texture",
        negative_prompt="low quality, blurry, distorted, deformed",
        workflow_path=workflow_path,
        seed=100,
        steps=20,
        cfg=7.5
    )

    if result:
        # Save result
        result.save("comfyui_test_result.png")
        print(f"\n✅ Test successful!")
        print(f"   Result saved to: comfyui_test_result.png")
    else:
        print(f"\n✗ Test failed")


if __name__ == "__main__":
    main()
