#!/usr/bin/env python3
"""
Speech-to-Clothing Pipeline with Rodin API (Direct)

Integrated pipeline that:
1. Calibrates microphone for ambient noise
2. Listens for user speech about dream dress
3. Transcribes with Whisper
4. Captures body frame with BodyPix segmentation
5. Generates clothing with ComfyUI (2D)
6. Generates 3D mesh with Rodin API (direct, no ComfyUI)
7. Saves all outputs (images, masks, GLB mesh)

Usage:
    # Set API key first
    export RODIN_API_KEY="your_api_key_here"

    # Run pipeline
    python src/modules/speech_to_clothing_with_rodin_api.py [--viewer] [--skip-3d]

Options:
    --viewer    Show OpenCV debug windows (for debugging only)
    --skip-3d   Skip 3D mesh generation (2D only)
"""

import sys
from pathlib import Path
import time
import json
import cv2
import numpy as np
import pyaudio
import threading
import queue
from PIL import Image
import argparse
import requests
import os

# Add paths
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / "src"))

# Add BlazePose path for OAK-D Pro
blazepose_path = project_root / "external" / "depthai_blazepose"
sys.path.insert(0, str(blazepose_path))

# Import existing modules
from modules.speechRecognition import SpeechRecognizer
from modules.comfyui_client import ComfyUIClient

# Import BodyPix
from tf_bodypix.api import download_model, load_model, BodyPixModelPaths

# Import OAK-D Pro camera
from BlazeposeDepthaiEdge import BlazeposeDepthai


class BodyPartSelector:
    """Selects body parts for dress"""

    DRESS_PARTS = [
        'torso_front', 'torso_back',
        'left_upper_arm_front', 'left_upper_arm_back',
        'left_lower_arm_front', 'left_lower_arm_back',
        'right_upper_arm_front', 'right_upper_arm_back',
        'right_lower_arm_front', 'right_lower_arm_back',
        'left_upper_leg_front', 'left_upper_leg_back',
        'right_upper_leg_front', 'right_upper_leg_back',
    ]


class SpeechToClothingPipeline:
    """Main pipeline orchestrator with direct Rodin API integration"""

    def __init__(self, comfyui_url="http://itp-ml.itp.tsoa.nyu.edu:9199", show_viewer=False, enable_3d=True, rodin_api_key=None):
        """Initialize pipeline components"""

        print("\n" + "="*70)
        print("Speech-to-Clothing Pipeline with Rodin API (Direct)")
        print("="*70)

        # Configuration
        self.comfyui_url = comfyui_url
        self.workflow_path_2d = "workflows/sdxl_inpainting_api.json"
        self.show_viewer = show_viewer
        self.enable_3d = enable_3d

        # Rodin API configuration
        self.rodin_api_key = rodin_api_key or os.environ.get('RODIN_API_KEY')
        self.rodin_base_url = "https://api.hyper3d.com/api/v2"

        if self.enable_3d and not self.rodin_api_key:
            print("\n⚠️  WARNING: RODIN_API_KEY not set!")
            print("   3D generation will fail without API key.")
            print("   Set it with: export RODIN_API_KEY='your_key'")
            print("   Get key from: https://hyperhuman.deemos.com/")

        # Components (lazy loaded)
        self.speech_recognizer = None
        self.comfyui_client = None
        self.bodypix_model = None

        # OAK-D Pro Camera
        self.tracker = None
        self.renderer = None

        # State
        self.current_state = "CALIBRATING"
        self.ambient_noise_level = 0
        self.volume_threshold = 0
        self.transcribed_text = ""
        self.generated_image = None

        # Audio recording
        self.audio_chunks = []
        self.recording_duration = 10.0
        self.recording_start_time = None
        self.mic_index = None

        # WebSocket for communication with viewer
        self.ws_server = None
        self.ws_clients = set()

        print("✓ Pipeline initialized")
        if self.enable_3d:
            print("✓ 3D mesh generation enabled (Rodin API Direct)")
            if self.rodin_api_key:
                print(f"✓ API key configured: {self.rodin_api_key[:8]}...{self.rodin_api_key[-4:]}")
        else:
            print("⚠ 3D mesh generation disabled")
        print("="*70 + "\n")

    def find_microphone_by_name(self, target_name="MacBook Pro Microphone"):
        """Find microphone device index by name"""
        audio = pyaudio.PyAudio()

        print("\n🎤 Available audio input devices:")
        for i in range(audio.get_device_count()):
            device_info = audio.get_device_info_by_index(i)
            if device_info['maxInputChannels'] > 0:
                device_name = device_info['name']
                print(f"  {i}: {device_name}")

                if target_name in device_name:
                    audio.terminate()
                    print(f"\n✓ Found '{target_name}' at index {i}")
                    return i

        audio.terminate()
        print(f"\n⚠ Could not find '{target_name}', using default device")
        return None

    def calibrate_microphone(self, duration=3.0):
        """Calibrate microphone by recording ambient noise"""
        print(f"\n🎤 Calibrating microphone ({duration}s)...")
        print("   Please stay quiet...")

        mic_index = self.find_microphone_by_name("MacBook Pro Microphone")
        if mic_index is None:
            print("⚠ Using default microphone")

        audio = pyaudio.PyAudio()

        try:
            stream = audio.open(
                format=pyaudio.paInt16,
                channels=1,
                rate=16000,
                input=True,
                frames_per_buffer=1024,
                input_device_index=mic_index
            )

            print(f"✓ Opened microphone (device {mic_index})")
            self.mic_index = mic_index

            volume_samples = []
            start_time = time.time()

            while (time.time() - start_time) < duration:
                try:
                    audio_data = stream.read(1024, exception_on_overflow=False)
                    audio_array = np.frombuffer(audio_data, dtype=np.int16)
                    audio_float = audio_array.astype(np.float32)
                    volume = np.sqrt(np.mean(audio_float**2))
                    volume_samples.append(volume)
                except Exception as e:
                    print(f"   Error reading audio: {e}")
                    continue

            self.ambient_noise_level = np.mean(volume_samples)
            self.volume_threshold = self.ambient_noise_level * 2.5

            if self.volume_threshold < 50:
                print(f"   ⚠ Calculated threshold {self.volume_threshold:.0f} is very low")
                print(f"   Setting minimum threshold of 50")
                self.volume_threshold = 50

            stream.stop_stream()
            stream.close()

            print(f"\n✓ Calibration complete!")
            print(f"   Ambient noise: {self.ambient_noise_level:.0f}")
            print(f"   Speech threshold: {self.volume_threshold:.0f}")

            return self.ambient_noise_level

        except Exception as e:
            print(f"✗ Calibration failed: {e}")
            import traceback
            traceback.print_exc()
            self.volume_threshold = 500
            self.mic_index = None
            return 0

        finally:
            audio.terminate()

    def initialize_camera(self):
        """Initialize OAK-D Pro camera with BlazePose"""
        print("\n📷 Initializing OAK-D Pro camera...")

        self.tracker = BlazeposeDepthai(
            input_src='rgb',
            lm_model='lite',
            xyz=True,
            smoothing=True,
            internal_fps=30,
            internal_frame_height=640,
            stats=False,
            trace=False
        )

        if self.show_viewer:
            from BlazeposeRenderer import BlazeposeRenderer
            self.renderer = BlazeposeRenderer(self.tracker, show_3d=None, output=None)

        print("✓ OAK-D Pro initialized")

    def is_body_detected(self, body):
        """Check if a body is detected by BlazePose"""
        if body and hasattr(body, 'landmarks_world'):
            return True
        return False

    def record_speech_fixed_duration(self):
        """Record speech for exactly 10 seconds once triggered"""
        print(f"\n🎙️  Recording for {self.recording_duration} seconds...")

        audio = pyaudio.PyAudio()

        try:
            stream = audio.open(
                format=pyaudio.paInt16,
                channels=1,
                rate=16000,
                input=True,
                frames_per_buffer=1024,
                input_device_index=self.mic_index
            )

            audio_chunks = []
            self.recording_start_time = time.time()

            while (time.time() - self.recording_start_time) < self.recording_duration:
                try:
                    audio_data = stream.read(1024, exception_on_overflow=False)
                    audio_chunks.append(audio_data)

                    elapsed = time.time() - self.recording_start_time
                    remaining = self.recording_duration - elapsed
                    self.send_update({
                        "type": "recording_progress",
                        "remaining": remaining
                    })

                except Exception as e:
                    print(f"   Error reading audio: {e}")
                    continue

            stream.stop_stream()
            stream.close()

            print(f"✓ Recording complete ({len(audio_chunks)} chunks)")
            return b''.join(audio_chunks)

        except Exception as e:
            print(f"✗ Recording failed: {e}")
            return None

        finally:
            audio.terminate()

    def transcribe_with_whisper(self, audio_data):
        """Transcribe audio using Whisper"""
        if not self.speech_recognizer:
            print("\n📝 Loading Whisper model...")
            self.speech_recognizer = SpeechRecognizer(modelSize="base")
            self.speech_recognizer.loadWhisperModel()

        print("\n📝 Transcribing with Whisper...")

        try:
            audio_array = np.frombuffer(audio_data, dtype=np.int16)
            audio_float = audio_array.astype(np.float32) / 32768.0

            inputs = self.speech_recognizer.processor(
                audio_float,
                sampling_rate=16000,
                return_tensors="pt"
            )

            inputs = inputs.input_features.to(self.speech_recognizer.device)

            with np.errstate(all='ignore'):
                import torch
                with torch.no_grad():
                    predicted_ids = self.speech_recognizer.model.generate(inputs)

            transcription = self.speech_recognizer.processor.batch_decode(
                predicted_ids,
                skip_special_tokens=True
            )[0].strip()

            print("="*70)
            print(f"✅ Transcription: '{transcription}'")
            print("="*70)

            return transcription

        except Exception as e:
            print(f"\n✗ Transcription failed: {e}")
            import traceback
            traceback.print_exc()
            return ""

    def capture_frame_with_bodypix(self, clean_frame):
        """Generate dress mask with BodyPix from clean frame"""
        print("\n📸 Processing captured frame with BodyPix...")

        if not self.bodypix_model:
            print("🎭 Loading BodyPix model...")
            self.bodypix_model = load_model(download_model(
                BodyPixModelPaths.MOBILENET_FLOAT_75_STRIDE_16
            ))
            print("✓ BodyPix loaded")

        print("🎭 Running BodyPix segmentation...")
        start_time = time.time()

        frame_rgb = cv2.cvtColor(clean_frame, cv2.COLOR_BGR2RGB)
        result = self.bodypix_model.predict_single(frame_rgb)

        person_mask = result.get_mask(threshold=0.75)
        dress_parts = BodyPartSelector.DRESS_PARTS
        body_part_mask = result.get_part_mask(person_mask, part_names=dress_parts)

        if hasattr(body_part_mask, 'numpy'):
            body_part_mask = body_part_mask.numpy()
        body_part_mask = np.squeeze(body_part_mask)

        mask = (body_part_mask > 0).astype(np.uint8) * 255

        elapsed = time.time() - start_time
        print(f"✓ BodyPix complete in {elapsed:.2f}s")

        return frame_rgb, mask

    def generate_clothing_with_comfyui(self, frame, mask, prompt):
        """Generate clothing image using ComfyUI"""
        if not self.comfyui_client:
            print("\n🌐 Initializing ComfyUI client...")
            self.comfyui_client = ComfyUIClient(self.comfyui_url)

        print(f"\n🎨 Generating 2D clothing with ComfyUI...")
        print(f"   Prompt: '{prompt}'")

        seed = 100
        steps = 35
        cfg = 9.5

        try:
            result = self.comfyui_client.generate_inpainting(
                image=frame,
                mask=mask,
                prompt=f"beautiful fashion {prompt}, haute couture, detailed fabric texture, elegant design, studio lighting, high quality, photorealistic",
                negative_prompt="low quality, blurry, distorted, deformed, ugly, bad anatomy, watermark, text, amateur, simple, plain",
                workflow_path=self.workflow_path_2d,
                seed=seed,
                steps=steps,
                cfg=cfg
            )

            return result

        except Exception as e:
            print(f"✗ Generation failed: {e}")
            import traceback
            traceback.print_exc()
            return None

    def submit_rodin_task(self, image_path):
        """
        Submit task to Rodin API (Regular tier).

        Args:
            image_path: Path to input image

        Returns:
            (task_uuid, subscription_key) tuple, or (None, None) if failed
        """
        url = f"{self.rodin_base_url}/rodin"

        try:
            # Read the image file
            with open(image_path, 'rb') as image_file:
                image_data = image_file.read()

            # Prepare multipart form data
            files = {
                'images': (os.path.basename(str(image_path)), image_data, 'image/png')
            }

            # Set tier to Regular with low-poly settings
            data = {
                'tier': 'Regular',
                'quality_override': '5000',      # ~5000 faces (low-poly)
                'material': 'PBR',               # Include texture files
                'mesh_mode': 'Raw',              # Triangular faces (simpler)
                'mesh_simplify': 'true',         # Simplify mesh
                'geometry_file_format': 'glb'   # GLB format
            }

            # Prepare headers
            headers = {
                'Authorization': f'Bearer {self.rodin_api_key}',
            }

            print(f"   Submitting to Rodin API...")
            response = requests.post(url, files=files, data=data, headers=headers, timeout=30)

            # Accept both 200 (OK) and 201 (Created) as success
            if response.status_code not in [200, 201]:
                print(f"   ✗ API error: HTTP {response.status_code}")
                print(f"   Response: {response.text}")
                return None, None

            result = response.json()

            # Check for errors
            if result.get('error'):
                print(f"   ✗ API error: {result['error']}")
                return None, None

            task_uuid = result.get('uuid')
            subscription_key = result.get('jobs', {}).get('subscription_key')

            if not task_uuid or not subscription_key:
                print(f"   ✗ Invalid response format")
                print(f"   Response: {result}")
                return None, None

            print(f"   ✓ Task submitted: {task_uuid}")
            return task_uuid, subscription_key

        except requests.exceptions.Timeout:
            print(f"   ✗ Request timed out")
            return None, None
        except Exception as e:
            print(f"   ✗ Submission failed: {e}")
            import traceback
            traceback.print_exc()
            return None, None

    def check_rodin_status(self, subscription_key):
        """
        Check status of Rodin task.

        Args:
            subscription_key: Subscription key from submit response

        Returns:
            List of job statuses, or None if failed
        """
        url = f"{self.rodin_base_url}/status"

        headers = {
            'Authorization': f'Bearer {self.rodin_api_key}',
            'Content-Type': 'application/json'
        }

        data = {
            'subscription_key': subscription_key
        }

        try:
            response = requests.post(url, headers=headers, json=data, timeout=10)

            # Accept both 200 (OK) and 201 (Created) as success
            if response.status_code not in [200, 201]:
                print(f"   ✗ Status check error: HTTP {response.status_code}")
                return None

            result = response.json()
            return result.get('jobs', [])

        except Exception as e:
            print(f"   ✗ Status check failed: {e}")
            return None

    def download_rodin_results(self, task_uuid, output_dir):
        """
        Download results from completed Rodin task.

        Args:
            task_uuid: Task UUID from submit response
            output_dir: Directory to save files

        Returns:
            Path to downloaded GLB file, or None if failed
        """
        url = f"{self.rodin_base_url}/download"

        headers = {
            'Authorization': f'Bearer {self.rodin_api_key}',
            'Content-Type': 'application/json'
        }

        data = {
            'task_uuid': task_uuid
        }

        try:
            response = requests.post(url, headers=headers, json=data, timeout=30)

            # Accept both 200 (OK) and 201 (Created) as success
            if response.status_code not in [200, 201]:
                print(f"   ✗ Download request error: HTTP {response.status_code}")
                return None

            result = response.json()
            download_list = result.get('list', [])

            if not download_list:
                print(f"   ✗ No files available for download")
                return None

            # Download each file
            glb_path = None
            for item in download_list:
                file_url = item.get('url')
                file_name = item.get('name')

                if not file_url or not file_name:
                    continue

                print(f"   Downloading: {file_name}")

                # Download file
                file_response = requests.get(file_url, timeout=60)

                if file_response.status_code == 200:
                    dest_path = Path(output_dir) / file_name
                    dest_path.parent.mkdir(parents=True, exist_ok=True)

                    with open(dest_path, 'wb') as f:
                        f.write(file_response.content)

                    print(f"   ✓ Saved: {dest_path} ({len(file_response.content)/1024:.1f} KB)")

                    # Track GLB file
                    if file_name.endswith('.glb'):
                        glb_path = dest_path
                else:
                    print(f"   ✗ Download failed: HTTP {file_response.status_code}")

            return glb_path

        except Exception as e:
            print(f"   ✗ Download failed: {e}")
            import traceback
            traceback.print_exc()
            return None

    def generate_3d_mesh_with_rodin_api(self, image_path):
        """
        Generate 3D mesh using Rodin API directly (not via ComfyUI).

        Args:
            image_path: Path to 2D clothing image

        Returns:
            Path to GLB file, or None if failed
        """
        print(f"\n🎲 Generating 3D mesh with Rodin API (Direct)...")
        print(f"   Input: {image_path}")
        print(f"   Tier: Regular")
        print(f"   This may take 60-90 seconds...")

        if not self.rodin_api_key:
            print("✗ RODIN_API_KEY not set!")
            return None

        # Step 1: Submit task
        task_uuid, subscription_key = self.submit_rodin_task(image_path)

        if not task_uuid or not subscription_key:
            print("✗ Failed to submit task")
            return None

        # Step 2: Poll status until done
        print(f"   ⏳ Waiting for generation...")
        start_time = time.time()
        poll_interval = 5  # seconds
        max_wait = 300  # 5 minutes

        status_list = []
        while (time.time() - start_time) < max_wait:
            time.sleep(poll_interval)

            status_list = self.check_rodin_status(subscription_key)

            if not status_list:
                print(f"   ⚠ Status check failed, retrying...")
                continue

            # Print status
            all_done = True
            for job in status_list:
                job_status = job.get('status', 'Unknown')
                job_uuid = job.get('uuid', '')[:8]
                print(f"   Job {job_uuid}: {job_status}")

                if job_status not in ['Done', 'Failed']:
                    all_done = False

            # Check if all done
            if all_done:
                # Check if any failed
                if any(job.get('status') == 'Failed' for job in status_list):
                    print("✗ Generation failed")
                    return None

                # All done successfully
                elapsed = time.time() - start_time
                print(f"✓ Generation complete in {elapsed:.1f}s")
                break
        else:
            # Timeout
            print(f"✗ Generation timed out after {max_wait}s")
            return None

        # Step 3: Download results
        print(f"   📥 Downloading mesh...")
        temp_dir = Path("/tmp") / f"rodin_output_{int(time.time())}"
        temp_dir.mkdir(parents=True, exist_ok=True)

        glb_path = self.download_rodin_results(task_uuid, temp_dir)

        if glb_path and glb_path.exists():
            print(f"✓ Mesh downloaded: {glb_path}")
            return glb_path
        else:
            print("✗ Failed to download mesh")
            return None

    def save_generated_outputs(self, frame_rgb, mask, generated_image, transcription, settings, mesh_path=None):
        """
        Save all outputs to organized folders.

        Args:
            frame_rgb: Original RGB frame
            mask: Body part mask
            generated_image: PIL Image result (2D)
            transcription: Prompt text
            settings: Dict with seed, steps, cfg
            mesh_path: Path to GLB mesh file (optional)
        """
        timestamp = int(time.time())
        output_dir = Path("comfyui_generated_mesh") / str(timestamp)
        output_dir.mkdir(parents=True, exist_ok=True)

        print(f"\n💾 Saving to: {output_dir}")

        # Save original frame
        frame_bgr = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)
        cv2.imwrite(str(output_dir / "original_frame.png"), frame_bgr)
        print(f"  ✓ original_frame.png")

        # Save mask
        cv2.imwrite(str(output_dir / "mask.png"), mask)
        print(f"  ✓ mask.png")

        # Save generated 2D clothing
        generated_image.save(str(output_dir / "generated_clothing.png"))
        print(f"  ✓ generated_clothing.png")

        # Save GLB mesh
        if mesh_path and mesh_path.exists():
            import shutil
            glb_dest = output_dir / "clothing_mesh.glb"
            shutil.copy(mesh_path, glb_dest)
            print(f"  ✓ clothing_mesh.glb ({mesh_path.stat().st_size / 1024:.1f} KB)")

        # Save metadata
        metadata = {
            "timestamp": timestamp,
            "transcription": transcription,
            "settings_2d": settings,
            "has_3d_mesh": mesh_path is not None,
            "3d_generation_method": "rodin_api_direct"
        }
        with open(output_dir / "metadata.json", 'w') as f:
            json.dump(metadata, f, indent=2)
        print(f"  ✓ metadata.json")

        print(f"\n✅ All files saved to: {output_dir}")

        return output_dir

    def send_update(self, data):
        """Send update to viewer"""
        if data["type"] == "recording_progress":
            print(f"   Recording... {data['remaining']:.1f}s remaining", end='\r')

    def run(self):
        """Main pipeline execution"""

        print("\n" + "="*70)
        print("Starting Pipeline")
        print("="*70)

        # Step 1-8: Same as original (calibrate, detect body, record, transcribe, capture, generate 2D)
        self.current_state = "CALIBRATING"
        self.calibrate_microphone(duration=3.0)

        self.initialize_camera()

        self.current_state = "WAITING_FOR_BODY"
        print("\n👤 Waiting for body detection...")
        print("   Stand in front of OAK-D camera...")

        body_detected = False
        while not body_detected:
            frame, body = self.tracker.next_frame()
            if frame is None:
                break

            body_detected = self.is_body_detected(body)

            if self.show_viewer:
                display_frame = self.renderer.draw(frame, body) if self.renderer else frame
                cv2.imshow("Speech to Clothing", display_frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    print("\nExiting...")
                    if self.show_viewer:
                        cv2.destroyAllWindows()
                    return

        print("✓ Body detected!")

        self.current_state = "WAITING_FOR_SPEECH"
        print("\n🎤 Listening for speech...")
        print("   Say: 'Describe your dream dress using your imagination'")
        print(f"   Speak when volume > {self.volume_threshold:.0f}")

        audio = pyaudio.PyAudio()
        stream = audio.open(
            format=pyaudio.paInt16,
            channels=1,
            rate=16000,
            input=True,
            frames_per_buffer=1024,
            input_device_index=self.mic_index
        )

        speech_detected = False
        while not speech_detected:
            try:
                audio_data = stream.read(1024, exception_on_overflow=False)
                audio_array = np.frombuffer(audio_data, dtype=np.int16)
                audio_float = audio_array.astype(np.float32)
                volume = np.sqrt(np.mean(audio_float**2))

                if volume > self.volume_threshold:
                    print(f"\n✓ Speech detected! (volume: {volume:.0f})")
                    speech_detected = True
                    break

                if self.show_viewer:
                    frame, body = self.tracker.next_frame()
                    if frame is not None:
                        display_frame = self.renderer.draw(frame, body) if self.renderer else frame
                        cv2.putText(display_frame, f"Volume: {volume:.0f} / {self.volume_threshold:.0f}",
                                   (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                        cv2.imshow("Speech to Clothing", display_frame)
                        if cv2.waitKey(1) & 0xFF == ord('q'):
                            stream.stop_stream()
                            stream.close()
                            audio.terminate()
                            cv2.destroyAllWindows()
                            return

            except Exception as e:
                continue

        stream.stop_stream()
        stream.close()
        audio.terminate()

        self.current_state = "RECORDING"
        audio_data = self.record_speech_fixed_duration()

        if not audio_data:
            print("✗ Recording failed, exiting")
            return

        self.transcribed_text = self.transcribe_with_whisper(audio_data)

        if not self.transcribed_text:
            print("✗ Transcription failed, using default prompt")
            self.transcribed_text = "elegant dress with floral patterns"

        self.current_state = "WAITING_FOR_POSE"
        print("\n🤸 Please stand in A-pose...")
        print("   3 second countdown starting...")

        clean_frame = None
        for i in range(3, 0, -1):
            print(f"   {i}...")

            frame, body = self.tracker.next_frame()
            if frame is not None:
                clean_frame = frame.copy()

                if self.show_viewer:
                    display_frame = frame.copy()
                    cv2.putText(display_frame, f"A-POSE: {i}", (display_frame.shape[1]//2 - 100, display_frame.shape[0]//2),
                               cv2.FONT_HERSHEY_SIMPLEX, 3, (0, 255, 255), 5)
                    cv2.imshow("Speech to Clothing", display_frame)
                    cv2.waitKey(1)

            time.sleep(1)

        if clean_frame is None:
            frame, body = self.tracker.next_frame()
            if frame is not None:
                clean_frame = frame.copy()

        if clean_frame is None:
            print("✗ Frame capture failed, exiting")
            return

        print("\n📸 Frame captured!")
        frame_rgb, mask = self.capture_frame_with_bodypix(clean_frame)

        if frame_rgb is None or mask is None:
            print("✗ BodyPix processing failed, exiting")
            return

        self.current_state = "GENERATING"
        result_image = self.generate_clothing_with_comfyui(frame_rgb, mask, self.transcribed_text)

        if not result_image:
            print("\n✗ 2D generation failed")
            return

        # Step 9: Generate 3D mesh with Rodin API (Direct)
        mesh_path = None
        if self.enable_3d:
            # Save temporary 2D image for Rodin input
            temp_2d_path = Path(f"/tmp/clothing_2d_{int(time.time())}.png")
            result_image.save(temp_2d_path)

            mesh_path = self.generate_3d_mesh_with_rodin_api(temp_2d_path)

            # Cleanup temp file
            if temp_2d_path.exists():
                temp_2d_path.unlink()

        # Step 10: Save all outputs
        self.current_state = "DONE"
        print("\n" + "="*70)
        print("✅ Generation Complete!")
        print("="*70)
        print(f"   Prompt: '{self.transcribed_text}'")
        if mesh_path:
            print(f"   3D Mesh: ✓ Generated")
        else:
            print(f"   3D Mesh: ✗ Skipped or failed")

        settings = {
            "seed": 100,
            "steps": 35,
            "cfg": 9.5
        }
        output_dir = self.save_generated_outputs(frame_rgb, mask, result_image, self.transcribed_text, settings, mesh_path)

        if self.show_viewer:
            result_np = np.array(result_image)
            result_bgr = cv2.cvtColor(result_np, cv2.COLOR_RGB2BGR)
            cv2.imshow("Generated Clothing", result_bgr)

            print("\n[Viewer Mode] Press any key to exit...")
            cv2.waitKey(0)
            cv2.destroyAllWindows()
        else:
            print("\n✅ Pipeline complete! Check output folder for results.")
            print(f"   Location: {output_dir}")

        if self.show_viewer:
            cv2.destroyAllWindows()


def main():
    """Entry point"""

    parser = argparse.ArgumentParser(description="Speech-to-Clothing Pipeline with Rodin API (Direct)")
    parser.add_argument('--viewer', action='store_true',
                       help='Show OpenCV debug windows')
    parser.add_argument('--skip-3d', action='store_true',
                       help='Skip 3D mesh generation (2D only)')
    parser.add_argument('--api-key', type=str,
                       help='Rodin API key (or set RODIN_API_KEY env var)')
    args = parser.parse_args()

    if args.viewer:
        print("\n🖼️  VIEWER MODE - OpenCV windows will be shown")
    else:
        print("\n🎥 HEADLESS MODE - Output via console")

    # Check for API key
    api_key = args.api_key or os.environ.get('RODIN_API_KEY')
    if not args.skip_3d and not api_key:
        print("\n⚠️  WARNING: No Rodin API key provided!")
        print("   Set with: export RODIN_API_KEY='your_key_here'")
        print("   Or use: --api-key 'your_key_here'")
        print("   Get key from: https://hyperhuman.deemos.com/")
        print("\n   3D generation will be skipped.")
        print("   Use --skip-3d flag to suppress this warning.\n")
        time.sleep(3)

    try:
        pipeline = SpeechToClothingPipeline(
            comfyui_url="http://itp-ml.itp.tsoa.nyu.edu:9199",
            show_viewer=args.viewer,
            enable_3d=not args.skip_3d,
            rodin_api_key=api_key
        )
        pipeline.run()

    except KeyboardInterrupt:
        print("\n\nInterrupted by user")
    except Exception as e:
        print(f"\n✗ Pipeline error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
