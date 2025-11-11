#!/usr/bin/env python3
"""
Speech-to-Clothing Pipeline with 3D Mesh Generation

Integrated pipeline that:
1. Calibrates microphone for ambient noise
2. Listens for user speech about dream dress
3. Transcribes with Whisper
4. Captures body frame with BodyPix segmentation
5. Generates clothing with ComfyUI
6. Generates 3D mesh with Rodin via ComfyUI
7. Saves all outputs (images, masks, GLB mesh)

Usage:
    python src/modules/speech_to_clothing_with_3d.py [--viewer] [--skip-3d]

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
    """Main pipeline orchestrator with 3D mesh generation"""

    def __init__(self, comfyui_url="http://itp-ml.itp.tsoa.nyu.edu:9199", show_viewer=False, enable_3d=True):
        """Initialize pipeline components"""

        print("\n" + "="*70)
        print("Speech-to-Clothing Pipeline with 3D Mesh Generation")
        print("="*70)

        # Configuration
        self.comfyui_url = comfyui_url
        self.workflow_path_2d = "workflows/sdxl_inpainting_api.json"
        self.workflow_path_3d = "workflows/rodin_regular_api.json"  # Using Rodin Regular (not Gen-2)
        self.show_viewer = show_viewer
        self.enable_3d = enable_3d

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
            print("✓ 3D mesh generation enabled (Rodin Regular)")
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

    def generate_3d_mesh_with_rodin(self, image_path):
        """
        Generate 3D mesh using Rodin Regular via ComfyUI.

        Args:
            image_path: Path to 2D clothing image

        Returns:
            Path to GLB file, or None if failed
        """
        if not self.comfyui_client:
            self.comfyui_client = ComfyUIClient(self.comfyui_url)

        print(f"\n🎲 Generating 3D mesh with Rodin Regular...")
        print(f"   Input: {image_path}")
        print(f"   This may take 60-90 seconds...")

        try:
            # Load workflow
            workflow = self.comfyui_client.load_workflow_template(self.workflow_path_3d)
            if not workflow:
                print(f"✗ Failed to load workflow: {self.workflow_path_3d}")
                return None

            # Upload image
            with open(image_path, 'rb') as f:
                image_data = f.read()

            image_pil = Image.open(image_path)
            image_np = np.array(image_pil)

            uploaded_name = self.comfyui_client.upload_image(image_np, f"rodin_input_{self.comfyui_client.client_id}.png")
            if not uploaded_name:
                print("✗ Failed to upload image")
                return None

            print(f"✓ Uploaded: {uploaded_name}")

            # Prepare workflow
            workflow_str = json.dumps(workflow)
            workflow_str = workflow_str.replace('"{{INPUT_IMAGE}}"', json.dumps(uploaded_name))
            workflow = json.loads(workflow_str)

            # Queue prompt
            print("🚀 Queueing Rodin Regular generation...")
            start_time = time.time()

            prompt_id = self.comfyui_client.queue_prompt(workflow)
            if not prompt_id:
                print("✗ Failed to queue prompt")
                return None

            print(f"✓ Queued: {prompt_id}")
            print("⏳ Waiting for generation (60-90s)...")

            # Wait for completion
            history = self.comfyui_client.wait_for_completion(prompt_id, timeout=300)
            if not history:
                print("✗ Generation timed out or failed")
                return None

            elapsed = time.time() - start_time
            print(f"✓ Generation complete in {elapsed:.1f}s")

            # Download GLB
            print("📥 Downloading mesh...")
            glb_data = self.download_rodin_mesh(history)

            if glb_data:
                # Save GLB
                timestamp = int(time.time())
                glb_path = Path(f"/tmp/rodin_mesh_{timestamp}.glb")
                with open(glb_path, 'wb') as f:
                    f.write(glb_data)

                print(f"✓ Mesh saved: {glb_path} ({len(glb_data)/1024:.1f} KB)")
                return glb_path
            else:
                print("✗ Failed to download mesh")
                return None

        except Exception as e:
            print(f"✗ 3D generation failed: {e}")
            import traceback
            traceback.print_exc()
            return None

    def download_rodin_mesh(self, history):
        """Download GLB mesh from Rodin generation history"""
        try:
            outputs = history.get('outputs', {})

            # Look for files in output (try multiple formats)
            for node_id, node_output in outputs.items():
                # Method 1: Check 'files' key (common for file outputs)
                if 'files' in node_output:
                    files = node_output['files']
                    for file_info in files:
                        filename = file_info.get('filename', file_info.get('name', ''))
                        if filename.endswith('.glb') or filename.endswith('.obj'):
                            print(f"   Found: {filename}")

                            # Download
                            import requests
                            subfolder = file_info.get('subfolder', '')
                            file_type = file_info.get('type', 'output')

                            response = requests.get(
                                f"{self.comfyui_client.base_url}/view",
                                params={
                                    'filename': filename,
                                    'subfolder': subfolder,
                                    'type': file_type
                                },
                                timeout=60
                            )

                            if response.status_code == 200:
                                return response.content
                            else:
                                print(f"   ✗ Download failed: HTTP {response.status_code}")

                # Method 2: Check direct model_file output (Rodin3D_Regular outputs this way)
                if 'model_file' in node_output or isinstance(node_output, dict):
                    # Look for URL or path in output
                    for key, value in node_output.items():
                        if isinstance(value, str) and (value.endswith('.glb') or value.endswith('.obj')):
                            print(f"   Found model: {value}")

                            import requests
                            response = requests.get(
                                f"{self.comfyui_client.base_url}/view",
                                params={'filename': value, 'type': 'output'},
                                timeout=60
                            )

                            if response.status_code == 200:
                                return response.content

            print("   ✗ No 3D model file found in output")
            print(f"   Debug: Output keys = {list(outputs.keys())}")
            for node_id, node_output in outputs.items():
                print(f"   Node {node_id}: {list(node_output.keys())}")
            return None

        except Exception as e:
            print(f"   ✗ Download error: {e}")
            import traceback
            traceback.print_exc()
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
            "has_3d_mesh": mesh_path is not None
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

        # Step 9: Generate 3D mesh (NEW)
        mesh_path = None
        if self.enable_3d:
            # Save temporary 2D image for Rodin input
            temp_2d_path = Path(f"/tmp/clothing_2d_{int(time.time())}.png")
            result_image.save(temp_2d_path)

            mesh_path = self.generate_3d_mesh_with_rodin(temp_2d_path)

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

    parser = argparse.ArgumentParser(description="Speech-to-Clothing Pipeline with 3D")
    parser.add_argument('--viewer', action='store_true',
                       help='Show OpenCV debug windows')
    parser.add_argument('--skip-3d', action='store_true',
                       help='Skip 3D mesh generation (2D only)')
    args = parser.parse_args()

    if args.viewer:
        print("\n🖼️  VIEWER MODE - OpenCV windows will be shown")
    else:
        print("\n🎥 HEADLESS MODE - Output via console")

    try:
        pipeline = SpeechToClothingPipeline(
            comfyui_url="http://itp-ml.itp.tsoa.nyu.edu:9199",
            show_viewer=args.viewer,
            enable_3d=not args.skip_3d
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
