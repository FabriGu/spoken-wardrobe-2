#!/usr/bin/env python3
"""
Speech-to-Clothing Pipeline

Integrated pipeline that:
1. Calibrates microphone for ambient noise
2. Listens for user speech about dream dress
3. Transcribes with Whisper
4. Captures body frame with BodyPix segmentation
5. Generates clothing with ComfyUI
6. Displays results in Three.js viewer

Usage:
    python src/modules/speech_to_clothing.py [--viewer]

Options:
    --viewer    Show OpenCV debug windows (for debugging only)
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

# NOTE: BlazeposeRenderer is NOT imported here to avoid segfault
# It will be imported conditionally in initialize_camera() if viewer mode is enabled


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
    """Main pipeline orchestrator"""

    def __init__(self, comfyui_url="http://itp-ml.itp.tsoa.nyu.edu:9199", show_viewer=False):
        """Initialize pipeline components"""

        print("\n" + "="*70)
        print("Speech-to-Clothing Pipeline")
        print("="*70)

        # Configuration
        self.comfyui_url = comfyui_url
        self.workflow_path = "workflows/sdxl_inpainting_api.json"
        self.show_viewer = show_viewer

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
        self.mic_index = None  # Will be set during calibration

        # WebSocket for communication with viewer
        self.ws_server = None
        self.ws_clients = set()

        print("✓ Pipeline initialized")
        print("="*70 + "\n")

    def find_microphone_by_name(self, target_name="MacBook Pro Microphone"):
        """
        Find microphone device index by name.

        Args:
            target_name: Name of microphone to find

        Returns:
            device_index: Index of the microphone, or None if not found
        """
        audio = pyaudio.PyAudio()

        print("\n🎤 Available audio input devices:")
        for i in range(audio.get_device_count()):
            device_info = audio.get_device_info_by_index(i)
            if device_info['maxInputChannels'] > 0:
                device_name = device_info['name']
                print(f"  {i}: {device_name}")

                # Check if this matches our target
                if target_name in device_name:
                    audio.terminate()
                    print(f"\n✓ Found '{target_name}' at index {i}")
                    return i

        audio.terminate()
        print(f"\n⚠ Could not find '{target_name}', using default device")
        return None

    def calibrate_microphone(self, duration=3.0):
        """
        Calibrate microphone by recording ambient noise for a few seconds.

        Args:
            duration: How long to record ambient noise (seconds)

        Returns:
            ambient_level: Average ambient noise level
        """
        print(f"\n🎤 Calibrating microphone ({duration}s)...")
        print("   Please stay quiet...")

        # Find MacBook Pro Microphone
        mic_index = self.find_microphone_by_name("MacBook Pro Microphone")
        if mic_index is None:
            print("⚠ Using default microphone")

        # Initialize audio
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

            # Store for later use
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

            # Calculate ambient noise level
            self.ambient_noise_level = np.mean(volume_samples)

            # Set threshold to 2x ambient noise (adjust multiplier as needed)
            self.volume_threshold = self.ambient_noise_level * 2.5

            # Fallback if threshold is too low
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
            # Fallback to default
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
            # Import BlazeposeRenderer only when needed to avoid segfault
            from BlazeposeRenderer import BlazeposeRenderer
            self.renderer = BlazeposeRenderer(self.tracker, show_3d=None, output=None)

        print("✓ OAK-D Pro initialized")

    def is_body_detected(self, body):
        """
        Check if a body is detected by BlazePose.

        Args:
            body: BlazePose body object from tracker.next_frame()

        Returns:
            bool: True if body detected
        """
        if body and hasattr(body, 'landmarks_world'):
            return True
        return False

    def record_speech_fixed_duration(self):
        """
        Record speech for exactly 10 seconds once triggered.

        Returns:
            audio_data: Recorded audio as bytes
        """
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

                    # Send progress update to viewer
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

            # Combine chunks
            return b''.join(audio_chunks)

        except Exception as e:
            print(f"✗ Recording failed: {e}")
            return None

        finally:
            audio.terminate()

    def transcribe_with_whisper(self, audio_data):
        """
        Transcribe audio using Whisper.

        Args:
            audio_data: Raw audio bytes

        Returns:
            transcription: Text string
        """
        if not self.speech_recognizer:
            print("\n📝 Loading Whisper model...")
            self.speech_recognizer = SpeechRecognizer(modelSize="base")
            self.speech_recognizer.loadWhisperModel()

        print("\n📝 Transcribing with Whisper...")

        try:
            # Convert audio to numpy array
            audio_array = np.frombuffer(audio_data, dtype=np.int16)
            audio_float = audio_array.astype(np.float32) / 32768.0

            # Process with Whisper
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
        """
        Generate dress mask with BodyPix from clean frame.

        Args:
            clean_frame: BGR image from OAK-D camera

        Returns:
            (frame_rgb, mask): Tuple of RGB frame and dress mask
        """
        print("\n📸 Processing captured frame with BodyPix...")

        # Load BodyPix if needed
        if not self.bodypix_model:
            print("🎭 Loading BodyPix model...")
            self.bodypix_model = load_model(download_model(
                BodyPixModelPaths.MOBILENET_FLOAT_75_STRIDE_16
            ))
            print("✓ BodyPix loaded")

        # Run BodyPix segmentation
        print("🎭 Running BodyPix segmentation...")
        start_time = time.time()

        # Convert BGR to RGB for BodyPix
        frame_rgb = cv2.cvtColor(clean_frame, cv2.COLOR_BGR2RGB)
        result = self.bodypix_model.predict_single(frame_rgb)

        # Get person mask
        person_mask = result.get_mask(threshold=0.75)

        # Get dress parts mask
        dress_parts = BodyPartSelector.DRESS_PARTS
        body_part_mask = result.get_part_mask(person_mask, part_names=dress_parts)

        # Convert to numpy
        if hasattr(body_part_mask, 'numpy'):
            body_part_mask = body_part_mask.numpy()
        body_part_mask = np.squeeze(body_part_mask)

        # Convert to uint8 mask
        mask = (body_part_mask > 0).astype(np.uint8) * 255

        elapsed = time.time() - start_time
        print(f"✓ BodyPix complete in {elapsed:.2f}s")

        return frame_rgb, mask

    def generate_clothing_with_comfyui(self, frame, mask, prompt):
        """
        Generate clothing image using ComfyUI.

        Args:
            frame: RGB image (numpy array)
            mask: Binary mask (numpy array)
            prompt: Text description

        Returns:
            PIL Image or None
        """
        if not self.comfyui_client:
            print("\n🌐 Initializing ComfyUI client...")
            self.comfyui_client = ComfyUIClient(self.comfyui_url)

        print(f"\n🎨 Generating clothing with ComfyUI...")
        print(f"   Prompt: '{prompt}'")

        # === SETTINGS PRESETS ===
        # Uncomment ONE of the following presets to use it:

        # PRESET 1: Creative & Imaginative (RECOMMENDED for creative prompts)
        # Higher CFG = follows prompt more closely
        # More steps = better quality
        seed = 100
        steps = 35
        cfg = 9.5

        # PRESET 2: Balanced Quality/Speed
        # seed = 100
        # steps = 25
        # cfg = 7.5

        # PRESET 3: Fast Preview
        # seed = 100
        # steps = 15
        # cfg = 6.0

        # PRESET 4: Highly Creative (less strict to prompt, more artistic freedom)
        # seed = 100
        # steps = 40
        # cfg = 12.0

        # PRESET 5: Realistic/Photographic
        # seed = 100
        # steps = 30
        # cfg = 8.0

        # === NOTES ON PARAMETERS ===
        # seed: Random seed (same seed = same result). Try different values: 42, 123, 999
        # steps: Inference steps (15-50). More = better quality but slower
        # cfg: Classifier-Free Guidance (5-15)
        #      - Lower (5-7): More creative, artistic freedom
        #      - Medium (7-10): Balanced
        #      - Higher (10-15): Strict adherence to prompt, less variation

        try:
            result = self.comfyui_client.generate_inpainting(
                image=frame,
                mask=mask,
                prompt=f"beautiful fashion {prompt}, haute couture, detailed fabric texture, elegant design, studio lighting, high quality, photorealistic",
                negative_prompt="low quality, blurry, distorted, deformed, ugly, bad anatomy, watermark, text, amateur, simple, plain",
                workflow_path=self.workflow_path,
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

    def save_generated_images(self, frame_rgb, mask, generated_image, transcription, settings):
        """
        Save images to organized folders.

        Args:
            frame_rgb: Original RGB frame
            mask: Body part mask
            generated_image: PIL Image result
            transcription: Prompt text
            settings: Dict with seed, steps, cfg
        """
        # Create timestamp folder
        timestamp = int(time.time())
        output_dir = Path("comfyui_generated_images") / str(timestamp)
        output_dir.mkdir(parents=True, exist_ok=True)

        print(f"\n💾 Saving to: {output_dir}")

        # Save original frame (convert RGB to BGR for cv2)
        frame_bgr = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)
        cv2.imwrite(str(output_dir / "original_frame.png"), frame_bgr)
        print(f"  ✓ original_frame.png")

        # Save mask
        cv2.imwrite(str(output_dir / "mask.png"), mask)
        print(f"  ✓ mask.png")

        # Save generated clothing
        generated_image.save(str(output_dir / "generated_clothing.png"))
        print(f"  ✓ generated_clothing.png")

        # Save metadata
        metadata = {
            "timestamp": timestamp,
            "transcription": transcription,
            "settings": settings
        }
        with open(output_dir / "metadata.json", 'w') as f:
            json.dump(metadata, f, indent=2)
        print(f"  ✓ metadata.json")

        print(f"\n✅ All files saved to: {output_dir}")

        return output_dir

    def send_update(self, data):
        """Send update to Three.js viewer via WebSocket"""
        # TODO: Implement WebSocket server
        # For now, just print updates
        if data["type"] == "recording_progress":
            print(f"   Recording... {data['remaining']:.1f}s remaining", end='\r')

    def run(self):
        """Main pipeline execution"""

        print("\n" + "="*70)
        print("Starting Pipeline")
        print("="*70)

        # Step 1: Calibrate microphone
        self.current_state = "CALIBRATING"
        self.calibrate_microphone(duration=3.0)

        # Step 2: Initialize camera
        self.initialize_camera()

        # Step 3: Wait for body detection
        self.current_state = "WAITING_FOR_BODY"
        print("\n👤 Waiting for body detection...")
        print("   Stand in front of OAK-D camera...")

        body_detected = False
        while not body_detected:
            frame, body = self.tracker.next_frame()
            if frame is None:
                break

            body_detected = self.is_body_detected(body)

            # Display frame if viewer enabled
            if self.show_viewer:
                display_frame = self.renderer.draw(frame, body) if self.renderer else frame
                cv2.imshow("Speech to Clothing", display_frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    print("\nExiting...")
                    if self.show_viewer:
                        cv2.destroyAllWindows()
                    return

        print("✓ Body detected!")

        # Step 4: Wait for speech
        self.current_state = "WAITING_FOR_SPEECH"
        print("\n🎤 Listening for speech...")
        print("   Say: 'Describe your dream dress using your imagination'")
        print(f"   Speak when volume > {self.volume_threshold:.0f}")

        # Monitor audio for speech trigger
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

                # Show live camera while waiting (if viewer enabled)
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

        # Step 5: Record for 10 seconds
        self.current_state = "RECORDING"
        audio_data = self.record_speech_fixed_duration()

        if not audio_data:
            print("✗ Recording failed, exiting")
            return

        # Step 6: Transcribe
        self.transcribed_text = self.transcribe_with_whisper(audio_data)

        if not self.transcribed_text:
            print("✗ Transcription failed, using default prompt")
            self.transcribed_text = "elegant dress with floral patterns"

        # Step 7: Wait for A-pose
        self.current_state = "WAITING_FOR_POSE"
        print("\n🤸 Please stand in A-pose...")
        print("   3 second countdown starting...")

        clean_frame = None
        for i in range(3, 0, -1):
            print(f"   {i}...")

            # Get frame from OAK-D
            frame, body = self.tracker.next_frame()
            if frame is not None:
                clean_frame = frame.copy()  # CRITICAL: Save clean frame before drawing

                # Show countdown on camera (if viewer enabled)
                if self.show_viewer:
                    display_frame = frame.copy()
                    cv2.putText(display_frame, f"A-POSE: {i}", (display_frame.shape[1]//2 - 100, display_frame.shape[0]//2),
                               cv2.FONT_HERSHEY_SIMPLEX, 3, (0, 255, 255), 5)
                    cv2.imshow("Speech to Clothing", display_frame)
                    cv2.waitKey(1)

            time.sleep(1)

        # Step 8: Capture final frame and generate mask
        if clean_frame is None:
            # Get one more frame if we don't have one
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

        # Step 9: Generate clothing
        self.current_state = "GENERATING"
        result_image = self.generate_clothing_with_comfyui(frame_rgb, mask, self.transcribed_text)

        if result_image:
            # Step 10: Save and display result
            self.current_state = "DONE"
            print("\n" + "="*70)
            print("✅ Clothing Generated Successfully!")
            print("="*70)
            print(f"   Prompt: '{self.transcribed_text}'")

            # Save to organized folder
            settings = {
                "seed": 100,
                "steps": 35,
                "cfg": 9.5
            }
            output_dir = self.save_generated_images(frame_rgb, mask, result_image, self.transcribed_text, settings)

            # Display result (if viewer enabled)
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

        else:
            print("\n✗ Generation failed")

        # Cleanup
        if self.show_viewer:
            cv2.destroyAllWindows()


def main():
    """Entry point"""

    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Speech-to-Clothing Pipeline")
    parser.add_argument('--viewer', action='store_true',
                       help='Show OpenCV debug windows (for debugging only)')
    args = parser.parse_args()

    if args.viewer:
        print("\n🖼️  VIEWER MODE - OpenCV windows will be shown")
    else:
        print("\n🎥 HEADLESS MODE - All output via console and Three.js viewer")

    try:
        pipeline = SpeechToClothingPipeline(
            comfyui_url="http://itp-ml.itp.tsoa.nyu.edu:9199",
            show_viewer=args.viewer
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
