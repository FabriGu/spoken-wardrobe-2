"""
PHASE 2: SPEECH RECOGNITION WITH WHISPER
=========================================

PURPOSE: Capture audio from microphone, detect when someone speaks, and
transcribe their words using OpenAI's Whisper model running locally.

LEARNING RESOURCES:
- PyAudio Tutorial: http://people.csail.mit.edu/hubert/pyaudio/docs/
- Whisper GitHub: https://github.com/openai/whisper
- HuggingFace Transformers: https://huggingface.co/docs/transformers/model_doc/whisper
- Audio Processing: https://realpython.com/playing-and-recording-sound-python/

WHAT YOU'RE BUILDING:
A system that continuously listens, detects when someone starts speaking,
records their speech, and converts it to text using AI. This text becomes
the prompt for generating custom clothing.
"""

import pyaudio  # For capturing audio from microphone
import wave  # For saving audio files (useful for debugging)
import numpy as np  # For audio processing and analysis
import threading  # For running audio capture in background
import queue  # For thread-safe communication between capture and processing
import time
from transformers import WhisperProcessor, WhisperForConditionalGeneration
import torch  # For running the AI model


class SpeechRecognizer:
    """
    This class handles everything related to capturing audio and converting
    speech to text using Whisper AI.
    
    HOW IT WORKS:
    1. Continuously monitor microphone volume
    2. When volume exceeds threshold, start recording
    3. When volume drops for 2 seconds, stop recording
    4. Send recorded audio to Whisper for transcription
    5. Return the transcribed text
    
    WHY WHISPER?
    Whisper is OpenAI's speech recognition model that works offline,
    supports multiple languages, and is reasonably accurate. Unlike cloud
    APIs, we can run it locally with no internet needed.
    """
    
    def __init__(self, model_size="base", volume_threshold=500, silence_duration=2.0):
        """
        Initialize the speech recognition system.
        
        PARAMETERS:
        - model_size: Whisper model size ("tiny", "base", "small", "medium", "large")
                     Smaller = faster but less accurate. "base" is good balance.
        - volume_threshold: How loud audio must be to count as speech (0-32767)
                           Lower = more sensitive. Start with 500 and adjust.
        - silence_duration: How many seconds of silence before we stop recording
        
        MODEL SIZES COMPARISON:
        - tiny: 39M params, ~1GB RAM, fastest, good for testing
        - base: 74M params, ~1GB RAM, good balance (RECOMMENDED FOR START)
        - small: 244M params, ~2GB RAM, better accuracy, slower
        - medium: 769M params, ~5GB RAM, very accurate, quite slow on Mac
        - large: 1550M params, ~10GB RAM, best accuracy, very slow
        
        For this project, start with "base". You can always upgrade later.
        """
        
        self.model_size = model_size
        self.volume_threshold = volume_threshold
        self.silence_duration = silence_duration
        
        # Audio settings - these are standard for speech recognition
        # Most speech recognition systems use these same values
        self.CHUNK = 1024  # How many samples to read at once (affects latency)
        self.FORMAT = pyaudio.paInt16  # 16-bit audio (standard for speech)
        self.CHANNELS = 1  # Mono audio (stereo not needed for speech)
        self.RATE = 16000  # 16kHz sample rate (Whisper's preferred rate)
        
        # PyAudio objects (initialized when we start)
        self.audio = None
        self.stream = None
        
        # Recording state
        self.is_listening = False  # Are we monitoring for speech?
        self.is_recording = False  # Are we actively recording?
        self.audio_buffer = []  # Stores recorded audio chunks
        
        # Threading for background audio capture
        self.audio_thread = None
        self.transcription_queue = queue.Queue()  # Thread-safe queue for results
        
        # Whisper model (loaded when we start)
        self.processor = None  # Handles audio preprocessing
        self.model = None  # The actual AI model
        self.device = None  # CPU, CUDA, or MPS
        
        print("SpeechRecognizer initialized")
        print(f"Model size: {model_size}")
        print(f"Volume threshold: {volume_threshold}")
        print(f"Silence duration: {silence_duration}s")
    
    
    def load_whisper_model(self):
        """
        Load the Whisper model from HuggingFace.
        
        IMPORTANT: This downloads the model on first run!
        Models are cached in ~/.cache/huggingface/ so subsequent runs are faster.
        
        The "base" model is about 290MB download.
        This can take a few minutes on first run depending on internet speed.
        
        WHY HUGGINGFACE?
        HuggingFace provides a nice Python API for Whisper that integrates
        well with PyTorch. It's easier to use than OpenAI's original repo.
        """
        
        print(f"\nLoading Whisper {self.model_size} model...")
        print("First time will download the model (this may take a few minutes)")
        print("Subsequent runs will load from cache and be much faster")
        
        # Determine which device to use for inference
        # MPS = Apple's Metal Performance Shaders (Mac GPU)
        # CUDA = NVIDIA GPU acceleration
        # CPU = Fallback if no GPU available
        if torch.backends.mps.is_available():
            self.device = "mps"
            print("✓ Using Mac GPU (MPS) acceleration")
        elif torch.cuda.is_available():
            self.device = "cuda"
            print("✓ Using NVIDIA GPU (CUDA) acceleration")
        else:
            self.device = "cpu"
            print("⚠ Using CPU (will be slower)")
        
        # Load the processor (handles audio preprocessing)
        # The processor converts raw audio to the format Whisper expects
        model_name = f"openai/whisper-{self.model_size}"
        print(f"Loading from: {model_name}")
        
        self.processor = WhisperProcessor.from_pretrained(model_name)
        
        # Load the model itself
        # This is the actual neural network that does speech recognition
        self.model = WhisperForConditionalGeneration.from_pretrained(model_name)
        self.model = self.model.to(self.device)
        
        # Set to evaluation mode (we're not training, just using it)
        self.model.eval()
        
        print("✓ Model loaded successfully!\n")
    
    
    def initialize_audio(self):
        """
        Initialize PyAudio and open the microphone stream.
        
        This connects to your computer's microphone and prepares to capture audio.
        
        ERRORS TO WATCH FOR:
        - No microphone found: Check connections
        - Permission denied: Grant microphone access in System Preferences
        - Device already in use: Close other apps using the microphone
        """
        
        # Create PyAudio instance
        # This is the main interface to your audio hardware
        self.audio = pyaudio.PyAudio()
        
        # Print available audio devices (helpful for debugging)
        # If you have multiple microphones, this shows which one will be used
        print("\nAvailable audio input devices:")
        for i in range(self.audio.get_device_count()):
            device_info = self.audio.get_device_info_by_index(i)
            # Only show devices that can record (have input channels)
            if device_info['maxInputChannels'] > 0:
                print(f"  {i}: {device_info['name']}")
        
        try:
            # Open the audio stream
            # This starts capturing audio from the default microphone
            self.stream = self.audio.open(
                format=self.FORMAT,  # 16-bit integers
                channels=self.CHANNELS,  # Mono
                rate=self.RATE,  # 16kHz sample rate
                input=True,  # We're capturing input (not playing output)
                frames_per_buffer=self.CHUNK,  # Read in chunks of 1024 samples
                stream_callback=None  # We'll read manually for more control
            )
            
            print("\n✓ Microphone opened successfully!")
            return True
            
        except Exception as e:
            print(f"\n✗ Error opening microphone: {e}")
            print("\nTROUBLESHOOTING:")
            print("1. Check microphone permissions in System Preferences > Security & Privacy")
            print("2. Make sure microphone is connected and not muted")
            print("3. Close other apps that might be using the microphone (Zoom, etc.)")
            print("4. Try restarting your computer")
            return False
    
    
    def calculate_volume(self, audio_data):
        """
        Calculate the volume (loudness) of audio data.
        
        HOW IT WORKS:
        Audio is represented as numbers (amplitude). Silence is near 0,
        loud sounds are high numbers. We calculate the Root Mean Square (RMS)
        which gives us a single number representing overall loudness.
        
        UNDERSTANDING AUDIO DATA:
        - Raw audio is a stream of numbers representing air pressure over time
        - Positive numbers = air pushing out, negative = air pulling in
        - Larger magnitude = louder sound
        - We use RMS to get a single "loudness" value from many samples
        
        PARAMETERS:
        - audio_data: Raw audio bytes from PyAudio
        
        RETURNS:
        - volume: A number representing loudness (0 = silence, higher = louder)
                 Typical values: silence <100, quiet speech 200-500, loud speech >1000
        """
        
        # Convert raw bytes to numpy array of 16-bit integers
        # PyAudio gives us bytes, but we need numbers to work with
        audio_array = np.frombuffer(audio_data, dtype=np.int16)
        
        # Calculate RMS (Root Mean Square) volume
        # RMS = square root of the average of squared values
        # This is a standard way to measure audio volume
        volume = np.sqrt(np.mean(audio_array**2))
        
        return volume
    
    
    def listen_loop(self):
        """
        Main listening loop that runs in a background thread.
        
        This is the heart of the speech recognition system. It continuously:
        1. Reads audio from microphone
        2. Calculates volume
        3. Decides if we should be recording
        4. Stores audio when recording
        5. Processes audio when recording stops
        
        This runs in a separate thread so it doesn't block the main video feed.
        
        STATE MACHINE:
        NOT RECORDING → (volume > threshold) → RECORDING
        RECORDING → (volume < threshold for silence_duration) → PROCESSING → NOT RECORDING
        """
        
        print("\n" + "="*60)
        print("LISTENING FOR SPEECH")
        print("="*60)
        print(f"Volume threshold: {self.volume_threshold}")
        print(f"Silence duration: {self.silence_duration}s")
        print("\nSpeak when ready...")
        print("Press Ctrl+C in terminal to stop")
        print("="*60 + "\n")
        
        silence_start_time = None  # Track when silence began
        
        while self.is_listening:
            try:
                # Read audio chunk from microphone
                # exception_on_overflow=False prevents crashes from buffer overflow
                audio_data = self.stream.read(self.CHUNK, exception_on_overflow=False)
                
                # Calculate how loud this chunk is
                volume = self.calculate_volume(audio_data)
                
                # ============================================================
                # STATE MACHINE FOR RECORDING
                # ============================================================
                
                if not self.is_recording:
                    # We're not currently recording - check if we should start
                    
                    if volume > self.volume_threshold:
                        # Volume exceeded threshold - someone is speaking!
                        print(f"\n🎤 Speech detected! (volume: {volume:.0f})")
                        print("Recording...")
                        
                        self.is_recording = True
                        self.audio_buffer = []  # Clear any old data
                        self.audio_buffer.append(audio_data)  # Start recording
                        silence_start_time = None  # Reset silence timer
                    
                    else:
                        # Still quiet - show volume occasionally so user knows it's working
                        # The modulo math makes this print every ~5 seconds
                        if int(time.time() * 2) % 10 == 0:
                            print(f"Volume: {volume:.0f} (threshold: {self.volume_threshold})", end='\r')
                
                else:
                    # We ARE recording - keep adding to buffer
                    self.audio_buffer.append(audio_data)
                    
                    if volume > self.volume_threshold:
                        # Still hearing speech - reset silence timer
                        silence_start_time = None
                        print(f"Recording... (volume: {volume:.0f})", end='\r')
                    
                    else:
                        # Volume dropped - might be silence between words or end of speech
                        if silence_start_time is None:
                            # Just started being quiet - start the timer
                            silence_start_time = time.time()
                        else:
                            # Check how long we've been quiet
                            silence_duration = time.time() - silence_start_time
                            
                            if silence_duration >= self.silence_duration:
                                # Been quiet long enough - stop recording and transcribe
                                print(f"\n\n✓ Silence detected after {silence_duration:.1f}s")
                                print("Processing speech...")
                                
                                self.is_recording = False
                                
                                # Process the recorded audio
                                self.process_audio_buffer()
                                
                                # Clear buffer for next recording
                                self.audio_buffer = []
                                silence_start_time = None
                                
                                print("\nListening for next speech...")
            
            except Exception as e:
                print(f"\nError in listen loop: {e}")
                continue
    
    
    def process_audio_buffer(self):
        """
        Convert recorded audio to text using Whisper.
        
        This is where the AI magic happens. We take the buffered audio chunks and:
        1. Combine them into one audio file
        2. Preprocess for Whisper (normalize, resample if needed)
        3. Run through the neural network
        4. Extract the transcription
        
        WHAT WHISPER DOES:
        Whisper is a transformer model trained on 680,000 hours of speech.
        It converts audio spectrograms to text tokens using encoder-decoder architecture.
        But you don't need to understand that - just know it's really good at speech!
        """
        
        if not self.audio_buffer:
            print("No audio to process")
            return
        
        print("Transcribing speech with Whisper...")
        start_time = time.time()
        
        try:
            # Combine all audio chunks into one continuous array
            audio_data = b''.join(self.audio_buffer)
            audio_array = np.frombuffer(audio_data, dtype=np.int16)
            
            # Convert to float32 in range [-1.0, 1.0]
            # Whisper expects audio normalized to this range
            audio_float = audio_array.astype(np.float32) / 32768.0
            
            # Preprocess audio for Whisper
            # This resamples to 16kHz if needed and extracts mel spectrogram features
            # The processor handles all the audio preprocessing magic
            inputs = self.processor(
                audio_float,
                sampling_rate=self.RATE,
                return_tensors="pt"  # Return PyTorch tensors
            )
            
            # Move inputs to the same device as model (MPS/CUDA/CPU)
            # This ensures everything runs on the same hardware (GPU or CPU)
            inputs = inputs.input_features.to(self.device)
            
            # Generate transcription
            # We use torch.no_grad() because we're not training (saves memory)
            with torch.no_grad():
                predicted_ids = self.model.generate(inputs)
            
            # Decode the generated tokens to text
            # The model outputs token IDs, we need to convert them to words
            transcription = self.processor.batch_decode(
                predicted_ids,
                skip_special_tokens=True  # Remove <start>, <end>, etc.
            )[0]
            
            # Clean up the transcription
            transcription = transcription.strip()
            
            elapsed_time = time.time() - start_time
            
            if transcription:
                print(f"\n{'='*60}")
                print(f"✓ TRANSCRIPTION: {transcription}")
                print(f"  (processed in {elapsed_time:.1f} seconds)")
                print(f"{'='*60}\n")
                
                # Put transcription in queue for main thread to retrieve
                # The queue is thread-safe, so multiple threads can access it
                self.transcription_queue.put(transcription)
            else:
                print(f"⚠ No speech detected in audio (silent or unintelligible)")
        
        except Exception as e:
            print(f"✗ Error during transcription: {e}")
            import traceback
            traceback.print_exc()
    
    
    def start(self):
        """
        Start the speech recognition system.
        
        This:
        1. Loads the Whisper model if not already loaded
        2. Initializes the audio stream
        3. Starts the listening loop in a background thread
        """
        
        # Load Whisper model if not already loaded
        if self.model is None:
            self.load_whisper_model()
        
        # Initialize audio if not already done
        if self.stream is None:
            if not self.initialize_audio():
                return False
        
        # Start listening in a background thread
        # daemon=True means thread will die when main program exits
        self.is_listening = True
        self.audio_thread = threading.Thread(target=self.listen_loop, daemon=True)
        self.audio_thread.start()
        
        print("✓ Speech recognition started!")
        return True
    
    
    def stop(self):
        """
        Stop the speech recognition system and clean up.
        
        IMPORTANT: Always call this when you're done!
        Not cleaning up can leave the microphone locked or create memory leaks.
        """
        
        print("\nStopping speech recognition...")
        self.is_listening = False
        
        # Wait for thread to finish (with timeout)
        if self.audio_thread is not None:
            self.audio_thread.join(timeout=2.0)
        
        # Close audio stream
        if self.stream is not None:
            self.stream.stop_stream()
            self.stream.close()
        
        # Terminate PyAudio
        if self.audio is not None:
            self.audio.terminate()
        
        print("✓ Speech recognition stopped")
    
    
    def get_transcription(self, block=True, timeout=None):
        """
        Get the next transcription from the queue.
        
        This is how the main application retrieves transcriptions from
        the background thread.
        
        PARAMETERS:
        - block: If True, wait until transcription is available
                If False, return immediately (returns None if queue empty)
        - timeout: How long to wait (None = wait forever)
        
        RETURNS:
        - transcription text (string), or None if no transcription available
        """
        
        try:
            return self.transcription_queue.get(block=block, timeout=timeout)
        except queue.Empty:
            return None
    
    
    def calibrate_threshold(self, duration=5):
        """
        Helper function to calibrate the volume threshold.
        
        Run this to see what volume levels your environment has.
        Speak normally during the test to see what threshold to use.
        
        This is VERY important because different microphones and rooms
        have different ambient noise levels.
        
        PARAMETERS:
        - duration: How long to calibrate for (seconds)
        """
        
        print(f"\n{'='*60}")
        print(f"CALIBRATING VOLUME THRESHOLD FOR {duration} SECONDS")
        print(f"{'='*60}")
        print("Please speak normally during this time.")
        print("Say a few sentences as you would when using the app.")
        print(f"{'='*60}\n")
        
        if self.stream is None:
            if not self.initialize_audio():
                return
        
        volumes = []
        start_time = time.time()
        
        while time.time() - start_time < duration:
            audio_data = self.stream.read(self.CHUNK, exception_on_overflow=False)
            volume = self.calculate_volume(audio_data)
            volumes.append(volume)
            print(f"Current volume: {volume:.0f}", end='\r')
            time.sleep(0.1)
        
        print("\n\n" + "="*60)
        print("CALIBRATION RESULTS")
        print("="*60)
        print(f"Minimum volume: {min(volumes):.0f}")
        print(f"Maximum volume: {max(volumes):.0f}")
        print(f"Average volume: {np.mean(volumes):.0f}")
        print(f"Standard deviation: {np.std(volumes):.0f}")
        print("\n" + "="*60)
        print(f"RECOMMENDED THRESHOLD: {np.mean(volumes) * 1.5:.0f}")
        print("="*60)
        print("\nThis is 1.5x your average speaking volume.")
        print("If you get false triggers, increase the threshold.")
        print("If speech isn't detected, decrease the threshold.")


# ============================================================================
# USAGE EXAMPLE
# ============================================================================

def main():
    """
    Test the speech recognition system.
    
    This is a standalone test to make sure Phase 2 works before integrating
    with the rest of the application.
    """
    
    print("=" * 60)
    print("PHASE 2: SPEECH RECOGNITION TEST")
    print("=" * 60)
    
    # First, let's calibrate to find a good threshold
    print("\nWould you like to calibrate the volume threshold first?")
    print("This is HIGHLY RECOMMENDED for first-time setup.")
    response = input("Calibrate? (y/n): ").strip().lower()
    
    recognizer = SpeechRecognizer(
        model_size="base",  # Start with base model
        volume_threshold=500,  # We might adjust this after calibration
        silence_duration=2.0
    )
    
    if response == 'y':
        print("\nStarting calibration...")
        recognizer.calibrate_threshold(duration=5)
        
        # Ask user if they want to adjust threshold
        print("\nCurrent threshold:", recognizer.volume_threshold)
        new_threshold = input("Enter new threshold (or press Enter to keep current): ").strip()
        if new_threshold:
            try:
                recognizer.volume_threshold = int(new_threshold)
                print(f"✓ Threshold set to: {recognizer.volume_threshold}")
            except ValueError:
                print("Invalid number, keeping current threshold")
    
    # Start speech recognition
    print("\n" + "="*60)
    print("STARTING CONTINUOUS SPEECH RECOGNITION")
    print("="*60)
    print("Speak normally when ready.")
    print("The system will transcribe your speech and display it.")
    print("Press Ctrl+C when done testing.")
    print("="*60)
    
    try:
        recognizer.start()
        
        # Keep running until user presses Ctrl+C
        # In the real app, this would be integrated with the main loop
        while True:
            time.sleep(0.1)
    
    except KeyboardInterrupt:
        print("\n\nStopping...")
    
    finally:
        recognizer.stop()
        print("\n" + "="*60)
        print("TEST COMPLETE!")
        print("="*60)


if __name__ == "__main__":
    main()


# ============================================================================
# TESTING CHECKLIST
# ============================================================================
#
# [ ] PyAudio installs without errors
#     If fails on Mac: brew install portaudio, then pip install pyaudio
# [ ] Microphone permission granted in System Preferences
# [ ] Calibration shows reasonable volume levels (not all 0 or all maxed)
# [ ] Speaking triggers recording (you see "Recording..." message)
# [ ] Silence stops recording after 2 seconds
# [ ] Transcription appears and is reasonably accurate
# [ ] Can do multiple transcriptions in a row without restart
# [ ] Ctrl+C cleanly exits without errors
# [ ] No "device busy" errors when restarting
#
# COMMON ISSUES & SOLUTIONS:
# - PyAudio won't install: Run "brew install portaudio" first
# - No transcription appears: Volume threshold might be too high, calibrate again
# - Transcription is gibberish: Try "small" model instead of "base"
# - Very slow (>10 seconds): Normal on Mac CPU, stick with "tiny" or "base"
# - Picks up background noise: Increase volume threshold
# - Doesn't trigger on speech: Decrease volume threshold
# - Model download fails: Check internet connection, try again
#
# TUNING TIPS:
# - Start with "base" model (good balance of speed and accuracy)
# - Adjust volume_threshold based on your environment
# - Use calibrate_threshold() function to find good values
# - Silence_duration of 2.0 seconds works well for most speech
# - For noisy environments, increase threshold significantly
# - For quiet speech, decrease threshold and use larger model
#
# EXPECTED PERFORMANCE:
# - "tiny" model: ~2-5 seconds per transcription on Mac
# - "base" model: ~3-7 seconds per transcription on Mac
# - "small" model: ~5-15 seconds per transcription on Mac
# - On PC with GPU: All models 2-5x faster
#
# WHAT THIS GIVES US:
# - Accurate speech-to-text conversion
# - Automatic speech detection (no button needed)
# - Local processing (no internet required)
# - Foundation for full "speak your wish" interaction
#
# NEXT STEPS:
# Once this works reliably, move to Phase 3 (Body Tracking) to detect
# where the user is, then Phase 4 (AI Generation) to create clothing
# from the transcribed speech!
# ============================================================================