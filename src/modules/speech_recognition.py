import pyaudio  # For capturing audio from microphone
import wave  # For saving audio files (useful for debugging)
import numpy as np  # For audio processing and analysis
import threading  # For running audio capture in background
import queue  # For thread-safe communication between capture and processing
import time
from transformers import WhisperProcessor, WhisperForConditionalGeneration
import torch  # For running the AI model

class SpeechRecognizer:
    def __init__(self, model_size="base", volume_threshold=500, silence_duration=2.0):
        # MODEL SIZES COMPARISON:
        # - tiny: 39M params, ~1GB RAM, fastest, good for testing
        # - base: 74M params, ~1GB RAM, good balance (RECOMMENDED FOR START)
        # - small: 244M params, ~2GB RAM, better accuracy, slower
        # - medium: 769M params, ~5GB RAM, very accurate, quite slow on Mac
        # - large: 1550M params, ~10GB RAM, best accuracy, very slow
        
        self.model_size = model_size
        self.volume_threshold = volume_threshold
        self.silence_duration = silence_duration

        #audio setting recommended by claude
        self.CHUNK = 1024  # How many samples to read at once (affects latency)
        self.FORMAT = pyaudio.paInt16  # 16-bit audio (standard for speech)
        self.CHANNELS = 1  # Mono audio (stereo not needed for speech)
        self.RATE = 16000  # 16kHz sample rate (Whisper's preferred rate)

        #PyAudio objects (initialized when start)
        self.audio = None
        self.stream = None

        #recording staes
        self.is_listening = False
        self.is_recording = False
        self.audio_buffer = [] #stores the recorded audio chunks

        #threading for background audio capture
        self.audio_thread = None
        self.transcription_queue = queue.Queue() #thread safe

        #load whisper model
        self.processesor = None
        self.model = None
        self.device = None

        print("SpeechRecognizer initialized")
        print(f"Model size: {model_size}")
        print(f"Volume threshold: {volume_threshold}")
        print(f"Silence duration: {silence_duration}s")

    def load_whisper_model(self):
        print(torch.backends.mps.is_available())

        self.device = "mps"

        model_name = f"openai/whisper-{self.model_size}"
        print(f"Loading from : {model_name}")

        #processor for audop
        #converts from raw audio to format whisper expects
        self.processor = WhisperProcessor.from_pretrained(model_name)

        # load the model itself 
        self.model = WhisperForConditionalGeneration.from_pretrained(model_name).to(self.device)

        self.model.eval()  # Set model to evaluation mode (do not want to train)

        print("whisper model has loaded!")

        return
    
    def initialize_audio(self):
        #connect to computer microphoe and prepare to capture audio

        #PyAudio instance
        self.audio = pyaudio.PyAudio()

        #print for debugging taken from claude
        print("\nAvailable audio input devices:")
        for i in range(self.audio.get_device_count()):
            device_info = self.audio.get_device_info_by_index(i)
            # Only show devices that can record (have input channels)
            if device_info['maxInputChannels'] > 0:
                print(f"  {i}: {device_info['name']}")


        try:
            #open audio stream 
            self.stream = self.audio.open(
                format=self.FORMAT,  # 16-bit integers
                channels=self.CHANNELS,  # Mono
                rate=self.RATE,  # 16kHz sample rate
                input=True,  # We're capturing input (not playing output)
                frames_per_buffer=self.CHUNK,  # Read in chunks of 1024 samples
                stream_callback=None,  # We'll read manually for more control
                input_device_index=3
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
        
    def calculate_volume(self, audioData):
        #calculate volume loudness to see if anyone is actually speaking

        #convert raw bytes to numpy array of 16 bit integers 
        #Pyaudio gives bytes we need numbers 
        audio_array = np.frombuffer(audioData, dtype=np.int16)

        #causing integer overflow when i spoke so convert to float first 
        audio_float = audio_array.astype(np.float32) 

        #calculate root means square volume 
        #rms is square root fo the average of squared values 
        volume = np.sqrt(np.mean(audio_float**2))

        return volume
    
    def listenLoop(self):
        #reads audio, calcualtes volume, decides if recording, stores audio if recording, processes audio when done recording
        #runs in sa separate thread to not block video feed

        silenceStartTime = None #to track silence

        while self.is_listening:
            try: 
                #read chunk from microphone
                # exception_on_overflow=False prevents crashes from buffer overflow
                audioData = self.stream.read(self.CHUNK, exception_on_overflow=False)
                
                #calculate loudness 
                volume = self.calculate_volume(audioData)

                if not self.is_recording:
                    #if not already recording check if we start

                    if volume > self.volume_threshold:
                        print(f"speech detected at volume {volume:.0f}")

                        self.is_recording = True
                        self.audio_buffer = []
                        self.audio_buffer.append(audioData)
                        silenceStartTime = None #reset

                    else:
                        #still quet 
                        #print debug message every 5 sec
                        if int(time.time() * 2 ) % 10 ==0:
                            print(f"volume: {volume:.0f} (threshold: {self.volume_threshold})")

                else:
                    #are recording add to buffer
                    self.audio_buffer.append(audioData)

                    if volume > self.volume_threshold:
                        #still hear speech
                        silenceStartTime = None
                        print(f"recording ... (volume: {volume:.0f})", end='\r')
                    else:
                        # First time detecting silence - record when it started
                        if silenceStartTime is None:
                            silenceStartTime = time.time()

                        #check quiet length
                        silence_duration = time.time() - silenceStartTime

                        if silence_duration >= self.silence_duration:
                            print(f"silence detected after {silence_duration:.1f}")

                            self.is_recording = False

                            self.processAudioBuffer()

                            self.audio_buffer = [] #clear buffer
                            silenceStartTime = None

                            print("waiting for next speech")

            except Exception as e:
                print(f"\nError in listen loop: {e}")
                continue

    def processAudioBuffer(self):
        #convert recording to text using Whisper

        if not self.audio_buffer:
            print("No audio to process")
            return
        
        print(f"\nProcessing {len(self.audio_buffer)} audio chunks...")
        startTime = time.time()

        try:
            #combine chunks into single byte array
            audioData = b''.join(self.audio_buffer)
            audio_array = np.frombuffer(audioData, dtype=np.int16)

            # Convert to float32 in range [-1.0, 1.0]
            # Whisper expects audio normalized to this range
            audio_float = audio_array.astype(np.float32) / 32768.0

            #prepare input for whisper model
            inputs = self.processor(
                audio_float,
                sampling_rate=self.RATE,
                return_tensors="pt"  # Return PyTorch tensors
            )

            #move to correct device
            inputs = inputs.input_features.to(self.device)
            
            # Generate transcription
            # We use torch.no_grad() because we're not training (saves memory)
            with torch.no_grad():
                predictedIds = self.model.generate(inputs)

            #generate transcription (greedy decoding)
            # The model outputs token IDs, we need to convert them to words
            transcription = self.processor.batch_decode(
                predictedIds,
                skip_special_tokens=True  # Remove <start>, <end>, etc.
            )[0]

            #clear up transcription
            transcription = transcription.strip()

            elapsedTime = time.time() - startTime

            if transcription:
                print(f"\n{'='*60}")
                print(f"✓ TRANSCRIPTION: {transcription}")
                print(f"  (processed in {elapsedTime:.1f} seconds)")
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
        #begin speech recognition system
        if self.model is None:
            self.load_whisper_model()
        
        if self.stream is None:
            if not self.initialize_audio():
                return False
            
        #start listening to in background thread
        self.is_listening = True
        self.audio_thread = threading.Thread(target=self.listenLoop, daemon=True)
        self.audio_thread.start()

        print("✓ Speech recognition started")
        return True
    
    def stop(self):
        #stop speech recognition system and clean up
        print("Stopping speech recognition...")

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
        return #maybe not needed?
    
    def getTranscription(self, block = True, timeout = None):
        #get the next transcription if available

        #This is how the main application retrieves transcriptions from the background thread.

        try:
            return self.transcription_queue.get(block=block, timeout=timeout)
        except queue.Empty:
            return None
        
    def calibrateThreshold(self, duration=5):
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


def main():
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
        recognizer.calibrateThreshold(duration=5)
        
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