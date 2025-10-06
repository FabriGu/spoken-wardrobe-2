"""
PHASE 8: MAIN APPLICATION - COMPLETE INTEGRATION
=================================================

PURPOSE: Bring together all components (video, speech, body tracking, AI generation,
and compositing) into one cohesive "Spoken Wardrobe" application.

This is the compilation pseudocode that orchestrates everything!

LEARNING RESOURCES:
- State Machines: https://python-3-patterns-idioms-test.readthedocs.io/en/latest/StateMachine.html
- Threading: https://docs.python.org/3/library/threading.html
- Multiprocessing: https://docs.python.org/3/library/multiprocessing.html
- Queue for thread communication: https://docs.python.org/3/library/queue.html

WHAT YOU'RE BUILDING:
The complete Spoken Wardrobe experience where users:
1. Stand in front of the mirror
2. Speak their wish (e.g., "flames")
3. See their words floating around them
4. Watch as AI generates custom clothing
5. See themselves wearing it for 15-20 seconds
6. System resets and waits for next person
"""

import cv2
import numpy as np
from PIL import Image
import threading
import multiprocessing
import queue
import time
from enum import Enum
import sys

# Import all our components from previous phases
# Make sure all phase files are in the src/modules/ folder
try:
    from modules.video_capture import VideoCapture
    from modules.speech_recognition import SpeechRecognizer
    from modules.body_tracking import BodyTracker
    from modules.ai_generation import ClothingGenerator
    from modules.compositing import ClothingCompositor
    # Phase 6 (text effects) would be imported here when complete
    # from modules.phase6_text_effects import TextEffectRenderer
except ImportError as e:
    print(f"Import error: {e}")
    print("\nMake sure all phase modules are in src/modules/ folder")
    print("Run each phase individually first to test them!")
    sys.exit(1)


# ============================================================================
# STATE MACHINE DEFINITION
# ============================================================================

class AppState(Enum):
    """
    The different states our application can be in.
    
    Think of this like a flowchart where the app moves from one state
    to another based on events (speech detected, generation complete, etc.)
    """
    IDLE = "idle"                 # Waiting for someone to speak
    LISTENING = "listening"        # Recording speech
    GENERATING = "generating"      # AI is creating clothing
    DISPLAYING = "displaying"      # Showing the generated clothing
    FADING_OUT = "fading_out"     # Removing clothing before reset


class SpokenWardrobeApp:
    """
    The main application class that orchestrates all components.
    
    ARCHITECTURE:
    - Main thread: Handles video display (must stay at 30fps)
    - Audio thread: Runs speech recognition (in SpeechRecognizer)
    - Generation process: Runs Stable Diffusion (multiprocessing)
    - Queues: For thread-safe communication between components
    
    WHY THIS STRUCTURE?
    Stable Diffusion is SLOW (20-40 seconds). If we ran it in the main thread,
    the video would freeze. By running it in a separate process, the video
    stays smooth while AI generates in the background.
    """
    
    def __init__(self, config=None):
        """
        Initialize the complete application.
        
        PARAMETERS:
        - config: Dictionary of configuration options (optional)
        
        CONFIG OPTIONS:
        {
            'camera_index': 0,
            'target_fps': 30,
            'speech_model': 'base',
            'volume_threshold': 500,
            'sd_model': 'runwayml/stable-diffusion-v1-5',
            'blend_alpha': 0.8,
            'display_duration': 15.0,  # How long to show clothing
            'fullscreen': False
        }
        """
        
        # Load configuration
        self.config = config or {}
        self.camera_index = self.config.get('camera_index', 0)
        self.target_fps = self.config.get('target_fps', 30)
        self.display_duration = self.config.get('display_duration', 15.0)
        self.fullscreen = self.config.get('fullscreen', False)
        
        # Application state
        self.current_state = AppState.IDLE
        self.state_start_time = None
        
        # Component initialization flags
        self.components_loaded = False
        
        # Video capture (Phase 1)
        self.video_capture = None
        
        # Speech recognition (Phase 2)
        self.speech_recognizer = None
        self.latest_transcription = None
        
        # Body tracking (Phase 3)
        self.body_tracker = None
        
        # AI generation (Phase 4) - runs in separate process
        self.generation_queue = multiprocessing.Queue()  # Send requests
        self.result_queue = multiprocessing.Queue()  # Receive results
        self.generation_process = None
        
        # Compositing (Phase 5)
        self.compositor = None
        
        # Text effects (Phase 6) - would be initialized here
        # self.text_renderer = None
        
        # Application control
        self.running = False
        self.display_start_time = None
        
        print("=" * 60)
        print("SPOKEN WARDROBE V2 - MAIN APPLICATION")
        print("=" * 60)
    
    
    def initialize_components(self):
        """
        Initialize all the components we need.
        
        This is done separately from __init__ so we can show loading progress
        and handle any initialization errors gracefully.
        """
        
        print("\nInitializing components...")
        print("This may take a few minutes on first run (model downloads)")
        print("-" * 60)
        
        try:
            # 1. Initialize video capture
            print("\n[1/5] Initializing video capture...")
            self.video_capture = VideoCapture(
                camera_index=self.camera_index,
                target_fps=self.target_fps
            )
            self.video_capture.start()
            print("✓ Video capture ready")
            
            # 2. Initialize body tracker
            print("\n[2/5] Initializing body tracker...")
            self.body_tracker = BodyTracker(
                min_detection_confidence=0.5,
                min_tracking_confidence=0.5
            )
            print("✓ Body tracker ready")
            
            # 3. Initialize speech recognizer
            print("\n[3/5] Initializing speech recognition...")
            self.speech_recognizer = SpeechRecognizer(
                model_size=self.config.get('speech_model', 'base'),
                volume_threshold=self.config.get('volume_threshold', 500),
                silence_duration=2.0
            )
            self.speech_recognizer.start()
            print("✓ Speech recognition ready")
            
            # 4. Initialize compositor
            print("\n[4/5] Initializing compositor...")
            self.compositor = ClothingCompositor(
                blend_alpha=self.config.get('blend_alpha', 0.8)
            )
            print("✓ Compositor ready")
            
            # 5. Start AI generation process
            print("\n[5/5] Starting AI generation process...")
            self.start_generation_process()
            print("✓ AI generator ready")
            
            print("\n" + "=" * 60)
            print("ALL COMPONENTS INITIALIZED SUCCESSFULLY!")
            print("=" * 60)
            
            self.components_loaded = True
            return True
            
        except Exception as e:
            print(f"\n✗ Error during initialization: {e}")
            print("\nTROUBLESHOOTING:")
            print("- Make sure all phase modules are tested individually")
            print("- Check camera permissions")
            print("- Check microphone permissions")
            print("- Ensure you have enough disk space for models (~5GB)")
            return False
    
    
    def start_generation_process(self):
        """
        Start the AI generation process.
        
        This runs in a completely separate process so it doesn't block
        the main video thread. It waits for generation requests on the queue,
        generates clothing, and sends results back.
        """
        
        # Define the worker function that runs in the separate process
        def generation_worker(request_queue, result_queue, config):
            """
            Worker function for AI generation process.
            
            This runs in its own process and continuously:
            1. Waits for generation requests
            2. Generates clothing with Stable Diffusion
            3. Sends result back through queue
            """
            
            # Import here (in the worker process)
            from modules.ai_generation import ClothingGenerator
            import torch
            
            print("[Generation Worker] Starting up...")
            
            # Initialize generator in this process
            generator = ClothingGenerator(
                model_id=config.get('sd_model', 'runwayml/stable-diffusion-v1-5')
            )
            generator.load_model()
            
            print("[Generation Worker] Ready to generate!")
            
            while True:
                try:
                    # Wait for a generation request
                    # Request format: (prompt_text, request_id)
                    request = request_queue.get(timeout=1.0)
                    
                    if request == "STOP":
                        print("[Generation Worker] Stopping...")
                        break
                    
                    prompt_text, request_id = request
                    print(f"[Generation Worker] Generating: '{prompt_text}'")
                    
                    # Generate the clothing
                    clothing_image = generator.generate_clothing_from_text(prompt_text)
                    
                    # Send result back
                    # Result format: (request_id, clothing_image_as_bytes)
                    if clothing_image is not None:
                        # Convert PIL Image to bytes for queue transfer
                        import io
                        img_byte_arr = io.BytesIO()
                        clothing_image.save(img_byte_arr, format='PNG')
                        img_bytes = img_byte_arr.getvalue()
                        
                        result_queue.put((request_id, img_bytes))
                        print(f"[Generation Worker] Completed: '{prompt_text}'")
                    else:
                        result_queue.put((request_id, None))
                        print(f"[Generation Worker] Failed: '{prompt_text}'")
                
                except queue.Empty:
                    # No request yet, keep waiting
                    continue
                except Exception as e:
                    print(f"[Generation Worker] Error: {e}")
                    continue
        
        # Start the worker process
        self.generation_process = multiprocessing.Process(
            target=generation_worker,
            args=(self.generation_queue, self.result_queue, self.config),
            daemon=True
        )
        self.generation_process.start()
    
    
    def change_state(self, new_state):
        """
        Change to a new application state.
        
        This handles all the transitions between states and triggers
        appropriate actions for each state.
        """
        
        print(f"\nState: {self.current_state.value} → {new_state.value}")
        
        self.current_state = new_state
        self.state_start_time = time.time()
        
        # State-specific actions
        if new_state == AppState.IDLE:
            # Reset everything
            self.latest_transcription = None
            # Could show "Speak your wish" text here
            
        elif new_state == AppState.LISTENING:
            # Speech detection already handles recording
            # We just visualize it with floating text (Phase 6)
            pass
        
        elif new_state == AppState.GENERATING:
            # Send generation request
            if self.latest_transcription:
                request_id = time.time()  # Use timestamp as ID
                self.generation_queue.put((self.latest_transcription, request_id))
                print(f"Sent generation request: '{self.latest_transcription}'")
        
        elif new_state == AppState.DISPLAYING:
            # Clothing should already be set by compositor
            self.display_start_time = time.time()
        
        elif new_state == AppState.FADING_OUT:
            # Start fade out animation
            if self.compositor:
                self.compositor.start_fade_out()
    
    
    def update_state_machine(self):
        """
        Update the state machine based on current conditions.
        
        This is called every frame and decides if we should transition
        to a new state based on events and timeouts.
        """
        
        # Check for new transcription from speech recognizer
        if self.speech_recognizer:
            new_transcription = self.speech_recognizer.get_transcription(block=False)
            if new_transcription:
                self.latest_transcription = new_transcription
                print(f"Received transcription: '{new_transcription}'")
                
                # If we're in IDLE or LISTENING, move to GENERATING
                if self.current_state in [AppState.IDLE, AppState.LISTENING]:
                    self.change_state(AppState.GENERATING)
        
        # Check for generation results
        try:
            result = self.result_queue.get_nowait()
            request_id, img_bytes = result
            
            if img_bytes is not None:
                # Convert bytes back to PIL Image
                import io
                clothing_image = Image.open(io.BytesIO(img_bytes))
                
                # Set the clothing in compositor
                if self.compositor:
                    self.compositor.set_clothing(clothing_image)
                
                # Move to DISPLAYING state
                self.change_state(AppState.DISPLAYING)
            else:
                print("Generation failed, returning to IDLE")
                self.change_state(AppState.IDLE)
        
        except queue.Empty:
            # No result yet
            pass
        
        # State-specific timeout checks
        if self.current_state == AppState.DISPLAYING:
            # Check if display duration is over
            if time.time() - self.display_start_time >= self.display_duration:
                print("Display duration complete")
                self.change_state(AppState.FADING_OUT)
        
        elif self.current_state == AppState.FADING_OUT:
            # Check if fade out is complete
            if self.compositor and not self.compositor.has_clothing():
                print("Fade out complete, returning to IDLE")
                self.change_state(AppState.IDLE)
    
    
    def process_frame(self):
        """
        Process one frame of video.
        
        This is the main loop function that:
        1. Captures video frame
        2. Runs body tracking
        3. Composites clothing if in DISPLAYING state
        4. Adds UI overlays
        5. Returns frame for display
        """
        
        # Capture frame
        success, frame = self.video_capture.read_frame()
        if not success:
            return None
        
        # Mirror the frame
        frame = self.video_capture.flip_horizontal(frame)
        
        # Run body tracking
        results, _ = self.body_tracker.process_frame(frame)
        
        # Get tracking data
        bounding_box = self.body_tracker.latest_bounding_box
        segmentation_mask = self.body_tracker.get_segmentation_mask(frame.shape)
        
        # Composite clothing if we have any
        if self.compositor and self.compositor.has_clothing():
            frame = self.compositor.composite_frame(
                frame,
                bounding_box,
                segmentation_mask
            )
        
        # Add UI overlays based on state
        frame = self.add_ui_overlay(frame)
        
        return frame
    
    
    def add_ui_overlay(self, frame):
        """
        Add UI elements to the frame based on current state.
        
        This shows the user what's happening:
        - IDLE: "Speak your wish"
        - LISTENING: Transcribed text floating around
        - GENERATING: Loading indicator
        - DISPLAYING: Nothing (focus on clothing)
        """
        
        # Add state indicator (for debugging)
        state_text = f"State: {self.current_state.value}"
        cv2.putText(frame, state_text, (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        # State-specific UI
        if self.current_state == AppState.IDLE:
            # Show prompt to speak
            prompt_text = "Speak your wish to conjure clothing"
            text_size = cv2.getTextSize(prompt_text, cv2.FONT_HERSHEY_SIMPLEX, 1.2, 2)[0]
            text_x = (frame.shape[1] - text_size[0]) // 2
            text_y = frame.shape[0] - 50
            
            # Add shadow for readability
            cv2.putText(frame, prompt_text, (text_x+2, text_y+2),
                       cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 0), 3)
            cv2.putText(frame, prompt_text, (text_x, text_y),
                       cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 255), 2)
        
        elif self.current_state == AppState.LISTENING:
            # Show that we're listening
            listen_text = "Listening..."
            cv2.putText(frame, listen_text, (10, 70),
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
            
            # Show transcription if available
            if self.latest_transcription:
                cv2.putText(frame, f"You said: {self.latest_transcription}", (10, 110),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        
        elif self.current_state == AppState.GENERATING:
            # Show loading animation
            gen_text = "Generating magical clothing..."
            elapsed = time.time() - self.state_start_time
            dots = "." * (int(elapsed * 2) % 4)
            full_text = gen_text + dots
            
            text_size = cv2.getTextSize(full_text, cv2.FONT_HERSHEY_SIMPLEX, 1, 2)[0]
            text_x = (frame.shape[1] - text_size[0]) // 2
            text_y = frame.shape[0] // 2
            
            cv2.putText(frame, full_text, (text_x, text_y),
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 200, 100), 2)
            
            # Show what we're generating
            if self.latest_transcription:
                prompt_text = f"'{self.latest_transcription}'"
                text_size2 = cv2.getTextSize(prompt_text, cv2.FONT_HERSHEY_SIMPLEX, 1.2, 2)[0]
                text_x2 = (frame.shape[1] - text_size2[0]) // 2
                cv2.putText(frame, prompt_text, (text_x2, text_y + 50),
                           cv2.FONT_HERSHEY_SIMPLEX, 1.2, (150, 150, 255), 2)
        
        elif self.current_state == AppState.DISPLAYING:
            # Minimal UI during display - let clothing be the focus
            # Could show countdown timer
            remaining = self.display_duration - (time.time() - self.display_start_time)
            if remaining > 3:  # Only show last 3 seconds
                pass
            else:
                countdown = f"{int(remaining) + 1}"
                cv2.putText(frame, countdown, (frame.shape[1] - 80, 80),
                           cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255), 3)
        
        return frame
    
    
    def run(self):
        """
        Main application loop.
        
        This runs continuously until the user quits:
        1. Process one frame
        2. Update state machine
        3. Display the frame
        4. Handle keyboard input
        """
        
        if not self.components_loaded:
            print("Error: Components not initialized. Call initialize_components() first.")
            return
        
        print("\n" + "=" * 60)
        print("SPOKEN WARDROBE V2 IS RUNNING!")
        print("=" * 60)
        print("\nControls:")
        print("- Q: Quit application")
        print("- F: Toggle fullscreen")
        print("- R: Force reset to IDLE state")
        print("- D: Toggle debug visualization")
        print("=" * 60)
        
        # Create window
        window_name = "Spoken Wardrobe V2"
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
        
        if self.fullscreen:
            cv2.setWindowProperty(window_name, cv2.WND_PROP_FULLSCREEN,
                                cv2.WINDOW_FULLSCREEN)
        
        # Set initial state
        self.change_state(AppState.IDLE)
        self.running = True
        
        # Main loop
        try:
            while self.running:
                # Update state machine
                self.update_state_machine()
                
                # Process frame
                frame = self.process_frame()
                
                if frame is None:
                    print("Failed to get frame")
                    continue
                
                # Display
                cv2.imshow(window_name, frame)
                
                # Handle keyboard input
                key = cv2.waitKey(1) & 0xFF
                
                if key == ord('q') or key == ord('Q'):
                    print("\nQuitting...")
                    self.running = False
                
                elif key == ord('f') or key == ord('F'):
                    # Toggle fullscreen
                    self.fullscreen = not self.fullscreen
                    if self.fullscreen:
                        cv2.setWindowProperty(window_name, cv2.WND_PROP_FULLSCREEN,
                                            cv2.WINDOW_FULLSCREEN)
                    else:
                        cv2.setWindowProperty(window_name, cv2.WND_PROP_FULLSCREEN,
                                            cv2.WINDOW_NORMAL)
                
                elif key == ord('r') or key == ord('R'):
                    # Force reset
                    print("Force reset to IDLE")
                    if self.compositor:
                        self.compositor.clear_clothing()
                    self.change_state(AppState.IDLE)
                
                elif key == ord('d') or key == ord('D'):
                    # Toggle debug mode (could show more info)
                    pass
        
        except KeyboardInterrupt:
            print("\nInterrupted by user")
        
        finally:
            self.shutdown()
    
    
    def shutdown(self):
        """
        Clean shutdown of all components.
        
        It's crucial to properly clean up:
        - Release camera
        - Stop speech recognition
        - Terminate generation process
        - Close windows
        """
        
        print("\n" + "=" * 60)
        print("SHUTTING DOWN SPOKEN WARDROBE")
        print("=" * 60)
        
        self.running = False
        
        # Stop video capture
        if self.video_capture:
            print("Stopping video capture...")
            self.video_capture.stop()
        
        # Stop speech recognizer
        if self.speech_recognizer:
            print("Stopping speech recognition...")
            self.speech_recognizer.stop()
        
        # Clean up body tracker
        if self.body_tracker:
            print("Cleaning up body tracker...")
            self.body_tracker.cleanup()
        
        # Stop generation process
        if self.generation_process and self.generation_process.is_alive():
            print("Stopping AI generation process...")
            self.generation_queue.put("STOP")
            self.generation_process.join(timeout=5.0)
            if self.generation_process.is_alive():
                print("Force terminating generation process...")
                self.generation_process.terminate()
        
        # Close windows
        cv2.destroyAllWindows()
        
        print("\n" + "=" * 60)
        print("SHUTDOWN COMPLETE")
        print("=" * 60)


# ============================================================================
# MAIN ENTRY POINT
# ============================================================================

def main():
    """
    Application entry point.
    
    This is what runs when you execute: python src/main.py
    """
    
    # Configuration
    config = {
        'camera_index': 0,
        'target_fps': 30,
        'speech_model': 'base',  # 'tiny', 'base', 'small', or 'medium'
        'volume_threshold': 500,  # Adjust based on your environment
        'sd_model': 'runwayml/stable-diffusion-v1-5',
        'blend_alpha': 0.8,
        'display_duration': 15.0,  # How long to show clothing (seconds)
        'fullscreen': False  # Start in windowed mode
    }
    
    # Create application
    app = SpokenWardrobeApp(config)
    
    # Initialize components
    if not app.initialize_components():
        print("\nFailed to initialize components. Exiting.")
        return 1
    
    # Run the application
    app.run()
    
    return 0


if __name__ == "__main__":
    sys.exit(main())


# ============================================================================
# DEPLOYMENT CHECKLIST
# ============================================================================
#
# BEFORE RUNNING THE FULL APP:
# [ ] All phase tests (1-5) completed successfully
# [ ] Camera permissions granted
# [ ] Microphone permissions granted
# [ ] ~5GB free disk space for models
# [ ] Good lighting in room
# [ ] Quiet environment for speech recognition
#
# FIRST RUN:
# [ ] Will download Whisper model (~300MB)
# [ ] Will download Stable Diffusion model (~4GB)
# [ ] May take 5-10 minutes to load everything
# [ ] Test with simple prompts first: "flames", "roses"
#
# TUNING FOR YOUR SPACE:
# [ ] Adjust volume_threshold for your room noise level
# [ ] Adjust display_duration based on use case
# [ ] Adjust blend_alpha for clothing transparency
# [ ] Consider speech_model size based on accuracy needs
#
# TROUBLESHOOTING:
# - App crashes on start: Check individual phases work first
# - No speech detection: Calibrate volume threshold
# - Slow generation: Normal, expected 20-40 seconds
# - Video stutters: Reduce target_fps or frame size
# - Clothing misaligned: Check body tracking (Phase 3)
#
# PERFORMANCE OPTIMIZATION:
# - On Mac: Use 'base' Whisper, expect 30-40s generation
# - On PC with Nvidia GPU: Can use 'small' Whisper, 10-15s generation
# - Reduce video resolution if frame rate drops
# - Close other applications for more resources
#
# FINAL INSTALLATION:
# - Test thoroughly in windowed mode first
# - Once working, enable fullscreen
# - Set up physical two-way mirror
# - Position camera for full body view
# - Adjust lighting for best results
# - Consider adding physical "Speak Now" button
#
# CONGRATULATIONS!
# You've built a complete AI-powered interactive art installation!
# ============================================================================