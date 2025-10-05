import cv2  # OpenCV - the main library for video capture and image processing
import time  # For calculating frame rate
import threading  # Will be needed later for smooth performance
import numpy as np  # For image array manipulation


class VideoCaptureObject:
    def __init__(self, cameraIndex=0, targetFps = 30):
        self.cameraIndex = cameraIndex # The camera we want to use
        self.targetFps = targetFps # The fps we want

        #initalize the opencv capture object as non 
        #start with none and create it when start capturing
        self.capture = None
        
        #store most recent frame captured
        #start with none since dont have any frames yet
        self.currentFrame = None
        
        #boolean flag to see if capture is running 
        self.isRunning = False

        #threading is a way to run tasks "simultaneously"
        #lock for thread-safe access to currentFrame
        #later multiple parts of code might try to read/write the frame at the same time 
        #lock prevents conflict
        self.frameLock = threading.Lock()

        #variables to monitor performance
        self.fps = 0
        self.frameCount = 0
        self.fpsStartTime = None

        print("VideoCapture initialized")
        print(f"Camera index: {cameraIndex}")
        print(f"Target FPS: {targetFps}")

    def start(self):
        self.capture = cv2.VideoCapture(self.cameraIndex)

        if not self.capture.isOpened():
            raise Exception(f"Could not open camera {self.cameraIndex}. "
                            "check if camera is connected and not in use")
    
        #set camear properties
        #potential place for debugging depending on system camera
        self.capture.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)  # 1280x720 is good HD quality
        self.capture.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        self.capture.set(cv2.CAP_PROP_FPS, self.targetFps)

        # Get actual camera properties (might differ from what we set)
        actualWidth = self.capture.get(cv2.CAP_PROP_FRAME_WIDTH)
        actualHeight = self.capture.get(cv2.CAP_PROP_FRAME_HEIGHT)
        actualFps = self.capture.get(cv2.CAP_PROP_FPS)

        print(f"Camera opened successfully!")
        print(f"Resolution: {int(actualWidth)}x{int(actualHeight)}")
        print(f"Camera FPS setting: {actualFps}")
        
        # Set the running flag to True
        self.isRunning = True
        print(f"it is {self.isRunning} that it is running")
        
        # Initialize FPS timing
        self.fpsStartTime = time.time()
        self.frameCount = 0
        
        return True

    def readFrame(self):
        #ret = return value (True if succsesfully got frame back)
        #frame is a numpy array of height, width, no. of color channels (BGR)
        ret, frame = self.capture.read()

        if not ret:
            print("Warning: Failed to read frame from camera")
            return False, None
        
        #now increment frame count 
        self.frameCount += 1

        #calculate FPS every second 
        currentTime = time.time()
        elapsed = currentTime - self.fpsStartTime
        if elapsed >= 1.0: #update every sec
            self.fps = self.frameCount / elapsed
            self.frameCount = 0
            self.fpsStartTime = currentTime

        return ret, frame
    
    def flipHorizontal(self, frame):
        return
    
    def addFpsOverlay(self, frame):
        #create text with fps data
        fpsText = f"FPS: {self.fps:.1f}"

        #draw text on frame
        #parameters are: image, text, position, font, scale, color, thickness
        cv2.putText(frame,
                    fpsText,
                    (10,30), #position x,y from top-left
                    cv2.FONT_HERSHEY_COMPLEX,
                    1.0, #font scale
                    (0,255,0), #color
                    2) # thickness
        return frame
    
    def display(self, windowName = "Spoken Wardrobe - Mirror"):
        cv2.namedWindow(windowName, cv2.WINDOW_NORMAL)

        print("\n starting video display...")
        print("press q to quite")
        print("press f to toggle fullscreen")

        isFullscreen = False

        while self.isRunning:
            #capture a frame 
            success, frame = self.readFrame()

            if not success:
                print("Fialed to capture frame, retrying")
                continue

            #create mirrored effect
            # Flip the frame horizontally (mirror effect)
            # flipCode = 0 for vertical flip
            # flipCode = 1 for horizontal flip
            # flipCode = -1 for both horizontal and vertical flip
            frameM = cv2.flip(frame, 1)

            #add fps overlay
            frameMF = self.addFpsOverlay(frameM)

            #show frame in the window
            cv2.imshow(windowName, frameMF) #for now is just frame because we have not mirrored or added text over it

            #wait one millisecond (needed for display to update)
            #read key press
            key = cv2.waitKey(1) & 0xFF #some fancy thing that masks out higher order bits so ensures a value between 0 and 255 to comapre with ord()

            if key == ord('q') or key == ord('Q'):
                # User pressed Q - quit the application
                print("\nQuitting...")
                self.isRunning = False
                break

            elif key == ord('f') or key == ord('F'):
                # User pressed F - toggle fullscreen
                if isFullscreen:
                    cv2.setWindowProperty(windowName, cv2.WND_PROP_FULLSCREEN, 
                                        cv2.WINDOW_NORMAL)
                    is_fullscreen = False
                    print("Windowed mode")
                else:
                    cv2.setWindowProperty(windowName, cv2.WND_PROP_FULLSCREEN, 
                                        cv2.WINDOW_FULLSCREEN)
                    isFullscreen = True
                    print("Fullscreen mode")

        self.stop()


    def stop(self):
        # self.isRunning
        if self.capture is not None:
            self.capture.release()
            print("cam released")

        #close openCV windows
        cv2.destroyAllWindows()

    
    
def main():
    try: 
        video = VideoCaptureObject(cameraIndex=0, targetFps=30)
        video.start()
        video.display()
    except Exception as e:
        print(f"\nError occurred: {e}")
        print("\nTROUBLESHOOTING:")
        print("1. Check camera permissions in System Preferences > Security & Privacy")
        print("2. Make sure no other app is using the camera")
        print("3. Try a different camera_index (1, 2, etc.) if you have multiple cameras")

    # video.release()
    # cv2.destroyAllWindows()
    if 'video' in locals():
        video.stop()
        print("\nTest complete!")

if __name__ == "__main__":
    main()
