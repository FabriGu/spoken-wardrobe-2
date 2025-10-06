import cv2
import mediapipe as mp
import numpy as np
import time

keypoints = {
  "LEFT_FACE": 0,
  "RIGHT_FACE": 1,
  "LEFT_UPPER_ARM_FRONT": 2,
  "LEFT_UPPER_ARM_BACK": 3,
  "RIGHT_UPPER_ARM_FRONT": 4,
  "RIGHT_UPPER_ARM_BACK": 5,
  "LEFT_LOWER_ARM_FRONT": 6,
  "LEFT_LOWER_ARM_BACK": 7,
  "RIGHT_LOWER_ARM_FRONT": 8,
  "RIGHT_LOWER_ARM_BACK": 9,
  "LEFT_HAND": 10,
  "RIGHT_HAND": 11,
  "TORSO_FRONT": 12,
  "TORSO_BACK": 13,
  "LEFT_UPPER_LEG_FRONT": 14,
  "LEFT_UPPER_LEG_BACK": 15,
  "RIGHT_UPPER_LEG_FRONT": 16,
  "RIGHT_UPPER_LEG_BACK": 17,
  "LEFT_LOWER_LEG_FRONT": 18,
  "LEFT_LOWER_LEG_BACK": 19,
  "RIGHT_LOWER_LEG_FRONT": 20,
  "RIGHT_LOWER_LEG_BACK": 21,
  "LEFT_FOOT": 22,
  "RIGHT_FOOT": 23
  }

class BodySegmenter:
    #body detection and segmentation using mediaPipe

    def __init__(self, min_detection_confidence=0.5, min_tracking_confidence=0.5):
        #min detection = confidence that mediapipe must reach to detect a person 
            # lower = more sensitive but more false positives
        #min_tracking_confidence = how sure to keep tracking 
            #lower = more stable but may track wrong person 

        self.min_detection_confidence = min_detection_confidence
        self.min_tracking_confidence = min_tracking_confidence

        #initialize mediapipe pose model
            # mp.solutions.pose is the pose detection module 
        self.mp_pose = mp.solutions.pose 
        self.mp_drawing = mp.solutions.drawing_utils #visualization
        self.mp_drawing_styles = mp.solutions.drawing_styles

        #create the pose detector 
        #model complexity: 0 = lite, 1 = full, 2 = heavy 
        #for real time mirror 1 should be good 
        self.pose = self.mp_pose.Pose(
            static_image_mode = False, #video mode for optimized tracking 
            model_complexity = 1,
            smooth_landmarks = True, #smooth jittery aspect
            enable_segmentation = True,
            min_detection_confidence = min_detection_confidence,
            min_tracking_confidence = min_tracking_confidence
        )

        #store the latest detection results 
        self.latest_landmarks = None 
        self.latest_segmentation_mask = None
        # self.latest_bounding_box = None

        self.mask_cache = {}

        #performance track 
        self.processing_time = 0

        print("Body tracking initialized")

    def process_frame(self, frame):
        #frame is inputted using BGR format from OpenCV

        start_time = time.time()

        #Mediapipe expects rgb but openCV uses BGR 
        #convert color space
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        #process frame with mediapipe 
        results = self.pose.process(frame_rgb)

        #calcualte processing itme 
        self.processing_time = (time.time() - start_time) * 1000 #ms conversion included

        #create copy for drawing vis 
        processed_frame = frame.copy()

        #is person there?
        if results.pose_landmarks:
            #person found from here on
            self.latest_landmarks = results.pose_landmarks

            #store segmentation mask 
            if results.segmentation_mask is not None:
                self.latest_segmentation_mask = results.segmentation_mask

            self.mp_drawing.draw_landamrks(
                processed_frame,
                self.latest_landmarks,
                self.mp_pose.POSE_CONNECTIONS,
                landmark_drawing_spec=self.mp_drawing_styles.get_default_pose_landmarks_style()
            )
            
        else:
            #no person
            self.latest_landmarks = None
            self.latest_segmentation_mask = None
            cv2.putText(processed_frame, "No person detected", (50,50), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2)
            
        #add processing time 
        cv2.putText(processed_frame, f"Processing time: {self.processing_time:.1f} ms", (10,30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,0), 2)
            
        return results, processed_frame
        
    def get_full_body_mask_headless(self, frame):
        #return binary mask of full body area (1 = body, 0 = background)
        #headless version without the head area
            #calc bounding box for torso area 