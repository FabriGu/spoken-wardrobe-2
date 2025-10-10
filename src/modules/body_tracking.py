import cv2
# import mediapipe as mp
import numpy as np
import time
from tf_bodypix.api import download_model, load_model, BodyPixModelPaths



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

    # Define preset configurations for different clothing types
    # These are the combinations you'll use most often
    #made by claude
    CLOTHING_PRESETS = {
        # Configuration 1: Full body except head (for full dresses, jumpsuits)
        'full_body': [
            'torso_front', 'torso_back',
            'left_upper_arm_front', 'left_upper_arm_back',
            'left_lower_arm_front', 'left_lower_arm_back',
            'right_upper_arm_front', 'right_upper_arm_back',
            'right_lower_arm_front', 'right_lower_arm_back',
            'left_hand', 'right_hand',
            'left_upper_leg_front', 'left_upper_leg_back',
            'left_lower_leg_front', 'left_lower_leg_back',
            'right_upper_leg_front', 'right_upper_leg_back',
            'right_lower_leg_front', 'right_lower_leg_back',
            'left_foot', 'right_foot'
        ],
        
        # Configuration 2: Torso + arms (for t-shirts, blouses, jackets)
        'torso_and_arms': [
            'torso_front', 'torso_back',
            'left_upper_arm_front', 'left_upper_arm_back',
            'left_lower_arm_front', 'left_lower_arm_back',
            'right_upper_arm_front', 'right_upper_arm_back',
            'right_lower_arm_front', 'right_lower_arm_back',
            'left_hand', 'right_hand'
        ],
        
        # Configuration 3: Just torso (for vests, corsets, tank tops)
        'torso_only': [
            'torso_front', 'torso_back'
        ],
        
        # Configuration 4: Torso + legs (for dresses that cover legs)
        'torso_and_legs': [
            'torso_front', 'torso_back',
            'left_upper_leg_front', 'left_upper_leg_back',
            'left_lower_leg_front', 'left_lower_leg_back',
            'right_upper_leg_front', 'right_upper_leg_back',
            'right_lower_leg_front', 'right_lower_leg_back',
            'left_foot', 'right_foot'
        ],
        
        # Configuration 5: Just legs (for pants, skirts)
        'legs_only': [
            'left_upper_leg_front', 'left_upper_leg_back',
            'left_lower_leg_front', 'left_lower_leg_back',
            'right_upper_leg_front', 'right_upper_leg_back',
            'right_lower_leg_front', 'right_lower_leg_back',
            'left_foot', 'right_foot'
        ]
    }

    def __init__(self, model_type = 'mobilenet_50'):
        #initialize the body segmenter 

        """
        MODEL SPEED COMPARISON (from claude):
        MobileNet 50: ~25-30 FPS on decent CPU (good for real-time)
        MobileNet 75: ~20-25 FPS
        MobileNet 100: ~15-20 FPS
        ResNet50: ~5-10 FPS (too slow for video, but most accurate)
        """


        self.model_type = model_type 
        self.model = None

        #mask confugiration preset 
        self.current_preset = 'torso_and_arms' 

        #store results 
        self.latest_mask = None 
        self.latest_colored_visualization = None

        #performance 
        self.processing_time_ms = 0

        print("initiaized body tracker ")
        print(self.CLOTHING_PRESETS[self.current_preset])

    def load_model(self):
        #load the actual model)
        #first run will download the model 

        print(f"\nLoading BodyPix model ({self.model_type})...")

        start_time = time.time()

        # Map our friendly names to BodyPix model paths (done by claude)
        model_map = {
            'mobilenet_50': BodyPixModelPaths.MOBILENET_FLOAT_50_STRIDE_16,
            'mobilenet_75': BodyPixModelPaths.MOBILENET_FLOAT_75_STRIDE_16,
            'mobilenet_100': BodyPixModelPaths.MOBILENET_FLOAT_100_STRIDE_16,
            # 'resnet50': BodyPixModelPaths.RESNET50_FLOAT_STRIDE_16
        }

        model_path = model_map.get(self.model_type, model_map['mobilenet_50'])

        #download the model 
        self.model = load_model(download_model(model_path))

        load_time = time.time() - start_time 
        print(f"Model loaded in {load_time:.1f} seconds!\n")

    def process_frame(self, frame, preset=None):
        if self.model is None:
            raise Exception("model not loaded")
        
        if preset is None:
            self.current_preset = preset

        start_time = time.time()

        #run bodypix 
        result = self.model.predict_single(frame)

        #get back binary person mask (separate person from background)
        #threshold is .75 -> 75% sure pixel is.  part of person
        person_mask = result.get_mask(threshold = 0.75)

        #get mask for specific body parts 
        part_names = self.CLOTHING_PRESETS[self.current_preset]
        clothing_mask = result.get_part_mask(person_mask, part_names = part_names)

        #convert tensor to numpy array and clean it up 
        if hasattr(clothing_mask, 'numpy)'):
            clothing_mask = clothing_mask.numpy()

        #remove extra dimensions and convert to binary 0/255
        clothing_mask = np.squeeze(clothing_mask) #remove single dimensions
        clothing_mask = (clothing_mask > 0).astype(np.uint8) *255

        #get colored vis 
        colored_viz = result.get_colored_part_mask(person_mask)
        if hasattr(colored_viz, 'numpy'):
            colored_viz = colored_viz.numpy()
        colored_viz = colored_viz.astype(np.uint8)

        #store results 
        self.latest_mask = clothing_mask
        self.latest_colored_visualization = colored_viz

        # Track performance
        self.processing_time_ms = (time.time() - start_time) * 1000
        
        return clothing_mask, colored_viz
    
    def get_mask_for_inpainting(self,frame, preset=None):
        #get raw mask
        mask, _ =self.process_frame(frame, preset)

        #claude mask cleanup 
        #close small holes 
        kernel_close = np.ones((5,5,), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel_close, iterations=2)

        #remove small noise 
        kernel_open = np.ones((3, 3), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel_open, iterations=1)

        #smooth edges
        mask = cv2.GaussianBlur(mask, (5,5), 0)

        #after blur convert back to pure 0/255
        mask = (mask>127).astype(np.uint8) *255

        return mask

    def visualize_mask_overlay(self, frame, mask, color=[0,255,0], alpha=0.6):

        #overlay mask on top of original frame 
        overlay = frame.copy()

        mask_bool = mask > 0

        # overlay[mask>0] = color
        overlay[mask_bool] = overlay[mask_bool] * (1 - alpha) + np.array(color) * alpha
        
        return overlay.astype(np.uint8)

    def set_preset(self, preset_name):
        if preset_name in self.CLOTHING_PRESETS:
            self.current_preset = preset_name
            print(f"preset changed to: {preset_name}")
        else:
            raise ValueError(f"Invalid preset name: {preset_name}. Available presets: {list(self.CLOTHING_PRESETS.keys())}")
        
    def get_available_presets(self):
        return list(self.CLOTHING_PRESETS.keys())
    
    def is_person_detected(self):
        return self.latest_mask is not None and np.any(self.latest_mask > 0)

        

def main():
    # Initialize video capture
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    
    if not cap.isOpened():
        print("Error: Could not open camera")
        return
    
    # Initialize segmenter
    segmenter = BodySegmenter(model_type='mobilenet_50')
    segmenter.load_model()
    segmenter.set_preset('full_body')
    
    # Display settings
    show_visualization = False
    save_counter = 0
    
    # FPS tracking
    fps = 0
    frame_count = 0
    fps_start_time = time.time()
    
    print("\nStarting body part segmentation...")
    print(f"Current preset: {segmenter.current_preset}")
    
    try:
        while True:
            # Read frame
            ret, frame = cap.read()
            if not ret:
                print("Failed to read frame")
                continue
            
            # Mirror the frame (feels more natural)
            frame = cv2.flip(frame, 1)
            
            # Get mask for inpainting
            mask = segmenter.get_mask_for_inpainting(frame, preset=segmenter.current_preset)
            
            # Create visualization
            if show_visualization:
                # Show the colored body parts
                display_frame = segmenter.latest_colored_visualization
            else:
                # Show mask overlay on original frame
                display_frame = segmenter.visualize_mask_overlay(
                    frame, mask, color=[255, 0, 255], alpha=0.5
                )
            
            # Add info text
            if segmenter.is_person_detected():
                status = f"Person detected | Preset: {segmenter.current_preset}"
                color = (0, 255, 0)
            else:
                status = "No person detected"
                color = (0, 0, 255)
            
            cv2.putText(display_frame, status, (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
            
            # Show processing time
            cv2.putText(display_frame, 
                       f"Processing: {segmenter.processing_time_ms:.1f}ms",
                       (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
            
            # Calculate and show FPS
            frame_count += 1
            elapsed = time.time() - fps_start_time
            if elapsed >= 1.0:
                fps = frame_count / elapsed
                frame_count = 0
                fps_start_time = time.time()
            
            cv2.putText(display_frame, f"FPS: {fps:.1f}",
                       (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
            
            # Show the mask in a separate window
            cv2.imshow("Mask (White = Clothing Area)", mask)
            cv2.imshow("Body Part Segmentation", display_frame)
            
            # Handle key presses
            key = cv2.waitKey(1) & 0xFF
            
            if key == ord('q') or key == ord('Q'):
                break
            
            elif key == ord('1'):
                segmenter.set_preset('full_body')
            
            elif key == ord('2'):
                segmenter.set_preset('torso_and_arms')
            
            elif key == ord('3'):
                segmenter.set_preset('torso_only')
            
            elif key == ord('4'):
                segmenter.set_preset('torso_and_legs')
            
            elif key == ord('5'):
                segmenter.set_preset('legs_only')
            
            elif key == ord('v') or key == ord('V'):
                show_visualization = not show_visualization
                mode = "colored parts" if show_visualization else "mask overlay"
                print(f"Visualization: {mode}")
            
            elif key == ord('s') or key == ord('S'):
                # Save current mask
                if segmenter.is_person_detected():
                    filename = f"mask_{segmenter.current_preset}_{save_counter}.png"
                    cv2.imwrite(filename, mask)
                    print(f"Saved: {filename}")
                    # save original frame too
                    orig_filename = f"frame_{segmenter.current_preset}_{save_counter}.png"
                    cv2.imwrite(orig_filename, frame)
                    print(f"Saved: {orig_filename}")
                    save_counter += 1
    
    except KeyboardInterrupt:
        print("\nInterrupted by user")
    
    finally:
        # Cleanup
        print("\nCleaning up...")
        cap.release()
        cv2.destroyAllWindows()
        print("Test complete!")


if __name__ == "__main__":
    main()


   