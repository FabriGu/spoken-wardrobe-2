import cv2
import numpy as np
import time
from tf_bodypix.api import download_model, load_model, BodyPixModelPaths

def main():
    print("Starting BodyPix test...")
    
    # Load model once at startup (MobileNet 50 for speed)
    print("Downloading/loading BodyPix model...")
    bodypix_model = load_model(download_model(
        BodyPixModelPaths.MOBILENET_FLOAT_50_STRIDE_16
    ))
    print("Model loaded successfully!")

    # Initialize camera using same pattern as your working code
    print("Opening camera...")
    cap = cv2.VideoCapture(0)
    
    if not cap.isOpened():
        raise Exception("Could not open camera 0. "
                       "Check if camera is connected and not in use")
    
    # Set camera properties
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    cap.set(cv2.CAP_PROP_FPS, 30)
    
    # Get actual camera properties
    actualWidth = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
    actualHeight = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
    actualFps = cap.get(cv2.CAP_PROP_FPS)
    
    print(f"Camera opened successfully!")
    print(f"Resolution: {int(actualWidth)}x{int(actualHeight)}")
    print(f"Camera FPS setting: {actualFps}")
    
    # Create windows
    cv2.namedWindow('Original', cv2.WINDOW_NORMAL)
    cv2.namedWindow('Body Parts', cv2.WINDOW_NORMAL)
    cv2.namedWindow('Custom Masks', cv2.WINDOW_NORMAL)
    
    print("\nStarting video display...")
    print("Press 'q' to quit")
    
    # FPS tracking
    fps = 0
    frameCount = 0
    fpsStartTime = time.time()
    
    isRunning = True
    
    while isRunning:
        # Read frame using same pattern as your working code
        ret, frame = cap.read()
        
        if not ret:
            print("Warning: Failed to read frame from camera, retrying...")
            continue
        
        # Update FPS counter
        frameCount += 1
        currentTime = time.time()
        elapsed = currentTime - fpsStartTime
        if elapsed >= 1.0:
            fps = frameCount / elapsed
            print(f"Processing at {fps:.1f} FPS")
            frameCount = 0
            fpsStartTime = currentTime
        
        print("Running BodyPix segmentation...", end='\r')
        
        # Run BodyPix segmentation
        result = bodypix_model.predict_single(frame)
        
        # Get binary person mask
        mask = result.get_mask(threshold=0.75)
        
        # Get colored visualization - convert to numpy and ensure uint8
        colored_mask = result.get_colored_part_mask(mask)
        # Convert tensor to numpy if needed and ensure it's uint8
        if hasattr(colored_mask, 'numpy'):
            colored_mask = colored_mask.numpy()
        colored_mask = colored_mask.astype(np.uint8)
        
        # Extract specific body parts
        left_arm_mask = result.get_part_mask(
            mask,
            part_names=[
                'left_upper_arm_front', 'left_upper_arm_back',
                'left_lower_arm_front', 'left_lower_arm_back'
            ]
        )
        
        torso_mask = result.get_part_mask(
            mask,
            part_names=['torso_front', 'torso_back']
        )
        
        # Convert masks to numpy arrays and squeeze to 2D
        if hasattr(left_arm_mask, 'numpy'):
            left_arm_mask = left_arm_mask.numpy()
        left_arm_mask = np.squeeze(left_arm_mask)  # Remove extra dimensions
        
        if hasattr(torso_mask, 'numpy'):
            torso_mask = torso_mask.numpy()
        torso_mask = np.squeeze(torso_mask)  # Remove extra dimensions
        
        # Apply different colors to different parts
        output = frame.copy()
        
        # Use proper boolean indexing for 2D mask on 3D image
        # Create the boolean mask and apply to each channel
        left_arm_bool = left_arm_mask > 0
        torso_bool = torso_mask > 0
        
        output[left_arm_bool] = [0, 255, 0]  # Green for left arm
        output[torso_bool] = [255, 0, 0]  # Blue for torso
        
        # Display frames
        cv2.imshow('Original', frame)
        cv2.imshow('Body Parts', colored_mask)
        cv2.imshow('Custom Masks', output)
        
        # Handle key presses
        key = cv2.waitKey(1) & 0xFF
        
        if key == ord('q') or key == ord('Q'):
            print("\nQuitting...")
            isRunning = False
            break
    
    # Cleanup
    if cap is not None:
        cap.release()
        print("Camera released")
    
    cv2.destroyAllWindows()
    print("Test complete!")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"\nError occurred: {e}")
        import traceback
        traceback.print_exc()
        print("\nTROUBLESHOOTING:")
        print("1. Check camera permissions in System Preferences > Security & Privacy")
        print("2. Make sure no other app is using the camera")
        print("3. Try a different camera index (1, 2, etc.) if you have multiple cameras")
        print("4. The GPU warning is normal on macOS - it will use CPU instead")