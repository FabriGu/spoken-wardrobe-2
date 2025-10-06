import cv2
import numpy as np
from tf_bodypix.api import download_model, load_model, BodyPixModelPaths

# Load model once at startup (MobileNet 50 for speed)
bodypix_model = load_model(download_model(
    BodyPixModelPaths.MOBILENET_FLOAT_50_STRIDE_16
))

cap = cv2.VideoCapture(0)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    
    # Run BodyPix segmentation
    result = bodypix_model.predict_single(frame)
    
    # Get binary person mask
    mask = result.get_mask(threshold=0.75)
    
    # Get colored visualization of all 24 body parts
    colored_mask = result.get_colored_part_mask(mask)
    
    # Extract specific body parts
    # Get just the left arm (upper + lower front and back)
    left_arm_mask = result.get_part_mask(
        mask,
        part_names=[
            'left_upper_arm_front', 'left_upper_arm_back',
            'left_lower_arm_front', 'left_lower_arm_back'
        ]
    )
    
    # Get just the torso
    torso_mask = result.get_part_mask(
        mask,
        part_names=['torso_front', 'torso_back']
    )
    
    # Get just the legs
    legs_mask = result.get_part_mask(
        mask,
        part_names=[
            'left_upper_leg_front', 'left_upper_leg_back',
            'left_lower_leg_front', 'left_lower_leg_back',
            'right_upper_leg_front', 'right_upper_leg_back',
            'right_lower_leg_front', 'right_lower_leg_back'
        ]
    )
    
    # Apply different colors to different parts
    output = frame.copy()
    output[left_arm_mask.numpy() > 0] = [0, 255, 0]  # Green for left arm
    output[torso_mask.numpy() > 0] = [255, 0, 0]  # Blue for torso
    
    cv2.imshow('Original', frame)
    cv2.imshow('Body Parts', colored_mask)
    cv2.imshow('Custom Masks', output)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()