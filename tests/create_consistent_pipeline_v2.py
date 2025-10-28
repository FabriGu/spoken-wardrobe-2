"""
Consistent Pipeline V2 - Complete Generation with Reference Data
=================================================================

This script runs the COMPLETE pipeline from scratch with consistent data:
1. Capture frame from camera
2. Run BodyPix segmentation on that frame
3. Run MediaPipe on that frame (reference keypoints)
4. Use BodyPix mask for Stable Diffusion inpainting
5. Generate 3D mesh with TripoSR
6. Save ALL reference data for cage generation

Everything uses the SAME FRAME → full consistency.

Usage:
    python tests/create_consistent_pipeline_v2.py

Author: AI Assistant
Date: October 26, 2025
"""

import cv2
import numpy as np
import time
import mediapipe as mp
import pickle
from pathlib import Path
from PIL import Image
import os

# Import existing modules
from tf_bodypix.api import download_model, load_model, BodyPixModelPaths

# Lazy import for ClothingGenerator (will be imported only when needed)
# This avoids dependency conflicts during initial setup
ClothingGenerator = None


class ConsistentPipelineV2:
    """
    Interactive pipeline that generates mesh with consistent reference data.
    """
    
    def __init__(self):
        """Initialize all components."""
        self.bodypix_model = None
        self.mp_pose = None
        self.pose_detector = None
        self.clothing_generator = None
        
        # Captured data
        self.captured_frame = None
        self.bodypix_result = None
        self.bodypix_masks = {}
        self.mediapipe_keypoints_2d = {}
        self.mediapipe_keypoints_3d = {}
        
        # Generation outputs
        self.selected_parts = []
        self.inpainted_full = None
        self.clothing_png = None
        self.mesh_path = None
        
        print("\n" + "="*70)
        print("CONSISTENT PIPELINE V2 - INTERACTIVE GENERATION")
        print("="*70)
        print("\nThis will guide you through:")
        print("  1. Capture frame from camera")
        print("  2. Run BodyPix segmentation")
        print("  3. Run MediaPipe pose estimation")
        print("  4. Select body parts for clothing")
        print("  5. Generate clothing with AI")
        print("  6. Generate 3D mesh")
        print("  7. Save reference data")
        print("="*70 + "\n")
    
    def setup_bodypix(self):
        """Load BodyPix model."""
        print("\n" + "-"*70)
        print("STEP 1: Loading BodyPix Model")
        print("-"*70)
        
        print("Loading BodyPix model (MobileNet 75 - better quality)...")
        self.bodypix_model = load_model(download_model(
            BodyPixModelPaths.MOBILENET_FLOAT_75_STRIDE_16  # Better quality than 50
        ))
        print("✓ BodyPix model loaded\n")
    
    def setup_mediapipe(self):
        """Load MediaPipe Pose."""
        print("-"*70)
        print("STEP 2: Loading MediaPipe Pose")
        print("-"*70)
        
        print("Loading MediaPipe Pose...")
        self.mp_pose = mp.solutions.pose
        self.pose_detector = self.mp_pose.Pose(
            static_image_mode=True,
            model_complexity=1,
            min_detection_confidence=0.5
        )
        print("✓ MediaPipe Pose loaded\n")
    
    def setup_clothing_generator(self):
        """Load Stable Diffusion model."""
        print("-"*70)
        print("STEP 3: Loading Stable Diffusion Model")
        print("-"*70)
        
        # Lazy import here to avoid dependency conflicts
        try:
            import sys
            sys.path.append(str(Path(__file__).parent.parent / "src" / "modules"))
            from ai_generation import ClothingGenerator as CG
            
            print("Initializing Clothing Generator...")
            self.clothing_generator = CG()
            self.clothing_generator.load_model()
            print("✓ Clothing Generator loaded\n")
        except Exception as e:
            print(f"✗ Failed to load Clothing Generator: {e}")
            print("\nThis is likely due to diffusers/transformers version conflicts.")
            print("You can skip clothing generation for now and test other parts.")
            raise
    
    def capture_frame(self):
        """Capture frame from camera with 5-second countdown and model warmup."""
        print("-"*70)
        print("STEP 4: Capture Frame from Camera")
        print("-"*70)
        print("\nStand in T-pose in front of the camera")
        print("Auto-capture in 5 seconds...")
        print("Press Q to quit")
        print("-"*70 + "\n")
        
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            raise Exception("Could not open camera")
        
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        
        # Warm up period - let camera stabilize and run BodyPix a few times
        print("Warming up camera and BodyPix model (30 frames)...")
        warmup_frames = 0
        max_warmup = 30  # ~1 second at 30fps - more warmup for better masks
        
        while warmup_frames < max_warmup:
            ret, frame = cap.read()
            if not ret:
                continue
            
            # Show camera feed during warmup
            display_frame = frame.copy()
            cv2.putText(display_frame, f"Warming up... {warmup_frames}/{max_warmup}", (10, 60),
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
            cv2.imshow("Capture Frame", display_frame)
            cv2.waitKey(1)
            
            # Run BodyPix to warm up the model (discard results)
            _ = self.bodypix_model.predict_single(frame)
            warmup_frames += 1
        
        print("✓ Warmup complete\n")
        
        # 5-second countdown
        start_time = time.time()
        countdown_duration = 5.0
        
        while True:
            ret, frame = cap.read()
            if not ret:
                continue
            
            elapsed = time.time() - start_time
            remaining = int(countdown_duration - elapsed) + 1
            
            display_frame = frame.copy()
            
            if remaining > 0:
                # Show countdown
                cv2.putText(display_frame, f"Capturing in {remaining}...", (10, 60),
                           cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0), 3)
                cv2.putText(display_frame, "Stand in T-pose!", (10, 120),
                           cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            else:
                # Capture!
                self.captured_frame = frame.copy()
                cv2.putText(display_frame, "CAPTURED!", (10, 60),
                           cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0), 3)
                cv2.imshow("Capture Frame", display_frame)
                cv2.waitKey(500)  # Show "CAPTURED!" for 0.5 seconds
                print("✓ Frame captured\n")
                break
            
            cv2.imshow("Capture Frame", display_frame)
            
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                cap.release()
                cv2.destroyAllWindows()
                print("Cancelled by user")
                return False
        
        cap.release()
        cv2.destroyAllWindows()
        return True
    
    def run_bodypix(self):
        """Run BodyPix on captured frame (model already warmed up)."""
        print("-"*70)
        print("STEP 5: Run BodyPix Segmentation")
        print("-"*70)
        
        print("Running BodyPix segmentation (model pre-warmed)...")
        
        # Single pass - model is already warmed up from 30 frames
        self.bodypix_result = self.bodypix_model.predict_single(self.captured_frame)
        
        # Get binary person mask - use 0.75 threshold (standard)
        mask = self.bodypix_result.get_mask(threshold=0.75)
        
        # Extract all body part masks
        body_parts = [
            'left_face', 'right_face',
            'left_upper_arm_front', 'left_upper_arm_back',
            'right_upper_arm_front', 'right_upper_arm_back',
            'left_lower_arm_front', 'left_lower_arm_back',
            'right_lower_arm_front', 'right_lower_arm_back',
            'left_hand', 'right_hand',
            'torso_front', 'torso_back',
            'left_upper_leg_front', 'left_upper_leg_back',
            'right_upper_leg_front', 'right_upper_leg_back',
            'left_lower_leg_front', 'left_lower_leg_back',
            'right_lower_leg_front', 'right_lower_leg_back',
            'left_foot', 'right_foot'
        ]
        
        for part_name in body_parts:
            part_mask = self.bodypix_result.get_part_mask(mask, part_names=[part_name])
            
            # Convert to numpy and ensure 2D
            if hasattr(part_mask, 'numpy'):
                part_mask = part_mask.numpy()
            part_mask = np.squeeze(part_mask)
            
            # Ensure uint8
            if part_mask.dtype != np.uint8:
                part_mask = (part_mask * 255).astype(np.uint8)
            
            self.bodypix_masks[part_name] = part_mask
        
        print(f"✓ Extracted {len(self.bodypix_masks)} body part masks\n")
    
    def run_mediapipe(self):
        """Run MediaPipe on captured frame."""
        print("-"*70)
        print("STEP 6: Run MediaPipe Pose Estimation")
        print("-"*70)
        
        print("Running MediaPipe pose estimation...")
        frame_rgb = cv2.cvtColor(self.captured_frame, cv2.COLOR_BGR2RGB)
        results = self.pose_detector.process(frame_rgb)
        
        if not results.pose_landmarks:
            print("⚠ Warning: No pose detected!")
            return
        
        # Extract keypoints
        h, w = self.captured_frame.shape[:2]
        
        keypoint_indices = {
            'nose': 0,
            'left_shoulder': 11,
            'right_shoulder': 12,
            'left_elbow': 13,
            'right_elbow': 14,
            'left_wrist': 15,
            'right_wrist': 16,
            'left_hip': 23,
            'right_hip': 24,
            'left_knee': 25,
            'right_knee': 26,
            'left_ankle': 27,
            'right_ankle': 28,
        }
        
        for name, idx in keypoint_indices.items():
            landmark = results.pose_landmarks.landmark[idx]
            
            # 2D keypoints (pixel coordinates + MediaPipe Z)
            x_px = landmark.x * w
            y_px = landmark.y * h
            self.mediapipe_keypoints_2d[name] = (x_px, y_px, landmark.z)
            
            # 3D keypoints (normalized coordinates)
            self.mediapipe_keypoints_3d[name] = (landmark.x, landmark.y, landmark.z)
        
        print(f"✓ Extracted {len(self.mediapipe_keypoints_2d)} keypoints\n")
    
    def select_body_parts(self):
        """Interactive body part selection."""
        print("-"*70)
        print("STEP 7: Select Body Parts for Clothing Generation")
        print("-"*70)
        
        print("\nAvailable body part groups:")
        print("  1. Dress")
        print("  2. Left Upper Arm")
        print("  3. Right Upper Arm")
        print("  4. Left Lower Arm")
        print("  5. Right Lower Arm")
        print("  6. Full Arms (upper + lower)")
        print("  7. T-Shirt (torso + upper arms)")
        print("  8. Long Sleeve Shirt (torso + full arms)")
        
        choice = input("\nSelect clothing type (1-8) or press ENTER for T-Shirt (7): ").strip()
        
        if choice == '1':
            # include dress selection which includes both full arms, torso, and legs
            self.selected_parts = ['left_upper_arm', 'right_upper_arm', 'left_lower_arm', 'right_lower_arm', 'torso', 'left_upper_leg', 'right_upper_leg', 'left_lower_leg', 'right_lower_leg']
            # self.selected_parts = ['left_upper_arm', 'right_upper_arm', 'left_lower_arm', 'right_lower_arm']
        elif choice == '2':
            self.selected_parts = ['left_upper_arm']
        elif choice == '3':
            self.selected_parts = ['right_upper_arm']
        elif choice == '4':
            self.selected_parts = ['left_lower_arm']
        elif choice == '5':
            self.selected_parts = ['right_lower_arm']
        elif choice == '6':
            self.selected_parts = ['left_upper_arm', 'right_upper_arm', 'left_lower_arm', 'right_lower_arm']
        elif choice == '8':
            self.selected_parts = ['torso', 'left_upper_arm', 'right_upper_arm', 'left_lower_arm', 'right_lower_arm']
        else:  # Default: T-shirt
            self.selected_parts = ['torso', 'left_upper_arm', 'right_upper_arm']
        
        print(f"\n✓ Selected: {', '.join(self.selected_parts)}\n")
    
    def generate_clothing_mask(self):
        """Create combined mask from selected body parts."""
        # Map anatomical names to BodyPix part names
        part_map = {
            'torso': ['torso_front', 'torso_back'],
            'left_upper_arm': ['left_upper_arm_front', 'left_upper_arm_back'],
            'right_upper_arm': ['right_upper_arm_front', 'right_upper_arm_back'],
            'left_lower_arm': ['left_lower_arm_front', 'left_lower_arm_back'],
            'right_lower_arm': ['right_lower_arm_front', 'right_lower_arm_back'],
            'left_upper_leg': ['left_upper_leg_front', 'left_upper_leg_back'],
            'right_upper_leg': ['right_upper_leg_front', 'right_upper_leg_back'],
            'left_lower_leg': ['left_lower_leg_front', 'left_lower_leg_back'],
            'right_lower_leg': ['right_lower_leg_front', 'right_lower_leg_back']
        }
        
        h, w = self.captured_frame.shape[:2]
        combined_mask = np.zeros((h, w), dtype=np.uint8)
        
        for section_name in self.selected_parts:
            if section_name in part_map:
                for part_name in part_map[section_name]:
                    if part_name in self.bodypix_masks:
                        combined_mask = np.maximum(combined_mask, self.bodypix_masks[part_name])
        
        return combined_mask
    
    def generate_clothing(self):
        """Generate clothing using Stable Diffusion."""
        print("-"*70)
        print("STEP 8: Generate Clothing with AI")
        print("-"*70)
        
        # Get prompt from user
        prompt = input("\nEnter clothing prompt (e.g., 'flames', 'roses') or press ENTER for 'flames': ").strip()
        if not prompt:
            prompt = "flames"
        
        print(f"\nGenerating clothing with prompt: '{prompt}'")
        
        # Create mask
        mask = self.generate_clothing_mask()
        
        # Generate
        self.inpainted_full, self.clothing_png = self.clothing_generator.generate_clothing_from_text(
            self.captured_frame, mask, prompt
        )
        
        if self.inpainted_full is None:
            print("✗ Clothing generation failed")
            return False
        
        print("✓ Clothing generated\n")
        return True
    
    def generate_mesh(self):
        """Generate 3D mesh using TripoSR."""
        print("-"*70)
        print("STEP 9: Generate 3D Mesh with TripoSR")
        print("-"*70)
        
        # Save clothing image temporarily
        temp_clothing_path = "temp_clothing_for_mesh.png"
        self.clothing_png.save(temp_clothing_path)
        
        # Generate output filename
        timestamp = int(time.time())
        output_dir = f"generated_meshes/{timestamp}"
        os.makedirs(output_dir, exist_ok=True)
        
        print("\nRunning TripoSR (this may take a minute)...")
        print(f"Output directory: {output_dir}")
        
        # Import TripoSR pipeline module
        from triposr_pipeline import generate_mesh_from_image
        
        try:
            # Generate mesh using Test 3 (Balanced) settings - best quality/performance
            self.mesh_path, corrected_mesh = generate_mesh_from_image(
                image_path=temp_clothing_path,
                output_dir=output_dir,
                z_scale=0.8,  # Adjustable - reduces "fatness"
                auto_orient=True,
                apply_flip=True,  # 180° flip to fix upside-down
                no_remove_bg=False,  # DO background removal (better for SD output)
                foreground_ratio=0.75,  # More padding than default (0.85)
                mc_resolution=196,  # Much better than 110 - fixes holes!
                chunk_size=8192,
                model_save_format="obj"
            )
            
            print(f"✓ Mesh generated: {self.mesh_path}\n")
            
        except Exception as e:
            print(f"✗ TripoSR failed: {e}")
            import traceback
            traceback.print_exc()
            return False
        
        finally:
            # Clean up temp file
            if os.path.exists(temp_clothing_path):
                os.remove(temp_clothing_path)
        
        return True
    
    def save_reference_data(self):
        """Save all reference data."""
        print("-"*70)
        print("STEP 10: Save Reference Data")
        print("-"*70)
        
        # Determine output filename
        mesh_dir = Path(self.mesh_path).parent
        reference_path = mesh_dir / "reference_data.pkl"
        
        reference_data = {
            'original_frame': self.captured_frame,
            'bodypix_masks': self.bodypix_masks,
            'selected_parts': self.selected_parts,
            'mediapipe_keypoints_2d': self.mediapipe_keypoints_2d,
            'mediapipe_keypoints_3d': self.mediapipe_keypoints_3d,
            'frame_shape': self.captured_frame.shape[:2],
            'mesh_path': self.mesh_path,
            'timestamp': time.time()
        }
        
        # Save
        with open(reference_path, 'wb') as f:
            pickle.dump(reference_data, f)
        
        # Also save frame as PNG
        frame_path = mesh_dir / "reference_frame.png"
        cv2.imwrite(str(frame_path), self.captured_frame)
        
        # Save generated images
        full_path = mesh_dir / "generated_full.png"
        clothing_path = mesh_dir / "generated_clothing.png"
        self.inpainted_full.save(str(full_path))
        self.clothing_png.save(str(clothing_path))
        
        print(f"\n✓ Reference data saved: {reference_path}")
        print(f"✓ Reference frame saved: {frame_path}")
        print(f"✓ Generated images saved: {full_path}, {clothing_path}")
        
        return reference_path
    
    def run(self):
        """Run the complete pipeline interactively."""
        try:
            # Setup
            self.setup_bodypix()
            self.setup_mediapipe()
            self.setup_clothing_generator()
            
            # Capture
            if not self.capture_frame():
                return
            
            # Process
            self.run_bodypix()
            self.run_mediapipe()
            
            # Generate
            self.select_body_parts()
            if not self.generate_clothing():
                return
            
            if not self.generate_mesh():
                return
            
            # Save
            reference_path = self.save_reference_data()
            
            # Final summary
            print("\n" + "="*70)
            print("PIPELINE COMPLETE!")
            print("="*70)
            print(f"\nGenerated files:")
            print(f"  Mesh: {self.mesh_path}")
            print(f"  Reference data: {reference_path}")
            print("\nNext step: Run test_integration_v2.py")
            print(f"\n  python tests/test_integration_v2.py \\")
            print(f"      --mesh {self.mesh_path} \\")
            print(f"      --reference {reference_path}")
            print("="*70 + "\n")
        
        except KeyboardInterrupt:
            print("\n\nPipeline interrupted by user")
        except Exception as e:
            print(f"\n\n✗ Error: {e}")
            import traceback
            traceback.print_exc()
        finally:
            if self.pose_detector:
                self.pose_detector.close()
            if self.clothing_generator:
                self.clothing_generator.cleanup()


def main():
    pipeline = ConsistentPipelineV2()
    pipeline.run()


if __name__ == "__main__":
    main()

