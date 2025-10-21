import cv2
import numpy as np
import torch
import time
from transformers import pipeline
import mediapipe as mp

# =============================================================
# CONFIG
# =============================================================
COUNTDOWN_SECONDS = 5
DEPTH_MODEL = "Intel/dpt-hybrid-midas"   # high-quality model
PARALLAX_STRENGTH = 0.5                  # scale factor for 3D illusion

device = "cuda" if torch.cuda.is_available() else (
    "mps" if torch.backends.mps.is_available() else "cpu"
)
print(f"[INFO] Using device: {device}")

# =============================================================
# 1. Load Depth Model (Hugging Face)
# =============================================================
print("[INFO] Loading depth estimation model...")
depth_estimator = pipeline(
    task="depth-estimation",
    model=DEPTH_MODEL,
    device=device,
)
print("[INFO] Depth model loaded successfully.")

# =============================================================
# 2. Initialize MediaPipe
# =============================================================
mp_drawing = mp.solutions.drawing_utils
mp_holistic = mp.solutions.holistic

holistic = mp_holistic.Holistic(
    static_image_mode=False,
    model_complexity=1,
    enable_segmentation=False,
    refine_face_landmarks=False,
)

# =============================================================
# Helper: Estimate depth map
# =============================================================
def compute_depth_map_hf(frame_bgr):
    """Compute normalized depth map using Hugging Face DPT model."""
    from PIL import Image
    img_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    pil_img = Image.fromarray(img_rgb)
    result = depth_estimator(pil_img)
    depth = np.array(result["depth"], dtype=np.float32)
    # Normalize to 0–1 (1=close, 0=far)
    depth = (depth - depth.min()) / (depth.max() - depth.min() + 1e-8)
    depth = 1.0 - depth
    return depth

# =============================================================
# Helper: Estimate pseudo 3D keypoints
# =============================================================
def get_3d_keypoints_from_mediapipe(results, depth_map, frame_shape):
    """Combine 2D mediapipe keypoints with reference depth map to get pseudo-3D."""
    h, w = frame_shape[:2]
    keypoints_3d = []

    if results.pose_landmarks:
        for lm in results.pose_landmarks.landmark:
            x_px = int(lm.x * w)
            y_px = int(lm.y * h)
            if 0 <= x_px < w and 0 <= y_px < h:
                z = float(depth_map[y_px, x_px]) * 2.0 - 1.0  # remap to [-1, 1]
                keypoints_3d.append((x_px, y_px, z))
    return keypoints_3d

# =============================================================
# Helper: Draw pseudo-3D skeleton
# =============================================================
def draw_3d_skeleton(frame, keypoints_3d):
    """Draws skeleton lines with depth-based parallax."""
    if not keypoints_3d:
        return frame

    overlay = frame.copy()
    h, w = frame.shape[:2]

    # simple parallax projection
    for (x, y, z) in keypoints_3d:
        parallax = int(z * PARALLAX_STRENGTH * 50)
        cv2.circle(overlay, (x + parallax, y), 10, (0, int((1 - z) * 255), int(z * 255)), -1)

    return cv2.addWeighted(overlay, 0.7, frame, 0.3, 0)

# =============================================================
# 3. Initialize Camera
# =============================================================
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

if not cap.isOpened():
    raise RuntimeError("Camera not accessible")

print("\n[INFO] Camera initialized.")
print("[INFO] Get into position for reference frame capture...")

# =============================================================
# 4. Countdown before capture
# =============================================================
for i in range(COUNTDOWN_SECONDS, 0, -1):
    ret, preview = cap.read()
    if ret:
        txt = f"Capturing reference frame in {i}..."
        preview = cv2.flip(preview, 1)
        cv2.putText(preview, txt, (40, 80), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 255), 3)
        cv2.imshow("Depth+Mediapipe 3D Test", preview)
        cv2.waitKey(1000)
    else:
        time.sleep(1)

print("[INFO] Capturing reference frame...")
ret, ref_frame = cap.read()
if not ret:
    raise RuntimeError("Failed to read from camera.")
ref_frame = cv2.flip(ref_frame, 1)
print("[INFO] Estimating reference depth...")
depth_ref = compute_depth_map_hf(ref_frame)
print("[INFO] Reference depth captured.\n")

# =============================================================
# 5. Main Loop
# =============================================================
print("[INFO] Running real-time 3D skeleton overlay (press 'q' to quit)...")
fps_start = time.time()
frame_count = 0
fps = 0

try:
    while True:
        ret, frame = cap.read()
        if not ret:
            continue

        frame = cv2.flip(frame, 1)
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Run MediaPipe Holistic
        results = holistic.process(rgb)
        keypoints_3d = get_3d_keypoints_from_mediapipe(results, depth_ref, frame.shape)
        frame_drawn = draw_3d_skeleton(frame, keypoints_3d)

        # FPS counter
        frame_count += 1
        elapsed = time.time() - fps_start
        if elapsed >= 1.0:
            fps = frame_count / elapsed
            frame_count = 0
            fps_start = time.time()

        cv2.putText(frame_drawn, f"FPS: {fps:.1f}", (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.imshow("Depth+Mediapipe 3D Test", frame_drawn)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break

except KeyboardInterrupt:
    pass

finally:
    cap.release()
    cv2.destroyAllWindows()
    holistic.close()
    print("[INFO] Test finished.")
