import cv2
import torch
import numpy as np
from torchvision import models, transforms
import time

# ==============================
# DEVICE
# ==============================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device:", device)

# ==============================
# LIGHTWEIGHT MODEL
# ==============================
model = models.segmentation.deeplabv3_mobilenet_v3_large(weights="DEFAULT").to(device)
model.eval()

# ==============================
# FAST TRANSFORM
# ==============================
transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize(320),
    transforms.CenterCrop(320),
    transforms.ToTensor(),
])

PERSON_CLASS = 15

# ==============================
# SMOOTHING
# ==============================
prev_density = 0

def smooth_density(current, alpha=0.6):
    global prev_density
    val = alpha * prev_density + (1 - alpha) * current
    prev_density = val
    return val
def generate_heatmap(mask, frame):
    heatmap = cv2.GaussianBlur(mask.astype(np.float32), (25, 25), 0)
    heatmap = cv2.normalize(heatmap, None, 0, 255, cv2.NORM_MINMAX)
    heatmap = heatmap.astype(np.uint8)

    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)

    overlay = cv2.addWeighted(frame, 0.6, heatmap, 0.4, 0)
    return overlay
# ==============================
# PROCESS FRAME (2 REGIONS)
# ==============================
def process_frame(frame):
    h, w, _ = frame.shape

    regions = [
        frame[0:h//2, :],
        frame[h//2:h, :]
    ]

    weights = [1.1, 0.9]

    total_human = 0
    total_pixels = 0

    full_mask = np.zeros((h, w), dtype=np.uint8)

    for i, region in enumerate(regions):
        input_tensor = transform(region).unsqueeze(0).to(device)

        with torch.no_grad():
            output = model(input_tensor)['out'][0]

        pred = output.argmax(0).byte().cpu().numpy()
        mask = (pred == PERSON_CLASS).astype(np.uint8)

        mask = cv2.resize(mask, (region.shape[1], region.shape[0]))

        # Light noise removal
        kernel = np.ones((2,2), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

        if i == 0:
            full_mask[0:h//2, :] = mask
        else:
            full_mask[h//2:h, :] = mask

        total_human += weights[i] * np.sum(mask)
        total_pixels += mask.size

    density = total_human / total_pixels if total_pixels > 0 else 0

    return density, full_mask

# ==============================
# CLASSIFY
# ==============================
def classify(d):
    if d < 0.03:
        return "LOW", (0,255,0)
    elif d < 0.5:
        return "MEDIUM", (0,255,255)
    else:
        return "HIGH", (0,0,255)

# ==============================
# VIDEO
# ==============================
cap = cv2.VideoCapture("test.mp4")

if not cap.isOpened():
    print("Camera error")
    exit()

frame_skip = 8
frame_count = 0
prev_time = 0

# ==============================
# LOOP
# ==============================
while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame_count += 1
    if frame_count % frame_skip != 0:
        continue

    frame = cv2.resize(frame, (640, 480))

    density, mask = process_frame(frame)
    density = smooth_density(density)

    label, color = classify(density)

    output = generate_heatmap(mask, frame)

    # FPS
    curr = time.time()
    fps = 1 / (curr - prev_time) if prev_time != 0 else 0
    prev_time = curr

    cv2.putText(output, f"Density: {density:.2f}", (10,30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

    cv2.putText(output, f"Level: {label}", (10,60),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

    cv2.putText(output, f"FPS: {int(fps)}", (10,90),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2)

    if label == "HIGH":
        cv2.putText(output, "⚠ OVERCROWDED!", (180, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 3)

    cv2.imshow("CrowdSense Fast", output)

    if cv2.waitKey(1) & 0xFF == 27:
        break



cap.release()
cv2.destroyAllWindows()