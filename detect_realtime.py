"""
=============================================================
 Robocon Real-Time Fake / Real Detector
 Uses: OpenCV webcam + trained model (PyTorch or Keras)
=============================================================

 USAGE:
   python detect_realtime.py --model model.pt --framework pytorch
   python detect_realtime.py --model model.h5 --framework keras

 CONTROLS (while running):
   Q        → quit
   S        → save current frame snapshot
   T        → toggle TTA on/off
   +/-      → increase/decrease confidence threshold
   SPACE    → freeze/unfreeze frame
"""

import cv2
import numpy as np
import argparse
import time
import os
from collections import deque

# ── Argument parsing ──────────────────────────────────────────────────────────
parser = argparse.ArgumentParser()
parser.add_argument("--model",     required=True, help="Path to model file (.pt or .h5)")
parser.add_argument("--framework", default="pytorch", choices=["pytorch","keras"],
                    help="Model framework")
parser.add_argument("--img-size",  default=224, type=int, help="Input image size")
parser.add_argument("--camera",    default=0,   type=int, help="Camera index")
parser.add_argument("--tta",       action="store_true",   help="Enable TTA by default")
args = parser.parse_args()

IMG_SIZE = args.img_size

# ── Load model ────────────────────────────────────────────────────────────────
print(f"[INFO] Loading {args.framework} model from {args.model} ...")

if args.framework == "pytorch":
    import torch
    import torchvision.transforms as T

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model  = torch.load(args.model, map_location=device)
    model.eval()
    print(f"[INFO] Running on {device}")

    base_tf = T.Compose([
        T.ToPILImage(),
        T.Resize((IMG_SIZE, IMG_SIZE)),
        T.ToTensor(),
        T.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225]),
    ])

    def predict_single(bgr_img):
        rgb = cv2.cvtColor(bgr_img, cv2.COLOR_BGR2RGB)
        x   = base_tf(rgb).unsqueeze(0).to(device)
        with torch.no_grad():
            out = model(x)
            prob = torch.softmax(out, dim=1)[0]
        # index 0 = fake, index 1 = real  (adjust if your labels differ)
        return float(prob[1].item())   # returns P(real)

else:
    import tensorflow as tf

    model = tf.keras.models.load_model(args.model)

    def predict_single(bgr_img):
        rgb   = cv2.cvtColor(bgr_img, cv2.COLOR_BGR2RGB)
        resized = cv2.resize(rgb, (IMG_SIZE, IMG_SIZE))
        x     = resized.astype("float32") / 255.0
        x     = np.expand_dims(x, 0)
        out   = model.predict(x, verbose=0)[0]
        # single sigmoid output → P(real)
        if len(out) == 1:
            return float(out[0])
        return float(out[1])   # softmax → index 1 = real

# ── Perspective warp helpers (same as augmentation pipeline) ──────────────────
def warp(img, src, dst):
    h, w = img.shape[:2]
    M = cv2.getPerspectiveTransform(np.float32(src), np.float32(dst))
    return cv2.warpPerspective(img, M, (w, h), borderMode=cv2.BORDER_REPLICATE)

def tta_variants(img):
    """Generate test-time augmentation variants for robust prediction."""
    h, w = img.shape[:2]
    m = 0.15
    variants = [img]  # original

    # Subtle perspective warps
    variants.append(warp(img,
        [[0,0],[w,0],[w,h],[0,h]],
        [[w*m,0],[w*(1-m),0],[w,h],[0,h]]))   # slight top tilt

    variants.append(warp(img,
        [[0,0],[w,0],[w,h],[0,h]],
        [[0,0],[w,0],[w*(1-m),h],[w*m,h]]))   # slight bottom tilt

    variants.append(warp(img,
        [[0,0],[w,0],[w,h],[0,h]],
        [[0,h*m],[w,0],[w,h],[0,h*(1-m)]]))   # slight left tilt

    variants.append(warp(img,
        [[0,0],[w,0],[w,h],[0,h]],
        [[0,0],[w,h*m],[w,h*(1-m)],[0,h]]))   # slight right tilt

    # Horizontal flip
    variants.append(cv2.flip(img, 1))

    # Slight brightness shift
    bright = np.clip(img.astype(np.int32) + 30, 0, 255).astype(np.uint8)
    dim    = np.clip(img.astype(np.int32) - 30, 0, 255).astype(np.uint8)
    variants.append(bright)
    variants.append(dim)

    return variants

def predict_tta(img):
    """Average predictions over all TTA variants."""
    scores = [predict_single(v) for v in tta_variants(img)]
    return float(np.mean(scores)), scores

# ── ROI crop: focus on center region (where target likely is) ─────────────────
def crop_roi(frame, roi_ratio=0.7):
    h, w = frame.shape[:2]
    dy = int(h * (1 - roi_ratio) / 2)
    dx = int(w * (1 - roi_ratio) / 2)
    return frame[dy:h-dy, dx:w-dx], (dx, dy, w-dx, h-dy)

# ── Drawing helpers ───────────────────────────────────────────────────────────
REAL_COLOR = (50, 220, 100)   # green
FAKE_COLOR = (50,  80, 230)   # red-ish
UNK_COLOR  = (180,180,180)

def draw_overlay(frame, label, confidence, threshold, fps, tta_on, frozen,
                 history, roi_box, scores=None):
    h, w = frame.shape[:2]
    dx, dy, dx2, dy2 = roi_box

    # ROI rectangle
    color = REAL_COLOR if label == "REAL" else (FAKE_COLOR if label == "FAKE" else UNK_COLOR)
    cv2.rectangle(frame, (dx, dy), (dx2, dy2), color, 2)

    # ── Top banner ────────────────────────────────────────────────
    cv2.rectangle(frame, (0,0), (w, 56), (20,20,20), -1)

    label_text = f"  {label}"
    cv2.putText(frame, label_text, (8, 38),
                cv2.FONT_HERSHEY_DUPLEX, 1.1, color, 2, cv2.LINE_AA)

    conf_text = f"{confidence*100:.1f}%"
    cv2.putText(frame, conf_text, (w-130, 38),
                cv2.FONT_HERSHEY_DUPLEX, 1.0, color, 2, cv2.LINE_AA)

    # ── Confidence bar ────────────────────────────────────────────
    bar_x, bar_y, bar_w, bar_h = 8, 48, w-16, 6
    cv2.rectangle(frame, (bar_x, bar_y), (bar_x+bar_w, bar_y+bar_h), (60,60,60), -1)
    fill = int(bar_w * confidence)
    cv2.rectangle(frame, (bar_x, bar_y), (bar_x+fill, bar_y+bar_h), color, -1)
    # threshold marker
    thresh_x = bar_x + int(bar_w * threshold)
    cv2.line(frame, (thresh_x, bar_y-2), (thresh_x, bar_y+bar_h+2), (255,255,80), 2)

    # ── Bottom status bar ─────────────────────────────────────────
    cv2.rectangle(frame, (0, h-36), (w, h), (20,20,20), -1)
    status_parts = [
        f"FPS:{fps:.0f}",
        f"THR:{threshold:.2f}",
        f"TTA:{'ON' if tta_on else 'OFF'}",
        "FROZEN" if frozen else "",
        "Q:quit  S:save  T:tta  +/-:thr  SPC:freeze",
    ]
    status = "  |  ".join(p for p in status_parts if p)
    cv2.putText(frame, status, (8, h-12),
                cv2.FONT_HERSHEY_SIMPLEX, 0.42, (180,180,180), 1, cv2.LINE_AA)

    # ── Confidence history graph ──────────────────────────────────
    if len(history) > 1:
        gx, gy, gw, gh = w-160, 60, 150, 60
        cv2.rectangle(frame, (gx, gy), (gx+gw, gy+gh), (30,30,30), -1)
        cv2.rectangle(frame, (gx, gy), (gx+gw, gy+gh), (70,70,70), 1)
        pts = []
        for i, val in enumerate(history):
            px = gx + int(i * gw / (len(history)-1))
            py = gy + gh - int(val * gh)
            pts.append((px, py))
        for i in range(1, len(pts)):
            cv2.line(frame, pts[i-1], pts[i], REAL_COLOR, 1)
        # threshold line on graph
        ty = gy + gh - int(threshold * gh)
        cv2.line(frame, (gx, ty), (gx+gw, ty), (255,255,80), 1)
        cv2.putText(frame, "P(real)", (gx+2, gy+10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.35, (150,150,150), 1)

    # ── TTA score breakdown ───────────────────────────────────────
    if tta_on and scores:
        bx, by = 8, 60
        cv2.putText(frame, "TTA variants:", (bx, by+12),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.38, (150,150,150), 1)
        labels_tta = ["orig","top","bot","left","right","flip","bright","dim"]
        for i, (s, lbl) in enumerate(zip(scores, labels_tta)):
            c = REAL_COLOR if s >= threshold else FAKE_COLOR
            cv2.putText(frame, f"{lbl}:{s:.2f}", (bx, by+28+i*14),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.34, c, 1)

    return frame

# ── Main loop ─────────────────────────────────────────────────────────────────
def main():
    cap = cv2.VideoCapture(args.camera)
    if not cap.isOpened():
        print(f"[ERROR] Cannot open camera index {args.camera}")
        return

    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    threshold   = 0.5
    tta_on      = args.tta
    frozen      = False
    frozen_frame= None
    history     = deque(maxlen=60)
    snap_idx    = 0

    prev_time   = time.time()
    fps         = 0.0
    last_scores = None

    os.makedirs("snapshots", exist_ok=True)
    print("[INFO] Camera open. Press Q to quit.")

    while True:
        if not frozen:
            ret, frame = cap.read()
            if not ret:
                print("[ERROR] Frame grab failed.")
                break
            display = frame.copy()
        else:
            display = frozen_frame.copy()

        # ── Predict ──────────────────────────────────────────────
        roi, roi_box = crop_roi(display)

        if tta_on:
            confidence, last_scores = predict_tta(roi)
        else:
            confidence = predict_single(roi)
            last_scores = None

        history.append(confidence)

        if   confidence >= threshold:         label = "REAL"
        elif confidence <= (1 - threshold):   label = "FAKE"
        else:                                 label = "UNCERTAIN"

        # ── FPS ───────────────────────────────────────────────────
        now = time.time()
        fps = 0.9 * fps + 0.1 * (1.0 / max(now - prev_time, 1e-6))
        prev_time = now

        # ── Draw ─────────────────────────────────────────────────
        draw_overlay(display, label, confidence, threshold, fps,
                     tta_on, frozen, history, roi_box, last_scores)

        cv2.imshow("Robocon Detector  [Q=quit]", display)

        # ── Key handling ─────────────────────────────────────────
        key = cv2.waitKey(1) & 0xFF

        if key == ord('q'):
            break
        elif key == ord('s'):
            path = f"snapshots/snap_{snap_idx:04d}_{label}_{confidence:.2f}.png"
            cv2.imwrite(path, display)
            print(f"[SAVE] {path}")
            snap_idx += 1
        elif key == ord('t'):
            tta_on = not tta_on
            print(f"[TTA] {'ON' if tta_on else 'OFF'}")
        elif key == ord('+') or key == ord('='):
            threshold = min(0.99, threshold + 0.05)
            print(f"[THR] {threshold:.2f}")
        elif key == ord('-'):
            threshold = max(0.01, threshold - 0.05)
            print(f"[THR] {threshold:.2f}")
        elif key == ord(' '):
            if not frozen:
                frozen       = True
                frozen_frame = display.copy()
                print("[FREEZE]")
            else:
                frozen = False
                print("[UNFREEZE]")

    cap.release()
    cv2.destroyAllWindows()
    print("[DONE]")

if __name__ == "__main__":
    main()