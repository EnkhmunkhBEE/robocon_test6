"""
=============================================================
 Robocon Real-Time Fake / Real Detector
 Uses: OpenCV webcam + trained model (PyTorch or Keras)
=============================================================
"""

import cv2
import numpy as np
import argparse
import time
import os
from collections import deque

parser = argparse.ArgumentParser()
parser.add_argument("--model",      required=True)
parser.add_argument("--framework",  default="pytorch", choices=["pytorch","keras"])
parser.add_argument("--img-size",   default=224,  type=int)
parser.add_argument("--camera",     default=0,    type=int)
parser.add_argument("--tta",        action="store_true")
parser.add_argument("--smooth",     default=10,   type=int)
parser.add_argument("--real-enter", default=0.70, type=float)
parser.add_argument("--real-exit",  default=0.45, type=float)
parser.add_argument("--confirm",    default=5,    type=int)
args = parser.parse_args()

IMG_SIZE   = args.img_size
SMOOTH_WIN = args.smooth
ENTER_THR  = args.real_enter
EXIT_THR   = args.real_exit
CONFIRM_N  = args.confirm

# ── Load model ────────────────────────────────────────────────────────────────
print(f"[INFO] Loading {args.framework} model: {args.model}")

if args.framework == "pytorch":
    import torch
    import torchvision.transforms as T

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model  = torch.load(args.model, map_location=device, weights_only=False)
    model.eval()
    print(f"[INFO] Device: {device}")

    base_tf = T.Compose([
        T.ToPILImage(),
        T.Resize((IMG_SIZE, IMG_SIZE)),
        T.ToTensor(),
        T.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225]),
    ])

    def predict_single(bgr):
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        x   = base_tf(rgb).unsqueeze(0).to(device)
        with torch.no_grad():
            prob = torch.softmax(model(x), dim=1)[0]
        return float(prob[1].item())

else:
    import tensorflow as tf
    model = tf.keras.models.load_model(args.model)

    def predict_single(bgr):
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        x   = np.expand_dims(
                cv2.resize(rgb,(IMG_SIZE,IMG_SIZE)).astype("float32")/255.0,0)
        out = model.predict(x, verbose=0)[0]
        return float(out[0]) if len(out)==1 else float(out[1])

# ── TTA ───────────────────────────────────────────────────────────────────────
def warp(img, src, dst):
    h,w = img.shape[:2]
    return cv2.warpPerspective(img,
        cv2.getPerspectiveTransform(np.float32(src),np.float32(dst)),
        (w,h), borderMode=cv2.BORDER_REPLICATE)

def tta_variants(img):
    h,w = img.shape[:2]; m=0.12
    return [
        img, cv2.flip(img,1),
        warp(img,[[0,0],[w,0],[w,h],[0,h]],[[w*m,0],[w*(1-m),0],[w,h],[0,h]]),
        warp(img,[[0,0],[w,0],[w,h],[0,h]],[[0,0],[w,0],[w*(1-m),h],[w*m,h]]),
        np.clip(img.astype(np.int32)+25,0,255).astype(np.uint8),
        np.clip(img.astype(np.int32)-25,0,255).astype(np.uint8),
    ]

def predict_tta(img):
    return float(np.mean([predict_single(v) for v in tta_variants(img)]))

# ── KFS box detection ─────────────────────────────────────────────────────────
KFS_COLOR_LOWER  = np.array([0,   120,  70])
KFS_COLOR_UPPER  = np.array([10,  255, 255])
KFS_COLOR_LOWER2 = np.array([170, 120,  70])
KFS_COLOR_UPPER2 = np.array([180, 255, 255])

def detect_kfs_box(frame):
    fh,fw = frame.shape[:2]; pad=10

    gray  = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(cv2.GaussianBlur(gray,(7,7),0),30,120)
    edges = cv2.dilate(edges,
                cv2.getStructuringElement(cv2.MORPH_RECT,(5,5)),iterations=2)
    cnts,_ = cv2.findContours(edges,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
    best,best_area = None,0
    for cnt in cnts:
        area = cv2.contourArea(cnt)
        if area < fw*fh*0.01 or area > fw*fh*0.95: continue
        approx = cv2.approxPolyDP(cnt,0.03*cv2.arcLength(cnt,True),True)
        if len(approx)==4 and area>best_area:
            x,y,w,h = cv2.boundingRect(approx)
            if 0.3 < w/max(h,1) < 3.5:
                best,best_area = (x,y,w,h),area
    if best:
        x,y,w,h = best
        return max(0,x-pad),max(0,y-pad),min(fw,x+w+pad),min(fh,y+h+pad)

    hsv  = cv2.cvtColor(frame,cv2.COLOR_BGR2HSV)
    mask = (cv2.inRange(hsv,KFS_COLOR_LOWER,KFS_COLOR_UPPER)+
            cv2.inRange(hsv,KFS_COLOR_LOWER2,KFS_COLOR_UPPER2))
    mask = cv2.morphologyEx(mask,cv2.MORPH_CLOSE,
               cv2.getStructuringElement(cv2.MORPH_RECT,(9,9)),iterations=3)
    cnts,_ = cv2.findContours(mask,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
    if cnts:
        cnt = max(cnts,key=cv2.contourArea)
        if cv2.contourArea(cnt)>fw*fh*0.005:
            x,y,w,h = cv2.boundingRect(cnt)
            return max(0,x-pad),max(0,y-pad),min(fw,x+w+pad),min(fh,y+h+pad)

    dy=int(fh*0.2); dx=int(fw*0.2)
    return dx,dy,fw-dx,fh-dy

# ── Stable label state machine ────────────────────────────────────────────────
class StableLabel:
    def __init__(self):
        self.smoothed   = 0.0
        self.state      = "FAKE"
        self.confirm_ct = 0
        self.locked     = False
        self.raw_buf    = deque(maxlen=SMOOTH_WIN)

    def update(self, raw_conf):
        self.raw_buf.append(raw_conf)
        self.smoothed = float(np.mean(self.raw_buf))

        if self.state == "FAKE":
            if self.smoothed >= ENTER_THR:
                self.confirm_ct += 1
            else:
                self.confirm_ct = 0
            if self.confirm_ct >= CONFIRM_N:
                self.state  = "REAL"
                self.locked = True
        else:
            if self.smoothed < EXIT_THR:
                self.confirm_ct = 0
                self.state      = "FAKE"
                self.locked     = False

        return self.state, self.smoothed, self.locked

# ── CSRT Tracker wrapper ──────────────────────────────────────────────────────
class PatternTracker:
    """
    Wraps OpenCV CSRT tracker.
    Once initialized with a bounding box, tracks the target across frames
    without re-running the classifier every frame — fast and smooth.
    """
    def __init__(self):
        self.tracker  = None
        self.active   = False
        self.lost_ct  = 0
        self.LOST_MAX = 15          # frames without update before declaring lost
        self.box      = None        # last known (x,y,w,h)

    def init(self, frame, x1, y1, x2, y2):
        """Start tracking from detected box."""
        self.tracker = cv2.TrackerCSRT_create()
        w = x2-x1; h = y2-y1
        self.tracker.init(frame, (x1, y1, w, h))
        self.box     = (x1, y1, x2, y2)
        self.active  = True
        self.lost_ct = 0
        print("[TRACKER] Initialized — now tracking REAL pattern")

    def update(self, frame):
        """
        Returns (x1,y1,x2,y2, tracking_ok).
        tracking_ok=False means target was lost.
        """
        if not self.active or self.tracker is None:
            return None, False

        ok, bbox = self.tracker.update(frame)

        if ok:
            x,y,w,h  = [int(v) for v in bbox]
            fh,fw    = frame.shape[:2]
            # Clamp to frame
            x1 = max(0,x);    y1 = max(0,y)
            x2 = min(fw,x+w); y2 = min(fh,y+h)
            # Sanity: reject degenerate boxes
            if (x2-x1) < 10 or (y2-y1) < 10:
                self.lost_ct += 1
            else:
                self.box     = (x1,y1,x2,y2)
                self.lost_ct = 0
        else:
            self.lost_ct += 1

        if self.lost_ct >= self.LOST_MAX:
            self.active  = False
            self.tracker = None
            print("[TRACKER] Lost target — returning to search")
            return self.box, False

        return self.box, True

    def reset(self):
        self.tracker = None
        self.active  = False
        self.lost_ct = 0
        self.box     = None

# ── Drawing ───────────────────────────────────────────────────────────────────
REAL_COLOR   = (50, 220, 100)
FAKE_COLOR   = (50,  80, 230)
LOST_COLOR   = (0,  165, 255)
SEARCH_COLOR = (180,180,180)

def draw_locked_box(frame, x1, y1, x2, y2, smoothed, tracking_ok):
    color = REAL_COLOR if tracking_ok else LOST_COLOR
    # Filled corner ticks
    tick = 22
    for cx,cy,dx,dy in [(x1,y1,1,1),(x2,y1,-1,1),(x1,y2,1,-1),(x2,y2,-1,-1)]:
        cv2.line(frame,(cx,cy),(cx+dx*tick,cy),color,4)
        cv2.line(frame,(cx,cy),(cx,cy+dy*tick),color,4)
    cv2.rectangle(frame,(x1,y1),(x2,y2),color,2)

    # Status badge
    status = "TRACKING" if tracking_ok else "RE-ACQUIRING"
    badge  = f"REAL  {smoothed*100:.1f}%  [{status}]"
    bw     = len(badge)*10+10
    cv2.rectangle(frame,(x1,y1-28),(x1+bw,y1),(0,0,0),-1)
    cv2.putText(frame, badge,(x1+4,y1-8),
                cv2.FONT_HERSHEY_DUPLEX,0.6,color,1,cv2.LINE_AA)

def draw_search_overlay(frame, confirm_ct):
    """Subtle scanning animation while searching."""
    h,w = frame.shape[:2]
    # Pulsing center crosshair
    t     = time.time()
    alpha = int(128 + 80*np.sin(t*4))
    cx,cy = w//2, h//2
    size  = 30
    col   = (alpha, alpha, alpha)
    cv2.line(frame,(cx-size,cy),(cx+size,cy),col,1)
    cv2.line(frame,(cx,cy-size),(cx,cy+size),col,1)
    cv2.putText(frame,f"Searching...  ({confirm_ct}/{CONFIRM_N})",
                (cx-80,cy+50),cv2.FONT_HERSHEY_SIMPLEX,0.55,col,1,cv2.LINE_AA)

def draw_hud(frame, app_state, smoothed, raw, fps, tta_on, frozen,
             raw_hist, smooth_hist, confirm_ct):
    h,w = frame.shape[:2]

    state_colors = {
        "SEARCHING": SEARCH_COLOR,
        "LOCKED":    REAL_COLOR,
        "LOST":      LOST_COLOR,
    }
    pc = state_colors.get(app_state, SEARCH_COLOR)

    # Top banner
    cv2.rectangle(frame,(0,0),(w,60),(20,20,20),-1)
    cv2.putText(frame, f"  {app_state}", (8,40),
                cv2.FONT_HERSHEY_DUPLEX,1.2,pc,2,cv2.LINE_AA)
    cv2.putText(frame,f"raw:{raw*100:.0f}%  smooth:{smoothed*100:.0f}%",
                (220,28),cv2.FONT_HERSHEY_SIMPLEX,0.52,(180,180,180),1,cv2.LINE_AA)
    cv2.putText(frame,f"confirm:{confirm_ct}/{CONFIRM_N}",
                (220,50),cv2.FONT_HERSHEY_SIMPLEX,0.48,
                (REAL_COLOR if confirm_ct>=CONFIRM_N else (255,180,0)),1,cv2.LINE_AA)

    # Smoothed bar
    bx,by,bw,bh = 8,52,w-16,6
    cv2.rectangle(frame,(bx,by),(bx+bw,by+bh),(60,60,60),-1)
    cv2.rectangle(frame,(bx,by),(bx+int(bw*smoothed),by+bh),pc,-1)
    ex = bx+int(bw*ENTER_THR)
    ox = bx+int(bw*EXIT_THR)
    cv2.line(frame,(ex,by-3),(ex,by+bh+3),(255,255,0),2)
    cv2.line(frame,(ox,by-3),(ox,by+bh+3),(0,165,255),2)

    # History graph
    gx,gy,gw,gh = w-170,64,160,65
    cv2.rectangle(frame,(gx,gy),(gx+gw,gy+gh),(25,25,25),-1)
    cv2.rectangle(frame,(gx,gy),(gx+gw,gy+gh),(70,70,70),1)

    def gpts(buf):
        return [(gx+int(i*gw/max(len(buf)-1,1)),
                 gy+gh-int(v*gh)) for i,v in enumerate(buf)]

    if len(raw_hist)>1:
        pts=gpts(raw_hist)
        for i in range(1,len(pts)): cv2.line(frame,pts[i-1],pts[i],(70,70,70),1)
    if len(smooth_hist)>1:
        pts=gpts(smooth_hist)
        for i in range(1,len(pts)): cv2.line(frame,pts[i-1],pts[i],REAL_COLOR,1)
    for thr,col in [(ENTER_THR,(255,255,0)),(EXIT_THR,(0,165,255))]:
        ty=gy+gh-int(thr*gh)
        cv2.line(frame,(gx,ty),(gx+gw,ty),col,1)
    cv2.putText(frame,"P(real)",(gx+2,gy+10),
                cv2.FONT_HERSHEY_SIMPLEX,0.3,REAL_COLOR,1)

    # Bottom status
    cv2.rectangle(frame,(0,h-36),(w,h),(20,20,20),-1)
    status=(f"FPS:{fps:.0f}  |  TTA:{'ON' if tta_on else 'OFF'}  |  "
            f"{'FROZEN  |  ' if frozen else ''}"
            f"Q:quit  S:save  T:tta  R:reset  SPC:freeze")
    cv2.putText(frame,status,(8,h-12),
                cv2.FONT_HERSHEY_SIMPLEX,0.37,(180,180,180),1,cv2.LINE_AA)

# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    cap = cv2.VideoCapture(args.camera)
    if not cap.isOpened():
        print(f"[ERROR] Cannot open camera {args.camera}"); return

    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    stable      = StableLabel()
    tracker     = PatternTracker()
    raw_hist    = deque(maxlen=80)
    smooth_hist = deque(maxlen=80)
    frozen      = False
    frozen_frame= None
    tta_on      = args.tta
    snap_idx    = 0
    prev_time   = time.time()
    fps         = 0.0

    # App-level state: "SEARCHING" | "LOCKED" | "LOST"
    app_state   = "SEARCHING"

    # How often to re-run classifier while LOCKED (every N frames)
    RECHECK_EVERY = 20
    frame_count   = 0

    os.makedirs("snapshots", exist_ok=True)
    print(f"[INFO] Enter REAL @ {ENTER_THR:.2f}  |  Exit REAL @ {EXIT_THR:.2f}  |  Confirm {CONFIRM_N} frames")
    print(f"[INFO] Q=quit  S=save  T=TTA  R=reset tracker  SPACE=freeze")

    while True:
        if not frozen:
            ret, frame = cap.read()
            if not ret:
                print("[ERROR] Frame grab failed."); break
            display = frame.copy()
        else:
            display = frozen_frame.copy()

        fh,fw = display.shape[:2]
        frame_count += 1

        # ═══════════════════════════════════════════════════════════
        #  STATE: SEARCHING — run classifier every frame
        # ═══════════════════════════════════════════════════════════
        if app_state == "SEARCHING":
            dy=int(fh*0.2); dx=int(fw*0.2)
            roi      = display[dy:fh-dy, dx:fw-dx]
            raw_conf = predict_tta(roi) if tta_on else predict_single(roi)
            label, smoothed, locked = stable.update(raw_conf)

            raw_hist.append(raw_conf)
            smooth_hist.append(smoothed)
            draw_search_overlay(display, stable.confirm_ct)

            if locked and label == "REAL":
                # Pattern confirmed — detect exact box and lock tracker
                x1,y1,x2,y2 = detect_kfs_box(display)
                tracker.init(display, x1, y1, x2, y2)
                app_state = "LOCKED"

        # ═══════════════════════════════════════════════════════════
        #  STATE: LOCKED — optical flow tracking, periodic recheck
        # ═══════════════════════════════════════════════════════════
        elif app_state == "LOCKED":
            box, tracking_ok = tracker.update(display)

            if not tracking_ok:
                # Tracker lost — re-search
                app_state = "LOST"
                stable    = StableLabel()   # reset classifier state
            else:
                x1,y1,x2,y2 = box
                draw_locked_box(display, x1, y1, x2, y2,
                                stable.smoothed, tracking_ok=True)

                # Periodically re-run classifier on tracked region
                # to verify it's still REAL (not drifted onto fake)
                if frame_count % RECHECK_EVERY == 0:
                    roi = display[y1:y2, x1:x2]
                    if roi.size > 0:
                        raw_conf = predict_tta(roi) if tta_on else predict_single(roi)
                        label, smoothed, _ = stable.update(raw_conf)
                        raw_hist.append(raw_conf)
                        smooth_hist.append(smoothed)

                        if label == "FAKE":
                            # Classifier says it's fake now — unlock
                            print("[RECHECK] Region is no longer REAL — releasing lock")
                            tracker.reset()
                            stable    = StableLabel()
                            app_state = "SEARCHING"
                else:
                    raw_hist.append(stable.smoothed)
                    smooth_hist.append(stable.smoothed)

        # ═══════════════════════════════════════════════════════════
        #  STATE: LOST — brief grace period then back to SEARCHING
        # ═══════════════════════════════════════════════════════════
        elif app_state == "LOST":
            if tracker.box:
                x1,y1,x2,y2 = tracker.box
                draw_locked_box(display, x1, y1, x2, y2,
                                stable.smoothed, tracking_ok=False)

            # Try to re-acquire for up to 30 frames
            dy=int(fh*0.2); dx=int(fw*0.2)
            roi      = display[dy:fh-dy, dx:fw-dx]
            raw_conf = predict_tta(roi) if tta_on else predict_single(roi)
            label, smoothed, locked = stable.update(raw_conf)
            raw_hist.append(raw_conf)
            smooth_hist.append(smoothed)

            if locked and label == "REAL":
                # Re-acquired — restart tracker
                x1,y1,x2,y2 = detect_kfs_box(display)
                tracker.init(display, x1, y1, x2, y2)
                app_state = "LOCKED"
                print("[TRACKER] Re-acquired — locked back on REAL pattern")
            elif len(raw_hist) > 30 and np.mean(list(raw_hist)[-30:]) < EXIT_THR:
                app_state = "SEARCHING"
                stable    = StableLabel()
                print("[TRACKER] Pattern gone — back to full search")

        # FPS
        now = time.time()
        fps = 0.9*fps + 0.1/max(now-prev_time,1e-6)
        prev_time = now

        draw_hud(display, app_state, stable.smoothed,
                 raw_hist[-1] if raw_hist else 0.0,
                 fps, tta_on, frozen, raw_hist, smooth_hist, stable.confirm_ct)

        cv2.imshow("Robocon — Lock & Track", display)

        key = cv2.waitKey(1) & 0xFF
        if   key == ord('q'): break
        elif key == ord('r'):
            # Manual reset
            tracker.reset()
            stable    = StableLabel()
            app_state = "SEARCHING"
            print("[RESET] Manual reset — searching again")
        elif key == ord('s'):
            path=f"snapshots/snap_{snap_idx:04d}_{app_state}_{stable.smoothed:.2f}.png"
            cv2.imwrite(path,display)
            print(f"[SAVE] {path}"); snap_idx+=1
        elif key == ord('t'):
            tta_on = not tta_on
            print(f"[TTA] {'ON' if tta_on else 'OFF'}")
        elif key == ord(' '):
            frozen = not frozen
            if frozen: frozen_frame = display.copy()
            print(f"[{'FREEZE' if frozen else 'UNFREEZE'}]")

    cap.release()
    cv2.destroyAllWindows()

if __name__=="__main__":
    main()
