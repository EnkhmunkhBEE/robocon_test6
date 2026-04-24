import cv2
import numpy as np
import albumentations as A
import os, glob

# ── 1. Perspective warp helpers ─────────────────────────────────────────────

def warp(img, src_pts, dst_pts):
    h, w = img.shape[:2]
    M = cv2.getPerspectiveTransform(
        np.float32(src_pts), np.float32(dst_pts))
    return cv2.warpPerspective(img, M, (w, h),
                               borderMode=cv2.BORDER_REPLICATE)

def angle_variants(img):
    h, w = img.shape[:2]
    variants = {}
    m = 0.25  # margin ratio for warp

    # Top-down (camera looking down at target)
    variants["top"] = warp(img,
        [[0,0],[w,0],[w,h],[0,h]],
        [[w*m,0],[w*(1-m),0],[w,h],[0,h]])

    # Bottom-up
    variants["bottom"] = warp(img,
        [[0,0],[w,0],[w,h],[0,h]],
        [[0,0],[w,0],[w*(1-m),h],[w*m,h]])

    # Left tilt
    variants["left"] = warp(img,
        [[0,0],[w,0],[w,h],[0,h]],
        [[0,h*m],[w,0],[w,h],[0,h*(1-m)]])

    # Right tilt
    variants["right"] = warp(img,
        [[0,0],[w,0],[w,h],[0,h]],
        [[0,0],[w,h*m],[w,h*(1-m)],[0,h]])

    # Bottom-left diagonal
    variants["diag_bl"] = warp(img,
        [[0,0],[w,0],[w,h],[0,h]],
        [[w*m,0],[w,0],[w*(1-m),h],[0,h]])

    # Bottom-right diagonal
    variants["diag_br"] = warp(img,
        [[0,0],[w,0],[w,h],[0,h]],
        [[0,0],[w*(1-m),0],[w,h],[w*m,h]])

    return variants

# ── 2. Lighting & environment augmentations ─────────────────────────────────

env_aug = A.Compose([
    A.RandomBrightnessContrast(p=0.8),
    A.RandomShadow(num_shadows_lower=1, num_shadows_upper=3, p=0.6),
    A.GaussNoise(var_limit=(10, 50), p=0.5),
    A.MotionBlur(blur_limit=7, p=0.4),
    A.ImageCompression(quality_lower=60, quality_upper=95, p=0.3),
    A.Rotate(limit=20, border_mode=cv2.BORDER_REPLICATE, p=0.5),
])

# ── 3. Distance simulation ───────────────────────────────────────────────────

def simulate_distance(img, level="far"):
    h, w = img.shape[:2]
    if level == "far":
        small = cv2.resize(img, (w//2, h//2))
        out = np.full_like(img, 128)
        y0 = h//4; x0 = w//4
        out[y0:y0+h//2, x0:x0+w//2] = small
        return out
    elif level == "close":
        crop_h, crop_w = int(h*0.6), int(w*0.6)
        y0 = (h-crop_h)//2; x0 = (w-crop_w)//2
        return cv2.resize(img[y0:y0+crop_h, x0:x0+crop_w], (w,h))
    return img  # "mid" — no change

# ── 4. Main pipeline ─────────────────────────────────────────────────────────

def augment_split(split_dir, out_dir, n_env=5):
    """
    split_dir: e.g. data/train
    out_dir:   e.g. data/train_aug
    """
    for label in ["fake", "real"]:
        imgs = glob.glob(f"{split_dir}/{label}/img*/*.png")
        out_label = os.path.join(out_dir, label)
        os.makedirs(out_label, exist_ok=True)
        idx = 0

        for img_path in imgs:
            img = cv2.imread(img_path)
            base = [img]  # start with original

            # a) angle variants
            for angle_img in angle_variants(img).values():
                base.append(angle_img)

            # b) distance variants for each
            all_views = []
            for v in base:
                all_views.extend([v,
                    simulate_distance(v, "far"),
                    simulate_distance(v, "close")])

            # c) environment augmentation
            for view in all_views:
                cv2.imwrite(f"{out_label}/{idx:05d}.png", view)
                idx += 1
                for _ in range(n_env):
                    aug = env_aug(image=view)["image"]
                    cv2.imwrite(f"{out_label}/{idx:05d}.png", aug)
                    idx += 1

        print(f"[{split_dir}/{label}] → {idx} images saved to {out_label}")

# ── Run ───────────────────────────────────────────────────────────────────────
augment_split("data/train", "data/train_aug", n_env=3)
augment_split("data/val",   "data/val_aug",   n_env=2)