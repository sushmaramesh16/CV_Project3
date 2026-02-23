
# Name: Sushma Ramesh, Dina Barua
# Date: February 22, 2026
# Purpose: Task 1 - Dynamic HSV thresholding using ISODATA algorithm written from scratch

"""
Project 3 – Task 1: Thresholding
=================================
Separates objects from a white-paper background in a translation, scale,
and rotation invariant manner.

Algorithm (written from scratch):
  1. Gaussian blur to reduce sensor noise
  2. Convert to HSV colour space
  3. Run ISODATA (K-means K=2) independently on the V channel and the S channel
     using a 1/16 random pixel sample to find the natural cluster boundary in
     each channel without any hard-coded constant.
  4. Foreground = pixels that are:
       - dark  (V < T_V − δ_V)  catches dark/black objects, OR
       - colourful (S > T_S + δ_S)  catches brightly coloured objects
     Both conditions are OR-combined so neither dark nor colourful objects
     are missed.

Usage (display mode – requires a display):
  python task1_threshold.py                     # show all dev images
  python task1_threshold.py --image path/to/img.png
  python task1_threshold.py --video 0           # live webcam

Output (headless / report mode):
  python task1_threshold.py --save_dir outputs/ # save side-by-side PNGs
"""

import cv2
import numpy as np
import argparse
import os
import sys


# ─────────────────────────────────────────────────────────────────────────────
#  ISODATA  (written from scratch – no OpenCV threshold functions used here)
# ─────────────────────────────────────────────────────────────────────────────

def isodata_threshold(channel: np.ndarray,
                      sample_frac: float = 0.0625,
                      seed: int = 0) -> float:
    """
    Iterative ISODATA / K-means (K=2) threshold.

    Takes a random sample of (sample_frac × N) pixels from *channel*,
    initialises T at the sample mean, then alternates between:
      - assigning pixels to 'below T' and 'above T' clusters
      - updating T to the midpoint of the two cluster means
    until T converges (change < 0.5 grey levels) or 200 iterations elapse.

    Parameters
    ----------
    channel : 2-D float array (H × W), values in [0, 255]
    sample_frac : fraction of pixels to sample (default 1/16 as suggested)
    seed : RNG seed for reproducibility

    Returns
    -------
    T : float – threshold separating the two dominant intensity clusters
    """
    flat = channel.flatten().astype(np.float32)
    n_sample = max(200, int(len(flat) * sample_frac))
    rng = np.random.default_rng(seed)
    samples = flat[rng.choice(len(flat), min(n_sample, len(flat)), replace=False)]

    T = float(samples.mean())
    for _ in range(200):
        below = samples[samples <= T]
        above = samples[samples > T]
        if len(below) == 0 or len(above) == 0:
            break
        T_new = (below.mean() + above.mean()) / 2.0
        if abs(T_new - T) < 0.5:
            break
        T = T_new

    return T


# ─────────────────────────────────────────────────────────────────────────────
#  MAIN THRESHOLD FUNCTION
# ─────────────────────────────────────────────────────────────────────────────

def threshold_frame(frame: np.ndarray,
                    delta_v: float = 15.0,
                    delta_s: float = 10.0) -> tuple[np.ndarray, dict]:
    """
    Threshold a BGR frame to produce a foreground binary mask.

    Steps
    -----
    1. Gaussian blur (7×7) to smooth noise before colour analysis.
    2. Convert to HSV – separates colour (H, S) from brightness (V).
    3. ISODATA on V channel → T_V  (separates bright paper from dark objects)
    4. ISODATA on S channel → T_S  (separates neutral paper from colourful objects)
    5. Foreground mask = (V < T_V − delta_v)  OR  (S > T_S + delta_s)

    Parameters
    ----------
    frame   : BGR image (H × W × 3, uint8)
    delta_v : gap below T_V to call a pixel 'dark'  (default 15)
    delta_s : gap above T_S to call a pixel 'colourful' (default 10)

    Returns
    -------
    binary : uint8 mask, 255 = foreground object, 0 = background
    info   : dict with threshold values and intermediate images for display
    """
    # Step 1 – blur
    blurred = cv2.GaussianBlur(frame, (7, 7), 0)

    # Step 2 – HSV conversion
    hsv = cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV)
    S = hsv[:, :, 1].astype(np.float32)   # saturation 0-255
    V = hsv[:, :, 2].astype(np.float32)   # value/brightness 0-255

    # Step 3 & 4 – ISODATA thresholds (written from scratch)
    T_V = isodata_threshold(V)   # cluster boundary in brightness
    T_S = isodata_threshold(S)   # cluster boundary in saturation

    # Step 5 – combine: dark OR colourful → foreground
    dark_mask   = (V < T_V - delta_v).astype(np.uint8) * 255
    colour_mask = (S > T_S + delta_s).astype(np.uint8) * 255
    binary      = cv2.bitwise_or(dark_mask, colour_mask)

    info = {
        'T_V': T_V, 'T_S': T_S,
        'dark_mask': dark_mask,
        'colour_mask': colour_mask,
        'blurred': blurred,
        'hsv': hsv,
    }
    return binary, info


# ─────────────────────────────────────────────────────────────────────────────
#  VISUALISATION
# ─────────────────────────────────────────────────────────────────────────────

def make_display(frame: np.ndarray, binary: np.ndarray, info: dict,
                 label: str = '') -> np.ndarray:
    """
    Build a 4-panel side-by-side image for reporting:
      [ Original | V channel | S channel | Binary (foreground) ]
    """
    H, W = frame.shape[:2]
    hsv   = info['hsv']
    V_img = cv2.cvtColor(hsv[:, :, 2], cv2.COLOR_GRAY2BGR)
    S_img = cv2.cvtColor(hsv[:, :, 1], cv2.COLOR_GRAY2BGR)

    T_V, T_S = info['T_V'], info['T_S']

    # Annotate panels
    def annotate(img, text, color=(50, 230, 50)):
        out = img.copy()
        cv2.putText(out, text, (10, 28),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 0, 0), 4)
        cv2.putText(out, text, (10, 28),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.65, color, 2)
        return out

    panel_orig   = annotate(frame, 'Original')
    panel_V      = annotate(V_img, f'V channel  T_V={T_V:.0f}')
    panel_S      = annotate(S_img, f'S channel  T_S={T_S:.0f}')
    panel_binary = annotate(cv2.cvtColor(binary, cv2.COLOR_GRAY2BGR),
                            'Thresholded (dark OR colour)')

    row = np.hstack([panel_orig, panel_V, panel_S, panel_binary])

    if label:
        cv2.putText(row, label, (10, H - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (180, 180, 180), 1)
    return row


# ─────────────────────────────────────────────────────────────────────────────
#  REPORT IMAGE GENERATION (save mode)
# ─────────────────────────────────────────────────────────────────────────────

def save_report_images(image_paths: list[str], save_dir: str):
    os.makedirs(save_dir, exist_ok=True)
    for path in image_paths:
        frame = cv2.imread(path)
        if frame is None:
            print(f'[WARN] Cannot read {path}')
            continue
        binary, info = threshold_frame(frame)
        disp = make_display(frame, binary, info, label=os.path.basename(path))
        out_path = os.path.join(save_dir,
                                'thresh_' + os.path.basename(path))
        cv2.imwrite(out_path, disp)
        print(f'[SAVED] {out_path}  T_V={info["T_V"]:.1f}  T_S={info["T_S"]:.1f}')


# ─────────────────────────────────────────────────────────────────────────────
#  INTERACTIVE / REAL-TIME DISPLAY
# ─────────────────────────────────────────────────────────────────────────────

def run_on_images(image_paths: list[str]):
    print("Press any key to advance, 'q' to quit.")
    for path in image_paths:
        frame = cv2.imread(path)
        if frame is None:
            continue
        binary, info = threshold_frame(frame)
        disp = make_display(frame, binary, info, label=os.path.basename(path))
        # Resize if too wide
        if disp.shape[1] > 1400:
            scale = 1400 / disp.shape[1]
            disp = cv2.resize(disp, (int(disp.shape[1]*scale),
                                     int(disp.shape[0]*scale)))
        cv2.imshow('Task 1 – Thresholding', disp)
        key = cv2.waitKey(0) & 0xFF
        if key == ord('q'):
            break
    cv2.destroyAllWindows()


def run_on_video(source):
    try:
        source = int(source)
    except (ValueError, TypeError):
        pass
    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        print(f'[ERROR] Cannot open {source}')
        return
    print("Press 'q' to quit, 's' to save screenshot.")
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        binary, info = threshold_frame(frame)
        disp = make_display(frame, binary, info)
        if disp.shape[1] > 1400:
            scale = 1400 / disp.shape[1]
            disp = cv2.resize(disp, (int(disp.shape[1]*scale),
                                     int(disp.shape[0]*scale)))
        cv2.imshow('Task 1 – Thresholding (live)', disp)
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('s'):
            cv2.imwrite('thresh_screenshot.png', disp)
            print('[INFO] Saved thresh_screenshot.png')
    cap.release()
    cv2.destroyAllWindows()


# ─────────────────────────────────────────────────────────────────────────────
#  ENTRY POINT
# ─────────────────────────────────────────────────────────────────────────────

DEFAULT_IMAGES = [
    'proj03/img1p3.png', 'proj03/img2P3.png', 'proj03/img3P3.png',
    'proj03/img4P3.png', 'proj03/img5P3.png',
]

if __name__ == '__main__':
    ap = argparse.ArgumentParser(description='Project 3 Task 1 – Thresholding')
    ap.add_argument('--image',    default=None, help='Path to a single image')
    ap.add_argument('--video',    default=None, help='Video file or camera index')
    ap.add_argument('--dir',      default=None, help='Directory of images')
    ap.add_argument('--save_dir', default=None, help='Save report images to this dir')
    ap.add_argument('--delta_v',  type=float, default=15.0,
                    help='Margin below T_V to flag as dark foreground (default 15)')
    ap.add_argument('--delta_s',  type=float, default=10.0,
                    help='Margin above T_S to flag as colourful foreground (default 10)')
    args = ap.parse_args()

    if args.video:
        run_on_video(args.video)
    else:
        # Collect images
        if args.image:
            images = [args.image]
        elif args.dir:
            images = sorted([
                os.path.join(args.dir, f)
                for f in os.listdir(args.dir)
                if f.lower().endswith(('.png', '.jpg', '.jpeg'))
            ])
        else:
            images = [p for p in DEFAULT_IMAGES if os.path.exists(p)]

        if not images:
            print('[ERROR] No images found.')
            sys.exit(1)

        if args.save_dir:
            save_report_images(images, args.save_dir)
        else:
            run_on_images(images)