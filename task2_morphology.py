# Name: Sushma Ramesh, Dina Barua
# Date: February 22, 2026
# Purpose: Task 2 - Morphological open and close filtering written from scratch using NumPy
"""
Project 3 – Task 2: Morphological Filtering
=============================================
Cleans the binary mask from Task 1 using morphological open and close,
both written from scratch using numpy sliding-window operations.

  - Morphological OPEN  (erode → dilate):  removes small noise blobs
  - Morphological CLOSE (dilate → erode):  fills holes inside objects

Written from scratch: erosion and dilation use pure numpy array slicing,
no cv2.erode / cv2.dilate / cv2.morphologyEx calls.

Usage:
  python3 task2_morphology.py --image IMGS/img1p3.png
  python3 task2_morphology.py --image IMGS/img2P3.png --save_dir outputs/
  python3 task2_morphology.py --image IMGS/img1p3.png --save_dir outputs/
  python3 task2_morphology.py --image IMGS/img2P3.png --save_dir outputs/
  python3 task2_morphology.py --image IMGS/img3P3.png --save_dir outputs/
"""

import cv2
import numpy as np
import argparse
import os
import sys

# ── reuse Task 1 threshold ────────────────────────────────────────────────────
sys.path.insert(0, os.path.dirname(__file__))
from task1_threshold import threshold_frame


# ─────────────────────────────────────────────────────────────────────────────
#  PRIMITIVE OPS  (written from scratch)
# ─────────────────────────────────────────────────────────────────────────────

def _erode(binary: np.ndarray, ksize: int) -> np.ndarray:
    """
    Morphological erosion – written from scratch.
    A pixel stays WHITE only if every pixel under the ksize×ksize
    square structuring element is also WHITE.

    Implementation: pad the image, then accumulate a boolean AND
    over all (ksize²) kernel offsets using numpy array slicing –
    no cv2 functions used.
    """
    pad = ksize // 2
    H, W = binary.shape
    padded = np.pad(binary, pad, constant_values=0)   # zeros at border

    # Start with all True; AND each offset slice in
    result = np.ones((H, W), dtype=bool)
    for dy in range(ksize):
        for dx in range(ksize):
            result &= (padded[dy: dy + H, dx: dx + W] == 255)

    return result.astype(np.uint8) * 255


def _dilate(binary: np.ndarray, ksize: int) -> np.ndarray:
    """
    Morphological dilation – written from scratch.
    A pixel becomes WHITE if ANY pixel under the ksize×ksize
    square structuring element is WHITE.

    Implementation: pad the image, then accumulate a boolean OR
    over all (ksize²) kernel offsets using numpy array slicing –
    no cv2 functions used.
    """
    pad = ksize // 2
    H, W = binary.shape
    padded = np.pad(binary, pad, constant_values=0)

    # Start with all False; OR each offset slice in
    result = np.zeros((H, W), dtype=bool)
    for dy in range(ksize):
        for dx in range(ksize):
            result |= (padded[dy: dy + H, dx: dx + W] == 255)

    return result.astype(np.uint8) * 255


# ─────────────────────────────────────────────────────────────────────────────
#  OPEN & CLOSE  (written from scratch, built on _erode / _dilate above)
# ─────────────────────────────────────────────────────────────────────────────

def morph_open(binary: np.ndarray, ksize: int = 5) -> np.ndarray:
    """
    Morphological OPEN = erode then dilate.
    Effect: removes small foreground blobs (noise) without shrinking
    large objects much.
    """
    return _dilate(_erode(binary, ksize), ksize)


def morph_close(binary: np.ndarray, ksize: int = 9) -> np.ndarray:
    """
    Morphological CLOSE = dilate then erode.
    Effect: fills small holes / gaps inside foreground objects.
    """
    return _erode(_dilate(binary, ksize), ksize)


def clean_binary(binary: np.ndarray,
                 open_ksize: int = 5,
                 close_ksize: int = 9) -> np.ndarray:
    """
    Full cleaning pipeline: open first (remove noise), then close (fill holes).
    """
    opened = morph_open(binary, open_ksize)
    closed = morph_close(opened, close_ksize)
    return closed


# ─────────────────────────────────────────────────────────────────────────────
#  VISUALISATION
# ─────────────────────────────────────────────────────────────────────────────

def make_display(frame, binary_raw, binary_opened, binary_cleaned,
                 label: str = '') -> np.ndarray:
    """
    5-panel display:
    Original | Raw binary | After Open | After Close | Overlay on original
    """
    def to_bgr(mask):
        return cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)

    def annotate(img, text):
        out = img.copy()
        cv2.putText(out, text, (8, 26), cv2.FONT_HERSHEY_SIMPLEX,
                    0.6, (0, 0, 0), 4)
        cv2.putText(out, text, (8, 26), cv2.FONT_HERSHEY_SIMPLEX,
                    0.6, (50, 230, 50), 2)
        return out

    # Overlay: green tint on detected foreground
    overlay = frame.copy()
    overlay[binary_cleaned == 255] = (
        overlay[binary_cleaned == 255] * 0.4 +
        np.array([0, 200, 0]) * 0.6
    ).astype(np.uint8)

    # Component counts for annotation
    n_raw,  _, s_raw,  _ = cv2.connectedComponentsWithStats(binary_raw)
    n_open, _, s_open, _ = cv2.connectedComponentsWithStats(binary_opened)
    n_clean,_, s_clean,_ = cv2.connectedComponentsWithStats(binary_cleaned)

    p1 = annotate(frame,           'Original')
    p2 = annotate(to_bgr(binary_raw),
                  f'Task1 binary ({n_raw-1} blobs)')
    p3 = annotate(to_bgr(binary_opened),
                  f'After Open k=5 ({n_open-1} blobs)')
    p4 = annotate(to_bgr(binary_cleaned),
                  f'After Close k=9 ({n_clean-1} blobs)')
    p5 = annotate(overlay,         'Cleaned overlay')

    row = np.hstack([p1, p2, p3, p4, p5])

    if label:
        H = row.shape[0]
        cv2.putText(row, label, (10, H - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (160, 160, 160), 1)
    return row


# ─────────────────────────────────────────────────────────────────────────────
#  PER-IMAGE PROCESSING
# ─────────────────────────────────────────────────────────────────────────────

def process_image(path: str, save_dir: str | None = None,
                  show: bool = True):
    frame = cv2.imread(path)
    if frame is None:
        print(f'[ERROR] Cannot read {path}')
        return

    # Task 1 – threshold
    binary_raw, _ = threshold_frame(frame)

    # Task 2 – morphological clean (written from scratch)
    binary_opened  = morph_open(binary_raw,   ksize=5)
    binary_cleaned = morph_close(binary_opened, ksize=9)

    disp = make_display(frame, binary_raw, binary_opened, binary_cleaned,
                        label=os.path.basename(path))

    # Resize if too wide for screen
    if disp.shape[1] > 1600:
        scale = 1600 / disp.shape[1]
        disp = cv2.resize(disp, (int(disp.shape[1] * scale),
                                  int(disp.shape[0] * scale)))

    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        name = 'morph_' + os.path.basename(path)
        out_path = os.path.join(save_dir, name)
        cv2.imwrite(out_path, disp)
        print(f'[SAVED] {out_path}')

    if show:
        cv2.imshow('Task 2 – Morphological Filtering', disp)
        print("Press any key to continue, 'q' to quit.")
        key = cv2.waitKey(0) & 0xFF
        cv2.destroyAllWindows()
        return key


# ─────────────────────────────────────────────────────────────────────────────
#  ENTRY POINT
# ─────────────────────────────────────────────────────────────────────────────

DEFAULT_IMAGES = [
    'IMGS/img1p3.png', 'IMGS/img2P3.png', 'IMGS/img3P3.png',
    'IMGS/img4P3.png', 'IMGS/img5P3.png',
]

if __name__ == '__main__':
    ap = argparse.ArgumentParser(
        description='Project 3 Task 2 – Morphological Filtering')
    ap.add_argument('--image',    default=None,
                    help='Path to a single image')
    ap.add_argument('--dir',      default=None,
                    help='Directory of images')
    ap.add_argument('--save_dir', default=None,
                    help='Save output images to this folder')
    args = ap.parse_args()

    # Collect image paths
    if args.image:
        images = [args.image]
    elif args.dir:
        images = sorted([
            os.path.join(args.dir, f)
            for f in os.listdir(args.dir)
            if f.lower().endswith(('.png', '.jpg', '.jpeg'))
            and not f.startswith('example')          # skip pre-thresholded
        ])
    else:
        images = [p for p in DEFAULT_IMAGES if os.path.exists(p)]

    if not images:
        print('[ERROR] No images found.')
        sys.exit(1)

    show = args.save_dir is None   # only show window if not saving

    for path in images:
        key = process_image(path, save_dir=args.save_dir, show=show)
        if key == ord('q'):
            break