"""
Project 3 – Task 3: Connected Components Analysis
===================================================
Uses OpenCV connectedComponentsWithStats on the cleaned binary from Task 2
to find and label distinct foreground regions.

Each valid region gets:
  - A unique colour in the component map
  - A bounding box drawn on the original image
  - Its centroid marked with a dot
  - Area and label printed as overlay text

Valid region filtering:
  - min_area = 3000 pixels  (removes tiny specks)
  - max_area = 40% of image (removes the large cardboard shadow blob)

Usage:
  python3 task3_components.py --image IMGS/img1p3.png
  python3 task3_components.py --image IMGS/img2P3.png --save_dir outputs/
  python3 task3_components.py --image IMGS/img1p3.png --save_dir outputs/
  python3 task3_components.py --image IMGS/img2P3.png --save_dir outputs/
  python3 task3_components.py --image IMGS/img3P3.png --save_dir outputs/
"""

import cv2
import numpy as np
import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(__file__))
from task1_threshold import threshold_frame
from task2_morphology import clean_binary


# ─────────────────────────────────────────────────────────────────────────────
#  COLOURS for labelling regions
# ─────────────────────────────────────────────────────────────────────────────

COLORS = [
    (255,  80,  80),   # red
    ( 80, 255,  80),   # green
    ( 80,  80, 255),   # blue
    (255, 255,  80),   # yellow
    (255,  80, 255),   # magenta
    ( 80, 255, 255),   # cyan
]


# ─────────────────────────────────────────────────────────────────────────────
#  CONNECTED COMPONENTS  (OpenCV – allowed for non-scratch tasks)
# ─────────────────────────────────────────────────────────────────────────────

def get_regions(binary: np.ndarray,
                min_area: int = 3000,
                max_area_ratio: float = 0.40):
    """
    Run connected components on a binary mask and return valid regions.

    Parameters
    ----------
    binary         : uint8 binary mask (255 = foreground)
    min_area       : minimum pixel area to keep a region
    max_area_ratio : regions larger than this fraction of the image are
                     treated as background (e.g. cardboard shadow blob)

    Returns
    -------
    list of dicts, each with:
        label_id  – component label integer
        area      – pixel area
        bbox      – (x, y, w, h) bounding rectangle
        centroid  – (cx, cy) float centroid
        mask      – uint8 binary mask of just this region
    """
    H, W = binary.shape
    max_area = int(H * W * max_area_ratio)

    n_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
        binary, connectivity=8)

    regions = []
    for i in range(1, n_labels):          # skip label 0 = background
        area = int(stats[i, cv2.CC_STAT_AREA])
        if area < min_area or area > max_area:
            continue

        x  = int(stats[i, cv2.CC_STAT_LEFT])
        y  = int(stats[i, cv2.CC_STAT_TOP])
        bw = int(stats[i, cv2.CC_STAT_WIDTH])
        bh = int(stats[i, cv2.CC_STAT_HEIGHT])
        cx, cy = float(centroids[i][0]), float(centroids[i][1])

        mask = (labels == i).astype(np.uint8) * 255

        regions.append({
            'label_id': i,
            'area':     area,
            'bbox':     (x, y, bw, bh),
            'centroid': (cx, cy),
            'mask':     mask,
        })

    return regions


# ─────────────────────────────────────────────────────────────────────────────
#  VISUALISATION
# ─────────────────────────────────────────────────────────────────────────────

def draw_regions(frame: np.ndarray, regions: list) -> np.ndarray:
    """Draw bounding boxes, centroids, and labels on the original frame."""
    vis = frame.copy()
    for i, r in enumerate(regions):
        c = COLORS[i % len(COLORS)]
        x, y, bw, bh = r['bbox']
        cx, cy = int(r['centroid'][0]), int(r['centroid'][1])

        # Bounding box
        cv2.rectangle(vis, (x, y), (x + bw, y + bh), c, 2)

        # Centroid dot
        cv2.circle(vis, (cx, cy), 7, c, -1)
        cv2.circle(vis, (cx, cy), 7, (0, 0, 0), 2)   # black outline

        # Label text
        label = f"Region {i+1}  area={r['area']}"
        cv2.putText(vis, label, (x, max(y - 8, 16)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 3)
        cv2.putText(vis, label, (x, max(y - 8, 16)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, c, 1)

    return vis


def make_cc_map(binary: np.ndarray, regions: list) -> np.ndarray:
    """
    Colour-coded connected component map.
    Each valid region gets its own colour; background stays black.
    """
    cc_map = np.zeros((*binary.shape, 3), dtype=np.uint8)
    for i, r in enumerate(regions):
        c = COLORS[i % len(COLORS)]
        cc_map[r['mask'] == 255] = c
    return cc_map


def make_display(frame, binary_clean, regions, label='') -> np.ndarray:
    """
    4-panel display:
    Original | Cleaned binary | CC colour map | Annotated original
    """
    def annotate(img, text):
        out = img.copy()
        cv2.putText(out, text, (8, 26), cv2.FONT_HERSHEY_SIMPLEX,
                    0.6, (0, 0, 0), 4)
        cv2.putText(out, text, (8, 26), cv2.FONT_HERSHEY_SIMPLEX,
                    0.6, (50, 230, 50), 2)
        return out

    cleaned_bgr = cv2.cvtColor(binary_clean, cv2.COLOR_GRAY2BGR)
    cc_map      = make_cc_map(binary_clean, regions)
    annotated   = draw_regions(frame, regions)

    p1 = annotate(frame,        'Original')
    p2 = annotate(cleaned_bgr,  'Cleaned binary (Task 2)')
    p3 = annotate(cc_map,       f'CC map  ({len(regions)} valid regions)')
    p4 = annotate(annotated,    'Bounding boxes + centroids')

    row = np.hstack([p1, p2, p3, p4])

    if label:
        cv2.putText(row, label, (10, row.shape[0] - 8),
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

    # Task 2 – morphological clean
    binary_clean = clean_binary(binary_raw)

    # Task 3 – connected components
    regions = get_regions(binary_clean)

    print(f'[{os.path.basename(path)}] Found {len(regions)} valid regions:')
    for i, r in enumerate(regions):
        cx, cy = r['centroid']
        print(f'   Region {i+1}: area={r["area"]}  '
              f'centroid=({cx:.0f},{cy:.0f})  bbox={r["bbox"]}')

    disp = make_display(frame, binary_clean, regions,
                        label=os.path.basename(path))

    if disp.shape[1] > 1400:
        scale = 1400 / disp.shape[1]
        disp = cv2.resize(disp, (int(disp.shape[1] * scale),
                                  int(disp.shape[0] * scale)))

    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        out_path = os.path.join(save_dir,
                                'cc_' + os.path.basename(path))
        cv2.imwrite(out_path, disp)
        print(f'[SAVED] {out_path}')

    if show:
        cv2.imshow('Task 3 – Connected Components', disp)
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
        description='Project 3 Task 3 – Connected Components')
    ap.add_argument('--image',    default=None)
    ap.add_argument('--dir',      default=None)
    ap.add_argument('--save_dir', default=None)
    args = ap.parse_args()

    if args.image:
        images = [args.image]
    elif args.dir:
        images = sorted([
            os.path.join(args.dir, f)
            for f in os.listdir(args.dir)
            if f.lower().endswith(('.png', '.jpg', '.jpeg'))
            and not f.startswith('example')
        ])
    else:
        images = [p for p in DEFAULT_IMAGES if os.path.exists(p)]

    if not images:
        print('[ERROR] No images found.')
        sys.exit(1)

    show = args.save_dir is None

    for path in images:
        key = process_image(path, save_dir=args.save_dir, show=show)
        if key == ord('q'):
            break