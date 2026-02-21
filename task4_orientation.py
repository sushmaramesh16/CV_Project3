"""
Project 3 – Task 4: Orientation & Moment Features
===================================================
Computes the principal axis orientation and moment-based features
for each detected region using central moments.

Orientation (written from scratch using central moments):
  angle = 0.5 * arctan2(2*mu11,  mu20 - mu02)

This gives the angle of the principal (longest) axis of the region,
which is rotation-invariant and requires no hard-coded assumptions
about object shape.

Features computed per region:
  - Orientation angle (degrees)
  - Centroid (cx, cy)
  - Area
  - 7 Hu moments (log-scaled, sign-preserved) — scale + rotation invariant
  - Aspect ratio of bounding box
  - Extent  (area / bounding box area)
  - Solidity (area / convex hull area)

Usage:
  python3 task4_orientation.py --image IMGS/img1p3.png
  python3 task4_orientation.py --image IMGS/img2P3.png --save_dir outputs/
  python3 task4_orientation.py --image IMGS/img1p3.png --save_dir outputs/
  python3 task4_orientation.py --image IMGS/img2P3.png --save_dir outputs/
  python3 task4_orientation.py --image IMGS/img3P3.png --save_dir outputs/
"""

import cv2
import numpy as np
import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(__file__))
from task1_threshold import threshold_frame
from task2_morphology import clean_binary
from task3_components import get_regions, COLORS


# ─────────────────────────────────────────────────────────────────────────────
#  ORIENTATION  (written from scratch using central moments)
# ─────────────────────────────────────────────────────────────────────────────

def compute_orientation(mask: np.ndarray) -> float:
    """
    Compute principal axis orientation from central moments.

    Uses the second-order central moments mu20, mu02, mu11:
        angle = 0.5 * arctan2(2*mu11, mu20 - mu02)

    This is the angle of the eigenvector of the covariance matrix of the
    pixel positions — i.e. the direction of maximum spread (longest axis).

    Returns angle in degrees in range (-90, 90].
    Written from scratch: no cv2.fitEllipse or cv2.minAreaRect used.
    """
    M = cv2.moments(mask)
    mu20 = M['mu20']
    mu02 = M['mu02']
    mu11 = M['mu11']

    angle_rad = 0.5 * np.arctan2(2.0 * mu11, mu20 - mu02)
    angle_deg = np.degrees(angle_rad)
    return angle_deg


# ─────────────────────────────────────────────────────────────────────────────
#  FEATURE EXTRACTION
# ─────────────────────────────────────────────────────────────────────────────

def extract_features(mask: np.ndarray) -> dict:
    """
    Extract a full set of shape features from a binary region mask.

    Returns a dict with:
      angle_deg  – principal axis orientation in degrees
      hu         – 7 log-scaled Hu moments (rotation+scale invariant)
      aspect     – min_side / max_side of bounding box  (0-1)
      extent     – area / bounding_box_area              (0-1)
      solidity   – area / convex_hull_area               (0-1)
      area       – pixel area
    """
    M = cv2.moments(mask)

    # Orientation (from scratch)
    angle_deg = compute_orientation(mask)

    # Hu moments (OpenCV computes them from central moments)
    hu_raw = cv2.HuMoments(M).flatten()
    # Log-scale: preserves sign, compresses dynamic range
    hu_log = -np.sign(hu_raw) * np.log10(np.abs(hu_raw) + 1e-10)

    # Shape descriptors from contour
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL,
                                   cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return {'angle_deg': angle_deg, 'hu': hu_log,
                'aspect': 0, 'extent': 0, 'solidity': 0,
                'area': 0}

    cnt   = max(contours, key=cv2.contourArea)
    area  = cv2.contourArea(cnt)
    x, y, bw, bh = cv2.boundingRect(cnt)

    aspect  = min(bw, bh) / max(bw, bh) if max(bw, bh) > 0 else 0
    extent  = area / (bw * bh)           if bw * bh > 0    else 0

    hull      = cv2.convexHull(cnt)
    hull_area = cv2.contourArea(hull)
    solidity  = area / hull_area          if hull_area > 0  else 0

    return {
        'angle_deg': angle_deg,
        'hu':        hu_log,
        'aspect':    aspect,
        'extent':    extent,
        'solidity':  solidity,
        'area':      int(area),
    }


# ─────────────────────────────────────────────────────────────────────────────
#  DRAWING
# ─────────────────────────────────────────────────────────────────────────────

def draw_oriented_bbox(img: np.ndarray, mask: np.ndarray, color) -> np.ndarray:
    """
    Draw the rotated (oriented) bounding box that wraps tightly around
    the region and rotates with the object using cv2.minAreaRect.
    """
    out = img.copy()
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL,
                                   cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return out
    cnt  = max(contours, key=cv2.contourArea)
    rect = cv2.minAreaRect(cnt)          # (center, (w,h), angle)
    box  = np.int32(cv2.boxPoints(rect)) # 4 corner points
    cv2.drawContours(out, [box], 0, color, 2)
    return out


def draw_orientation(img: np.ndarray, cx: float, cy: float,
                     angle_deg: float, color,
                     length: int = 80) -> np.ndarray:
    """
    Draw the principal axis arrow (both directions) and perpendicular
    minor axis line from the centroid.
    """
    out = img.copy()
    angle_rad = np.radians(angle_deg)

    # Major axis arrows (both directions)
    ex  = int(cx + length * np.cos(angle_rad))
    ey  = int(cy + length * np.sin(angle_rad))
    ex2 = int(cx - length * np.cos(angle_rad))
    ey2 = int(cy - length * np.sin(angle_rad))
    cv2.arrowedLine(out, (int(cx), int(cy)), (ex,  ey),
                    color, 3, tipLength=0.25)
    cv2.arrowedLine(out, (int(cx), int(cy)), (ex2, ey2),
                    color, 3, tipLength=0.25)

    # Minor axis line (perpendicular)
    perp = angle_rad + np.pi / 2
    px  = int(cx + (length // 2) * np.cos(perp))
    py  = int(cy + (length // 2) * np.sin(perp))
    px2 = int(cx - (length // 2) * np.cos(perp))
    py2 = int(cy - (length // 2) * np.sin(perp))
    cv2.line(out, (px, py), (px2, py2), color, 2)

    # Centroid dot
    cv2.circle(out, (int(cx), int(cy)), 7, color, -1)
    cv2.circle(out, (int(cx), int(cy)), 7, (0, 0, 0), 2)

    return out


def annotate_features(img: np.ndarray, region: dict,
                      features: dict, color, idx: int) -> np.ndarray:
    """
    Print the full feature vector as text on the image next to the region.
    Shows: angle, area, aspect, extent, solidity, and first 4 Hu moments.
    """
    out = img.copy()
    x, y, bw, bh = region['bbox']

    hu = features['hu']
    lines = [
        f"R{idx+1}  angle={features['angle_deg']:.1f}deg",
        f"  area={features['area']}",
        f"  aspect={features['aspect']:.2f}  extent={features['extent']:.2f}",
        f"  solidity={features['solidity']:.2f}",
        f"  Hu1={hu[0]:.2f}  Hu2={hu[1]:.2f}",
        f"  Hu3={hu[2]:.2f}  Hu4={hu[3]:.2f}",
    ]

    # Place text above the bounding box; clamp to image top
    start_y = max(y - len(lines) * 17, 14)
    for i, line in enumerate(lines):
        ypos = start_y + i * 17
        cv2.putText(out, line, (x, ypos),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 0), 3)
        cv2.putText(out, line, (x, ypos),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)
    return out


# ─────────────────────────────────────────────────────────────────────────────
#  VISUALISATION
# ─────────────────────────────────────────────────────────────────────────────

def make_display(frame, binary_clean, regions, all_features, label=''):
    """
    4-panel display:
    Original | CC map | Orientation axes | Feature text
    """
    def annotate_title(img, text):
        out = img.copy()
        cv2.putText(out, text, (8, 26), cv2.FONT_HERSHEY_SIMPLEX,
                    0.6, (0, 0, 0), 4)
        cv2.putText(out, text, (8, 26), cv2.FONT_HERSHEY_SIMPLEX,
                    0.6, (50, 230, 50), 2)
        return out

    # Panel 1 – original
    p1 = annotate_title(frame, 'Original')

    # Panel 2 – CC map
    cc_map = np.zeros_like(frame)
    for i, r in enumerate(regions):
        cc_map[r['mask'] == 255] = COLORS[i % len(COLORS)]
    p2 = annotate_title(cc_map, 'CC regions')

    # Panel 3 – orientation axes + rotated bounding box
    p3 = frame.copy()
    for i, (r, feat) in enumerate(zip(regions, all_features)):
        c = COLORS[i % len(COLORS)]
        cx, cy = r['centroid']
        # Rotated bounding box (rotates with the object)
        p3 = draw_oriented_bbox(p3, r['mask'], c)
        # Orientation axes
        p3 = draw_orientation(p3, cx, cy, feat['angle_deg'], c, length=70)
        # Angle label
        cv2.putText(p3, f"{feat['angle_deg']:.1f}deg",
                    (int(cx)+10, int(cy)-10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0,0,0), 3)
        cv2.putText(p3, f"{feat['angle_deg']:.1f}deg",
                    (int(cx)+10, int(cy)-10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, c, 1)
    p3 = annotate_title(p3, 'Axis + oriented bbox')

    # Panel 4 – feature annotations
    p4 = frame.copy()
    for i, (r, feat) in enumerate(zip(regions, all_features)):
        c = COLORS[i % len(COLORS)]
        p4 = annotate_features(p4, r, feat, c, i)
        # Draw centroid
        cv2.circle(p4, (int(r['centroid'][0]), int(r['centroid'][1])),
                   6, c, -1)
    p4 = annotate_title(p4, 'Shape features')

    row = np.hstack([p1, p2, p3, p4])

    if label:
        cv2.putText(row, label, (10, row.shape[0] - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (160, 160, 160), 1)
    return row


# ─────────────────────────────────────────────────────────────────────────────
#  PER-IMAGE PROCESSING
# ─────────────────────────────────────────────────────────────────────────────

def process_image(path: str, save_dir=None, show=True):
    frame = cv2.imread(path)
    if frame is None:
        print(f'[ERROR] Cannot read {path}'); return

    binary_raw, _  = threshold_frame(frame)
    binary_clean   = clean_binary(binary_raw)
    regions        = get_regions(binary_clean)
    all_features   = [extract_features(r['mask']) for r in regions]

    print(f'\n[{os.path.basename(path)}] {len(regions)} regions:')
    for i, (r, feat) in enumerate(zip(regions, all_features)):
        print(f'  Region {i+1}: angle={feat["angle_deg"]:.1f}°  '
              f'area={feat["area"]}  '
              f'aspect={feat["aspect"]:.2f}  '
              f'extent={feat["extent"]:.2f}  '
              f'solidity={feat["solidity"]:.2f}')
        print(f'    Hu (log): {np.round(feat["hu"][:4], 2)}...')

    disp = make_display(frame, binary_clean, regions, all_features,
                        label=os.path.basename(path))

    if disp.shape[1] > 1400:
        scale = 1400 / disp.shape[1]
        disp = cv2.resize(disp, (int(disp.shape[1]*scale),
                                  int(disp.shape[0]*scale)))

    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        out_path = os.path.join(save_dir,
                                'orient_' + os.path.basename(path))
        cv2.imwrite(out_path, disp)
        print(f'[SAVED] {out_path}')

    if show:
        cv2.imshow('Task 4 – Orientation & Features', disp)
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
        description='Project 3 Task 4 – Orientation & Moments')
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
        print('[ERROR] No images found.'); sys.exit(1)

    show = args.save_dir is None

    for path in images:
        key = process_image(path, save_dir=args.save_dir, show=show)
        if key == ord('q'):
            break