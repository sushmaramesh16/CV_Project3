# Name: Sushma Ramesh, Dina Barua
# Date: February 22, 2026
# Purpose: Extension 1 - Unknown object detection using scaled Euclidean distance thresholding
"""
Extension: Unknown Object Detection
=====================================
Classifies objects using scaled Euclidean distance (nearest-neighbour).
If the best match distance exceeds a threshold, the object is labelled
"UNKNOWN" instead of a known category.

This builds directly on Tasks 1-5.

Distance metric (scaled Euclidean):
    d(x1, x2) = sqrt( sum( ((x1_i - x2_i) / std_i)^2 ) )

Dividing by std per feature normalises each dimension so no single
feature dominates due to scale differences.

Usage:
  python3 extension_unknown.py --image IMGS/img1p3.png
  python3 extension_unknown.py --dir IMGS/
  python3 extension_unknown.py --image IMGS/img1p3.png --save_dir outputs/
  python3 extension_unknown.py --threshold 3.5
"""

import cv2
import numpy as np
import argparse
import os
import sys
import json

sys.path.insert(0, os.path.dirname(__file__))
from task1_threshold import threshold_frame
from task2_morphology import clean_binary
from task3_components import get_regions, COLORS
from task4_orientation import extract_features, draw_orientation, draw_oriented_bbox
from task5_training import load_db, pick_object_region

DB_PATH = 'object_db.json'


# ─────────────────────────────────────────────────────────────────────────────
#  CLASSIFIER  (scaled Euclidean distance / nearest-neighbour)
# ─────────────────────────────────────────────────────────────────────────────

def build_classifier(db: list):
    """
    Pre-compute the per-feature standard deviation across all training
    samples so we can use scaled Euclidean distance at run time.
    Returns (vectors, labels, stds).
    """
    def to_vec(e):
        return np.array(e['hu'] + [e['aspect'], e['extent'], e['solidity']],
                        dtype=np.float32)

    vectors = np.array([to_vec(e) for e in db])
    labels  = [e['label'] for e in db]
    stds    = vectors.std(axis=0)
    stds[stds < 1e-6] = 1.0   # avoid divide-by-zero for constant features
    return vectors, labels, stds


def classify(features: dict, vectors, labels, stds,
             threshold: float = 3.0):
    """
    Classify using TWO distance metrics for comparison:

    1. Scaled Euclidean (recommended):
       d = sqrt( sum( ((x1-x2)/std)^2 ) )
       Normalises each feature by std so no single feature dominates.

    2. Plain Euclidean:
       d = sqrt( sum( (x1-x2)^2 ) )
       No normalisation — features with larger values dominate.

    Returns (label, scaled_dist, plain_dist, plain_label)
    label = 'UNKNOWN' if best scaled distance > threshold.
    """
    query = np.array(list(features['hu']) + [
        features['aspect'],
        features['extent'],
        features['solidity'],
    ], dtype=np.float32)

    scaled_dists = []
    plain_dists  = []

    for vec, lbl in zip(vectors, labels):
        diff_s = (query - vec) / stds
        scaled_dists.append((float(np.sqrt(np.sum(diff_s**2))), lbl))
        diff_p = query - vec
        plain_dists.append((float(np.sqrt(np.sum(diff_p**2))), lbl))

    best_s_dist, best_s_label = min(scaled_dists, key=lambda x: x[0])
    best_p_dist, best_p_label = min(plain_dists,  key=lambda x: x[0])

    label = 'UNKNOWN' if best_s_dist > threshold else best_s_label
    return label, best_s_dist, best_p_dist, best_p_label


# ─────────────────────────────────────────────────────────────────────────────
#  VISUALISATION
# ─────────────────────────────────────────────────────────────────────────────

def draw_result(frame, region, features, label, scaled_dist,
                plain_dist, plain_label):
    """
    Draw oriented bbox, axis, and classification result on the frame.
    Shows BOTH distance metrics side by side for comparison.
    Known objects → green.  UNKNOWN → red.
    """
    vis    = frame.copy()
    color  = (0, 255, 0) if label != 'UNKNOWN' else (0, 0, 255)
    cx, cy = region['centroid']

    # Oriented bounding box
    vis = draw_oriented_bbox(vis, region['mask'], color)

    # Orientation axis
    vis = draw_orientation(vis, cx, cy, features['angle_deg'],
                           color, length=80)

    # Main label (scaled Euclidean result)
    x, y, bw, bh = region['bbox']
    banner = f"{label}  (scaled_d={scaled_dist:.2f})"
    cv2.putText(vis, banner, (x, max(y - 32, 20)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 4)
    cv2.putText(vis, banner, (x, max(y - 32, 20)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

    # Plain Euclidean result (comparison)
    match = '✓' if plain_label == label else '✗'
    plain_txt = f"plain Euclidean → {plain_label} (d={plain_dist:.2f}) {match}"
    cv2.putText(vis, plain_txt, (x, max(y - 10, 38)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 0), 3)
    cv2.putText(vis, plain_txt, (x, max(y - 10, 38)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 200, 80), 1)

    # Feature summary at bottom
    feat_txt = (f"aspect={features['aspect']:.2f}  "
                f"extent={features['extent']:.2f}  "
                f"solidity={features['solidity']:.2f}  "
                f"angle={features['angle_deg']:.1f}deg")
    cv2.putText(vis, feat_txt, (10, vis.shape[0] - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 0), 3)
    cv2.putText(vis, feat_txt, (10, vis.shape[0] - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 200, 200), 1)

    return vis


# ─────────────────────────────────────────────────────────────────────────────
#  PER-IMAGE PROCESSING
# ─────────────────────────────────────────────────────────────────────────────

def process_image(path, vectors, labels, stds,
                  threshold=3.0, save_dir=None, show=True):

    frame = cv2.imread(path)
    if frame is None:
        print(f'[ERROR] Cannot read {path}'); return

    binary, _    = threshold_frame(frame)
    cleaned      = clean_binary(binary)
    regions      = get_regions(cleaned)
    obj          = pick_object_region(regions)

    if obj is None:
        print(f'[SKIP] No region in {path}'); return

    features              = extract_features(obj['mask'])
    label, s_dist, p_dist, p_label = classify(features, vectors, labels,
                                               stds, threshold)

    match = '✓' if p_label == label else '✗ DISAGREE'
    print(f'[{os.path.basename(path)}]')
    print(f'  Scaled Euclidean → {label:12s} dist={s_dist:.2f}')
    print(f'  Plain  Euclidean → {p_label:12s} dist={p_dist:.2f}  {match}')

    result = draw_result(frame, obj, features, label,
                         s_dist, p_dist, p_label)

    if result.shape[1] > 900:
        scale = 900 / result.shape[1]
        result = cv2.resize(result, (int(result.shape[1]*scale),
                                     int(result.shape[0]*scale)))

    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        out = os.path.join(save_dir, 'classified_' + os.path.basename(path))
        cv2.imwrite(out, result)
        print(f'  [SAVED] {out}')

    if show:
        cv2.imshow(f'Classification: {label}', result)
        cv2.waitKey(0)
        cv2.destroyAllWindows()


# ─────────────────────────────────────────────────────────────────────────────
#  ENTRY POINT
# ─────────────────────────────────────────────────────────────────────────────

DEFAULT_IMAGES = [
    'IMGS/img1p3.png', 'IMGS/img2P3.png', 'IMGS/img3P3.png',
    'IMGS/img4P3.png', 'IMGS/img5P3.png',
]

if __name__ == '__main__':
    ap = argparse.ArgumentParser(
        description='Extension – Unknown Object Detection')
    ap.add_argument('--image',     default=None)
    ap.add_argument('--dir',       default=None)
    ap.add_argument('--save_dir',  default=None)
    ap.add_argument('--db',        default=DB_PATH)
    ap.add_argument('--threshold', type=float, default=3.0,
                    help='Distance threshold above which object = UNKNOWN')
    args = ap.parse_args()

    # Load DB
    db = load_db(args.db)
    if not db:
        print(f'[ERROR] DB empty or not found at {args.db}')
        print('        Run task5_training.py first.')
        sys.exit(1)

    vectors, labels, stds = build_classifier(db)
    print(f'[DB] Loaded {len(db)} entries: {sorted(set(labels))}')
    print(f'[DB] Unknown threshold = {args.threshold}\n')

    # Collect images
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
        process_image(path, vectors, labels, stds,
                      threshold=args.threshold,
                      save_dir=args.save_dir,
                      show=show)