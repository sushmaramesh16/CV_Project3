"""
Project 3 – Task 5: Collect Training Data
==========================================
Enables collection of feature vectors from known objects,
attaches labels, and stores them in a JSON database file for
later use in classification.

Two modes:
  1. INTERACTIVE (--image or --dir):
     Shows each image, detects the object region, displays features,
     prompts you to type a label → saves to DB.

  2. AUTO (--auto):
     Reads a folder of images where the filename IS the label
     e.g. triangle_01.png → label "triangle"
     Useful for batch-labelling a training set quickly.

DB format (object_db.json):
  [ { "label": "triangle",
      "hu": [...],
      "aspect": 0.98,
      "extent": 0.50,
      "solidity": 0.98,
      "angle": -47.2,
      "area": 27512 }, ... ]

Usage:
  # Label images one by one interactively
  python3 task5_training.py --image IMGS/img1p3.png
  python3 task5_training.py --dir IMGS/

  # Auto-label from filenames  (filename = label_anything.png)
  python3 task5_training.py --auto --dir IMGS/

  # View what's in the DB
  python3 task5_training.py --show_db
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


DB_PATH = 'object_db.json'


# ─────────────────────────────────────────────────────────────────────────────
#  DB HELPERS
# ─────────────────────────────────────────────────────────────────────────────

def load_db(path: str = DB_PATH) -> list:
    if os.path.exists(path):
        with open(path) as f:
            return json.load(f)
    return []


def save_db(db: list, path: str = DB_PATH):
    with open(path, 'w') as f:
        json.dump(db, f, indent=2)
    print(f'[DB] Saved {len(db)} entries to {path}')


def add_entry(db: list, label: str, features: dict) -> dict:
    entry = {
        'label':    label,
        'hu':       [round(float(h), 6) for h in features['hu']],
        'aspect':   round(features['aspect'],   4),
        'extent':   round(features['extent'],   4),
        'solidity': round(features['solidity'], 4),
        'angle':    round(features['angle_deg'], 2),
        'area':     features['area'],
    }
    db.append(entry)
    return entry


# ─────────────────────────────────────────────────────────────────────────────
#  FEATURE VECTOR → numpy  (for classifier use)
# ─────────────────────────────────────────────────────────────────────────────

def entry_to_vector(entry: dict) -> np.ndarray:
    """Convert a DB entry dict to a flat numpy feature vector."""
    return np.array(entry['hu'] + [
        entry['aspect'],
        entry['extent'],
        entry['solidity'],
    ], dtype=np.float32)


# ─────────────────────────────────────────────────────────────────────────────
#  PICK BEST REGION  (the actual object, not cardboard shadow)
# ─────────────────────────────────────────────────────────────────────────────

def pick_object_region(regions: list) -> dict | None:
    """
    From the valid regions, pick the one most likely to be the object:
    - Smallest valid area (avoids the large cardboard shadow blob)
    - Must be > 3000 px (already filtered by get_regions)
    """
    if not regions:
        return None
    return sorted(regions, key=lambda r: r['area'])[0]


# ─────────────────────────────────────────────────────────────────────────────
#  VISUALISATION
# ─────────────────────────────────────────────────────────────────────────────

def make_training_display(frame, region, features, label='?') -> np.ndarray:
    """
    Show the detected region with oriented bbox, axis, and feature vector.
    Used during interactive labelling so you can confirm before saving.
    """
    vis = frame.copy()
    c = (0, 255, 0)
    cx, cy = region['centroid']

    # Oriented bounding box
    vis = draw_oriented_bbox(vis, region['mask'], c)

    # Orientation axis
    vis = draw_orientation(vis, cx, cy, features['angle_deg'], c, length=80)

    # Feature text
    hu = features['hu']
    lines = [
        f"Label: {label}",
        f"angle={features['angle_deg']:.1f}  area={features['area']}",
        f"aspect={features['aspect']:.2f}  extent={features['extent']:.2f}  solidity={features['solidity']:.2f}",
        f"Hu: {hu[0]:.2f}  {hu[1]:.2f}  {hu[2]:.2f}  {hu[3]:.2f}",
    ]
    for i, line in enumerate(lines):
        y = 28 + i * 22
        cv2.putText(vis, line, (10, y), cv2.FONT_HERSHEY_SIMPLEX,
                    0.6, (0, 0, 0), 4)
        cv2.putText(vis, line, (10, y), cv2.FONT_HERSHEY_SIMPLEX,
                    0.6, (50, 230, 50), 2)
    return vis


# ─────────────────────────────────────────────────────────────────────────────
#  PROCESS ONE IMAGE
# ─────────────────────────────────────────────────────────────────────────────

def process_image(path: str, db: list,
                  auto_label: str | None = None,
                  show: bool = True) -> bool:
    """
    Detect object in image, extract features, label and store.
    Returns True if an entry was saved, False if skipped.
    """
    frame = cv2.imread(path)
    if frame is None:
        print(f'[ERROR] Cannot read {path}'); return False

    binary, _    = threshold_frame(frame)
    cleaned      = clean_binary(binary)
    regions      = get_regions(cleaned)
    obj          = pick_object_region(regions)

    if obj is None:
        print(f'[SKIP] No valid region in {path}'); return False

    features = extract_features(obj['mask'])

    if auto_label:
        label = auto_label
        entry = add_entry(db, label, features)
        print(f'[AUTO] {os.path.basename(path)} → "{label}"  '
              f'hu[:2]={features["hu"][:2].round(2)}')
        return True

    # Interactive mode
    disp = make_training_display(frame, obj, features, label='?')
    if disp.shape[1] > 900:
        scale = 900 / disp.shape[1]
        disp = cv2.resize(disp, (int(disp.shape[1]*scale),
                                  int(disp.shape[0]*scale)))

    cv2.imshow('Task 5 – Training  (press S=save, Q=skip)', disp)
    print(f'\n[{os.path.basename(path)}]')
    print(f'  angle={features["angle_deg"]:.1f}  area={features["area"]}  '
          f'aspect={features["aspect"]:.2f}  '
          f'extent={features["extent"]:.2f}  '
          f'solidity={features["solidity"]:.2f}')
    print(f'  Hu: {features["hu"].round(3)}')

    key = cv2.waitKey(0) & 0xFF
    cv2.destroyAllWindows()

    if key == ord('q') or key == ord('Q'):
        print('  [SKIP]')
        return False

    # Prompt for label in terminal
    label = input('  Enter label (or blank to skip): ').strip()
    if not label:
        print('  [SKIP]')
        return False

    entry = add_entry(db, label, features)
    print(f'  [SAVED] label="{label}"')
    return True


# ─────────────────────────────────────────────────────────────────────────────
#  SHOW DB SUMMARY
# ─────────────────────────────────────────────────────────────────────────────

def show_db(path: str = DB_PATH):
    db = load_db(path)
    if not db:
        print('[DB] Empty or not found.')
        return
    print(f'\n[DB] {len(db)} entries in {path}:\n')
    from collections import Counter
    counts = Counter(e['label'] for e in db)
    for label, count in sorted(counts.items()):
        print(f'  {label:20s}  {count} sample(s)')
    print()
    print('Full entries:')
    for e in db:
        print(f'  {e["label"]:15s}  '
              f'hu[:3]={e["hu"][:3]}  '
              f'aspect={e["aspect"]:.2f}  '
              f'solidity={e["solidity"]:.2f}')


# ─────────────────────────────────────────────────────────────────────────────
#  ENTRY POINT
# ─────────────────────────────────────────────────────────────────────────────

DEFAULT_IMAGES = [
    ('IMGS/img1p3.png',  'triangle'),
    ('IMGS/img2P3.png',  'tbar'),
    ('IMGS/img3P3.png',  'lkey'),
    ('IMGS/img4P3.png',  'chisel'),
    ('IMGS/img5P3.png',  'rod'),
]

if __name__ == '__main__':
    ap = argparse.ArgumentParser(
        description='Project 3 Task 5 – Collect Training Data')
    ap.add_argument('--image',    default=None,
                    help='Single image to label')
    ap.add_argument('--dir',      default=None,
                    help='Directory of images to label')
    ap.add_argument('--auto',     action='store_true',
                    help='Auto-label using filename as label')
    ap.add_argument('--db',       default=DB_PATH,
                    help=f'DB file path (default: {DB_PATH})')
    ap.add_argument('--show_db',  action='store_true',
                    help='Print DB contents and exit')
    ap.add_argument('--seed',     action='store_true',
                    help='Seed DB from built-in dev images (quick start)')
    args = ap.parse_args()

    if args.show_db:
        show_db(args.db)
        sys.exit(0)

    db = load_db(args.db)

    # ── SEED mode: auto-label from dev images ──────────────────────────────
    if args.seed:
        print('[SEED] Auto-labelling dev images...')
        for path, label in DEFAULT_IMAGES:
            if os.path.exists(path):
                process_image(path, db, auto_label=label, show=False)
            else:
                print(f'  [SKIP] {path} not found')
        save_db(db, args.db)
        show_db(args.db)
        sys.exit(0)

    # ── Collect image paths ────────────────────────────────────────────────
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
        # Default: seed from dev images automatically
        print('[INFO] No input specified — seeding from dev images.')
        print('       Use --image, --dir, or --seed for custom input.\n')
        for path, label in DEFAULT_IMAGES:
            if os.path.exists(path):
                process_image(path, db, auto_label=label, show=False)
        save_db(db, args.db)
        show_db(args.db)
        sys.exit(0)

    # ── Process each image ─────────────────────────────────────────────────
    show = not args.auto
    for path in images:
        if args.auto:
            # Derive label from filename: "triangle_01.png" → "triangle"
            base  = os.path.splitext(os.path.basename(path))[0]
            label = base.split('_')[0]
            process_image(path, db, auto_label=label, show=False)
        else:
            process_image(path, db, auto_label=None, show=True)

    save_db(db, args.db)
    show_db(args.db)