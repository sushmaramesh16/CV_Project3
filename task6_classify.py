"""
Task 6 - Classify new images
"""

import cv2
import numpy as np
import argparse
import os
import sys
import json

# allow imports from other task files in the same folder
sys.path.insert(0, os.path.dirname(__file__))

# import functions from previous tasks
from task1_threshold import threshold_frame
from task2_morphology import clean_binary
from task3_components import get_regions
from task4_orientation import extract_features, draw_orientation, draw_oriented_bbox
from task5_training import load_db, entry_to_vector, pick_object_region

DB_PATH = 'object_db.json'


def build_classifier(db):
    # take all saved feature entries from the database
    # and turn them into numeric vectors
    vectors = np.array([entry_to_vector(e) for e in db], dtype=np.float32)
    
    # store the labels (triangle, tbar, etc.)
    labels = [e['label'] for e in db]

    # compute standard deviation for each feature
    # this helps normalize the distance calculation
    stdevs = vectors.std(axis=0)

    # make sure we never divide by zero
    stdevs = np.where(stdevs < 1e-6, 1e-6, stdevs)

    return vectors, labels, stdevs


def scaled_euclidean_distance(query, vectors, stdevs):
    # calculate scaled Euclidean distance
    # formula: sqrt(sum(((x1-x2)/stdev)^2))
    diff = (query - vectors) / stdevs
    return np.sqrt((diff ** 2).sum(axis=1))


def classify(query_features, vectors, labels, stdevs):
    # convert feature dictionary into a flat vector
    q = entry_to_vector({
        'hu': list(query_features['hu']),
        'aspect': query_features['aspect'],
        'extent': query_features['extent'],
        'solidity': query_features['solidity'],
    })

    # compute distances to all database entries
    dists = scaled_euclidean_distance(q, vectors, stdevs)

    # find the closest match
    best = int(np.argmin(dists))

    # return predicted label and distance score
    return labels[best], float(dists[best])


def draw_label(frame, region, features, label, dist):
    # make a copy so we don’t modify the original image
    vis = frame.copy()

    # get center of the object
    cx, cy = region['centroid']

    # assign a color for each object type
    color_map = {
        'triangle': (0, 200, 255),
        'tbar':     (255, 100, 0),
        'lkey':     (0, 255, 100),
        'chisel':   (200, 0, 255),
        'carkey':   (0, 180, 180),
    }

    # default color if label isn’t in map
    color = color_map.get(label, (50, 230, 50))

    # draw oriented bounding box
    vis = draw_oriented_bbox(vis, region['mask'], color)

    # draw orientation line
    vis = draw_orientation(vis, cx, cy, features['angle_deg'], color, length=75)

    # create text label with distance score
    text = f"{label}  (d={dist:.2f})"
    font = cv2.FONT_HERSHEY_SIMPLEX

    # measure text size so we can draw background box
    (tw, th), _ = cv2.getTextSize(text, font, 0.75, 2)

    # position text above the object
    tx = max(int(cx) - tw // 2, 4)
    ty = max(int(cy) - 90, th + 4)

    # draw black background rectangle for readability
    cv2.rectangle(vis, (tx - 4, ty - th - 4), (tx + tw + 4, ty + 4), (0, 0, 0), -1)

    # draw the text
    cv2.putText(vis, text, (tx, ty), font, 0.75, color, 2)

    return vis


def process_image(path, vectors, labels, stdevs, save_dir=None, show=True):
    # read image from disk
    frame = cv2.imread(path)

    if frame is None:
        print(f'could not read {path}')
        return None

    # full pipeline:
    # 1. threshold image
    # 2. clean binary image
    # 3. find connected components
    # 4. pick main object
    binary = threshold_frame(frame)[0]
    cleaned = clean_binary(binary)
    regions = get_regions(cleaned)
    obj = pick_object_region(regions)

    if obj is None:
        print(f'no object found in {path}')
        return None

    # extract features from object mask
    features = extract_features(obj['mask'])

    # classify object
    pred_label, dist = classify(features, vectors, labels, stdevs)

    print(f'{os.path.basename(path)}  ->  "{pred_label}"  dist={dist:.3f}')

    # draw prediction result on image
    vis = draw_label(frame, obj, features, pred_label, dist)

    # show original and labeled image side by side
    out_img = np.hstack([frame, vis])

    # optionally save the output image
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        cv2.imwrite(os.path.join(save_dir, 'classified_' + os.path.basename(path)), out_img)

    # optionally display image
    if show:
        cv2.imshow(f'classified: {os.path.basename(path)}', out_img)
        key = cv2.waitKey(0) & 0xFF
        cv2.destroyAllWindows()

        # press q to quit early
        if key == ord('q'):
            return {'label': pred_label, 'dist': dist, 'quit': True}

    return {'label': pred_label, 'dist': dist, 'path': path}


# default test images
DEFAULT_IMAGES = [
    'IMGS/img1p3.png', 'IMGS/img2P3.png', 'IMGS/img3P3.png',
    'IMGS/img4P3.png', 'IMGS/img5P3.png',
]


if __name__ == '__main__':
    # set up command line arguments
    ap = argparse.ArgumentParser()
    ap.add_argument('--image', default=None)
    ap.add_argument('--dir', default=None)
    ap.add_argument('--db', default=DB_PATH)
    ap.add_argument('--save_dir', default=None)
    args = ap.parse_args()

    # load trained database
    db = load_db(args.db)

    if not db:
        print('DB is empty - run task5_training.py first')
        sys.exit(1)

    # build classifier from database
    vectors, labels, stdevs = build_classifier(db)
    print(f'loaded {len(db)} entries: {sorted(set(labels))}')

    # decide which images to classify
    if args.image:
        images = [args.image]
    elif args.dir:
        images = sorted([
            os.path.join(args.dir, f) for f in os.listdir(args.dir)
            if f.lower().endswith(('.png', '.jpg', '.jpeg'))
            and not f.startswith('example')
        ])
    else:
        images = [p for p in DEFAULT_IMAGES if os.path.exists(p)]

    # if saving, don't show pop-up window
    show = args.save_dir is None

    # classify each image
    for path in images:
        result = process_image(path, vectors, labels, stdevs,
                               save_dir=args.save_dir, show=show)
        if result and result.get('quit'):
            break