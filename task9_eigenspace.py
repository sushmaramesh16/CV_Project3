# Name: Sushma Ramesh, Dina Barua
# Date: February 22, 2026
# Purpose: Task 9 - One-shot classification using eigenspace embeddings (PCA-based)

import cv2
import numpy as np
import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(__file__))
from task1_threshold import threshold_frame
from task2_morphology import clean_binary
from task3_components import get_regions
from task4_orientation import extract_features, draw_oriented_bbox, draw_orientation
from task5_training import load_db, pick_object_region
from task6_classify import build_classifier, classify as classify_handbuilt
from task7_evaluate import build_confusion_matrix, print_confusion_matrix, save_confusion_matrix_image

DB_PATH = 'object_db.json'
EIGEN_DB_PATH = 'eigenspace_db.npz'

N_EVEC = 6        # how many eigenvectors to use for the embedding
RESIZE_LONG = 160  # resize so long side = 160px (matches professor's embedding.py)

# training and test image sets
DEFAULT_TRAIN = [
    ('IMGS/img1p3.png', 'triangle'),
    ('IMGS/img2P3.png', 'tbar'),
    ('IMGS/img3P3.png', 'lkey'),
    ('IMGS/img4P3.png', 'chisel'),
    ('IMGS/img5P3.png', 'carkey'),
]

DEFAULT_TEST = [
    ('IMGS/img1p3.png', 'triangle'),
    ('IMGS/img2P3.png', 'tbar'),
    ('IMGS/img3P3.png', 'lkey'),
    ('IMGS/img4P3.png', 'chisel'),
    ('IMGS/img5P3.png', 'carkey'),
]


# --- pre-processing helpers ---

def get_axis_extents(mask, cx, cy, theta_deg):
    """
    Project each foreground pixel onto the primary and secondary axes
    to get the bounding extents - this matches the professor's utilities.cpp
    (minE1, maxE1, minE2, maxE2)
    """
    theta = np.radians(theta_deg)
    e1 = np.array([np.cos(theta), np.sin(theta)])   # primary axis direction
    e2 = np.array([-np.sin(theta), np.cos(theta)])  # secondary axis direction

    ys, xs = np.where(mask > 0)
    if len(xs) == 0:
        return 0, 0, 0, 0

    # offset from centroid
    dx = xs.astype(np.float32) - cx
    dy = ys.astype(np.float32) - cy

    proj1 = dx * e1[0] + dy * e1[1]
    proj2 = dx * e2[0] + dy * e2[1]

    return float(proj1.min()), float(proj1.max()), float(proj2.min()), float(proj2.max())


def get_roi_vector(frame, region, features):
    """
    Rotate the image so the object's primary axis points right,
    then crop just the object using the axis extents.
    Matches professor's prepEmbeddingImage() from utilities.cpp.
    Returns a flattened float32 vector (green channel only).
    """
    h, w = frame.shape[:2]
    cx, cy = region['centroid']
    theta = features['angle_deg']

    minE1, maxE1, minE2, maxE2 = get_axis_extents(region['mask'], cx, cy, theta)

    # rotate image so primary axis is horizontal (note: -theta like the professor's code)
    largest = int(1.414 * max(w, h))
    M = cv2.getRotationMatrix2D((cx, cy), -theta, 1.0)
    rotated = cv2.warpAffine(frame, M, (largest, largest),
                             flags=cv2.INTER_LINEAR,
                             borderMode=cv2.BORDER_REPLICATE)

    # crop box using the professor's exact formula from utilities.cpp
    left = int(cx + minE1)
    top = int(cy - maxE2)
    width = int(maxE1 - minE1)
    height = int(maxE2 - minE2)

    # bounds check
    if left < 0:
        width += left
        left = 0
    if top < 0:
        height += top
        top = 0
    if left + width >= rotated.shape[1]:
        width = rotated.shape[1] - 1 - left
    if top + height >= rotated.shape[0]:
        height = rotated.shape[0] - 1 - top

    if width <= 0 or height <= 0:
        return None

    roi = rotated[top:top + height, left:left + width]
    if roi.size == 0:
        return None

    # resize so long side = 160, keep green channel only (matches embedding.py)
    scale = RESIZE_LONG / max(roi.shape[0], roi.shape[1])
    roi_small = cv2.resize(roi, (int(roi.shape[1] * scale), int(roi.shape[0] * scale)),
                           interpolation=cv2.INTER_AREA)

    return roi_small[:, :, 1].astype(np.float32).flatten()


# --- eigenspace functions ---

def build_eigenspace(train_pairs, n_evec=N_EVEC, save_path=EIGEN_DB_PATH):
    """
    Build the PCA eigenspace from training images.
    Stacks all ROI vectors into a matrix, computes SVD, stores eigenvectors.
    """
    vectors = []
    labels = []
    org_len = None

    print('building eigenspace from training images...')
    for path, label in train_pairs:
        frame = cv2.imread(path)
        if frame is None:
            print(f'  could not read {path}')
            continue

        binary = threshold_frame(frame)[0]
        cleaned = clean_binary(binary)
        regions = get_regions(cleaned)
        obj = pick_object_region(regions)
        if obj is None:
            print(f'  no region found in {path}')
            continue

        features = extract_features(obj['mask'])
        vec = get_roi_vector(frame, obj, features)
        if vec is None:
            print(f'  ROI extraction failed for {path}')
            continue

        # all vectors need to be the same length - use the first image as reference
        if org_len is None:
            org_len = vec.shape[0]

        if vec.shape[0] != org_len:
            vec = cv2.resize(vec.reshape(-1, 1), (1, org_len),
                             interpolation=cv2.INTER_LINEAR).flatten().astype(np.float32)

        vectors.append(vec)
        labels.append(label)
        print(f'  processed [{label}]  vec length={vec.shape[0]}')

    if len(vectors) < 2:
        print('need at least 2 training images')
        return None

    # stack into matrix A (n_images x n_pixels)
    A = np.vstack(vectors).astype(np.float32)
    meanvec = np.mean(A, axis=0)

    # difference matrix - subtract mean (n_pixels x n_images, like professor's code)
    D = (A - meanvec).T

    print('running SVD...')
    U, s, V = np.linalg.svd(D, full_matrices=False)

    eigenvalues = s ** 2 / (D.shape[0] - 1)
    print(f'top eigenvalues: {eigenvalues[:6].round(1)}')

    # project each training image into the eigenspace to get its embedding
    train_embeds = np.array([(A[i] - meanvec) @ U[:, :n_evec]
                              for i in range(A.shape[0])], dtype=np.float32)

    np.savez(save_path,
             meanvec=meanvec,
             U=U,
             train_embeds=train_embeds,
             labels=np.array(labels),
             n_evec=np.array(n_evec),
             org_len=np.array(org_len))
    print(f'saved eigenspace -> {save_path}')

    return {'meanvec': meanvec, 'U': U, 'train_embeds': train_embeds,
            'labels': labels, 'n_evec': n_evec, 'org_len': org_len}


def load_eigenspace(path=EIGEN_DB_PATH):
    if not os.path.exists(path):
        return None
    d = np.load(path, allow_pickle=True)
    return {'meanvec': d['meanvec'], 'U': d['U'],
            'train_embeds': d['train_embeds'],
            'labels': list(d['labels']),
            'n_evec': int(d['n_evec']),
            'org_len': int(d['org_len'])}


def get_embedding(vec, eigen):
    # make sure lengths match then project onto eigenvectors
    if vec.shape[0] != eigen['org_len']:
        vec = cv2.resize(vec.reshape(-1, 1), (1, eigen['org_len']),
                         interpolation=cv2.INTER_LINEAR).flatten()
    diff = (vec - eigen['meanvec']).astype(np.float32)
    return diff @ eigen['U'][:, :eigen['n_evec']]


def classify_eigen(query_embed, eigen):
    # find closest training embedding using sum-squared difference
    best_label = None
    best_dist = float('inf')
    for i, emb in enumerate(eigen['train_embeds']):
        diff = query_embed - emb
        dist = float(np.dot(diff, diff))
        if dist < best_dist:
            best_dist = dist
            best_label = eigen['labels'][i]
    return best_label, best_dist


# --- run both classifiers on one image ---

def process_one(path, eigen, hb_vectors, hb_labels, hb_stdevs):
    frame = cv2.imread(path)
    if frame is None:
        return None

    binary = threshold_frame(frame)[0]
    cleaned = clean_binary(binary)
    regions = get_regions(cleaned)
    obj = pick_object_region(regions)
    if obj is None:
        return None

    features = extract_features(obj['mask'])

    # eigenspace prediction
    vec = get_roi_vector(frame, obj, features)
    if vec is None:
        return None
    embed = get_embedding(vec, eigen)
    eigen_pred, ed = classify_eigen(embed, eigen)

    # hand-built feature prediction from task 6
    hb_pred, hd = classify_handbuilt(features, hb_vectors, hb_labels, hb_stdevs)

    return {
        'path': path,
        'eigen_pred': eigen_pred,
        'eigen_dist': ed,
        'hb_pred': hb_pred,
        'hb_dist': hd,
        'frame': frame,
        'region': obj,
        'features': features,
        'embed': embed,
    }


def draw_result(result):
    frame = result['frame']
    obj = result['region']
    features = result['features']

    vis = frame.copy()
    vis = draw_oriented_bbox(vis, obj['mask'], (0, 200, 255))
    cx, cy = obj['centroid']
    vis = draw_orientation(vis, cx, cy, features['angle_deg'], (0, 200, 255), 70)

    # show both predictions on the image
    for text, yoff, col in [
        (f"Eigenspace: {result['eigen_pred']}", 30, (0, 200, 255)),
        (f"Hand-built: {result['hb_pred']}", 58, (50, 230, 50)),
    ]:
        cv2.putText(vis, text, (10, yoff), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 4)
        cv2.putText(vis, text, (10, yoff), cv2.FONT_HERSHEY_SIMPLEX, 0.7, col, 2)

    return vis


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--build', action='store_true', help='build eigenspace from training images')
    ap.add_argument('--image', default=None, help='classify a single image')
    ap.add_argument('--evaluate', action='store_true', help='evaluate and compare both classifiers')
    ap.add_argument('--n_evec', type=int, default=N_EVEC)
    ap.add_argument('--db', default=DB_PATH)
    ap.add_argument('--eigen_db', default=EIGEN_DB_PATH)
    ap.add_argument('--save_dir', default=None)
    args = ap.parse_args()

    # build mode - compute and save eigenspace
    if args.build:
        build_eigenspace(DEFAULT_TRAIN, n_evec=args.n_evec, save_path=args.eigen_db)
        sys.exit(0)

    # load eigenspace
    eigen = load_eigenspace(args.eigen_db)
    if eigen is None:
        print('eigenspace not found - run with --build first')
        sys.exit(1)
    print(f'loaded eigenspace: {len(eigen["labels"])} training images, {eigen["n_evec"]} eigenvectors')

    # load hand-built classifier for comparison
    hb_db = load_db(args.db)
    if not hb_db:
        print(f'feature DB not found at {args.db}')
        sys.exit(1)
    hb_vectors, hb_labels, hb_stdevs = build_classifier(hb_db)
    class_names = sorted(set(hb_labels))

    # classify a single image
    if args.image:
        result = process_one(args.image, eigen, hb_vectors, hb_labels, hb_stdevs)
        if result:
            print(f'eigenspace  ->  {result["eigen_pred"]}  (dist={result["eigen_dist"]:.1f})')
            print(f'hand-built  ->  {result["hb_pred"]}  (dist={result["hb_dist"]:.3f})')
            vis = draw_result(result)
            if args.save_dir:
                os.makedirs(args.save_dir, exist_ok=True)
                cv2.imwrite(os.path.join(args.save_dir, 'eigen_' + os.path.basename(args.image)), vis)
            else:
                cv2.imshow('task 9 result', vis)
                cv2.waitKey(0)
                cv2.destroyAllWindows()
        sys.exit(0)

    # evaluate both classifiers and compare
    if args.evaluate:
        print(f'\nevaluating {len(DEFAULT_TEST)} images...\n')
        true_list = []
        eigen_preds = []
        hb_preds = []

        for path, true_label in DEFAULT_TEST:
            if not os.path.exists(path):
                continue
            result = process_one(path, eigen, hb_vectors, hb_labels, hb_stdevs)
            if result is None:
                continue

            ep = result['eigen_pred']
            hp = result['hb_pred']
            print(f'  true={true_label:12s}  eigen={ep:12s}{"correct" if ep==true_label else "WRONG"}  '
                  f'hbuilt={hp:12s}{"correct" if hp==true_label else "WRONG"}')

            true_list.append(true_label)
            eigen_preds.append(ep)
            hb_preds.append(hp)

            if args.save_dir:
                os.makedirs(args.save_dir, exist_ok=True)
                vis = draw_result(result)
                cv2.imwrite(os.path.join(args.save_dir, f'eigen_{os.path.basename(path)}'), vis)

        print('\n=== Eigenspace Classifier ===')
        cm_eigen = build_confusion_matrix(true_list, eigen_preds, class_names)
        print_confusion_matrix(cm_eigen, class_names)

        print('\n=== Hand-built Feature Classifier ===')
        cm_hb = build_confusion_matrix(true_list, hb_preds, class_names)
        print_confusion_matrix(cm_hb, class_names)

        if args.save_dir:
            save_confusion_matrix_image(cm_eigen, class_names,
                                        os.path.join(args.save_dir, 'confusion_eigenspace.png'))
            save_confusion_matrix_image(cm_hb, class_names,
                                        os.path.join(args.save_dir, 'confusion_handbuilt.png'))
        sys.exit(0)

    print('no mode given - use --build, --image, or --evaluate')