"""
Task 9 - One-shot classification using eigenspace embeddings (PCA)

Builds a PCA eigenspace from training object images.
Then it classifies new objects by projecting them into that eigenspace
and comparing embeddings using distance.

"""

import cv2
import numpy as np
import argparse
import os
import sys

# allow importing from other task files
sys.path.insert(0, os.path.dirname(__file__))

# import earlier pipeline functions
from task1_threshold import threshold_frame
from task2_morphology import clean_binary
from task3_components import get_regions
from task4_orientation import extract_features, draw_oriented_bbox, draw_orientation
from task5_training import load_db, pick_object_region
from task6_classify import build_classifier, classify as classify_handbuilt
from task7_evaluate import build_confusion_matrix, print_confusion_matrix, save_confusion_matrix_image

DB_PATH = 'object_db.json'
EIGEN_DB_PATH = 'eigenspace_db.npz'

# how many principal components to keep
N_EVEC = 6

# resize object ROI so longest side is 160 pixels
RESIZE_LONG = 160


# -----------------------------
# Pre-processing helpers
# -----------------------------

def get_axis_extents(mask, cx, cy, theta_deg):
    """
    Projects all object pixels onto the main axis and perpendicular axis.
    This gives us bounding limits so we can crop tightly around the object.
    """
    theta = np.radians(theta_deg)

    # primary axis direction
    e1 = np.array([np.cos(theta), np.sin(theta)])

    # secondary axis direction (perpendicular)
    e2 = np.array([-np.sin(theta), np.cos(theta)])

    ys, xs = np.where(mask > 0)
    if len(xs) == 0:
        return 0, 0, 0, 0

    # shift pixels relative to centroid
    dx = xs.astype(np.float32) - cx
    dy = ys.astype(np.float32) - cy

    # project pixels onto axes
    proj1 = dx * e1[0] + dy * e1[1]
    proj2 = dx * e2[0] + dy * e2[1]

    return float(proj1.min()), float(proj1.max()), float(proj2.min()), float(proj2.max())


def get_roi_vector(frame, region, features):
    """
    Rotate object so its main axis is horizontal.
    Crop it tightly.
    Resize it.
    Use only green channel.
    Flatten into 1D vector.
    """
    h, w = frame.shape[:2]
    cx, cy = region['centroid']
    theta = features['angle_deg']

    minE1, maxE1, minE2, maxE2 = get_axis_extents(region['mask'], cx, cy, theta)

    # rotate image so object is horizontal
    largest = int(1.414 * max(w, h))
    M = cv2.getRotationMatrix2D((cx, cy), -theta, 1.0)
    rotated = cv2.warpAffine(frame, M, (largest, largest),
                             flags=cv2.INTER_LINEAR,
                             borderMode=cv2.BORDER_REPLICATE)

    # crop tight box
    left = int(cx + minE1)
    top = int(cy - maxE2)
    width = int(maxE1 - minE1)
    height = int(maxE2 - minE2)

    # make sure crop stays inside image
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

    # resize object
    scale = RESIZE_LONG / max(roi.shape[0], roi.shape[1])
    roi_small = cv2.resize(roi,
                           (int(roi.shape[1] * scale),
                            int(roi.shape[0] * scale)),
                           interpolation=cv2.INTER_AREA)

    # return green channel flattened
    return roi_small[:, :, 1].astype(np.float32).flatten()


# -----------------------------
# Eigenspace (PCA) functions
# -----------------------------

def build_eigenspace(train_pairs, n_evec=N_EVEC, save_path=EIGEN_DB_PATH):
    """
    Builds PCA eigenspace using SVD.
    Saves eigenvectors and training embeddings.
    """
    vectors = []
    labels = []
    org_len = None

    print('building eigenspace from training images...')

    for path, label in train_pairs:
        frame = cv2.imread(path)
        if frame is None:
            continue

        # run segmentation pipeline
        binary = threshold_frame(frame)[0]
        cleaned = clean_binary(binary)
        regions = get_regions(cleaned)
        obj = pick_object_region(regions)
        if obj is None:
            continue

        features = extract_features(obj['mask'])
        vec = get_roi_vector(frame, obj, features)
        if vec is None:
            continue

        # ensure all vectors same length
        if org_len is None:
            org_len = vec.shape[0]

        if vec.shape[0] != org_len:
            vec = cv2.resize(vec.reshape(-1, 1), (1, org_len),
                             interpolation=cv2.INTER_LINEAR).flatten().astype(np.float32)

        vectors.append(vec)
        labels.append(label)

        print(f'  processed [{label}]')

    if len(vectors) < 2:
        print('need at least 2 training images')
        return None

    # stack into matrix
    A = np.vstack(vectors).astype(np.float32)

    # compute mean image
    meanvec = np.mean(A, axis=0)

    # subtract mean
    D = (A - meanvec).T

    print('running SVD...')
    U, s, V = np.linalg.svd(D, full_matrices=False)

    # compute eigenvalues
    eigenvalues = s ** 2 / (D.shape[0] - 1)
    print(f'top eigenvalues: {eigenvalues[:6].round(1)}')

    # project training images into eigenspace
    train_embeds = np.array([(A[i] - meanvec) @ U[:, :n_evec]
                              for i in range(A.shape[0])],
                             dtype=np.float32)

    # save everything
    np.savez(save_path,
             meanvec=meanvec,
             U=U,
             train_embeds=train_embeds,
             labels=np.array(labels),
             n_evec=np.array(n_evec),
             org_len=np.array(org_len))

    print(f'saved eigenspace -> {save_path}')

    return {'meanvec': meanvec, 'U': U,
            'train_embeds': train_embeds,
            'labels': labels,
            'n_evec': n_evec,
            'org_len': org_len}


def load_eigenspace(path=EIGEN_DB_PATH):
    # load saved PCA model
    if not os.path.exists(path):
        return None
    d = np.load(path, allow_pickle=True)
    return {'meanvec': d['meanvec'],
            'U': d['U'],
            'train_embeds': d['train_embeds'],
            'labels': list(d['labels']),
            'n_evec': int(d['n_evec']),
            'org_len': int(d['org_len'])}


def get_embedding(vec, eigen):
    # project new image vector into eigenspace
    if vec.shape[0] != eigen['org_len']:
        vec = cv2.resize(vec.reshape(-1, 1),
                         (1, eigen['org_len']),
                         interpolation=cv2.INTER_LINEAR).flatten()

    diff = (vec - eigen['meanvec']).astype(np.float32)
    return diff @ eigen['U'][:, :eigen['n_evec']]


def classify_eigen(query_embed, eigen):
    # find closest training embedding using squared distance
    best_label = None
    best_dist = float('inf')

    for i, emb in enumerate(eigen['train_embeds']):
        diff = query_embed - emb
        dist = float(np.dot(diff, diff))

        if dist < best_dist:
            best_dist = dist
            best_label = eigen['labels'][i]

    return best_label, best_dist


# -----------------------------
# Compare eigenspace vs hand-built
# -----------------------------

def process_one(path, eigen, hb_vectors, hb_labels, hb_stdevs):
    # classify using both methods
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

    # hand-built feature prediction
    hb_pred, hd = classify_handbuilt(features,
                                     hb_vectors,
                                     hb_labels,
                                     hb_stdevs)

    return {
        'path': path,
        'eigen_pred': eigen_pred,
        'eigen_dist': ed,
        'hb_pred': hb_pred,
        'hb_dist': hd,
        'frame': frame,
        'region': obj,
        'features': features,
    }


def draw_result(result):
    # draw predictions on image
    frame = result['frame']
    obj = result['region']
    features = result['features']

    vis = frame.copy()
    vis = draw_oriented_bbox(vis, obj['mask'], (0, 200, 255))

    cx, cy = obj['centroid']
    vis = draw_orientation(vis, cx, cy,
                           features['angle_deg'],
                           (0, 200, 255), 70)

    # display both classifier results
    for text, yoff, col in [
        (f"Eigenspace: {result['eigen_pred']}", 30, (0, 200, 255)),
        (f"Hand-built: {result['hb_pred']}", 58, (50, 230, 50)),
    ]:
        cv2.putText(vis, text, (10, yoff),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                    (0, 0, 0), 4)
        cv2.putText(vis, text, (10, yoff),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                    col, 2)

    return vis


# -----------------------------
# Main
# -----------------------------

if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--build', action='store_true')
    ap.add_argument('--image', default=None)
    ap.add_argument('--evaluate', action='store_true')
    ap.add_argument('--n_evec', type=int, default=N_EVEC)
    ap.add_argument('--db', default=DB_PATH)
    ap.add_argument('--eigen_db', default=EIGEN_DB_PATH)
    ap.add_argument('--save_dir', default=None)
    args = ap.parse_args()

    # build eigenspace mode
    if args.build:
        build_eigenspace(DEFAULT_TRAIN,
                         n_evec=args.n_evec,
                         save_path=args.eigen_db)
        sys.exit(0)

    # load eigenspace
    eigen = load_eigenspace(args.eigen_db)
    if eigen is None:
        print('eigenspace not found - run with --build first')
        sys.exit(1)

    print(f'loaded eigenspace with {eigen["n_evec"]} eigenvectors')

    # load hand-built classifier
    hb_db = load_db(args.db)
    if not hb_db:
        print('feature DB not found')
        sys.exit(1)

    hb_vectors, hb_labels, hb_stdevs = build_classifier(hb_db)

    print('use --build, --image, or --evaluate')