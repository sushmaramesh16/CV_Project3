# Name: Sushma Ramesh, Dina Barua
# Date: February 22, 2026
# Purpose: Task 7 - Performance evaluation and confusion matrix generation
import cv2
import numpy as np
import argparse
import os
import sys
import json
from collections import defaultdict

# allow imports from other task files in this folder
sys.path.insert(0, os.path.dirname(__file__))

# import pipeline + classifier functions from previous tasks
from task1_threshold import threshold_frame
from task2_morphology import clean_binary
from task3_components import get_regions
from task4_orientation import extract_features
from task5_training import load_db, entry_to_vector, pick_object_region
from task6_classify import build_classifier, classify

DB_PATH = 'object_db.json'


def evaluate_image(path, true_label, vectors, labels, stdevs):
    # load image
    frame = cv2.imread(path)
    if frame is None:
        print(f'could not read {path}')
        return None

    # same processing pipeline as before:
    # threshold -> clean -> find regions -> pick object
    binary = threshold_frame(frame)[0]
    cleaned = clean_binary(binary)
    regions = get_regions(cleaned)
    obj = pick_object_region(regions)

    if obj is None:
        print(f'no region found in {path}')
        return None

    # extract features and classify
    features = extract_features(obj['mask'])
    pred, dist = classify(features, vectors, labels, stdevs)

    # check if prediction matches true label
    mark = 'correct' if pred == true_label else 'WRONG'
    print(f'  [{mark}]  true={true_label}  pred={pred}  ({os.path.basename(path)})')

    return pred


def build_confusion_matrix(true_labels, pred_labels, class_names):
    # number of classes
    n = len(class_names)

    # map class name -> index number
    idx = {name: i for i, name in enumerate(class_names)}

    # create empty n x n matrix
    cm = np.zeros((n, n), dtype=int)

    # fill matrix:
    # row = true label
    # col = predicted label
    for t, p in zip(true_labels, pred_labels):
        if t in idx and p in idx:
            cm[idx[t]][idx[p]] += 1

    return cm


def print_confusion_matrix(cm, class_names):
    # pretty print confusion matrix in terminal
    w = max(len(n) for n in class_names) + 2
    pad = ' ' * (w + 2)

    print('\nConfusion Matrix  (rows = true,  cols = predicted)')
    print('-' * (w + 2 + len(class_names) * (w + 1)))
    print(pad + ''.join(f'{n:>{w}}' for n in class_names))
    print('-' * (w + 2 + len(class_names) * (w + 1)))

    # print each row
    for i, name in enumerate(class_names):
        row = f'{name:>{w}}  ' + ''.join(f'{cm[i, j]:>{w}}' for j in range(len(class_names)))
        print(row)

    # calculate overall accuracy
    total = cm.sum()
    correct = np.trace(cm)
    acc = correct / total if total > 0 else 0
    print(f'\nAccuracy: {correct}/{total} = {acc*100:.1f}%\n')


def save_confusion_matrix_image(cm, class_names, path):
    # draw a visual confusion matrix image
    # green = correct predictions (diagonal)
    # red = wrong predictions (off-diagonal)

    n = len(class_names)
    cell = 80
    margin = 160

    # compute image size
    h = margin + n * cell + 20
    w = margin + n * cell + 20

    # light gray background
    img = np.ones((h, w, 3), dtype=np.uint8) * 245
    font = cv2.FONT_HERSHEY_SIMPLEX

    # draw predicted label headers (top)
    for j, name in enumerate(class_names):
        x = margin + j * cell + cell // 2
        cv2.putText(img, name, (x - 28, margin - 12), font, 0.52, (30, 30, 30), 1)

    # get max value for color scaling
    max_val = cm.max() if cm.max() > 0 else 1

    for i, name in enumerate(class_names):
        # draw true label (left side)
        cv2.putText(img, name, (4, margin + i * cell + cell // 2 + 8),
                    font, 0.52, (30, 30, 30), 1)

        for j in range(n):
            x0 = margin + j * cell
            y0 = margin + i * cell
            val = cm[i, j]

            # normalize color intensity
            filled = val / max_val

            if i == j:
                # correct prediction -> green
                g = int(80 + 175 * filled)
                bg = (60, g, 60)
            elif val > 0:
                # wrong prediction -> red
                r = int(80 + 175 * filled)
                bg = (60, 60, r)
            else:
                # empty cell
                bg = (230, 230, 230)

            # draw colored cell
            cv2.rectangle(img, (x0, y0), (x0 + cell, y0 + cell), bg, -1)

            # draw border
            cv2.rectangle(img, (x0, y0), (x0 + cell, y0 + cell), (180, 180, 180), 1)

            # draw count number inside cell
            txt = str(val)
            (tw, th), _ = cv2.getTextSize(txt, font, 0.65, 2)
            cv2.putText(img, txt,
                        (x0 + (cell - tw) // 2, y0 + (cell + th) // 2),
                        font, 0.65, (255, 255, 255), 2)

    # save image file
    cv2.imwrite(path, img)
    print(f'saved confusion matrix image -> {path}')


# default test set with known labels
DEFAULT_TEST = [
    ('IMGS/img1p3.png', 'triangle'),
    ('IMGS/triangle_02.png', 'triangle'),
    ('IMGS/triangle_03.png', 'triangle'),
    ('IMGS/img2P3.png', 'tbar'),
    ('IMGS/tbar_02.png', 'tbar'),
    ('IMGS/tbar_03.png', 'tbar'),
    ('IMGS/img3P3.png', 'lkey'),
    ('IMGS/lkey_02.png', 'lkey'),
    ('IMGS/lkey_03.png', 'lkey'),
    ('IMGS/img4P3.png', 'chisel'),
    ('IMGS/chisel_02.png', 'chisel'),
    ('IMGS/chisel_03.png', 'chisel'),
    ('IMGS/img5P3.png', 'carkey'),
    ('IMGS/carkey_02.png', 'carkey'),
    ('IMGS/carkey_03.png', 'carkey'),
]


if __name__ == '__main__':
    # command line arguments
    ap = argparse.ArgumentParser()
    ap.add_argument('--db', default=DB_PATH)
    ap.add_argument('--dir', default=None)
    ap.add_argument('--auto', action='store_true',
                    help='get true label from filename (e.g. triangle_01.png)')
    ap.add_argument('--save_dir', default=None)
    args = ap.parse_args()

    # load trained database
    db = load_db(args.db)
    if not db:
        print(f'DB not found - run task5 first')
        sys.exit(1)

    # build classifier
    vectors, labels, stdevs = build_classifier(db)
    class_names = sorted(set(labels))

    # decide which test images to use
    if args.dir and args.auto:
        # automatically extract label from filename
        files = sorted([
            os.path.join(args.dir, f) for f in os.listdir(args.dir)
            if f.lower().endswith(('.png', '.jpg', '.jpeg'))
            and not f.startswith('example')
        ])
        test_pairs = [(p, os.path.splitext(os.path.basename(p))[0].split('_')[0])
                      for p in files]
    else:
        # use default test set
        test_pairs = [(p, l) for p, l in DEFAULT_TEST if os.path.exists(p)]

    print(f'\ntesting {len(test_pairs)} images...\n')

    true_list = []
    pred_list = []

    # evaluate each image
    for path, true_label in test_pairs:
        pred = evaluate_image(path, true_label, vectors, labels, stdevs)
        if pred is not None:
            true_list.append(true_label)
            pred_list.append(pred)

    # build confusion matrix
    cm = build_confusion_matrix(true_list, pred_list, class_names)

    # print results
    print_confusion_matrix(cm, class_names)

    # optionally save visual confusion matrix image
    if args.save_dir:
        os.makedirs(args.save_dir, exist_ok=True)
        save_confusion_matrix_image(cm, class_names,
                                    os.path.join(args.save_dir, 'confusion_matrix.png'))