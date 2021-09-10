import json
import cv2
import numpy as np
from collections import defaultdict
from draw_keypoints import draw
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

images_dir_path = "data/mpii/images/"
gt = 'data/mpii/annot/val.json'

# images_dir_path = 'data/coco/images'
# gt = 'data/coco/annotations/person_keypoints_val2017.json'

import scipy.io as scio

det = scio.loadmat(r"mpii_val_pred.mat")
# coco = COCO(gt)
# det = coco.loadRes("coco_val_pred.json")
# keypoints = det.dataset['keypoints']
# print(keypoints)

det = det['preds']

with open(gt, 'r') as f:
    gt = json.load(f)

assert len(gt) == len(det)

images_path = []
for id in range(len(gt)):

    file_name = gt[id]['image']
    file_name = images_dir_path + file_name

    img = cv2.imread(file_name)
    if file_name not in images_path:
        img = cv2.imread(file_name)
        images_path.append(file_name)
        print(file_name)
    else:
        print(gt[id]['image'])
        continue

    kpts = det[id]
    #
    # gt_kpts = np.array(gt[id]['joints'])

    img = draw(gt[id]['image'], img, kpts)

    # cv2.imshow('img', img)
    # cv2.waitKey(300)
    # cv2.imwrite('mpii_results/pred/{}'.format(gt[id]['image']), img)

    # img = draw(gt[id]['image'], img, gt_kpts)
    # cv2.imwrite('mpii_results/gt/{}'.format(gt[id]['image']), img)
