import time

import argparse
from pyimagesearch.helpers import convert_and_trim_bb
import dlib
import cv2
import numpy as np
from tqdm import tqdm
from utils import *
import matplotlib.pyplot as plt


limit = 100
width = 300

model = "mmod_human_face_detector.dat"
upsample = 1
cnn_detector = dlib.cnn_face_detection_model_v1(model)
hog_detector = dlib.get_frontal_face_detector()

annotation_file = "images/wider_face_split/wider_face_val_bbx_gt.txt"
image_base = "images/WIDER_val/images"

img_dataset = parse_wider_annotations(annotation_file, image_base, limit=limit)

# img_csv_path = "images/FDDB copy/FDDB-folds/FDDB-fold-01-ellipseList.csv"
# img_csv = pd.read_csv(img_csv_path)
# img_path = "images/FDDB copy/originalPics"
# img_path_list = []
# for path in img_csv["img_path"]:
#     img_path_list.append(img_path + path)

def train_cnn(detector, dataset):
    recall = []
    precision = []
    speed = []
    for image_path, true_boxes in tqdm(dataset, desc="Evaluating CNN", unit="img"):
        img = cv2.imread(image_path)
        img, true_boxes = scale_by_width(img, true_boxes, width)
        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        start_time = time.time()
        results = detector(rgb, upsample)
        end_time = time.time() - start_time
        speed.append(img.size / end_time)
        boxes = [convert_and_trim_bb(img, r.rect) for r in results]
        sub_recall = compute_recall(boxes, true_boxes)
        sub_precision = compute_precision(boxes, true_boxes)
        recall.append(sub_recall)
        precision.append(sub_precision)
    print("CNN mean recall:", np.mean(recall), "CNN mean compliance:", np.mean(precision))
    return recall, precision, speed

def train_hog(detector, dataset):
    recall = []
    precision = []
    speed = []
    for image_path, true_boxes in tqdm(dataset, desc="Evaluating HOG", unit="img"):
        img = cv2.imread(image_path)
        img, true_boxes = scale_by_width(img, true_boxes, width)
        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        start_time = time.time()
        results = detector(rgb, upsample)
        end_time = time.time() - start_time
        speed.append(img.size / end_time)
        boxes = [convert_and_trim_bb(img, r) for r in results]
        sub_recall = compute_recall(boxes, true_boxes)
        sub_precision = compute_precision(boxes, true_boxes)
        recall.append(sub_recall)
        precision.append(sub_precision)
    print("Hog mean recall:", np.mean(recall), "Hog mean compliance:", np.mean(precision))
    return recall, precision, speed


cnn_recall, cnn_precision, cnn_speed = train_cnn(cnn_detector, img_dataset)
print("------------------------------------")
hog_recall, hog_precision, hog_speed = train_hog(hog_detector, img_dataset)


savefig_path = "./plots/"
if not os.path.exists(savefig_path):
    os.mkdir(savefig_path)

plt.plot(cnn_recall, color="crimson", label="CNN")
plt.plot(hog_recall, color="cornflowerblue", label="HOG")
plt.title("Recall per image")
plt.legend(loc="best")
plt.savefig(savefig_path + "Recall.png")
plt.show()

plt.plot(cnn_precision, color="crimson", label="CNN")
plt.plot(hog_precision, color="cornflowerblue", label="HOG")
plt.title("Compliance per image")
plt.legend(loc="best")
plt.savefig(savefig_path + "Compliance.png")
plt.show()

plt.plot(cnn_speed, color="crimson", label="CNN")
plt.plot(hog_speed, color="cornflowerblue", label="HOG")
plt.title("Pixels per second")
plt.legend(loc="best")
plt.savefig(savefig_path + "Speed.png")
plt.show()