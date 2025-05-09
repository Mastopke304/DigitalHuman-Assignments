from pyimagesearch.helpers import convert_and_trim_bb
import dlib
import cv2
from tqdm import tqdm
from utils import *


limit = 5
width = 300

model = "mmod_human_face_detector.dat"
upsample = 1
cnn_detector = dlib.cnn_face_detection_model_v1(model)
hog_detector = dlib.get_frontal_face_detector()

annotation_file = "images/demo/annotations.txt"
image_base = "images/demo"

img_dataset = parse_wider_annotations(annotation_file, image_base, limit=limit)



for image_path, true_boxes in tqdm(img_dataset, desc="Evaluating HOG", unit="img"):
    img = cv2.imread(image_path)
    img, true_boxes = scale_by_width(img, true_boxes, width)
    rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    cnn_results = cnn_detector(rgb, upsample)
    hog_results = hog_detector(rgb, upsample)
    cnn_boxes = [convert_and_trim_bb(img, r.rect) for r in cnn_results]
    hog_boxes = [convert_and_trim_bb(img, r) for r in hog_results]
    for (x, y, w, h) in cnn_boxes:
        cv2.rectangle(img, (int(x), int(y)), (int(x + w), int(y + h)), (255, 0, 0), 2)
    for (x, y, w, h) in hog_boxes:
        cv2.rectangle(img, (int(x), int(y)), (int(x + w), int(y + h)), (0, 255, 0), 2)
    for (x, y, w, h) in true_boxes:
        cv2.rectangle(img, (int(x), int(y)), (int(x + w), int(y + h)), (0, 0, 255), 2)
    cv2.imshow("CNN: Blue, HOG: Green, GT: Red", img)
    cv2.waitKey(0)