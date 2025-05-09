import os
import imutils



def parse_wider_annotations(
    annotation_file: str, base_image_path: str, limit: int = 100
):
    with open(annotation_file, "r") as f:
        lines = f.readlines()

    dataset = []
    i = 0
    while i < len(lines) and len(dataset) < limit:
        filename = lines[i].strip()
        i += 1
        face_count = int(lines[i].strip())
        i += 1
        boxes = []
        for _ in range(face_count):
            x, y, w, h, *_ = map(float, lines[i].strip().split())
            boxes.append((x, y, w, h))
            i += 1

        full_path = os.path.join(base_image_path, filename)
        dataset.append((full_path, boxes))
    return dataset

def scale_by_width(img, true_boxes, width):
    h_orig, w_orig = img.shape[:2]

    img_resized = imutils.resize(img, width=width)
    h_new, w_new = img_resized.shape[:2]

    scale_x = w_new / w_orig
    scale_y = h_new / h_orig

    scaled_boxes = [
        (x * scale_x, y * scale_y, w * scale_x, h * scale_y)
        for (x, y, w, h) in true_boxes
    ]

    return img_resized, scaled_boxes

def compute_iou(boxA, boxB):
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[0] + boxA[2], boxB[0] + boxB[2])
    yB = min(boxA[1] + boxA[3], boxB[1] + boxB[3])

    interW = max(0, xB - xA)
    interH = max(0, yB - yA)
    interArea = interW * interH

    boxAArea = boxA[2] * boxA[3]
    boxBArea = boxB[2] * boxB[3]
    unionArea = boxAArea + boxBArea - interArea

    return interArea / unionArea if unionArea != 0 else 0

def compute_recall(predicted_boxes, true_boxes, iou_threshold=0.5):
    matched = set()
    for true_box in true_boxes:
        for pred_box in predicted_boxes:
            if (
                compute_iou(pred_box, true_box) >= iou_threshold
                and true_box not in matched
            ):
                matched.add(true_box)
                break
    TP = len(matched)
    FN = len(true_boxes) - TP
    return TP / (TP + FN) if (TP + FN) > 0 else 0

def compute_precision(predicted_boxes, true_boxes, iou_threshold=0.5):
    matched = set()
    for pred_box in predicted_boxes:
        for true_box in true_boxes:
            if (
                compute_iou(pred_box, true_box) >= iou_threshold
                and true_box not in matched
            ):
                matched.add(true_box)
                break
    TP = len(matched)
    FP = len(predicted_boxes) - TP
    return TP / (TP + FP) if (TP + FP) > 0 else 0