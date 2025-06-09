from ultralytics import YOLO

model = YOLO(
    "runs/train/yolo_train_20250608_200028/weights/best.pt"
)  # load a pretrained model

# Predict with the model
results = model("interface/tests/test12.jpg")  # predict on an image
for result in results:
    boxes = result.boxes  # Boxes object for bounding box outputs
    masks = result.masks  # Masks object for segmentation masks outputs
    keypoints = result.keypoints  # Keypoints object for pose outputs
    probs = result.probs  # Probs object for classification outputs
    obb = result.obb  # Oriented boxes object for OBB outputs
    result.save(filename="interface/result.jpg")  # save to disk
