import cv2
from ultralytics import YOLO

# Load a model
model = YOLO("yolov8n.yaml")  # build a new model from scratch
modelP = YOLO("/Users/towfiislam/PycharmProjects/Resnet/runs/detect/train5/weights/best.pt")  # load a pretrained model (recommended for training)

# Use the model
model.train(data="/Users/towfiislam/Downloads/Recycle-Detection.v3i.yolov8/data.yaml", epochs=15)  # train the model
metrics = model.val()  # evaluate model performance on the validation set
results = model("https://ultralytics.com/images/bus.jpg")  # predict on an image
path = model.export(format="pt")  # export the model to ONNX format
