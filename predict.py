from ultralytics import YOLO
import cv2

model_path = 'runs/segment/train21/weights/best.pt'
image_path = 'C:/Users/NESAC/Desktop/123.jpg'

model = YOLO(model_path)
results = model.predict(image_path, save_conf=True, show_boxes=False, save=True)