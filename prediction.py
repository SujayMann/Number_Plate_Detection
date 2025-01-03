import torch
import streamlit as st
import easyocr
import cv2
from ultralytics import YOLO
import numpy as np

model = YOLO('best.torchscript', task='detect')
reader = easyocr.Reader(['en'])

def predict_box(image):
    bbox = []
    with torch.inference_mode():
        results = model(image)
        for res in results:
            boxes = res.boxes
            for box in boxes:
                if box:
                    xmin, ymin, xmax, ymax = box.xyxy.tolist()[0]
                    bbox.append([int(xmin), int(ymin), int(xmax), int(ymax)])
        if bbox:
            return bbox[0]
        else:
            return []

def crop_image(bbox, image):
    xmin, ymin, xmax, ymax = bbox
    new_image = image[ymin:ymax, xmin:xmax]
    return new_image

def predict(img_file):
    bytes_data = img_file.getvalue()
    img = cv2.imdecode(np.frombuffer(bytes_data, np.uint8), cv2.IMREAD_COLOR)
    bbox = predict_box(img)
    if not bbox:
        return [['', 'Not found']]
    image = crop_image(bbox, img)
    results = reader.readtext(image)
    return results

def prediction_page():
    st.title('Number Plate Detection')
    choice = st.radio(label='Choose an option', options=('Take a picture', 'Upload a picture'))
    if choice == 'Take a picture':
        enable = st.checkbox('Enable Camera')
        img_file = st.camera_input('Cam', disabled = not enable)
    else:
        img_file = st.file_uploader('Upload a picture', type=['png', 'jpg', 'jpeg'])
    if img_file is not None:
        st.image(img_file)
        results = predict(img_file)
        st.write(f'License Plate: {results[0][1]}')
    
if __name__ == '__page__':
    prediction_page()
