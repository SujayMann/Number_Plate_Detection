import torch
import streamlit as st
import easyocr
import cv2
from ultralytics import YOLO
import numpy as np
import os

if not os.path.exists('best.torchscript'):
    st.write("Model file not found!")
    
model = YOLO('best.torchscript', task='detect')
reader = easyocr.Reader(['en'], model_storage_directory='.')

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

def get_text(img):
    results = reader.readtext(img)
    if results:
        return results[0][1]
    return 'Not Found'

def draw_box_and_text(frame, bbox, text):
    xmin, ymin, xmax, ymax = bbox
    frame = cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), (0, 255, 0), 2)
    frame = cv2.putText(frame, text, (xmin, ymin - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2, cv2.LINE_AA)
    return frame

def process_frame(frame):
    bbox = predict_box(frame)
    if bbox:
        cropped_img = crop_image(bbox, frame)
        text = get_text(cropped_img)
        frame = draw_box_and_text(frame, bbox, text)
    else:
        cv2.putText(frame, "No number plate detected.", (20, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)

    return frame

def live_video():
    st.subheader('Live Detection')
    if st.checkbox('Enable Camera', value=True):
        capture = cv2.VideoCapture(0)

        if not capture.isOpened():
            st.error('Error: Could not access the camera.')
            return
        
        frame_count = 0
        n_skips = 5

        stframe = st.empty()
        while True:
            ret, frame = capture.read()
            if not ret:
                st.write('Failed to capture image.')
                break
            
            if frame_count % n_skips == 0:
                frame = process_frame(frame)

            frame_count += 1
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            stframe.image(frame_rgb, channels='RGB', use_container_width=True)
        
        cap.release()        
    
    else:
        st.write('Camera is disabled. Please enable it to start the live video.')

def prediction_page():
    st.title('Number Plate Detection')
    choice = st.radio(label='Choose an option', options=('Upload a picture', 'Live Video Feed'))

    if choice == 'Upload a picture':
        st.subheader('File Upload')
        img_file = st.file_uploader('Upload a picture', type=['png', 'jpg', 'jpeg'])
        if img_file is not None:
            bytes_data = img_file.getvalue()
            img = cv2.imdecode(np.frombuffer(bytes_data, np.uint8), cv2.IMREAD_COLOR)
            bbox = predict_box(img)
            if bbox:
                cropped_image = crop_image(bbox, img)
                text = get_text(cropped_image)
                img = draw_box_and_text(img, bbox, text)
                img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

                st.image(img_rgb, caption='Processed Image')
                st.write(f'License Plate: {text}')
            else:
                st.write('License Plate Not Found.')

    elif choice == 'Live Video Feed':
        live_video()

if __name__ == '__page__':
    prediction_page()
