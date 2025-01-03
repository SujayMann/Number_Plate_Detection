# Number_Plate_Detection

This project is a Number Plate Detector built using a fine tuned **YOLO11m** Model in **PyTorch**. The model can detect the number plate and then extract the text using **EasyOCR**. The app is deployed on **Streamlit** for an easy-to-use interface.
## Features

* **Image Upload and Capture:** Upload or Capture an image using streamlit's interface.
* **License Plate Detection:** The app uses a **YOLO11m** model for detecting a license plate in the image.
* **Text Recognition**: The app uses **EasyOCR** to output the text of the detected number plate.
* **Real-time Results:** Get immediate results once the image is uploaded.

## Demo

Try the app by visiting [app](https://number-plate-detection-sm.streamlit.app/).

## Requirements

To run the app locally, some dependencies are needed.
* streamlit
* torch
* opencv-python
* easyocr
* ultralytics
* numpy

Download the EasyOCR model `english_g2` from [https://www.jaided.ai/easyocr/modelhub/](https://www.jaided.ai/easyocr/modelhub/)

Install the dependencies with:
```
pip install -r requirements.txt
```

## Run the app

1. Clone the repository
```
git clone https://github.com/SujayMann/Number_Plate_Detection.git
cd Number_Plate_Detection
```
2. Install the dependencies
```
pip install -r requirements.txt
```
3. Run the streamlit app
```
streamlit run app.py
```

## How It Works

1. The app uses a **YOLO11m** model fine-tuned on **Car License Plate Detection** dataset on **Kaggle**.
2. It also uses **EasyOCR** for text recognition.
3. Once the image is uploaded, the model detects the number plate and passes it to the **EasyOCR** model.
4. The model outputs the predicted text immediately.

## Dataset

This project uses the **Car License Plate Detection** dataset from **Kaggle**. The dataset consists of license plate images and annotations.
* Dataset link: [Car License Plate Detection](https://www.kaggle.com/datasets/andrewmvd/car-plate-detection).
