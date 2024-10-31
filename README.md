# Eczema Extraction

Welcome to **Eczema Extraction**! This is a Streamlit-based web application designed for the detection and analysis of Eczema features from medical images using image processing and machine learning techniques.

## Features

- **Image Upload**: Upload medical images for analysis.
- **Image Segmentation**: The app performs segmentation on uploaded images to isolate regions relevant to Eczema analysis.
- **Feature Extraction**: Extracts various features such as area, color, texture, and other morphological data for further study.
- **Classification**: Uses machine learning models to classify the extracted features and provide predictions related to Eczema severity or type.
- **Visualization**: Displays processed images and key metrics from the analysis.

## Getting Started

1. **Access the Application**: Visit the app at [https://eczemaextraction.streamlit.app](https://eczemaextraction.streamlit.app).
2. **Upload an Image**: Use the upload button to upload a medical image for Eczema analysis.
3. **Run Analysis**: Click "Analyze" to initiate image processing, segmentation, and feature extraction.
4. **View Results**: The results, including segmented images and classification details, will be displayed on the screen.

## Technologies Used

- **Streamlit**: For creating the web interface.
- **OpenCV**: For image processing and feature extraction.
- **Scikit-Learn / TensorFlow**: For the machine learning model to classify features.
- **Pandas & NumPy**: For data manipulation and calculations.

## Installation (Optional for Local Setup)

To run the application locally, follow these steps:

1. Clone this repository:
   ```bash
   git clone https://github.com/username/eczema-extraction.git
   ```
2. Navigate to the project directory:
  ```bash
  Copy code
  cd eczema-extraction
   ```
3. Install the required dependencies:
  ```bash
  Copy code
  pip install -r requirements.txt
   ```
4. Run the Streamlit app:
  ```bash
  Copy code
  streamlit run app.py
   ```
