# Eczema Extraction

Welcome to **Eczema Extraction**! This is a Streamlit-based web application designed for the detection and analysis of Eczema features from medical images using image processing.

## Features

- **Image Upload**: Supports uploading Eczema images in `.jpg`, `.jpeg`, or `.png` formats (up to 200MB per file).
- **Pre-Processing**: Applies Otsu's thresholding to generate contour and binary images for initial analysis.
- **Image Processing**: Offers tools for edge detection, segmentation, and other image enhancement features.
- **Feature Extraction**: Extracts essential features from Eczema images to assist in diagnosis and research.
- **Data Visualization**: Displays original and processed images side by side, along with calculated metrics.
- **Integrated Chatbot**: Provides assistance and answers to questions related to the app and Eczema.

## Getting Started

1. **Access the Application**: Visit the app at [https://eczemaextraction.streamlit.app](https://eczemaextraction.streamlit.app).
2. **Upload an Image**: Use the upload button to upload a medical image for Eczema analysis.
3. **View Results**: The results, including segmented images and classification details, will be displayed on the screen.

## Technologies Used

- **Streamlit**: For creating the interactive web interface.
- **streamlit-option-menu**: For enhanced menu options in the app.
- **SciPy**: For advanced mathematical and scientific computations.
- **scikit-image**: For image processing tasks.
- **openpyxl**: For handling Excel files.
- **Matplotlib**: For data visualization and plotting.
- **Pandas**: For data manipulation and handling.
- **NumPy**: For numerical computations.
- **fuzzywuzzy**: For string matching and similarity scoring.

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
