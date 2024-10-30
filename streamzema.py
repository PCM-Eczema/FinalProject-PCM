import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import streamlit as st
from streamlit_option_menu import option_menu
from skimage.measure import regionprops_table
from skimage import io, color, measure, img_as_ubyte, feature, filters, exposure, morphology
import math
from skimage.measure import label, regionprops
from matplotlib.colors import ListedColormap
from scipy import ndimage as ndi

# Load the image and process it once, making `img` available for all menu options
image_path = r"eczema-subacute-35-NoWM.jpeg"
im = io.imread(image_path)
img = color.rgb2gray(im)  # Convert image to grayscale
img = img_as_ubyte(img)   # Convert to uint8

# Sidebar menu
with st.sidebar:
    selected = option_menu("Pengolahan Citra Medika", ["Home", "Encyclopedia", "Feature Extraction", "Chatbot"], default_index=0)

# Home Page
if selected == "Home":
    st.title('Final Project ✨')
    st.subheader("Anggota kelompok")
    group_members = [
        "Farhan Majid - 5023211049",
        "Leony Purba - 50232110",
        "Benedicta Sabdaningtyas - 50232110",
        "Adelia Safira - 50232110"
    ]
    for member in group_members:
        st.markdown(f"<p style='font-family:Georgia; color: black; font-size: 20px;'>{member}</p>", unsafe_allow_html=True)

# Encyclopedia Page
elif selected == "Encyclopedia":
    st.markdown("<h1 style='text-align: center; color: red;'>🫀ENCYCLOPEDIA</h1>", unsafe_allow_html=True)
    questions = [
        ("1. Apa itu Eczema Subacute?", "Eczema subacute adalah bentuk eksim yang ditandai dengan gejala peradangan kulit yang sedang berlangsung, muncul setelah fase akut eksim. Pada tahap ini, gejala yang umum terjadi meliputi rasa gatal yang mengganggu, kemerahan dan pembengkakan pada kulit, kekeringan dan pengelupasan, serta lesi yang dapat mengeluarkan cairan atau membentuk keropeng saat mengering."),
        ("2. Apa gejala umum yang terjadi pada eczema subacute?", "Eczema subacute dapat dipicu oleh berbagai faktor, seperti alergen (serbuk sari atau debu), iritan (deterjen atau sabun), perubahan cuaca yang ekstrem, dan stres emosional. Pengobatan untuk eczema subacute biasanya melibatkan penggunaan krim atau salep kortikosteroid untuk mengurangi peradangan dan gatal, serta pelembap untuk menjaga kelembapan kulit. Penting juga untuk mengidentifikasi dan menghindari alergen atau iritan yang dapat memperburuk kondisi. Jika tidak ditangani dengan baik, eczema subacute dapat berkembang menjadi eczema kronis, sehingga mencari perawatan yang tepat dan mengikuti saran dokter sangatlah penting untuk mengelola gejala dan mencegah kekambuhan."),
    ]
    for title, description in questions:
        st.markdown(f"<p style='font-family:Georgia; color:yellow; font-size: 23px; text-align: left;'>{title}</p>", unsafe_allow_html=True)
        st.markdown(f"<p style='font-family:Georgia; color:white; font-size: 20px; text-align: justify;'>{description}</p>", unsafe_allow_html=True)
    st.markdown("""<iframe width="560" height="315" src="https://www.youtube.com/embed/fmurdUlmaIg" frameborder="0" allow="accelerometer; autoplay; encrypted-media; gyroscope; picture-in-picture" allowfullscreen></iframe>""", unsafe_allow_html=True)

# Feature Extraction Page
elif selected == "Feature Extraction":
    st.title("Eczema Image Feature Extraction 🧬")
    selected2 = option_menu(None, ["Image", "Image Processing", "Edge Detection", "Image Segmentation", "Data"], icons=['image', 'adjust', 'filter', 'table'], menu_icon="cast", default_index=0, orientation="horizontal")

    if selected2 == "Image":
        st.subheader("Image Section")
        st.image(im, caption="Loaded Eczema Image", use_column_width=True)
        st.subheader("Grayscale Image (Converted to uint8)")
        st.image(img, caption="Grayscale Image", use_column_width=True)

    elif selected2 == "Image Processing":
        st.subheader("Pre-Processing")
        threshold = filters.threshold_otsu(img)
        st.write(f"Otsu's threshold value: {threshold}")

        img_hieq = exposure.equalize_adapthist(img, clip_limit=0.9) * 255  
        img_hieq = img_hieq.astype('uint8')

        fig, ax = plt.subplots(ncols=2, figsize=(12, 6))
        ax[0].imshow(img, cmap='gray')
        ax[0].contour(img, levels=[threshold], colors='red')
        ax[0].set_title('Original Image with Otsu Contour')
        ax[1].imshow(img < threshold, cmap='gray')
        ax[1].set_title('Binary Image (Otsu Threshold Applied)')
        st.pyplot(fig)

    elif selected2 == "Edge Detection":
        st.subheader("Edge Detection Filters")
        roberts = filters.roberts(img)
        sobel = filters.sobel(img)
        prewitt = filters.prewitt(img)
        canny = feature.canny(img, sigma=1)

        fig, ax = plt.subplots(2, 2, figsize=(8, 8))
        ax[0, 0].imshow(roberts, cmap='gray')
        ax[0, 0].set_title('Roberts')
        ax[0, 1].imshow(sobel, cmap='gray')
        ax[0, 1].set_title('Sobel')
        ax[1, 0].imshow(prewitt, cmap='gray')
        ax[1, 0].set_title('Prewitt')
        ax[1, 1].imshow(canny, cmap='gray')
        ax[1, 1].set_title(r'Canny $\sigma=1$')
        st.pyplot(fig)

    elif selected2 == "Image Segmentation":
        st.subheader("Contour Image and Labeling")
        label_img = label(img < filters.threshold_otsu(img))
        regions = regionprops(label_img)
        
        fig, ax = plt.subplots()
        ax.imshow(img, cmap=plt.cm.gray)

        for props in regions:
            y0, x0 = props.centroid
            orientation = props.orientation
            x1 = x0 + math.cos(orientation) * 0.5 * props.minor_axis_length
            y1 = y0 - math.sin(orientation) * 0.5 * props.minor_axis_length
            ax.plot((x0, x1), (y0, y1), '-r', linewidth=2.5)
            ax.plot(x0, y0, '.g', markersize=15)
        st.pyplot(fig)

    elif selected2 == "Data":
        st.subheader("Extracted Data")
        excel_path = r"C:\Users\50232\Downloads\extract_features.xlsx"
        if os.path.exists(excel_path):
            df_existing = pd.read_excel(excel_path)
            st.write("Existing Extracted Features:")
            st.write(df_existing)

        label_img = measure.label(img < filters.threshold_otsu(img))
        props = regionprops_table(label_img, properties=('centroid', 'orientation', 'major_axis_length', 'minor_axis_length'))
        df_new = pd.DataFrame(props)
        st.write("Newly Extracted Features:")
        st.write(df_new)

# Chatbot Page
elif selected == "Chatbot":
    st.title("Eczema Chatbot 🤖")

    if "messages" not in st.session_state:
        st.session_state["messages"] = []

    qa_pairs = {
        "What is eczema?": "Eczema is a condition that makes your skin red and itchy.",
        "How can I prevent eczema flare-ups?": "To prevent flare-ups, keep your skin moisturized, avoid triggers, and avoid scratching.",
    }

    st.write("### Ask me about Eczema:")
    user_question = st.selectbox("Choose a question:", [""] + list(qa_pairs.keys()))

    if user_question:
        with st.chat_message("user"):
            st.markdown(user_question)
        st.session_state.messages.append({"role": "user", "content": user_question})

        bot_response = qa_pairs.get(user_question, "I'm sorry, I don't have an answer.")
        with st.chat_message("assistant"):
            st.markdown(bot_response)
        st.session_state.messages.append({"role": "assistant", "content": bot_response})

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
