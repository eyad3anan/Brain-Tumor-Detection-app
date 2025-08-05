import streamlit as st
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.resnet_v2 import preprocess_input
import numpy as np
from PIL import Image
import cv2

# Configure Streamlit page
st.set_page_config(
    page_title="Brain Tumor Detection",
    page_icon="🧠",
    layout="centered"
)

# Cache the model loading to improve performance
@st.cache_resource
def load_brain_tumor_model():
    """Load the pre-trained ResNet50V2 model"""
    try:
        model = load_model("E:\\Downloads\\Brain_tumor_model\\ResNet50V2_model.h5")
        return model
    except Exception as e:
        st.error(f"Error loading model: {str(e)}")
        return None

# Class labels based on your training data
CLASS_LABELS = ['Glioma', 'Meningioma', 'No Tumor', 'Pituitary']

def preprocess_image(uploaded_image):
    """Preprocess the uploaded image for model prediction"""
    try:
        # Open and convert image
        img = Image.open(uploaded_image)
        
        # Convert to RGB if necessary
        if img.mode != 'RGB':
            img = img.convert('RGB')
        
        # Resize to model input size
        img = img.resize((224, 224))
        
        # Convert to array
        img_array = np.array(img)
        
        # Expand dimensions to match model input
        img_array = np.expand_dims(img_array, axis=0)
        
        # Preprocess for ResNet50V2
        img_array = preprocess_input(img_array)
        
        return img_array, img
    except Exception as e:
        st.error(f"Error preprocessing image: {str(e)}")
        return None, None

def make_prediction(model, processed_image):
    """Make prediction using the loaded model"""
    try:
        predictions = model.predict(processed_image, verbose=0)
        predicted_class = np.argmax(predictions[0])
        confidence = float(predictions[0][predicted_class])
        
        return predicted_class, confidence, predictions[0]
    except Exception as e:
        st.error(f"Error making prediction: {str(e)}")
        return None, None, None

def main():
    # App title and description
    st.title("🧠 Brain Tumor Detection System")
    st.markdown("""
    This application uses a ResNet50V2 deep learning model to classify brain MRI images 
    into four categories: **Glioma**, **Meningioma**, **No Tumor**, and **Pituitary**.
    """)
    
    # Load model
    with st.spinner("Loading model..."):
        model = load_brain_tumor_model()
    
    if model is None:
        st.error("Failed to load the model. Please check if the model file exists.")
        return
    
    st.success("Model loaded successfully!")
    
    # File uploader
    st.subheader("Upload MRI Image")
    uploaded_file = st.file_uploader(
        "Choose a brain MRI image...",
        type=['jpg', 'jpeg', 'png'],
        help="Upload a brain MRI image in JPG, JPEG, or PNG format"
    )
    
    if uploaded_file is not None:
        # Create two columns for layout
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Uploaded Image")
            # Display uploaded image
            display_image = Image.open(uploaded_file)
            st.image(display_image, caption="Original Image", use_column_width=True)
        
        with col2:
            st.subheader("Prediction Results")
            
            # Process image and make prediction
            with st.spinner("Analyzing image..."):
                processed_image, processed_pil = preprocess_image(uploaded_file)
                
                if processed_image is not None:
                    predicted_class, confidence, all_predictions = make_prediction(model, processed_image)
                    
                    if predicted_class is not None:
                        # Display main prediction
                        predicted_label = CLASS_LABELS[predicted_class]
                        
                        # Color coding for results
                        if predicted_label == "No Tumor":
                            st.success(f"**Prediction: {predicted_label}**")
                        else:
                            st.warning(f"**Prediction: {predicted_label}**")
                        
                        st.metric("Confidence", f"{confidence:.2%}")
                        
                        # Display all probabilities
                        st.subheader("Detailed Probabilities")
                        for i, (label, prob) in enumerate(zip(CLASS_LABELS, all_predictions)):
                            st.write(f"**{label}:** {prob:.3f} ({prob*100:.1f}%)")
                            st.progress(float(prob))
        
        # Additional information
        st.markdown("---")
        st.subheader("ℹ️ Important Notes")
        st.warning("""
        **Disclaimer:** This tool is for educational and research purposes only. 
        It should not be used as a substitute for professional medical diagnosis. 
        Always consult with qualified healthcare professionals for medical advice.
        """)
        
        with st.expander("About the Model"):
            st.markdown("""
            - **Architecture:** ResNet50V2
            - **Input Size:** 224×224 pixels
            - **Classes:** Glioma, Meningioma, No Tumor, Pituitary
            - **Training:** The model was trained on brain MRI images
            """)

if __name__ == "__main__":
    main()