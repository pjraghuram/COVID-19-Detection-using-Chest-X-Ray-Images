#!/usr/bin/env python3
"""
COVID-19 X-Ray Classification - Streamlit Frontend
Professional web interface for X-ray image classification
"""

import streamlit as st
import numpy as np
import pandas as pd
from PIL import Image, ImageOps
import tensorflow as tf
import io
import sys
from pathlib import Path
import plotly.graph_objects as go
import plotly.express as px

# Add src to Python path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root / 'src'))

# Page configuration
st.set_page_config(
    page_title="COVID-19 X-Ray Classifier",
    page_icon="🫁",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        text-align: center;
        padding: 2rem 0;
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        color: white;
        border-radius: 10px;
        margin-bottom: 2rem;
    }
    .prediction-box {
        border: 2px solid #ddd;
        border-radius: 10px;
        padding: 1rem;
        margin: 1rem 0;
        background-color: #f8f9fa;
    }
    .covid-positive {
        border-color: #dc3545;
        background-color: #f8d7da;
    }
    .covid-negative {
        border-color: #28a745;
        background-color: #d4edda;
    }
    .confidence-bar {
        height: 20px;
        border-radius: 10px;
        text-align: center;
        line-height: 20px;
        color: white;
        font-weight: bold;
    }
    .warning-box {
        background-color: #fff3cd;
        border: 1px solid #ffeaa7;
        border-radius: 5px;
        padding: 1rem;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

@st.cache_resource
def load_model():
    """Load the trained COVID-19 classification model"""
    try:
        # Try to load the best model first
        model_paths = [
            project_root / 'saved_models' / 'best_model.h5',
            project_root / 'saved_models' / 'covid_model.h5'
        ]
        
        for model_path in model_paths:
            if model_path.exists():
                model = tf.keras.models.load_model(str(model_path))
                return model, str(model_path)
        
        return None, None
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None, None

def preprocess_image(image, target_size=(64, 64)):
    """
    Preprocess uploaded image for model prediction
    """
    try:
        # Convert to grayscale
        if image.mode != 'L':
            image = ImageOps.grayscale(image)
        
        # Resize to model input size
        image = image.resize(target_size)
        
        # Convert to numpy array
        img_array = np.asarray(image)
        
        # Normalize pixel values
        img_array = img_array / 255.0
        
        # Reshape for model input (batch_size, height, width, channels)
        img_array = img_array.reshape(1, target_size[0], target_size[1], 1)
        
        return img_array
    except Exception as e:
        st.error(f"Error preprocessing image: {e}")
        return None

def make_prediction(model, processed_image):
    """
    Make prediction using the loaded model
    """
    try:
        # Get prediction probability
        prediction_prob = model.predict(processed_image, verbose=0)[0][0]
        
        # Determine class and confidence
        if prediction_prob > 0.5:
            prediction_class = "COVID-19 Positive"
            confidence = prediction_prob
            is_covid = True
        else:
            prediction_class = "COVID-19 Negative"
            confidence = 1 - prediction_prob
            is_covid = False
        
        return {
            'class': prediction_class,
            'confidence': confidence,
            'probability': prediction_prob,
            'is_covid': is_covid
        }
    except Exception as e:
        st.error(f"Error making prediction: {e}")
        return None

def create_confidence_chart(confidence, is_covid):
    """
    Create a confidence visualization
    """
    colors = ['#dc3545' if is_covid else '#28a745']
    
    fig = go.Figure(go.Bar(
        x=[confidence],
        y=['Confidence'],
        orientation='h',
        marker=dict(color=colors),
        text=[f'{confidence:.1%}'],
        textposition='inside'
    ))
    
    fig.update_layout(
        title="Prediction Confidence",
        xaxis=dict(range=[0, 1], tickformat='.0%'),
        height=150,
        margin=dict(l=10, r=10, t=40, b=10)
    )
    
    return fig

def display_model_info():
    """
    Display information about the model
    """
    st.sidebar.markdown("### 📊 Model Information")
    
    model_info = {
        "Architecture": "VGG16 + Custom Classifier",
        "Input Size": "64×64 Grayscale",
        "Training Accuracy": "91.97%",
        "Precision": "71.69%",
        "Recall": "87.55%",
        "F1-Score": "78.83%",
        "AUC-ROC": "96.69%"
    }
    
    for key, value in model_info.items():
        st.sidebar.metric(key, value)


def main():
    """
    Main Streamlit application
    """
    
    # Header
    st.markdown("""
    <div class="main-header">
        <h1>🫁 COVID-19 X-Ray Classifier</h1>
        <p>AI-Powered Chest X-Ray Analysis for COVID-19 Screening</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Load model
    with st.spinner("Loading AI model..."):
        model, model_path = load_model()
    
    if model is None:
        st.error("❌ Failed to load the trained model. Please ensure the model files exist in 'saved_models/' directory.")
        st.stop()
    
    st.success(f"✅ Model loaded successfully from: {Path(model_path).name}")
    
    # Sidebar
    display_model_info()
    
    # Main content
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.markdown("### 📤 Upload X-Ray Image")
        
        # File uploader
        uploaded_file = st.file_uploader(
            "Choose a chest X-ray image...",
            type=['png', 'jpg', 'jpeg'],
            help="Upload a chest X-ray image in PNG, JPG, or JPEG format"
        )
        
        if uploaded_file is not None:
            # Display uploaded image
            image = Image.open(uploaded_file)
            st.image(image, caption="Uploaded X-Ray Image", use_column_width=True)
            
            # Image info
            st.markdown(f"""
            **Image Details:**
            - Filename: {uploaded_file.name}
            - Size: {image.size[0]} × {image.size[1]} pixels
            - Mode: {image.mode}
            - Format: {image.format}
            """)
            
            # Prediction button
            if st.button("🔍 Analyze X-Ray", type="primary", use_container_width=True):
                with st.spinner("Analyzing image..."):
                    # Preprocess image
                    processed_image = preprocess_image(image)
                    
                    if processed_image is not None:
                        # Make prediction
                        result = make_prediction(model, processed_image)
                        
                        if result is not None:
                            # Store result in session state for display in col2
                            st.session_state['prediction_result'] = result
                            st.session_state['original_image'] = image
                            st.rerun()
    
    with col2:
        st.markdown("### 🎯 Analysis Results")
        
        if 'prediction_result' in st.session_state:
            result = st.session_state['prediction_result']
            
            # Prediction result box
            box_class = "covid-positive" if result['is_covid'] else "covid-negative"
            
            st.markdown(f"""
            <div class="prediction-box {box_class}">
                <h3>🏥 Diagnosis: {result['class']}</h3>
                <p><strong>Confidence:</strong> {result['confidence']:.1%}</p>
                <p><strong>Raw Probability:</strong> {result['probability']:.4f}</p>
            </div>
            """, unsafe_allow_html=True)
            
            # Confidence chart
            confidence_chart = create_confidence_chart(result['confidence'], result['is_covid'])
            st.plotly_chart(confidence_chart, use_container_width=True)
            
            # Interpretation
            st.markdown("### 📋 Interpretation")
            
            if result['is_covid']:
                st.markdown("""
                🔴 **COVID-19 Positive Indication**
                - The model suggests potential COVID-19 patterns in the X-ray
                - Recommend immediate consultation with healthcare professionals
                - Consider additional diagnostic tests (RT-PCR, CT scan)
                - Follow isolation protocols as advised by medical professionals
                """)
            else:
                st.markdown("""
                🟢 **COVID-19 Negative Indication**
                - The model suggests no obvious COVID-19 patterns detected
                - This does not rule out other respiratory conditions
                - Clinical correlation is always recommended
                - Consult healthcare professionals for complete evaluation
                """)
            
            # Additional metrics
            st.markdown("### Model Performance Metrics")
            
            metrics_col1, metrics_col2, metrics_col3 = st.columns(3)
            
            with metrics_col1:
                st.metric("Sensitivity", "87.55%", help="Ability to correctly identify COVID-19 cases")
            
            with metrics_col2:
                st.metric("Specificity", "92.88%", help="Ability to correctly identify non-COVID cases")
            
            with metrics_col3:
                st.metric("AUC-ROC", "96.69%", help="Overall discriminative ability")
        
        else:
            st.info("👆 Upload an X-ray image and click 'Analyze' to see results here.")
            
            # Sample images section
            st.markdown("### 🖼️ Sample X-Ray Categories")
            
            sample_info = {
                "COVID-19": "Shows ground-glass opacities, consolidations",
                "Normal": "Clear lung fields, normal cardiac silhouette",
                "Viral Pneumonia": "Bilateral infiltrates, inflammatory changes",
                "Lung Opacity": "Various opacification patterns"
            }
            
            for condition, description in sample_info.items():
                st.markdown(f"**{condition}:** {description}")
    
    
    # Footer
    st.markdown("---")
    st.markdown("""
    <div style="text-align: center; color: #666;">
        <p>Built with ❤️ using Streamlit and TensorFlow | 
        Model Accuracy: 91.97% | 
        For Educational Use Only</p>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()