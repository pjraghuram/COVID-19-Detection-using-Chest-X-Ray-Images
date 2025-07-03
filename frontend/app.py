#!/usr/bin/env python3
"""
COVID-19 X-Ray Classification - Uses Actual best_model.h5
Simple and direct implementation
"""

import streamlit as st
import numpy as np
import pandas as pd
from PIL import Image, ImageOps
import tensorflow as tf
import os
from pathlib import Path
import plotly.graph_objects as go

# Page configuration
st.set_page_config(
    page_title="COVID-19 X-Ray Classifier",
    page_icon="🫁",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
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
    """Load the trained COVID-19 model from best_model.h5"""
    
    # Define possible paths for the model
    possible_paths = [
        "saved_models/best_model.h5",
        "best_model.h5",
        "../saved_models/best_model.h5",
        "saved_models/covid_model.h5"
    ]
    
    for model_path in possible_paths:
        if os.path.exists(model_path):
            try:
                st.info(f"Loading model from: {model_path}")
                model = tf.keras.models.load_model(model_path)
                st.success("✅ Model loaded successfully!")
                return model, model_path
            except Exception as e:
                st.error(f"Error loading model from {model_path}: {e}")
    
    st.error("❌ Could not find best_model.h5 file. Please ensure it's in the correct location.")
    return None, None

def preprocess_image(image):
    """
    Preprocess image exactly as done during training
    """
    # Convert to grayscale
    if image.mode != 'L':
        image = ImageOps.grayscale(image)
    
    # Resize to 64x64 (same as training)
    image = image.resize((64, 64))
    
    # Convert to numpy array
    img_array = np.asarray(image)
    
    # Normalize (divide by 255)
    img_array = img_array / 255.0
    
    # Reshape for model input: (1, 64, 64, 1)
    img_array = img_array.reshape(1, 64, 64, 1)
    
    return img_array

def make_prediction(model, processed_image):
    """
    Make prediction using the loaded model
    """
    # Get prediction probability
    prediction_prob = model.predict(processed_image, verbose=0)[0][0]
    
    # Determine class (threshold = 0.5)
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

def create_confidence_chart(confidence, is_covid):
    """Create confidence visualization"""
    color = '#dc3545' if is_covid else '#28a745'
    
    fig = go.Figure(go.Bar(
        x=[confidence],
        y=['Confidence'],
        orientation='h',
        marker=dict(color=color),
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
    """Display model information in sidebar"""
    st.sidebar.markdown("### 📊 Model Information")
    
    model_metrics = {
        "Architecture": "VGG16 + Custom Classifier",
        "Input Size": "64×64 Grayscale",
        "Training Accuracy": "91.97%",
        "Precision": "71.69%",
        "Recall": "87.55%",
        "F1-Score": "78.83%",
        "AUC-ROC": "96.69%",
        "Specificity": "92.88%"
    }
    
    for metric, value in model_metrics.items():
        st.sidebar.metric(metric, value)

def main():
    """Main application"""
    
    # Header
    st.markdown("""
    <div class="main-header">
        <h1>🫁 COVID-19 X-Ray Classifier</h1>
        <p>AI-Powered Chest X-Ray Analysis using Trained VGG16 Model</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Load model
    model, model_path = load_model()
    
    if model is None:
        st.stop()
    
    # Display model info
    display_model_info()
    
    # Show model details
    st.info(f"🤖 Using trained model: {model_path}")
    st.info(f"📊 Model parameters: {model.count_params():,}")
    
    # Main content
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.markdown("### 📤 Upload X-Ray Image")
        
        uploaded_file = st.file_uploader(
            "Choose a chest X-ray image...",
            type=['png', 'jpg', 'jpeg'],
            help="Upload a chest X-ray image for COVID-19 analysis"
        )
        
        if uploaded_file is not None:
            # Display image
            image = Image.open(uploaded_file)
            st.image(image, caption="Uploaded X-Ray Image", use_column_width=True)
            
            # Image details
            st.markdown(f"""
            **Image Details:**
            - Filename: {uploaded_file.name}
            - Size: {image.size[0]} × {image.size[1]} pixels
            - Mode: {image.mode}
            """)
            
            # Analyze button
            if st.button("🔍 Analyze X-Ray", type="primary", use_container_width=True):
                with st.spinner("Analyzing with trained model..."):
                    
                    # Preprocess image
                    processed_image = preprocess_image(image)
                    
                    # Make prediction
                    result = make_prediction(model, processed_image)
                    
                    # Store in session state
                    st.session_state['prediction_result'] = result
                    st.session_state['original_image'] = image
                    st.rerun()
    
    with col2:
        st.markdown("### 🎯 Analysis Results")
        
        if 'prediction_result' in st.session_state:
            result = st.session_state['prediction_result']
            
            # Model status
            st.success("✅ **REAL MODEL**: Using your trained model with 91.97% accuracy")
            
            # Prediction box
            box_class = "covid-positive" if result['is_covid'] else "covid-negative"
            
            st.markdown(f"""
            <div class="prediction-box {box_class}">
                <h3>🏥 Prediction: {result['class']}</h3>
                <p><strong>Confidence:</strong> {result['confidence']:.1%}</p>
                <p><strong>Raw Probability:</strong> {result['probability']:.4f}</p>
            </div>
            """, unsafe_allow_html=True)
            
            # Confidence chart
            confidence_chart = create_confidence_chart(result['confidence'], result['is_covid'])
            st.plotly_chart(confidence_chart, use_container_width=True)
            
            # Interpretation
            st.markdown("### 📋 Clinical Interpretation")
            
            if result['is_covid']:
                st.markdown("""
                🔴 **COVID-19 Positive Indication**
                - The trained model detected patterns consistent with COVID-19
                - **Immediate action recommended:**
                  - Consult healthcare professionals immediately
                  - Consider RT-PCR confirmation
                  - Follow isolation protocols
                  - Monitor symptoms closely
                """)
            else:
                st.markdown("""
                🟢 **COVID-19 Negative Indication**
                - The trained model suggests no obvious COVID-19 patterns
                - **Important notes:**
                  - Does not rule out other respiratory conditions
                  - Clinical correlation always recommended
                  - Consult healthcare professionals for complete evaluation
                """)
            
            # Performance metrics
            st.markdown("### 📊 Model Performance")
            
            metrics_col1, metrics_col2, metrics_col3 = st.columns(3)
            
            with metrics_col1:
                st.metric("Sensitivity", "87.55%", help="Correctly identifies COVID-19 cases")
            
            with metrics_col2:
                st.metric("Specificity", "92.88%", help="Correctly identifies non-COVID cases")
            
            with metrics_col3:
                st.metric("Overall Accuracy", "91.97%", help="Overall model accuracy")
                
        else:
            st.info("👆 Upload an X-ray image and click 'Analyze' to see results")
            
            # Model description
            st.markdown("### 🤖 About the Model")
            st.markdown("""
            **Your Trained Model:**
            - **Architecture**: VGG16 with custom classifier layers
            - **Training**: 21,000+ chest X-ray images
            - **Classes**: COVID-19 vs Non-COVID (Normal, Viral Pneumonia, Lung Opacity)
            - **Preprocessing**: 64×64 grayscale, normalized
            - **Performance**: 91.97% accuracy, 96.69% AUC-ROC
            
            **Training Process:**
            - Transfer learning from ImageNet-pretrained VGG16
            - SMOTE balancing for class imbalance
            - Data augmentation and regularization
            - Adam optimizer with early stopping
            """)
    
    # Medical disclaimer
    st.markdown("""
    <div class="warning-box">
        <h4>⚠️ Medical Disclaimer</h4>
        <p>This AI tool is for <strong>research and educational purposes only</strong>. 
        It should <strong>NOT</strong> be used as a substitute for professional medical diagnosis. 
        Always consult with qualified healthcare professionals for medical advice and treatment decisions.</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Footer
    st.markdown("---")
    st.markdown("""
    <div style="text-align: center; color: #666;">
        <p>🧠 Powered by Your Trained VGG16 Model | 
        🎯 91.97% Accuracy | 
        🏥 For Educational Use Only</p>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()