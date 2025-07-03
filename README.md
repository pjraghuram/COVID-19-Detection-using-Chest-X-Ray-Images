# 🫁 COVID-19 X-Ray Classification

An AI-powered chest X-ray analysis tool for COVID-19 screening using deep learning.

## 🎯 Project Overview

This project uses a VGG16-based deep learning model to classify chest X-ray images and detect potential COVID-19 indicators. The model achieves **91.97% accuracy** with excellent sensitivity and specificity for medical screening applications.

## 📊 Model Performance

- **Accuracy**: 91.97%
- **Precision**: 71.69%
- **Recall (Sensitivity)**: 87.55%
- **Specificity**: 92.88%
- **F1-Score**: 78.83%
- **AUC-ROC**: 96.69%

## 🏗️ Architecture

- **Base Model**: VGG16 (pre-trained on ImageNet)
- **Custom Layers**: Dense layers with dropout for classification
- **Input**: 64×64 grayscale chest X-ray images
- **Output**: Binary classification (COVID/Non-COVID)

## 🖼️ Dataset

The model was trained on the COVID-19 Radiography Database containing:
- COVID-19 positive cases
- Normal chest X-rays
- Viral pneumonia cases
- Lung opacity cases

## 🔧 Technical Stack

- **Framework**: TensorFlow/Keras
- **Frontend**: Streamlit
- **Data Processing**: NumPy, Pandas, PIL
- **Visualization**: Plotly, Matplotlib
- **Deployment**: Streamlit Cloud

## Features

- 📤 **Image Upload**: Drag & drop X-ray images
- 🎯 **AI Analysis**: Real-time COVID-19 detection
- 📊 **Confidence Scores**: Visual confidence indicators
- 🏥 **Medical Guidance**: Interpretation and recommendations
- 📈 **Performance Metrics**: Model accuracy statistics

## Local Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/yourusername/covid19-xray-classifier.git
   cd covid19-xray-classifier
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Run the application**
   ```bash
   streamlit run frontend/app.py
   ```

## Project Structure

```
covid19-xray-classifier/
├── frontend/
│   └── app.py              # Streamlit web application
├── src/
│   ├── preprocessing.py    # Data preprocessing
│   ├── model.py           # Model architecture
│   ├── train.py           # Training pipeline
│   └── predict.py         # Prediction utilities
├── requirements.txt        # Dependencies
├── README.md              # Project documentation
└── .gitignore            # Git ignore rules
```


## This project demonstrates:
- Transfer learning with VGG16
- Medical image classification
- Class imbalance handling with SMOTE
- Model evaluation metrics
- Web deployment with Streamlit

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Submit a pull request


