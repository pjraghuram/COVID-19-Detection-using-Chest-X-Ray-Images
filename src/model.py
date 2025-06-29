"""
COVID-19 X-Ray Classification - Model Architecture Module
Creates VGG16-based transfer learning model for binary classification
"""

import tensorflow as tf
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import (Dense, Conv2D, MaxPooling2D, 
                                   BatchNormalization, Activation, 
                                   Flatten, Dropout, Input)
from tensorflow.keras.applications import VGG16
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import plot_model
import os

def create_vgg16_base(input_shape=(64, 64, 1)):
    """
    Create VGG16 base model with custom input layer for grayscale images
    
    Args:
        input_shape (tuple): Shape of input images (height, width, channels)
    
    Returns:
        Model: VGG16 base model with custom input
    """
    
    # Create input tensor
    input_tensor = Input(shape=input_shape)
    
    # Convert grayscale to RGB (VGG16 expects 3 channels)
    # Add a Conv2D layer to convert 1 channel to 3 channels
    x = Conv2D(3, (3, 3), padding='same', name='grayscale_to_rgb')(input_tensor)
    
    # Load VGG16 without top layers
    vgg16 = VGG16(
        weights='imagenet',
        include_top=False,
        input_tensor=None,
        input_shape=(input_shape[0], input_shape[1], 3)
    )
    
    # Apply VGG16 to the converted input
    vgg_output = vgg16(x)
    
    # Create the base model
    base_model = Model(inputs=input_tensor, outputs=vgg_output, name='vgg16_base')
    
    return base_model, vgg16

def create_vgg16_model(input_shape=(64, 64, 1), 
                      dense_units=[512, 512, 512],
                      dropout_rate=0.2,
                      learning_rate=0.001):
    """
    Create complete VGG16-based model for COVID-19 classification
    
    Args:
        input_shape (tuple): Shape of input images
        dense_units (list): Number of units in dense layers
        dropout_rate (float): Dropout rate for regularization
        learning_rate (float): Learning rate for optimizer
    
    Returns:
        Model: Compiled model ready for training
    """
    
    print("🏗️ Creating VGG16-based model...")
    
    # Create base model
    base_model, vgg16 = create_vgg16_base(input_shape)
    
    # Freeze VGG16 layers
    print("🔒 Freezing VGG16 layers...")
    base_model.trainable = False
    for layer in base_model.layers:
        layer.trainable = False
    
    # Count frozen parameters
    frozen_params = sum([layer.count_params() for layer in base_model.layers if not layer.trainable])
    print(f"   Frozen parameters: {frozen_params:,}")
    
    # Create the complete model
    model = Sequential(name='covid19_vgg16_classifier')
    
    # Add base model
    model.add(base_model)
    
    # Add flatten layer
    model.add(Flatten())
    
    # Add dense layers with dropout
    for i, units in enumerate(dense_units):
        model.add(Dense(units, activation='relu', name=f'dense_{i+1}'))
        model.add(Dropout(dropout_rate, name=f'dropout_{i+1}'))
    
    # Add output layer for binary classification
    model.add(Dense(1, activation='sigmoid', name='output'))
    
    # Compile model
    optimizer = Adam(learning_rate=learning_rate)
    model.compile(
        optimizer=optimizer,
        loss='binary_crossentropy',
        metrics=['accuracy', 'precision', 'recall']
    )
    
    print("✅ Model created and compiled successfully!")
    
    return model

def create_custom_cnn_model(input_shape=(64, 64, 1),
                           learning_rate=0.001):
    """
    Create a custom CNN model (alternative to VGG16)
    
    Args:
        input_shape (tuple): Shape of input images
        learning_rate (float): Learning rate for optimizer
    
    Returns:
        Model: Compiled custom CNN model
    """
    
    print("🏗️ Creating custom CNN model...")
    
    model = Sequential(name='covid19_custom_cnn')
    
    # First convolutional block
    model.add(Conv2D(32, (3, 3), activation='relu', input_shape=input_shape))
    model.add(BatchNormalization())
    model.add(MaxPooling2D((2, 2)))
    model.add(Dropout(0.25))
    
    # Second convolutional block
    model.add(Conv2D(64, (3, 3), activation='relu'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D((2, 2)))
    model.add(Dropout(0.25))
    
    # Third convolutional block
    model.add(Conv2D(128, (3, 3), activation='relu'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D((2, 2)))
    model.add(Dropout(0.25))
    
    # Fourth convolutional block
    model.add(Conv2D(256, (3, 3), activation='relu'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D((2, 2)))
    model.add(Dropout(0.25))
    
    # Flatten and dense layers
    model.add(Flatten())
    model.add(Dense(512, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(256, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(1, activation='sigmoid'))
    
    # Compile model
    optimizer = Adam(learning_rate=learning_rate)
    model.compile(
        optimizer=optimizer,
        loss='binary_crossentropy',
        metrics=['accuracy', 'precision', 'recall']
    )
    
    print("✅ Custom CNN model created successfully!")
    
    return model

def print_model_summary(model, save_plot=False, plot_path='results/model_architecture.png'):
    """
    Print detailed model summary and optionally save architecture plot
    
    Args:
        model: Keras model
        save_plot (bool): Whether to save model architecture plot
        plot_path (str): Path to save the plot
    """
    
    print("\n📊 MODEL ARCHITECTURE SUMMARY")
    print("=" * 60)
    
    # Print summary
    model.summary()
    
    # Count parameters
    total_params = model.count_params()
    trainable_params = sum([layer.count_params() for layer in model.layers if layer.trainable])
    non_trainable_params = total_params - trainable_params
    
    print(f"\n📈 PARAMETER SUMMARY:")
    print(f"   Total parameters: {total_params:,}")
    print(f"   Trainable parameters: {trainable_params:,}")
    print(f"   Non-trainable parameters: {non_trainable_params:,}")
    
    # Save model plot if requested
    if save_plot:
        try:
            os.makedirs(os.path.dirname(plot_path), exist_ok=True)
            plot_model(model, to_file=plot_path, show_shapes=True, show_layer_names=True)
            print(f"   Model architecture saved to: {plot_path}")
        except Exception as e:
            print(f"   Could not save model plot: {e}")

def load_saved_model(model_path):
    """
    Load a saved model
    
    Args:
        model_path (str): Path to the saved model
    
    Returns:
        Model: Loaded Keras model
    """
    
    try:
        print(f"📂 Loading model from {model_path}...")
        model = tf.keras.models.load_model(model_path)
        print("✅ Model loaded successfully!")
        return model
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        return None

def save_model(model, save_path='saved_models/covid19_model.h5'):
    """
    Save model to disk
    
    Args:
        model: Keras model to save
        save_path (str): Path where to save the model
    """
    
    try:
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        
        # Save model
        model.save(save_path)
        print(f"💾 Model saved successfully to: {save_path}")
        
        # Print file size
        file_size = os.path.getsize(save_path) / (1024 * 1024)  # MB
        print(f"   File size: {file_size:.2f} MB")
        
    except Exception as e:
        print(f"❌ Error saving model: {e}")

def compare_models():
    """
    Create and compare different model architectures
    """
    
    print("🔍 Comparing model architectures...")
    
    input_shape = (64, 64, 1)
    
    # VGG16 model
    vgg16_model = create_vgg16_model(input_shape)
    print(f"\nVGG16 Model Parameters: {vgg16_model.count_params():,}")
    
    # Custom CNN model
    custom_model = create_custom_cnn_model(input_shape)
    print(f"Custom CNN Parameters: {custom_model.count_params():,}")
    
    # Clean up
    del vgg16_model, custom_model

if __name__ == "__main__":
    # Test model creation
    print("🧪 Testing model creation...")
    
    try:
        # Test VGG16 model
        model = create_vgg16_model(input_shape=(64, 64, 1))
        print_model_summary(model)
        
        print("\n✅ Model creation test successful!")
        
    except Exception as e:
        print(f"❌ Model creation test failed: {e}")