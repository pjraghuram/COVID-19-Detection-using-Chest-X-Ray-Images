"""
COVID-19 X-Ray Classification - Data Preprocessing Module
Handles image loading, preprocessing, balancing, and train-test split
"""

import numpy as np
import pandas as pd
import os
from PIL import Image, ImageOps
from tqdm import tqdm
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler

def load_images_from_folder(folder_path, image_size=(64, 64)):
    """
    Load and preprocess images from a folder
    
    Args:
        folder_path (str): Path to the folder containing images
        image_size (tuple): Target size for resizing images
    
    Returns:
        list: List of preprocessed image arrays
    """
    images = []
    
    if not os.path.exists(folder_path):
        print(f"Warning: Path {folder_path} does not exist")
        return images
    
    print(f"Loading images from {folder_path}")
    
    for image_name in tqdm(os.listdir(folder_path), desc=f"Loading {os.path.basename(folder_path)}"):
        try:
            # Load image
            img_path = os.path.join(folder_path, image_name)
            img = Image.open(img_path)
            
            # Convert to grayscale
            img = ImageOps.grayscale(img)
            
            # Resize
            img = img.resize(image_size)
            
            # Convert to numpy array
            img_array = np.asarray(img)
            
            # Reshape to include channel dimension
            img_array = img_array.reshape(image_size + (1,))
            
            images.append(img_array)
            
        except Exception as e:
            print(f"Error loading image {image_name}: {e}")
            continue
    
    return images

def load_all_images(data_path, image_size=(64, 64)):
    """
    Load all images from the dataset
    
    Args:
        data_path (str): Base path to the COVID-19 dataset
        image_size (tuple): Target size for resizing images
    
    Returns:
        tuple: (all_images, labels)
    """
    
    # Define paths
    covid_path = os.path.join(data_path, "COVID")
    normal_path = os.path.join(data_path, "Normal")
    viral_pneumonia_path = os.path.join(data_path, "Viral Pneumonia")
    lung_opacity_path = os.path.join(data_path, "Lung_Opacity")
    
    # Check if paths exist
    paths = {
        "COVID": covid_path,
        "Normal": normal_path,
        "Viral Pneumonia": viral_pneumonia_path,
        "Lung Opacity": lung_opacity_path
    }
    
    print("Dataset Overview:")
    for name, path in paths.items():
        if os.path.exists(path):
            count = len(os.listdir(path))
            print(f"  {name}: {count} images")
        else:
            print(f"  {name}: Path not found!")
    
    # Load images
    all_images = []
    
    # Load COVID images (label = 1)
    covid_images = load_images_from_folder(covid_path, image_size)
    all_images.extend(covid_images)
    covid_count = len(covid_images)
    
    # Load Normal images (label = 0)
    normal_images = load_images_from_folder(normal_path, image_size)
    all_images.extend(normal_images)
    normal_count = len(normal_images)
    
    # Load Viral Pneumonia images (label = 0)
    viral_images = load_images_from_folder(viral_pneumonia_path, image_size)
    all_images.extend(viral_images)
    viral_count = len(viral_images)
    
    # Load Lung Opacity images (label = 0)
    opacity_images = load_images_from_folder(lung_opacity_path, image_size)
    all_images.extend(opacity_images)
    opacity_count = len(opacity_images)
    
    # Create labels (COVID = 1, Others = 0)
    labels = ([1] * covid_count + 
              [0] * normal_count + 
              [0] * viral_count + 
              [0] * opacity_count)
    
    print(f"\nTotal images loaded: {len(all_images)}")
    print(f"Total labels created: {len(labels)}")
    print(f"Label distribution: {np.unique(labels, return_counts=True)}")
    
    return all_images, labels

def normalize_and_convert(images, labels):
    """
    Normalize images and convert to numpy arrays
    
    Args:
        images (list): List of image arrays
        labels (list): List of labels
    
    Returns:
        tuple: (normalized_images_array, labels_array)
    """
    
    # Convert to numpy arrays
    images_array = np.array(images)
    labels_array = np.array(labels)
    
    # Normalize pixel values to [0, 1]
    images_array = images_array / 255.0
    
    print(f"Images shape: {images_array.shape}")
    print(f"Labels shape: {labels_array.shape}")
    print(f"Pixel value range: [{images_array.min():.3f}, {images_array.max():.3f}]")
    
    return images_array, labels_array

def balance_dataset(X, y, oversample_ratio=0.7, undersample_ratio=1.0):
    """
    Balance the dataset using SMOTE and Random Under Sampling
    
    Args:
        X (np.array): Feature array
        y (np.array): Label array
        oversample_ratio (float): SMOTE sampling strategy
        undersample_ratio (float): Under sampling strategy
    
    Returns:
        tuple: (balanced_X, balanced_y)
    """
    
    print("Original distribution:", np.unique(y, return_counts=True))
    
    # Reshape for SMOTE (flatten images)
    X_reshaped = X.reshape(X.shape[0], -1)
    
    # Apply SMOTE
    print("Applying SMOTE...")
    oversample = SMOTE(sampling_strategy=oversample_ratio, random_state=42)
    X_over, y_over = oversample.fit_resample(X_reshaped, y)
    
    print("After SMOTE:", np.unique(y_over, return_counts=True))
    
    # Apply Random Under Sampling
    print("Applying Random Under Sampling...")
    undersample = RandomUnderSampler(sampling_strategy=undersample_ratio, random_state=42)
    X_balanced, y_balanced = undersample.fit_resample(X_over, y_over)
    
    print("After balancing:", np.unique(y_balanced, return_counts=True))
    
    # Reshape back to original image dimensions
    X_balanced = X_balanced.reshape(X_balanced.shape[0], X.shape[1], X.shape[2], X.shape[3])
    
    return X_balanced, y_balanced

def load_and_preprocess_data(data_path, image_size=(64, 64), test_size=0.2, balance_data=True):
    """
    Complete data loading and preprocessing pipeline
    
    Args:
        data_path (str): Path to the dataset
        image_size (tuple): Target image size
        test_size (float): Proportion of test data
        balance_data (bool): Whether to balance the dataset
    
    Returns:
        tuple: (X_train, X_test, y_train, y_test)
    """
    
    print("🔄 Starting data preprocessing pipeline...")
    
    # Step 1: Load all images
    print("\n1️⃣ Loading images...")
    images, labels = load_all_images(data_path, image_size)
    
    if len(images) == 0:
        raise ValueError("No images loaded! Check your data path.")
    
    # Step 2: Normalize and convert
    print("\n2️⃣ Normalizing images...")
    X, y = normalize_and_convert(images, labels)
    
    # Step 3: Shuffle data
    print("\n3️⃣ Shuffling data...")
    X, y = shuffle(X, y, random_state=42)
    
    # Step 4: Split into train and test
    print(f"\n4️⃣ Splitting data (test_size={test_size})...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, stratify=y, random_state=42
    )
    
    print(f"Training set: {X_train.shape[0]} samples")
    print(f"Test set: {X_test.shape[0]} samples")
    print(f"Train distribution: {np.unique(y_train, return_counts=True)}")
    print(f"Test distribution: {np.unique(y_test, return_counts=True)}")
    
    # Step 5: Balance training data (optional)
    if balance_data:
        print("\n5️⃣ Balancing training data...")
        X_train, y_train = balance_dataset(X_train, y_train)
        print(f"Balanced training set: {X_train.shape[0]} samples")
    
    print("\n✅ Data preprocessing completed successfully!")
    
    return X_train, X_test, y_train, y_test

def load_single_image(image_path, image_size=(64, 64)):
    """
    Load and preprocess a single image for prediction
    
    Args:
        image_path (str): Path to the image
        image_size (tuple): Target image size
    
    Returns:
        np.array: Preprocessed image array ready for prediction
    """
    
    try:
        # Load image
        img = Image.open(image_path)
        
        # Convert to grayscale
        img = ImageOps.grayscale(img)
        
        # Resize
        img = img.resize(image_size)
        
        # Convert to numpy array
        img_array = np.asarray(img)
        
        # Reshape and normalize
        img_array = img_array.reshape(1, image_size[0], image_size[1], 1)
        img_array = img_array / 255.0
        
        return img_array
        
    except Exception as e:
        print(f"Error loading image {image_path}: {e}")
        return None

if __name__ == "__main__":
    # Test the preprocessing module
    data_path = "data/COVID-19_Radiography_Dataset"
    
    try:
        X_train, X_test, y_train, y_test = load_and_preprocess_data(data_path)
        print("✅ Preprocessing test successful!")
    except Exception as e:
        print(f"❌ Preprocessing test failed: {e}")