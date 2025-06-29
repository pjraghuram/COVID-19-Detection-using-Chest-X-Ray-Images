#!/usr/bin/env python3
"""
COVID-19 X-Ray Classification - Main Training Script
This is the main entry point that orchestrates the entire training pipeline
"""

import os
import sys
import numpy as np
from pathlib import Path

# Add src to Python path for imports
project_root = Path(__file__).parent
sys.path.append(str(project_root / 'src'))

# Import your custom modules
from preprocessing import load_and_preprocess_data
from model import create_vgg16_model
from train import train_model
from predict import evaluate_model

def main():
    """Main training pipeline"""
    print("🚀 Starting COVID-19 X-Ray Classification Training")
    print("=" * 60)
    
    # Configuration
    CONFIG = {
        'data_path': 'data/COVID-19_Radiography_Dataset',
        'image_size': (64, 64),
        'batch_size': 32,
        'epochs': 20,
        'learning_rate': 0.001,
        'test_size': 0.2,
        'model_save_path': 'saved_models/covid_model.h5'
    }
    
    try:
        # Step 1: Load and preprocess data
        print("\n📁 Step 1: Loading and preprocessing data...")
        X_train, X_test, y_train, y_test = load_and_preprocess_data(
            data_path=CONFIG['data_path'],
            image_size=CONFIG['image_size'],
            test_size=CONFIG['test_size']
        )
        
        print(f"✅ Data loaded successfully!")
        print(f"   Training samples: {len(X_train)}")
        print(f"   Test samples: {len(X_test)}")
        print(f"   Image shape: {X_train[0].shape}")
        
        # Step 2: Create model
        print("\n🏗️  Step 2: Creating VGG16 model...")
        model = create_vgg16_model(input_shape=CONFIG['image_size'] + (1,))
        print("✅ Model created successfully!")
        
        # Step 3: Train model
        print("\n🎯 Step 3: Training model...")
        history = train_model(
            model=model,
            X_train=X_train,
            y_train=y_train,
            epochs=CONFIG['epochs'],
            batch_size=CONFIG['batch_size']
        )
        print("✅ Training completed!")
        
        # Step 4: Evaluate model
        print("\n📊 Step 4: Evaluating model...")
        metrics = evaluate_model(model, X_test, y_test)
        
        print("\n🎉 FINAL RESULTS:")
        print("-" * 30)
        for metric, value in metrics.items():
            print(f"{metric}: {value:.4f}")
        
        # Step 5: Save model
        print(f"\n💾 Step 5: Saving model to {CONFIG['model_save_path']}...")
        os.makedirs('saved_models', exist_ok=True)
        model.save(CONFIG['model_save_path'])
        print("✅ Model saved successfully!")
        
        print("\n🎊 Training pipeline completed successfully!")
        
    except Exception as e:
        print(f"\n❌ Error during training: {str(e)}")
        print("Please check the error details above.")
        return False
    
    return True

if __name__ == "__main__":
    success = main()
    if success:
        print("\n✨ You can now run the Streamlit app: streamlit run frontend/app.py")
    else:
        print("\n⚠️  Please fix the errors above before proceeding.")