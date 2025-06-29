"""
COVID-19 X-Ray Classification - Training Module
Handles model training with callbacks and monitoring
"""

import tensorflow as tf
from tensorflow.keras.callbacks import (EarlyStopping, ReduceLROnPlateau, 
                                      ModelCheckpoint, CSVLogger)
import numpy as np
import matplotlib.pyplot as plt
import os
import time
from datetime import datetime

def create_callbacks(model_save_path='saved_models/best_model.h5',
                    patience=5,
                    monitor='val_accuracy',
                    save_logs=True):
    """
    Create training callbacks for monitoring and optimization
    
    Args:
        model_save_path (str): Path to save the best model
        patience (int): Patience for early stopping
        monitor (str): Metric to monitor
        save_logs (bool): Whether to save training logs
    
    Returns:
        list: List of callback objects
    """
    
    callbacks = []
    
    # Early Stopping
    early_stopping = EarlyStopping(
        monitor=monitor,
        patience=patience,
        restore_best_weights=True,
        verbose=1,
        mode='max' if 'accuracy' in monitor else 'min'
    )
    callbacks.append(early_stopping)
    
    # Reduce Learning Rate on Plateau
    reduce_lr = ReduceLROnPlateau(
        monitor=monitor,
        factor=0.5,
        patience=patience//2,
        min_lr=1e-7,
        verbose=1,
        mode='max' if 'accuracy' in monitor else 'min'
    )
    callbacks.append(reduce_lr)
    
    # Model Checkpoint
    os.makedirs(os.path.dirname(model_save_path), exist_ok=True)
    checkpoint = ModelCheckpoint(
        model_save_path,
        monitor=monitor,
        save_best_only=True,
        save_weights_only=False,
        verbose=1,
        mode='max' if 'accuracy' in monitor else 'min'
    )
    callbacks.append(checkpoint)
    
    # CSV Logger
    if save_logs:
        log_path = 'results/training_log.csv'
        os.makedirs(os.path.dirname(log_path), exist_ok=True)
        csv_logger = CSVLogger(log_path, append=True)
        callbacks.append(csv_logger)
    
    return callbacks

def train_model(model, X_train, y_train, X_val=None, y_val=None,
               epochs=20, batch_size=32, validation_split=0.2,
               use_callbacks=True, verbose=1):
    """
    Train the model with optional validation and callbacks
    
    Args:
        model: Compiled Keras model
        X_train: Training images
        y_train: Training labels
        X_val: Validation images (optional)
        y_val: Validation labels (optional)
        epochs (int): Number of training epochs
        batch_size (int): Batch size for training
        validation_split (float): Validation split if no validation data provided
        use_callbacks (bool): Whether to use training callbacks
        verbose (int): Verbosity level
    
    Returns:
        History: Training history object
    """
    
    print(f"🎯 Starting model training...")
    print(f"   Training samples: {len(X_train)}")
    print(f"   Epochs: {epochs}")
    print(f"   Batch size: {batch_size}")
    
    # Prepare validation data
    validation_data = None
    if X_val is not None and y_val is not None:
        validation_data = (X_val, y_val)
        validation_split = 0
        print(f"   Validation samples: {len(X_val)}")
    elif validation_split > 0:
        print(f"   Validation split: {validation_split}")
    
    # Setup callbacks
    callbacks = []
    if use_callbacks:
        callbacks = create_callbacks()
        print(f"   Using {len(callbacks)} callbacks")
    
    # Record training start time
    start_time = time.time()
    
    try:
        # Train the model
        history = model.fit(
            X_train, y_train,
            epochs=epochs,
            batch_size=batch_size,
            validation_data=validation_data,
            validation_split=validation_split,
            callbacks=callbacks,
            verbose=verbose,
            shuffle=True
        )
        
        # Calculate training time
        training_time = time.time() - start_time
        print(f"\n⏱️ Training completed in {training_time:.2f} seconds ({training_time/60:.2f} minutes)")
        
        # Print final metrics
        if hasattr(history.history, 'loss'):
            final_loss = history.history['loss'][-1]
            final_accuracy = history.history['accuracy'][-1]
            print(f"   Final training loss: {final_loss:.4f}")
            print(f"   Final training accuracy: {final_accuracy:.4f}")
            
            if 'val_loss' in history.history:
                final_val_loss = history.history['val_loss'][-1]
                final_val_accuracy = history.history['val_accuracy'][-1]
                print(f"   Final validation loss: {final_val_loss:.4f}")
                print(f"   Final validation accuracy: {final_val_accuracy:.4f}")
        
        return history
        
    except Exception as e:
        print(f"❌ Training failed: {e}")
        return None

def plot_training_history(history, save_path='results/training_history.png'):
    """
    Plot training history metrics
    
    Args:
        history: Keras training history object
        save_path (str): Path to save the plot
    """
    
    if history is None:
        print("❌ No history to plot")
        return
    
    try:
        # Create figure with subplots
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle('Model Training History', fontsize=16)
        
        # Plot loss
        axes[0, 0].plot(history.history['loss'], label='Training Loss', color='blue')
        if 'val_loss' in history.history:
            axes[0, 0].plot(history.history['val_loss'], label='Validation Loss', color='red')
        axes[0, 0].set_title('Model Loss')
        axes[0, 0].set_xlabel('Epoch')
        axes[0, 0].set_ylabel('Loss')
        axes[0, 0].legend()
        axes[0, 0].grid(True)
        
        # Plot accuracy
        axes[0, 1].plot(history.history['accuracy'], label='Training Accuracy', color='blue')
        if 'val_accuracy' in history.history:
            axes[0, 1].plot(history.history['val_accuracy'], label='Validation Accuracy', color='red')
        axes[0, 1].set_title('Model Accuracy')
        axes[0, 1].set_xlabel('Epoch')
        axes[0, 1].set_ylabel('Accuracy')
        axes[0, 1].legend()
        axes[0, 1].grid(True)
        
        # Plot precision if available
        if 'precision' in history.history:
            axes[1, 0].plot(history.history['precision'], label='Training Precision', color='blue')
            if 'val_precision' in history.history:
                axes[1, 0].plot(history.history['val_precision'], label='Validation Precision', color='red')
            axes[1, 0].set_title('Model Precision')
            axes[1, 0].set_xlabel('Epoch')
            axes[1, 0].set_ylabel('Precision')
            axes[1, 0].legend()
            axes[1, 0].grid(True)
        
        # Plot recall if available
        if 'recall' in history.history:
            axes[1, 1].plot(history.history['recall'], label='Training Recall', color='blue')
            if 'val_recall' in history.history:
                axes[1, 1].plot(history.history['val_recall'], label='Validation Recall', color='red')
            axes[1, 1].set_title('Model Recall')
            axes[1, 1].set_xlabel('Epoch')
            axes[1, 1].set_ylabel('Recall')
            axes[1, 1].legend()
            axes[1, 1].grid(True)
        
        # Adjust layout and save
        plt.tight_layout()
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"📊 Training history plot saved to: {save_path}")
        
    except Exception as e:
        print(f"❌ Error plotting training history: {e}")

def save_training_summary(history, model, save_path='results/training_summary.txt'):
    """
    Save a text summary of the training process
    
    Args:
        history: Keras training history
        model: Trained model
        save_path (str): Path to save the summary
    """
    
    try:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        
        with open(save_path, 'w') as f:
            f.write("COVID-19 X-Ray Classification - Training Summary\n")
            f.write("=" * 60 + "\n")
            f.write(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            
            # Model architecture summary
            f.write("MODEL ARCHITECTURE:\n")
            f.write("-" * 30 + "\n")
            f.write(f"Model name: {model.name}\n")
            f.write(f"Total parameters: {model.count_params():,}\n")
            
            trainable_params = sum([layer.count_params() for layer in model.layers if layer.trainable])
            f.write(f"Trainable parameters: {trainable_params:,}\n")
            f.write(f"Non-trainable parameters: {model.count_params() - trainable_params:,}\n\n")
            
            # Training summary
            if history:
                f.write("TRAINING SUMMARY:\n")
                f.write("-" * 30 + "\n")
                f.write(f"Epochs completed: {len(history.history['loss'])}\n")
                
                # Final metrics
                final_loss = history.history['loss'][-1]
                final_accuracy = history.history['accuracy'][-1]
                f.write(f"Final training loss: {final_loss:.4f}\n")
                f.write(f"Final training accuracy: {final_accuracy:.4f}\n")
                
                if 'val_loss' in history.history:
                    final_val_loss = history.history['val_loss'][-1]
                    final_val_accuracy = history.history['val_accuracy'][-1]
                    f.write(f"Final validation loss: {final_val_loss:.4f}\n")
                    f.write(f"Final validation accuracy: {final_val_accuracy:.4f}\n")
                
                # Best metrics
                best_train_acc = max(history.history['accuracy'])
                f.write(f"Best training accuracy: {best_train_acc:.4f}\n")
                
                if 'val_accuracy' in history.history:
                    best_val_acc = max(history.history['val_accuracy'])
                    f.write(f"Best validation accuracy: {best_val_acc:.4f}\n")
        
        print(f"📝 Training summary saved to: {save_path}")
        
    except Exception as e:
        print(f"❌ Error saving training summary: {e}")

def fine_tune_model(model, X_train, y_train, X_val=None, y_val=None,
                   unfreeze_layers=10, fine_tune_epochs=5, 
                   fine_tune_lr=1e-5):
    """
    Fine-tune the pre-trained model by unfreezing some layers
    
    Args:
        model: Pre-trained model
        X_train: Training data
        y_train: Training labels
        X_val: Validation data
        y_val: Validation labels
        unfreeze_layers (int): Number of layers to unfreeze from the top
        fine_tune_epochs (int): Number of epochs for fine-tuning
        fine_tune_lr (float): Learning rate for fine-tuning
    
    Returns:
        History: Fine-tuning history
    """
    
    print(f"🔧 Starting fine-tuning...")
    print(f"   Unfreezing top {unfreeze_layers} layers")
    print(f"   Fine-tune learning rate: {fine_tune_lr}")
    
    # Unfreeze the top layers
    for layer in model.layers[-unfreeze_layers:]:
        if hasattr(layer, 'layers'):  # For nested models
            for sublayer in layer.layers[-unfreeze_layers:]:
                sublayer.trainable = True
        else:
            layer.trainable = True
    
    # Recompile with lower learning rate
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=fine_tune_lr),
        loss='binary_crossentropy',
        metrics=['accuracy', 'precision', 'recall']
    )
    
    print(f"   Trainable parameters after unfreezing: {sum([layer.count_params() for layer in model.layers if layer.trainable]):,}")
    
    # Fine-tune training
    validation_data = (X_val, y_val) if X_val is not None else None
    
    history = model.fit(
        X_train, y_train,
        epochs=fine_tune_epochs,
        validation_data=validation_data,
        callbacks=create_callbacks(model_save_path='saved_models/fine_tuned_model.h5'),
        verbose=1
    )
    
    print("✅ Fine-tuning completed!")
    return history

if __name__ == "__main__":
    # Test training module (placeholder)
    print("🧪 Training module test...")
    print("✅ Training module loaded successfully!")