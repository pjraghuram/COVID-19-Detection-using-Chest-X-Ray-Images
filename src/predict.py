"""
COVID-19 X-Ray Classification - Prediction and Evaluation Module
Handles model evaluation, predictions, and performance metrics
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (accuracy_score, precision_score, recall_score, 
                           f1_score, confusion_matrix, classification_report,
                           roc_auc_score, roc_curve, precision_recall_curve)
import pandas as pd
import os
from datetime import datetime

def make_predictions(model, X_test, threshold=0.5):
    """
    Make predictions using the trained model
    
    Args:
        model: Trained Keras model
        X_test: Test data
        threshold (float): Classification threshold
    
    Returns:
        tuple: (predictions, probabilities)
    """
    
    print(f"🔮 Making predictions on {len(X_test)} samples...")
    
    # Get prediction probabilities
    probabilities = model.predict(X_test, verbose=0)
    
    # Convert probabilities to binary predictions
    predictions = (probabilities > threshold).astype(int).flatten()
    probabilities = probabilities.flatten()
    
    print(f"   Predictions shape: {predictions.shape}")
    print(f"   Probability range: [{probabilities.min():.3f}, {probabilities.max():.3f}]")
    
    return predictions, probabilities

def calculate_metrics(y_true, y_pred, y_prob):
    """
    Calculate comprehensive evaluation metrics
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
        y_prob: Prediction probabilities
    
    Returns:
        dict: Dictionary of metrics
    """
    
    metrics = {}
    
    # Basic metrics
    metrics['accuracy'] = accuracy_score(y_true, y_pred)
    metrics['precision'] = precision_score(y_true, y_pred, zero_division=0)
    metrics['recall'] = recall_score(y_true, y_pred, zero_division=0)
    metrics['f1_score'] = f1_score(y_true, y_pred, zero_division=0)
    
    # AUC-ROC
    try:
        metrics['auc_roc'] = roc_auc_score(y_true, y_prob)
    except ValueError:
        metrics['auc_roc'] = 0.0
    
    # Specificity (True Negative Rate)
    cm = confusion_matrix(y_true, y_pred)
    if cm.shape == (2, 2):
        tn, fp, fn, tp = cm.ravel()
        metrics['specificity'] = tn / (tn + fp) if (tn + fp) > 0 else 0
        metrics['sensitivity'] = tp / (tp + fn) if (tp + fn) > 0 else 0  # Same as recall
        
        # Additional metrics
        metrics['positive_predictive_value'] = tp / (tp + fp) if (tp + fp) > 0 else 0  # Same as precision
        metrics['negative_predictive_value'] = tn / (tn + fn) if (tn + fn) > 0 else 0
    
    return metrics

def evaluate_model(model, X_test, y_test, threshold=0.5, verbose=True):
    """
    Comprehensive model evaluation
    
    Args:
        model: Trained model
        X_test: Test features
        y_test: Test labels
        threshold (float): Classification threshold
        verbose (bool): Print detailed results
    
    Returns:
        dict: Evaluation metrics
    """
    
    if verbose:
        print("📊 Evaluating model performance...")
        print(f"   Test samples: {len(X_test)}")
        print(f"   Test distribution: {np.unique(y_test, return_counts=True)}")
    
    # Make predictions
    y_pred, y_prob = make_predictions(model, X_test, threshold)
    
    # Calculate metrics
    metrics = calculate_metrics(y_test, y_pred, y_prob)
    
    if verbose:
        print("\n📈 EVALUATION RESULTS:")
        print("-" * 40)
        for metric, value in metrics.items():
            print(f"   {metric.replace('_', ' ').title()}: {value:.4f}")
    
    return metrics

def plot_confusion_matrix(y_true, y_pred, class_names=['Non-COVID', 'COVID'], 
                         save_path='results/confusion_matrix.png'):
    """
    Plot and save confusion matrix
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
        class_names (list): Names of classes
        save_path (str): Path to save the plot
    """
    
    try:
        # Calculate confusion matrix
        cm = confusion_matrix(y_true, y_pred)
        
        # Create plot
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                   xticklabels=class_names, yticklabels=class_names)
        plt.title('Confusion Matrix')
        plt.xlabel('Predicted Label')
        plt.ylabel('True Label')
        
        # Add percentage annotations
        cm_percent = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis] * 100
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                plt.text(j + 0.5, i + 0.7, f'{cm_percent[i, j]:.1f}%', 
                        ha='center', va='center', fontsize=10, color='red')
        
        # Save plot
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"📊 Confusion matrix saved to: {save_path}")
        
    except Exception as e:
        print(f"❌ Error plotting confusion matrix: {e}")

def plot_roc_curve(y_true, y_prob, save_path='results/roc_curve.png'):
    """
    Plot ROC curve
    
    Args:
        y_true: True labels
        y_prob: Prediction probabilities
        save_path (str): Path to save the plot
    """
    
    try:
        # Calculate ROC curve
        fpr, tpr, thresholds = roc_curve(y_true, y_prob)
        auc_score = roc_auc_score(y_true, y_prob)
        
        # Create plot
        plt.figure(figsize=(8, 6))
        plt.plot(fpr, tpr, color='darkorange', lw=2, 
                label=f'ROC curve (AUC = {auc_score:.3f})')
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', 
                label='Random classifier')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver Operating Characteristic (ROC) Curve')
        plt.legend(loc="lower right")
        plt.grid(True)
        
        # Save plot
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"📈 ROC curve saved to: {save_path}")
        
    except Exception as e:
        print(f"❌ Error plotting ROC curve: {e}")

def plot_precision_recall_curve(y_true, y_prob, save_path='results/precision_recall_curve.png'):
    """
    Plot Precision-Recall curve
    
    Args:
        y_true: True labels
        y_prob: Prediction probabilities
        save_path (str): Path to save the plot
    """
    
    try:
        # Calculate PR curve
        precision, recall, thresholds = precision_recall_curve(y_true, y_prob)
        
        # Create plot
        plt.figure(figsize=(8, 6))
        plt.plot(recall, precision, color='blue', lw=2)
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.title('Precision-Recall Curve')
        plt.grid(True)
        
        # Add baseline
        baseline = np.sum(y_true) / len(y_true)
        plt.axhline(y=baseline, color='red', linestyle='--', 
                   label=f'Baseline (Positive class rate: {baseline:.3f})')
        plt.legend()
        
        # Save plot
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"📊 Precision-Recall curve saved to: {save_path}")
        
    except Exception as e:
        print(f"❌ Error plotting PR curve: {e}")

def generate_classification_report(y_true, y_pred, class_names=['Non-COVID', 'COVID'],
                                 save_path='results/classification_report.txt'):
    """
    Generate and save detailed classification report
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
        class_names (list): Names of classes
        save_path (str): Path to save the report
    """
    
    try:
        # Generate report
        report = classification_report(y_true, y_pred, target_names=class_names, digits=4)
        
        # Save to file
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        with open(save_path, 'w') as f:
            f.write("COVID-19 X-Ray Classification Report\n")
            f.write("=" * 50 + "\n")
            f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            f.write(report)
        
        print(f"📋 Classification report saved to: {save_path}")
        print("\nClassification Report:")
        print(report)
        
    except Exception as e:
        print(f"❌ Error generating classification report: {e}")

def predict_single_image(model, image_array, threshold=0.5, class_names=['Non-COVID', 'COVID']):
    """
    Predict on a single image
    
    Args:
        model: Trained model
        image_array: Preprocessed image array
        threshold (float): Classification threshold
        class_names (list): Class names
    
    Returns:
        dict: Prediction results
    """
    
    # Make prediction
    probability = model.predict(image_array, verbose=0)[0][0]
    prediction = int(probability > threshold)
    confidence = probability if prediction == 1 else 1 - probability
    
    result = {
        'prediction': prediction,
        'class_name': class_names[prediction],
        'probability': probability,
        'confidence': confidence
    }
    
    return result

def batch_predict(model, image_paths, preprocess_func, threshold=0.5):
    """
    Make predictions on a batch of images
    
    Args:
        model: Trained model
        image_paths (list): List of image file paths
        preprocess_func: Function to preprocess images
        threshold (float): Classification threshold
    
    Returns:
        list: List of prediction results
    """
    
    results = []
    
    print(f"🔮 Making predictions on {len(image_paths)} images...")
    
    for i, image_path in enumerate(image_paths):
        try:
            # Preprocess image
            image_array = preprocess_func(image_path)
            
            if image_array is not None:
                # Make prediction
                result = predict_single_image(model, image_array, threshold)
                result['image_path'] = image_path
                result['image_name'] = os.path.basename(image_path)
                results.append(result)
            else:
                print(f"   ⚠️  Failed to process image: {image_path}")
                
        except Exception as e:
            print(f"   ❌ Error processing {image_path}: {e}")
    
    print(f"✅ Completed predictions on {len(results)} images")
    return results

def save_predictions_to_csv(predictions, save_path='results/predictions.csv'):
    """
    Save prediction results to CSV file
    
    Args:
        predictions (list): List of prediction dictionaries
        save_path (str): Path to save CSV file
    """
    
    try:
        df = pd.DataFrame(predictions)
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        df.to_csv(save_path, index=False)
        print(f"💾 Predictions saved to: {save_path}")
    except Exception as e:
        print(f"❌ Error saving predictions: {e}")

def comprehensive_evaluation(model, X_test, y_test, threshold=0.5):
    """
    Perform comprehensive evaluation with all plots and reports
    
    Args:
        model: Trained model
        X_test: Test features
        y_test: Test labels
        threshold (float): Classification threshold
    
    Returns:
        dict: Complete evaluation results
    """
    
    print("🎯 Starting comprehensive evaluation...")
    
    # Make predictions
    y_pred, y_prob = make_predictions(model, X_test, threshold)
    
    # Calculate metrics
    metrics = calculate_metrics(y_test, y_pred, y_prob)
    
    # Generate plots
    plot_confusion_matrix(y_test, y_pred)
    plot_roc_curve(y_test, y_prob)
    plot_precision_recall_curve(y_test, y_prob)
    
    # Generate reports
    generate_classification_report(y_test, y_pred)
    
    print("✅ Comprehensive evaluation completed!")
    
    return {
        'metrics': metrics,
        'predictions': y_pred,
        'probabilities': y_prob,
        'true_labels': y_test
    }

if __name__ == "__main__":
    # Test prediction module
    print("🧪 Testing prediction module...")
    print("✅ Prediction module loaded successfully!")