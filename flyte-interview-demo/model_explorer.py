#!/usr/bin/env python3
"""
Model Explorer Tool for inspecting model.pkl

This script provides various ways to explore and visualize the saved ML model.
Perfect for Customer Success Engineers to help customers understand their models.
"""

import joblib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.tree import export_text, plot_tree
# from sklearn.inspection import plot_partial_dependence  # Not available in this sklearn version
import pickle
import os

def load_model(model_path='model.pkl'):
    """Load the saved model"""
    return joblib.load(model_path)

def basic_model_info(model):
    """Display basic model information"""
    print("=== BASIC MODEL INFORMATION ===")
    print(f"Model Type: {type(model).__name__}")
    print(f"Number of features: {model.n_features_in_}")
    print(f"Feature names: {list(model.feature_names_in_)}")
    print(f"Classes: {model.classes_}")
    print(f"Number of estimators: {model.n_estimators}")
    print(f"Max depth: {model.max_depth}")
    print(f"Random state: {model.random_state}")
    print()

def model_parameters(model):
    """Display all model parameters"""
    print("=== MODEL PARAMETERS ===")
    params = model.get_params()
    for param, value in params.items():
        print(f"  {param}: {value}")
    print()

def feature_importance_analysis(model):
    """Analyze and visualize feature importance"""
    print("=== FEATURE IMPORTANCE ===")
    importances = model.feature_importances_
    feature_names = model.feature_names_in_
    
    # Print importance scores
    for name, importance in zip(feature_names, importances):
        print(f"  {name}: {importance:.4f} ({importance*100:.1f}%)")
    
    # Create importance plot
    plt.figure(figsize=(10, 6))
    plt.subplot(1, 2, 1)
    plt.bar(feature_names, importances)
    plt.title('Feature Importance')
    plt.xlabel('Features')
    plt.ylabel('Importance')
    plt.xticks(rotation=45)
    
    # Create pie chart
    plt.subplot(1, 2, 2)
    plt.pie(importances, labels=feature_names, autopct='%1.1f%%')
    plt.title('Feature Importance Distribution')
    
    plt.tight_layout()
    plt.savefig('feature_importance.png', dpi=150, bbox_inches='tight')
    plt.show()
    print("Feature importance plot saved as 'feature_importance.png'")
    print()

def model_performance_analysis(model, data_path='data/sample_data.csv'):
    """Analyze model performance on the dataset"""
    print("=== MODEL PERFORMANCE ANALYSIS ===")
    
    # Load data
    data = pd.read_csv(data_path)
    X = data[['feature1', 'feature2', 'feature3']]
    y = data['target']
    
    # Make predictions
    predictions = model.predict(X)
    probabilities = model.predict_proba(X)
    
    # Calculate accuracy
    accuracy = (predictions == y).mean()
    print(f"Accuracy on dataset: {accuracy:.3f}")
    
    # Confusion matrix
    from sklearn.metrics import confusion_matrix, classification_report
    cm = confusion_matrix(y, predictions)
    print(f"\nConfusion Matrix:")
    print(cm)
    
    print(f"\nClassification Report:")
    print(classification_report(y, predictions))
    
    # Visualize predictions
    plt.figure(figsize=(15, 5))
    
    # Plot 1: Actual vs Predicted
    plt.subplot(1, 3, 1)
    plt.scatter(range(len(y)), y, alpha=0.7, label='Actual', color='blue')
    plt.scatter(range(len(predictions)), predictions, alpha=0.7, label='Predicted', color='red')
    plt.xlabel('Sample Index')
    plt.ylabel('Class')
    plt.title('Actual vs Predicted')
    plt.legend()
    
    # Plot 2: Prediction Confidence
    plt.subplot(1, 3, 2)
    confidence = np.max(probabilities, axis=1)
    plt.bar(range(len(confidence)), confidence)
    plt.xlabel('Sample Index')
    plt.ylabel('Prediction Confidence')
    plt.title('Model Confidence')
    
    # Plot 3: Feature distributions by class
    plt.subplot(1, 3, 3)
    for i, feature in enumerate(['feature1', 'feature2', 'feature3']):
        for class_val in [0, 1]:
            class_data = data[data['target'] == class_val][feature]
            plt.hist(class_data, alpha=0.6, label=f'Class {class_val} - {feature}', bins=5)
    plt.xlabel('Feature Value')
    plt.ylabel('Frequency')
    plt.title('Feature Distributions by Class')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig('model_performance.png', dpi=150, bbox_inches='tight')
    plt.show()
    print("Performance analysis plot saved as 'model_performance.png'")
    print()

def explore_individual_trees(model, n_trees=3):
    """Explore individual decision trees in the Random Forest"""
    print(f"=== EXPLORING {n_trees} INDIVIDUAL TREES ===")
    
    for i in range(min(n_trees, len(model.estimators_))):
        print(f"\n--- Tree {i+1} ---")
        tree = model.estimators_[i]
        
        # Print tree rules (first few lines only)
        tree_rules = export_text(tree, feature_names=model.feature_names_in_)
        lines = tree_rules.split('\n')[:20]  # First 20 lines
        print('\n'.join(lines))
        if len(tree_rules.split('\n')) > 20:
            print("... (truncated)")
        
        # Visualize tree structure
        plt.figure(figsize=(15, 10))
        plot_tree(tree, 
                 feature_names=model.feature_names_in_,
                 class_names=['Class 0', 'Class 1'],
                 filled=True,
                 max_depth=3)  # Limit depth for readability
        plt.title(f'Decision Tree {i+1} (max_depth=3 for visualization)')
        plt.savefig(f'tree_{i+1}.png', dpi=150, bbox_inches='tight')
        plt.show()
        print(f"Tree {i+1} visualization saved as 'tree_{i+1}.png'")

def model_file_analysis(model_path='model.pkl'):
    """Analyze the model file itself"""
    print("=== MODEL FILE ANALYSIS ===")
    
    # File size
    file_size = os.path.getsize(model_path)
    print(f"File size: {file_size:,} bytes ({file_size/1024:.1f} KB)")
    
    # Pickle protocol analysis
    with open(model_path, 'rb') as f:
        first_bytes = f.read(10)
        print(f"First 10 bytes (hex): {first_bytes.hex()}")
    
    # Try to inspect pickle contents (advanced)
    try:
        import pickletools
        print("\nPickle structure (first 1000 characters):")
        with open(model_path, 'rb') as f:
            pickletools.dis(f)
    except Exception as e:
        print(f"Could not analyze pickle structure: {e}")
    
    print()

def interactive_prediction_tool(model):
    """Interactive tool for making predictions"""
    print("=== INTERACTIVE PREDICTION TOOL ===")
    print("Enter feature values to get predictions (or 'quit' to exit)")
    
    while True:
        try:
            user_input = input("\nEnter feature1,feature2,feature3 (e.g., 1.5,1.0,2.5): ")
            if user_input.lower() == 'quit':
                break
                
            values = [float(x.strip()) for x in user_input.split(',')]
            if len(values) != 3:
                print("Please enter exactly 3 values")
                continue
                
            # Make prediction
            features = np.array(values).reshape(1, -1)
            prediction = model.predict(features)[0]
            probability = model.predict_proba(features)[0]
            
            print(f"Prediction: Class {prediction}")
            print(f"Probabilities: Class 0: {probability[0]:.3f}, Class 1: {probability[1]:.3f}")
            
        except ValueError:
            print("Invalid input. Please enter numbers separated by commas.")
        except KeyboardInterrupt:
            break
    print()

def main():
    """Main function to run all analysis tools"""
    print("🔍 MODEL EXPLORER TOOL")
    print("=" * 50)
    
    # Load model
    try:
        model = load_model()
        print("✅ Model loaded successfully!\n")
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        return
    
    # Run all analyses
    basic_model_info(model)
    model_parameters(model)
    feature_importance_analysis(model)
    model_performance_analysis(model)
    explore_individual_trees(model, n_trees=2)
    model_file_analysis()
    
    # Optional interactive tool
    print("🎯 For interactive predictions, run: python -c \"import joblib; m=joblib.load('model.pkl'); print('Example prediction:', m.predict([[1.5,1.0,2.5]]))\"")
    
    print("✅ Model exploration complete!")
    print("Generated files: feature_importance.png, model_performance.png, tree_1.png, tree_2.png")

if __name__ == "__main__":
    main()