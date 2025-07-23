"""
ML Pipeline Workflow for Union.ai Interview Demo

This demonstrates:
- Data processing workflows
- ML model training and evaluation
- Error handling and debugging scenarios
- Customer-facing documentation patterns
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
import joblib
from typing import NamedTuple, Tuple, Any

from flytekit import task, workflow, Resources
from flytekit.types.file import FlyteFile
from flytekit.types.directory import FlyteDirectory


class ModelMetrics(NamedTuple):
    """Model evaluation metrics"""
    accuracy: float
    precision: float
    recall: float
    f1_score: float


@task(
    requests=Resources(cpu="500m", mem="1Gi"),
    limits=Resources(cpu="1", mem="2Gi")
)
def load_and_validate_data(data_path: str = "data/sample_data.csv") -> pd.DataFrame:
    """
    Load and validate input data
    
    Common customer issues addressed:
    - File path resolution
    - Data validation
    - Memory management
    """
    try:
        df = pd.read_csv(data_path)
        
        # Data validation - common customer pain point
        required_columns = ['feature1', 'feature2', 'feature3', 'target']
        missing_cols = [col for col in required_columns if col not in df.columns]
        
        if missing_cols:
            raise ValueError(f"Missing required columns: {missing_cols}")
        
        # Check for data quality issues
        if df.isnull().sum().sum() > 0:
            print(f"Warning: Found {df.isnull().sum().sum()} null values")
            df = df.dropna()
        
        print(f"Loaded {len(df)} rows of data")
        return df
        
    except FileNotFoundError:
        raise FileNotFoundError(f"Data file not found at {data_path}. Check file path configuration.")
    except Exception as e:
        raise RuntimeError(f"Data loading failed: {str(e)}")


@task(
    requests=Resources(cpu="200m", mem="500Mi")
)
def preprocess_data(df: pd.DataFrame, test_size: float = 0.2) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """
    Preprocess data for training
    
    Demonstrates:
    - Data splitting strategies
    - Feature scaling (if needed)
    - Common preprocessing errors
    """
    try:
        # Separate features and target
        X = df.drop('target', axis=1)
        y = df['target']
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=42, stratify=y
        )
        
        print(f"Training set: {len(X_train)} samples")
        print(f"Test set: {len(X_test)} samples")
        
        return X_train, X_test, y_train, y_test
        
    except Exception as e:
        raise RuntimeError(f"Data preprocessing failed: {str(e)}")


@task(
    requests=Resources(cpu="1", mem="2Gi"),
    limits=Resources(cpu="2", mem="4Gi")
)
def train_model(X_train: pd.DataFrame, y_train: pd.Series) -> Any:
    """
    Train ML model
    
    Common customer scenarios:
    - Resource allocation for training
    - Model hyperparameters
    - Training time optimization
    """
    try:
        # Model configuration - often a customer customization point
        model = RandomForestClassifier(
            n_estimators=100,
            max_depth=10,
            random_state=42,
            n_jobs=-1  # Use all available cores
        )
        
        print("Starting model training...")
        model.fit(X_train, y_train)
        print("Model training completed")
        
        return model
        
    except Exception as e:
        raise RuntimeError(f"Model training failed: {str(e)}")


@task(
    requests=Resources(cpu="500m", mem="1Gi")
)
def evaluate_model(model: Any, X_test: pd.DataFrame, y_test: pd.Series) -> ModelMetrics:
    """
    Evaluate model performance
    
    Demonstrates:
    - Model evaluation patterns
    - Metrics collection
    - Performance monitoring
    """
    try:
        predictions = model.predict(X_test)
        
        # Calculate metrics
        accuracy = accuracy_score(y_test, predictions)
        
        # Get detailed classification report
        report = classification_report(y_test, predictions, output_dict=True)
        
        metrics = ModelMetrics(
            accuracy=float(accuracy),
            precision=float(report['weighted avg']['precision']),
            recall=float(report['weighted avg']['recall']),
            f1_score=float(report['weighted avg']['f1-score'])
        )
        
        print(f"Model Evaluation Results:")
        print(f"Accuracy: {metrics.accuracy:.3f}")
        print(f"Precision: {metrics.precision:.3f}")
        print(f"Recall: {metrics.recall:.3f}")
        print(f"F1-Score: {metrics.f1_score:.3f}")
        
        return metrics
        
    except Exception as e:
        raise RuntimeError(f"Model evaluation failed: {str(e)}")


@task(
    requests=Resources(cpu="200m", mem="500Mi")
)
def save_model(model: Any, model_path: str = "model.pkl") -> str:
    """
    Save trained model
    
    Common customer needs:
    - Model versioning
    - Model artifacts management
    - Deployment preparation
    """
    try:
        joblib.dump(model, model_path)
        print(f"Model saved to {model_path}")
        return model_path
        
    except Exception as e:
        raise RuntimeError(f"Model saving failed: {str(e)}")


@workflow
def ml_training_pipeline(data_path: str = "data/sample_data.csv", test_size: float = 0.2) -> ModelMetrics:
    """
    End-to-end ML training pipeline
    
    This workflow demonstrates:
    - Data loading and validation
    - Data preprocessing
    - Model training
    - Model evaluation
    - Model saving
    
    Common customer use cases:
    - Batch training workflows
    - Model retraining pipelines
    - A/B testing scenarios
    """
    # Load and validate data
    df = load_and_validate_data(data_path=data_path)
    
    # Preprocess data
    X_train, X_test, y_train, y_test = preprocess_data(df=df, test_size=test_size)
    
    # Train model
    model = train_model(X_train=X_train, y_train=y_train)
    
    # Evaluate model
    metrics = evaluate_model(model=model, X_test=X_test, y_test=y_test)
    
    # Save model
    model_path = save_model(model=model)
    
    return metrics


# Additional workflow for batch prediction - common customer pattern
@task
def load_model(model_path: str) -> Any:
    """Load saved model for inference"""
    try:
        model = joblib.load(model_path)
        return model
    except Exception as e:
        raise RuntimeError(f"Model loading failed: {str(e)}")


@task
def batch_predict(model: Any, data: pd.DataFrame) -> pd.DataFrame:
    """Perform batch predictions"""
    try:
        predictions = model.predict(data.drop('target', axis=1, errors='ignore'))
        data_with_predictions = data.copy()
        data_with_predictions['predictions'] = predictions
        return data_with_predictions
    except Exception as e:
        raise RuntimeError(f"Batch prediction failed: {str(e)}")


@workflow
def batch_prediction_pipeline(data_path: str, model_path: str = "model.pkl") -> pd.DataFrame:
    """
    Batch prediction pipeline
    
    Common customer scenario:
    - Regular inference jobs
    - Batch scoring workflows
    """
    # Load data and model
    df = load_and_validate_data(data_path=data_path)
    model = load_model(model_path=model_path)
    
    # Make predictions
    results = batch_predict(model=model, data=df)
    
    return results


if __name__ == "__main__":
    # Local execution for testing
    print("Running ML Pipeline locally...")
    metrics = ml_training_pipeline()
    print(f"Pipeline completed with accuracy: {metrics.accuracy:.3f}")