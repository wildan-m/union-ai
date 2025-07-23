"""
Common Customer Debugging Scenarios for Union.ai

This file demonstrates typical issues customers face and how to debug them.
Essential for Customer Success Engineer role.
"""

import pandas as pd
import time
from typing import Optional
from flytekit import task, workflow, Resources, current_context
from flytekit.exceptions.user import FlyteRecoverableException


# Scenario 1: Resource Exhaustion
@task(
    requests=Resources(cpu="100m", mem="128Mi"),
    limits=Resources(cpu="200m", mem="256Mi"),
    retries=2
)
def memory_intensive_task(size: int = 1000000) -> str:
    """
    Simulates memory issues customers often face
    
    Common customer issues:
    - Underestimating memory requirements
    - Not setting proper resource limits
    - Memory leaks in data processing
    """
    try:
        # This will fail with small memory limits
        large_data = list(range(size))
        processed_data = [x * 2 for x in large_data]
        return f"Processed {len(processed_data)} items"
    except MemoryError:
        ctx = current_context()
        raise FlyteRecoverableException(
            f"Memory exhausted processing {size} items. "
            f"Current task: {ctx.execution_id}. "
            f"Recommendation: Increase memory allocation or reduce batch size."
        )


# Scenario 2: Timeout Issues
@task(
    requests=Resources(cpu="200m", mem="500Mi"),
    timeout=60,
    retries=1
)
def slow_processing_task(delay_seconds: int = 30) -> str:
    """
    Simulates timeout scenarios
    
    Common customer issues:
    - Underestimating processing time
    - Network timeouts
    - Database connection timeouts
    """
    try:
        print(f"Starting long-running process ({delay_seconds}s)...")
        time.sleep(delay_seconds)
        return f"Completed after {delay_seconds} seconds"
    except Exception as e:
        raise FlyteRecoverableException(
            f"Task timed out after {delay_seconds} seconds. "
            f"Recommendation: Increase timeout or optimize processing logic."
        )


# Scenario 3: Data Validation Failures
@task
def validate_customer_data(data_path: str) -> pd.DataFrame:
    """
    Common data validation issues customers face
    
    Typical problems:
    - Schema mismatches
    - Missing required fields
    - Data type issues
    - Encoding problems
    """
    try:
        df = pd.read_csv(data_path)
    except FileNotFoundError:
        raise ValueError(
            f"Data file not found: {data_path}. "
            f"Common causes: "
            f"1. Incorrect file path in workflow config "
            f"2. File not uploaded to correct location "
            f"3. Permission issues accessing file"
        )
    except pd.errors.EmptyDataError:
        raise ValueError(
            f"Data file is empty: {data_path}. "
            f"Check data pipeline and file generation process."
        )
    except UnicodeDecodeError:
        raise ValueError(
            f"Encoding issue with file: {data_path}. "
            f"Try specifying encoding parameter: pd.read_csv(path, encoding='utf-8')"
        )
    
    # Schema validation
    required_columns = ['feature1', 'feature2', 'feature3', 'target']
    missing_columns = [col for col in required_columns if col not in df.columns]
    
    if missing_columns:
        raise ValueError(
            f"Missing required columns: {missing_columns}. "
            f"Expected columns: {required_columns}. "
            f"Found columns: {list(df.columns)}. "
            f"Check data generation process and column naming."
        )
    
    # Data type validation
    numeric_columns = ['feature1', 'feature2', 'feature3']
    for col in numeric_columns:
        if not pd.api.types.is_numeric_dtype(df[col]):
            try:
                df[col] = pd.to_numeric(df[col], errors='coerce')
            except:
                raise ValueError(
                    f"Column '{col}' contains non-numeric data. "
                    f"Check data source and cleaning process."
                )
    
    # Check for reasonable data ranges
    for col in numeric_columns:
        if df[col].min() < -1000 or df[col].max() > 1000:
            print(f"Warning: Column '{col}' has extreme values. "
                  f"Range: {df[col].min()} to {df[col].max()}")
    
    return df


# Scenario 4: Dependency Issues
@task
def check_dependencies() -> dict:
    """
    Check for common dependency issues
    
    Customer problems:
    - Missing packages
    - Version conflicts
    - Import errors
    """
    results = {}
    
    try:
        import sklearn
        results['sklearn'] = sklearn.__version__
    except ImportError:
        raise ImportError(
            "scikit-learn not found. "
            "Add 'scikit-learn>=1.3.0' to requirements.txt"
        )
    
    try:
        import pandas as pd
        results['pandas'] = pd.__version__
    except ImportError:
        raise ImportError(
            "pandas not found. "
            "Add 'pandas>=2.0.0' to requirements.txt"
        )
    
    try:
        import numpy as np
        results['numpy'] = np.__version__
    except ImportError:
        raise ImportError(
            "numpy not found. "
            "Add 'numpy>=1.24.0' to requirements.txt"
        )
    
    # Version compatibility checks
    if sklearn.__version__ < '1.0.0':
        print(f"Warning: scikit-learn version {sklearn.__version__} is outdated. "
              f"Consider upgrading to avoid compatibility issues.")
    
    return results


# Scenario 5: Configuration Issues
@task
def check_configuration(config: Optional[dict] = None) -> str:
    """
    Common configuration problems
    
    Customer issues:
    - Missing environment variables
    - Incorrect parameter types
    - Invalid configuration values
    """
    if config is None:
        config = {}
    
    # Check required configuration
    required_configs = ['model_type', 'data_source', 'output_path']
    missing_configs = [key for key in required_configs if key not in config]
    
    if missing_configs:
        raise ValueError(
            f"Missing required configuration: {missing_configs}. "
            f"Add these to your workflow parameters or config file."
        )
    
    # Validate configuration values
    valid_models = ['random_forest', 'svm', 'neural_network']
    if config.get('model_type') not in valid_models:
        raise ValueError(
            f"Invalid model_type: {config.get('model_type')}. "
            f"Valid options: {valid_models}"
        )
    
    return "Configuration validated successfully"


# Scenario 6: Network and External Service Issues
@task(retries=3, interruptible=True)
def external_service_call(api_endpoint: str) -> dict:
    """
    Simulate external service failures
    
    Common customer issues:
    - API rate limiting
    - Network timeouts
    - Authentication failures
    - Service unavailability
    """
    import random
    
    # Simulate different failure modes
    failure_mode = random.choice(['timeout', 'rate_limit', 'auth_error', 'success'])
    
    if failure_mode == 'timeout':
        raise FlyteRecoverableException(
            f"Timeout connecting to {api_endpoint}. "
            f"Check network connectivity and endpoint availability. "
            f"Consider increasing timeout or using circuit breaker pattern."
        )
    elif failure_mode == 'rate_limit':
        raise FlyteRecoverableException(
            f"Rate limit exceeded for {api_endpoint}. "
            f"Implement exponential backoff or request throttling. "
            f"Check API documentation for rate limits."
        )
    elif failure_mode == 'auth_error':
        raise ValueError(
            f"Authentication failed for {api_endpoint}. "
            f"Check API credentials and permissions. "
            f"Verify environment variables are set correctly."
        )
    
    return {"status": "success", "data": "mock_response"}


@workflow
def debugging_scenarios_workflow() -> dict:
    """
    Workflow that demonstrates common debugging scenarios
    
    This helps Customer Success Engineers understand:
    - Common failure patterns
    - Error messages customers see
    - Debugging approaches
    - Resolution strategies
    """
    results = {}
    
    # Check dependencies first
    deps = check_dependencies()
    results['dependencies'] = deps
    
    # Validate sample configuration
    sample_config = {
        'model_type': 'random_forest',
        'data_source': 'data/sample_data.csv',
        'output_path': 'outputs/'
    }
    config_status = check_configuration(config=sample_config)
    results['configuration'] = config_status
    
    # Try data validation
    try:
        data = validate_customer_data(data_path="data/sample_data.csv")
        results['data_validation'] = f"Validated {len(data)} rows"
    except Exception as e:
        results['data_validation_error'] = str(e)
    
    # Try resource-constrained task (may fail intentionally)
    try:
        memory_result = memory_intensive_task(size=100000)  # Smaller size for demo
        results['memory_task'] = memory_result
    except Exception as e:
        results['memory_task_error'] = str(e)
    
    # Try external service call
    try:
        service_result = external_service_call(api_endpoint="https://api.example.com")
        results['external_service'] = service_result
    except Exception as e:
        results['external_service_error'] = str(e)
    
    return results


if __name__ == "__main__":
    # Local testing
    print("Testing debugging scenarios...")
    
    # Test dependency checking
    try:
        deps = check_dependencies()
        print(f"Dependencies OK: {deps}")
    except Exception as e:
        print(f"Dependency error: {e}")
    
    # Test data validation
    try:
        data = validate_customer_data("data/sample_data.csv")
        print(f"Data validation passed: {len(data)} rows")
    except Exception as e:
        print(f"Data validation error: {e}")
    
    print("Debugging scenarios test completed.")