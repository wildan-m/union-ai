# Flyte Troubleshooting Guide

## Common Customer Issues and Solutions

This guide covers the most frequent issues Union.ai customers encounter when working with Flyte workflows.

### 1. Workflow Execution Issues

#### Problem: Workflow fails to start
**Error Message:** `Failed to launch workflow execution`

**Common Causes:**
- Invalid workflow configuration
- Missing required parameters
- Authentication issues
- Resource quota exceeded

**Solution Steps:**
1. Check workflow registration: `flytectl get workflows`
2. Verify parameters match workflow definition
3. Confirm authentication: `flytectl config current-context`
4. Check resource quotas in cluster

#### Problem: Tasks stuck in "QUEUED" state
**Symptoms:** Tasks remain in queue without executing

**Common Causes:**
- Insufficient cluster resources
- Node affinity/tolerations issues
- Resource requests too high
- Cluster autoscaling delays

**Debugging:**
```bash
# Check cluster resources
kubectl top nodes
kubectl describe nodes

# Check pending pods
kubectl get pods -n flyte --field-selector=status.phase=Pending

# Check resource quotas
kubectl describe resourcequota -n flyte
```

**Solutions:**
- Reduce resource requests in task definitions
- Configure cluster autoscaling
- Review node selectors and affinity rules

### 2. Data Processing Issues

#### Problem: FileNotFoundError in data tasks
**Error:** `FileNotFoundError: [Errno 2] No such file or directory`

**Common Scenarios:**
- Incorrect file paths in workflow inputs
- Missing data uploads
- Permission issues
- Path resolution problems

**Solutions:**
```python
# Use absolute paths
@task
def load_data(data_path: str) -> pd.DataFrame:
    import os
    abs_path = os.path.abspath(data_path)
    if not os.path.exists(abs_path):
        raise FileNotFoundError(f"File not found: {abs_path}")
    return pd.read_csv(abs_path)
```

#### Problem: Memory issues with large datasets
**Error:** `MemoryError` or `OOMKilled`

**Solutions:**
1. Increase task memory allocation:
```python
@task(requests=Resources(mem="4Gi"), limits=Resources(mem="8Gi"))
def process_large_data(df: pd.DataFrame) -> pd.DataFrame:
    # Process in chunks
    chunk_size = 10000
    results = []
    for chunk in pd.read_csv(file_path, chunksize=chunk_size):
        processed_chunk = process_chunk(chunk)
        results.append(processed_chunk)
    return pd.concat(results)
```

2. Use data streaming approaches
3. Implement data partitioning

### 3. Model Training Issues

#### Problem: Training tasks timing out
**Error:** `Task exceeded maximum execution time`

**Solutions:**
```python
@task(timeout="3600s")  # 1 hour timeout
def train_model(X: pd.DataFrame, y: pd.Series) -> model:
    # Add progress logging
    model = RandomForestClassifier(verbose=1)
    model.fit(X, y)
    return model
```

#### Problem: Model artifacts not persisting
**Issue:** Trained models lost between tasks

**Solution:**
```python
from flytekit.types.file import FlyteFile
import joblib

@task
def save_model(model: Any) -> FlyteFile:
    model_path = "model.pkl"
    joblib.dump(model, model_path)
    return FlyteFile(model_path)

@task
def load_model(model_file: FlyteFile) -> Any:
    return joblib.load(model_file)
```

### 4. Resource Configuration

#### Problem: Tasks failing due to resource constraints
**Symptoms:** `OOMKilled`, `CPU throttling`, `Evicted pods`

**Best Practices:**
```python
# Conservative resource allocation
@task(
    requests=Resources(cpu="500m", mem="1Gi"),
    limits=Resources(cpu="2", mem="4Gi")
)
def data_processing_task():
    pass

# For ML training
@task(
    requests=Resources(cpu="2", mem="4Gi", gpu="1"),
    limits=Resources(cpu="4", mem="8Gi", gpu="1")
)
def train_gpu_model():
    pass
```

#### Problem: GPU tasks not scheduling
**Common Issues:**
- GPU resources not available
- Incorrect GPU resource specification
- Node selector issues

**Solutions:**
1. Check GPU availability: `kubectl describe nodes | grep nvidia`
2. Verify GPU resource specification:
```python
@task(requests=Resources(gpu="1"))
def gpu_task():
    import torch
    assert torch.cuda.is_available()
```

### 5. Dependency and Environment Issues

#### Problem: ImportError in tasks
**Error:** `ModuleNotFoundError: No module named 'package'`

**Solutions:**
1. Update requirements.txt:
```txt
scikit-learn>=1.3.0
pandas>=2.0.0
numpy>=1.24.0
```

2. Rebuild container images after dependency changes
3. Use explicit image references:
```python
@task(container_image="my-registry/my-image:v1.0")
def ml_task():
    pass
```

#### Problem: Version conflicts
**Symptoms:** Incompatible package versions, runtime errors

**Debugging:**
```python
@task
def check_versions():
    import sys
    import pkg_resources
    
    installed_packages = [d for d in pkg_resources.working_set]
    for package in installed_packages:
        print(f"{package.project_name}=={package.version}")
```

### 6. Performance Optimization

#### Problem: Slow workflow execution
**Common Causes:**
- Inefficient data processing
- Lack of parallelization
- Resource underutilization

**Solutions:**
1. Use map tasks for parallel processing:
```python
from flytekit import map_task

@map_task
def process_batch(item: dict) -> dict:
    # Process individual items in parallel
    return processed_item

@workflow
def parallel_processing_workflow(items: list[dict]) -> list[dict]:
    return process_batch(item=items)
```

2. Optimize resource allocation
3. Use appropriate instance types

### 7. Monitoring and Debugging

#### Essential Debugging Commands
```bash
# Check workflow status
flytectl get executions -p <project> -d <domain>

# Get detailed execution info
flytectl get execution -p <project> -d <domain> <execution-id>

# View task logs
flytectl get task-logs -p <project> -d <domain> <execution-id> <task-name>

# Check cluster resources
kubectl top nodes
kubectl top pods -n flyte
```

#### Adding Debug Information
```python
@task
def debug_task():
    import os
    import psutil
    
    # Log environment info
    print(f"Working directory: {os.getcwd()}")
    print(f"Available memory: {psutil.virtual_memory().available / 1024**3:.2f} GB")
    print(f"CPU count: {psutil.cpu_count()}")
    
    # Log environment variables
    for key, value in os.environ.items():
        if 'FLYTE' in key:
            print(f"{key}: {value}")
```

### 8. Best Practices for Customer Success

#### Preventive Measures
1. **Resource Planning:** Always set appropriate resource requests and limits
2. **Error Handling:** Use try-catch blocks with informative error messages
3. **Logging:** Add comprehensive logging to track workflow progress
4. **Testing:** Test workflows locally before deploying
5. **Documentation:** Document workflow parameters and expected inputs

#### Customer Communication Templates

**Resource Issue Response:**
```
I see your workflow is failing due to memory constraints. The error indicates that your task needs more than the allocated 1Gi of memory. 

Here's how to fix this:

1. Update your task definition:
```python
@task(requests=Resources(mem="4Gi"), limits=Resources(mem="8Gi"))
def your_task():
    pass
```

2. If you're processing large datasets, consider using chunked processing to reduce memory usage.

Would you like me to help you implement chunked processing for your specific use case?
```

**Data Issue Response:**
```
The FileNotFoundError suggests that the workflow can't find the input data file. Let's troubleshoot this step by step:

1. Verify the file path in your workflow configuration
2. Check if the file was uploaded to the correct location
3. Ensure the file has the correct permissions

Can you share your current workflow configuration so I can help identify the issue?
```