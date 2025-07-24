## Repository Overview

This repository contains multiple Union.ai/Flyte projects demonstrating ML workflows and Union platform features:

- **flyte-interview-demo/**: Interview demonstration project showcasing Flyte ML pipelines and customer debugging scenarios
- **union-cloud/**: Simple Union cloud project with hello world workflow
- **union-app/**: Streamlit quickstart application for Union platform

## Common Development Commands

### Flyte ML Pipeline Demo (flyte-interview-demo/)
```bash
# Install dependencies
pip install -r flyte-interview-demo/requirements.txt

# Run ML pipeline locally
cd flyte-interview-demo
python workflows/ml_pipeline.py

# Test debugging scenarios
python workflows/debugging_scenarios.py

# Build container for deployment
docker build -t flyte-ml-demo flyte-interview-demo/

# Run workflows with Flyte (requires Flyte cluster)
pyflyte run workflows/ml_pipeline.py ml_training_pipeline
pyflyte run workflows/debugging_scenarios.py debugging_scenarios_workflow
```

### Union Cloud Project (union-cloud/)
```bash
# Install union CLI and dependencies (uses uv for dependency management)
cd union-cloud
uv sync

# Run workflow locally
union run hello_world.py hello_world_wf

# Deploy to Union cloud (requires union login)
union register hello_world.py
```

### Union Streamlit App (union-app/)
```bash
# Deploy as Union app
cd union-app
union deploy apps quickstart.py streamlit-quickstart 

# Or with specific project/domain
union deploy apps -p myproject -d production quickstart.py
```

## Architecture and Code Structure

### Flyte ML Pipeline Demo
- **Primary Purpose**: Demonstrates customer success scenarios for Union.ai interview
- **Key Components**:
  - `workflows/ml_pipeline.py`: Complete ML training pipeline with data loading, preprocessing, training, and evaluation
  - `workflows/debugging_scenarios.py`: Common customer issues (memory, timeouts, data validation, dependencies)
  - `docs/troubleshooting-guide.md`: Customer-facing documentation
  - `config/flyte_config.yaml`: Local Flyte cluster configuration

### Task Resource Management
All Flyte tasks use explicit resource definitions:
- CPU requests/limits (e.g., "500m", "1")  
- Memory requests/limits (e.g., "1Gi", "2Gi")
- Retries and timeout configurations
- Resource-aware error handling

### Error Handling Patterns
- Uses `FlyteRecoverableException` for retryable failures
- Customer-friendly error messages with debugging guidance
- Comprehensive validation for data, dependencies, and configuration
- Proper exception chaining and context preservation

### Development Workflow
1. Local testing first (`python workflows/script.py`)
2. Container testing (`docker build` and run)
3. Flyte execution (`pyflyte run`)
4. Union cloud deployment (`union create`)

## Testing Strategy

No formal test framework is configured. Testing is done through:
- Direct script execution for local validation
- Manual workflow execution with sample data
- Container builds to verify deployment readiness
- Resource constraint testing for debugging scenarios

## Dependencies and Environment

### Python Environment
- Python 3.10+ required
- Primary ML stack: pandas, scikit-learn, numpy, matplotlib
- Flyte/Union: flytekit>=1.10.0, union
- Container base: python:3.10-slim

### Development Tools
- Docker for containerization
- uv for modern Python package management (union-cloud project)
- pip for traditional dependency management (flyte-interview-demo)

## Configuration Files

- `flyte-interview-demo/config/flyte_config.yaml`: Local Flyte cluster setup with minio storage
- `union-cloud/pyproject.toml`: Union project configuration with dependency specifications
- `flyte-interview-demo/requirements.txt`: Traditional pip requirements
- Dockerfiles for containerized deployments