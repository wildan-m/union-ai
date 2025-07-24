# Jupyter Python 3.12.11 Setup Fix Documentation

## Problem Description
VS Code was showing the error: `Running cells with 'Python 3.12.11' requires the ipykernel package` despite Python 3.12.11 being available on the system.

## Root Cause Analysis
1. Python 3.12.11 was managed by `uv` and marked as "externally managed"
2. `ipykernel` was not installed for the Python 3.12.11 installation
3. Jupyter kernel was not registered for Python 3.12.11
4. Required ML packages were missing from Python 3.12.11 environment

## Complete Fix Script Sequence

### Step 1: Diagnose the Environment

```bash
# Check Python version and location
python --version && which python

# Check if ipykernel is installed (this will fail)
pip list | grep ipykernel

# Check available Jupyter kernels
jupyter kernelspec list
```

**Expected Results:**
- Python 3.12.11 located at `/Users/wildan/.local/bin/python3.12`
- ipykernel not found in pip list
- Only basic kernels listed in jupyter kernelspec

### Step 2: Install ipykernel for Python 3.12.11

```bash
# Install ipykernel (override externally-managed restriction)
/Users/wildan/.local/bin/python3.12 -m pip install ipykernel -U --user --force-reinstall --break-system-packages
```

**Expected Output:**
```
Successfully installed appnope-0.1.4 asttokens-3.0.0 comm-0.2.2 debugpy-1.8.15 
decorator-5.2.1 executing-2.2.0 ipykernel-6.30.0 ipython-9.4.0 
ipython-pygments-lexers-1.1.1 jedi-0.19.2 jupyter-client-8.6.3 
jupyter-core-5.8.1 matplotlib-inline-0.1.7 nest-asyncio-1.6.0 
packaging-25.0 parso-0.8.4 pexpect-4.9.0 platformdirs-4.3.8 
prompt_toolkit-3.0.51 psutil-7.0.0 ptyprocess-0.7.0 pure-eval-0.2.3 
pygments-2.19.2 python-dateutil-2.9.0.post0 pyzmq-27.0.0 six-1.17.0 
stack_data-0.6.3 tornado-6.5.1 traitlets-5.14.3 wcwidth-0.2.13
```

### Step 3: Register Jupyter Kernel

```bash
# Register Python 3.12.11 as a Jupyter kernel
/Users/wildan/.local/bin/python3.12 -m ipykernel install --user --name=python312 --display-name="Python 3.12.11"
```

**Expected Output:**
```
Installed kernelspec python312 in /Users/wildan/Library/Jupyter/kernels/python312
```

### Step 4: Verify Kernel Registration

```bash
# Verify the kernel is now available
jupyter kernelspec list
```

**Expected Output:**
```
Available kernels:
  python312    /Users/wildan/Library/Jupyter/kernels/python312
  python3      /Users/wildan/.local/share/jupyter/kernels/python3
```

### Step 5: Install Required ML Packages

```bash
# Install all required packages for the notebook
/Users/wildan/.local/bin/python3.12 -m pip install pandas numpy matplotlib seaborn scikit-learn joblib --break-system-packages
```

**Expected Output:**
```
Successfully installed contourpy-1.3.2 cycler-0.12.1 fonttools-4.59.0 
joblib-1.5.1 kiwisolver-1.4.8 matplotlib-3.10.3 numpy-2.3.1 
pandas-2.3.1 pillow-11.3.0 pyparsing-3.2.3 pytz-2025.2 
scikit-learn-1.7.1 scipy-1.16.0 seaborn-0.13.2 threadpoolctl-3.6.0 
tzdata-2025.2
```

### Step 6: Test Package Installation

```bash
# Test that all packages are working
/Users/wildan/.local/bin/python3.12 -c "import pandas as pd; import numpy as np; import matplotlib.pyplot as plt; import seaborn as sns; from sklearn.ensemble import RandomForestClassifier; import joblib; print('All required packages imported successfully!')"
```

**Expected Output:**
```
All required packages imported successfully!
Matplotlib is building the font cache; this may take a moment.
```

### Step 7: Configure VS Code Settings

Create or update `.vscode/settings.json`:

```json
{
    "python.defaultInterpreterPath": "/Users/wildan/.local/bin/python3.12",
    "jupyter.kernels.filter": [
        {
            "path": "/Users/wildan/.local/bin/python3.12",
            "type": "pythonEnvironment"
        }
    ],
    "jupyter.defaultKernel": "python312"
}
```

## Complete One-Shot Fix Script

Save this as `fix-jupyter-python312.sh`:

```bash
#!/bin/bash

echo "🔧 Fixing Jupyter Python 3.12.11 Setup"
echo "======================================"

# Step 1: Install ipykernel
echo "📦 Installing ipykernel..."
/Users/wildan/.local/bin/python3.12 -m pip install ipykernel -U --user --force-reinstall --break-system-packages

# Step 2: Register kernel
echo "🔗 Registering Jupyter kernel..."
/Users/wildan/.local/bin/python3.12 -m ipykernel install --user --name=python312 --display-name="Python 3.12.11"

# Step 3: Install ML packages
echo "📊 Installing ML packages..."
/Users/wildan/.local/bin/python3.12 -m pip install pandas numpy matplotlib seaborn scikit-learn joblib --break-system-packages

# Step 4: Verify installation
echo "✅ Testing installation..."
/Users/wildan/.local/bin/python3.12 -c "import pandas as pd; import numpy as np; import matplotlib.pyplot as plt; import seaborn as sns; from sklearn.ensemble import RandomForestClassifier; import joblib; print('✅ All packages working!')"

# Step 5: List available kernels
echo "📋 Available Jupyter kernels:"
jupyter kernelspec list

echo ""
echo "🎉 Setup complete! Next steps:"
echo "1. Restart VS Code or reload window (Cmd+Shift+P → 'Developer: Reload Window')"
echo "2. In notebook, select 'Python 3.12.11' kernel from kernel selector"
echo "3. Run your notebook cells"
```

Make it executable and run:

```bash
chmod +x fix-jupyter-python312.sh
./fix-jupyter-python312.sh
```

## VS Code Configuration Script

Save this as `configure-vscode.sh`:

```bash
#!/bin/bash

echo "⚙️ Configuring VS Code for Python 3.12.11"
echo "========================================"

# Create .vscode directory if it doesn't exist
mkdir -p .vscode

# Create settings.json
cat > .vscode/settings.json << 'EOF'
{
    "python.defaultInterpreterPath": "/Users/wildan/.local/bin/python3.12",
    "jupyter.kernels.filter": [
        {
            "path": "/Users/wildan/.local/bin/python3.12",
            "type": "pythonEnvironment"
        }
    ],
    "jupyter.defaultKernel": "python312"
}
EOF

echo "✅ VS Code configured!"
echo "📋 Settings written to .vscode/settings.json"
echo ""
echo "🔄 Please restart VS Code to apply changes"
```

## Troubleshooting Common Issues

### Issue 1: "externally-managed-environment" Error
**Solution:** Use `--break-system-packages` flag
```bash
/Users/wildan/.local/bin/python3.12 -m pip install package --break-system-packages
```

### Issue 2: Kernel Not Appearing in VS Code
**Solutions:**
1. Restart VS Code completely
2. Refresh kernels: `Cmd+Shift+P` → "Python: Refresh Kernels"
3. Check kernel exists: `jupyter kernelspec list`

### Issue 3: Wrong Python Path
**Solution:** Use full absolute path to Python executable
```bash
# Find correct path
which python3.12
# Use full path in commands
/full/path/to/python3.12 -m pip install package
```

### Issue 4: Packages Not Found in Notebook
**Solution:** Ensure packages are installed in the same Python environment as the kernel
```bash
# Check which Python the kernel uses
cat /Users/wildan/Library/Jupyter/kernels/python312/kernel.json
# Install packages in that exact Python
/path/from/kernel.json -m pip install package --break-system-packages
```

## Verification Checklist

After running the fix scripts, verify:

- [ ] `jupyter kernelspec list` shows `python312` kernel
- [ ] VS Code shows "Python 3.12.11" in kernel selector
- [ ] Test cell runs without ipykernel error: `print("Hello World")`
- [ ] ML packages import successfully: `import pandas, numpy, sklearn`
- [ ] `.vscode/settings.json` exists with correct Python path

## Environment Details

This fix was tested on:
- **OS:** macOS (Darwin 24.3.0)
- **Python:** 3.12.11 (managed by uv)
- **Python Path:** `/Users/wildan/.local/bin/python3.12`
- **VS Code:** with Python and Jupyter extensions
- **Package Manager:** pip with `--break-system-packages`

## Security Note

The `--break-system-packages` flag is used because Python 3.12.11 is managed by `uv` and marked as externally managed. This is safe in a development environment but use with caution in production systems.