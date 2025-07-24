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