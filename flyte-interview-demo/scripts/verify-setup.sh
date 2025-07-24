#!/bin/bash

echo "🔍 Verifying Jupyter Python 3.12.11 Setup"
echo "========================================="

# Check Python version
echo "🐍 Python version:"
/Users/wildan/.local/bin/python3.12 --version

echo ""
echo "📋 Available Jupyter kernels:"
jupyter kernelspec list

echo ""
echo "📦 Testing package imports:"
/Users/wildan/.local/bin/python3.12 -c "
try:
    import pandas as pd
    print('✅ pandas:', pd.__version__)
except ImportError as e:
    print('❌ pandas:', e)

try:
    import numpy as np
    print('✅ numpy:', np.__version__)
except ImportError as e:
    print('❌ numpy:', e)

try:
    import matplotlib
    print('✅ matplotlib:', matplotlib.__version__)
except ImportError as e:
    print('❌ matplotlib:', e)

try:
    import seaborn as sns
    print('✅ seaborn:', sns.__version__)
except ImportError as e:
    print('❌ seaborn:', e)

try:
    import sklearn
    print('✅ scikit-learn:', sklearn.__version__)
except ImportError as e:
    print('❌ scikit-learn:', e)

try:
    import joblib
    print('✅ joblib:', joblib.__version__)
except ImportError as e:
    print('❌ joblib:', e)

try:
    import ipykernel
    print('✅ ipykernel:', ipykernel.__version__)
except ImportError as e:
    print('❌ ipykernel:', e)
"

echo ""
echo "⚙️ VS Code settings:"
if [ -f ".vscode/settings.json" ]; then
    echo "✅ .vscode/settings.json exists"
    echo "📄 Content:"
    cat .vscode/settings.json
else
    echo "❌ .vscode/settings.json not found"
    echo "💡 Run: ./scripts/configure-vscode.sh"
fi

echo ""
echo "🎯 Setup Status:"
PYTHON_OK=$(/Users/wildan/.local/bin/python3.12 -c "import sys; print('OK' if sys.version_info >= (3, 12) else 'FAIL')" 2>/dev/null || echo "FAIL")
KERNEL_OK=$(jupyter kernelspec list | grep -q "python312" && echo "OK" || echo "FAIL")
PACKAGES_OK=$(/Users/wildan/.local/bin/python3.12 -c "import pandas, numpy, matplotlib, seaborn, sklearn, joblib, ipykernel; print('OK')" 2>/dev/null || echo "FAIL")
VSCODE_OK=$([ -f ".vscode/settings.json" ] && echo "OK" || echo "FAIL")

echo "Python 3.12.11: $PYTHON_OK"
echo "Jupyter Kernel: $KERNEL_OK"
echo "ML Packages: $PACKAGES_OK"
echo "VS Code Config: $VSCODE_OK"

if [ "$PYTHON_OK" = "OK" ] && [ "$KERNEL_OK" = "OK" ] && [ "$PACKAGES_OK" = "OK" ] && [ "$VSCODE_OK" = "OK" ]; then
    echo ""
    echo "🎉 ALL CHECKS PASSED!"
    echo "✅ Your Jupyter notebook should now work with Python 3.12.11"
    echo ""
    echo "📋 Next steps:"
    echo "1. Restart VS Code: Cmd+Shift+P → 'Developer: Reload Window'"
    echo "2. Open your notebook"
    echo "3. Select 'Python 3.12.11' from kernel selector (top right)"
    echo "4. Run your cells!"
else
    echo ""
    echo "❌ SOME CHECKS FAILED"
    echo "💡 Try running: ./scripts/fix-jupyter-python312.sh"
fi