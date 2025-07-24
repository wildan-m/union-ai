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