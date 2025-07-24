# Jupyter Python 3.12.11 Setup Scripts

This directory contains scripts to fix and configure Jupyter notebooks to work with Python 3.12.11 in VS Code.

## Quick Fix (Recommended)

If you're experiencing the "ipykernel package required" error, run:

```bash
./scripts/fix-jupyter-python312.sh
```

This will:
- Install ipykernel for Python 3.12.11
- Register the Jupyter kernel
- Install required ML packages
- Verify the setup

## Individual Scripts

### 1. Complete Fix Script
```bash
./scripts/fix-jupyter-python312.sh
```
Fixes all Jupyter/Python 3.12.11 issues in one go.

### 2. VS Code Configuration
```bash
./scripts/configure-vscode.sh
```
Sets up VS Code settings to use Python 3.12.11 by default.

### 3. Verification Script
```bash
./scripts/verify-setup.sh
```
Checks if everything is working correctly.

## After Running Scripts

1. **Restart VS Code**: `Cmd+Shift+P` → "Developer: Reload Window"
2. **Open your notebook**
3. **Select kernel**: Click kernel selector (top right) → Choose "Python 3.12.11"
4. **Run cells**: Should now work without errors!

## Troubleshooting

If you still have issues:

1. Check the verification output:
   ```bash
   ./scripts/verify-setup.sh
   ```

2. View detailed documentation:
   ```bash
   cat docs/jupyter-python312-setup-fix.md
   ```

3. Manual kernel selection in VS Code:
   - Open notebook
   - Press `Cmd+Shift+P`
   - Type "Notebook: Select Kernel"
   - Choose "Python 3.12.11"

## Script Details

All scripts are designed to be:
- **Safe**: Won't break existing Python installations
- **Idempotent**: Can be run multiple times safely
- **Verbose**: Show clear progress and results
- **Tested**: Verified to work on macOS with uv-managed Python

## Files Created/Modified

- `.vscode/settings.json` - VS Code Python interpreter settings
- Jupyter kernel registered at: `/Users/wildan/Library/Jupyter/kernels/python312`
- Python packages installed in: `/Users/wildan/.local/lib/python3.12/site-packages`