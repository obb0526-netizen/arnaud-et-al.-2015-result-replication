# Quick Start: Upload to GitHub

## ⚠️ IMPORTANT: Save Notebook Outputs First!

Your `main_analysis.ipynb` currently has outputs in 4/10 code cells. To make ALL results visible on GitHub:

### Run and Save the Notebook

**Option 1: Using Jupyter (Easiest)**
```bash
# Open in Jupyter
jupyter notebook main_analysis.ipynb
# Or
jupyter lab main_analysis.ipynb
```
Then:
1. Go to `Cell` → `Run All`
2. Wait for all cells to execute
3. `File` → `Save` (or `Cmd+S`)

**Option 2: Command Line**
```bash
# Execute all cells and save outputs
jupyter nbconvert --to notebook --execute --inplace main_analysis.ipynb
```

This will embed all outputs so they're visible on GitHub.

## Quick Upload Steps

```bash
# 1. Initialize git
git init

# 2. Add files
git add .

# 3. Commit
git commit -m "Initial commit: EEG analysis project"

# 4. Create repo on GitHub.com, then:
git remote add origin https://github.com/YOUR_USERNAME/REPO_NAME.git
git branch -M main
git push -u origin main
```

For detailed instructions, see `GITHUB_UPLOAD_GUIDE.md`.

