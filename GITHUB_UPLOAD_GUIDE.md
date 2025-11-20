# Guide: Uploading EEG Project to GitHub

This guide will help you upload your EEG project to GitHub and ensure that notebook results are visible on the first page.

## Prerequisites

1. A GitHub account (sign up at https://github.com)
2. Git installed on your system (check with `git --version`)

## Step 1: Ensure Notebook Outputs Are Saved

**IMPORTANT:** Before uploading, make sure `main_analysis.ipynb` has all outputs saved so results are visible on GitHub.

### Option A: Using Jupyter Interface (Recommended)

1. Open `main_analysis.ipynb` in Jupyter Notebook/Lab
2. Run all cells: `Cell` → `Run All` (or press `Shift + Enter` through all cells)
3. Save the notebook: `File` → `Save` (or `Cmd+S` / `Ctrl+S`)
4. This ensures all outputs are embedded in the notebook

### Option B: Using Command Line

```bash
# Install jupyter if needed
pip install jupyter nbconvert

# Execute and save notebook with outputs
jupyter nbconvert --to notebook --execute --inplace main_analysis.ipynb
```

### Verify Outputs Are Saved

Check that cells have outputs:
```bash
python3 -c "import json; nb=json.load(open('main_analysis.ipynb')); print(f\"Cells with outputs: {sum(1 for c in nb['cells'] if c.get('cell_type')=='code' and c.get('outputs'))}\")"
```

## Step 2: Initialize Git Repository

```bash
# Navigate to project directory
cd /Users/leeyelim/Documents/EEG

# Initialize git repository
git init

# Add all files (respecting .gitignore)
git add .

# Create initial commit
git commit -m "Initial commit: EEG analysis project"
```

## Step 3: Create GitHub Repository

1. Go to https://github.com and sign in
2. Click the **"+"** icon in the top right → **"New repository"**
3. Repository name: e.g., `eeg-analysis` or `EEG`
4. Description: "EEG memory recognition study analysis"
5. Choose **Public** or **Private**
6. **DO NOT** initialize with README, .gitignore, or license (you already have these)
7. Click **"Create repository"**

## Step 4: Connect and Push to GitHub

GitHub will show you commands. Use these (replace `YOUR_USERNAME` and `REPO_NAME`):

```bash
# Add remote repository
git remote add origin https://github.com/YOUR_USERNAME/REPO_NAME.git

# Rename default branch to main (if needed)
git branch -M main

# Push to GitHub
git push -u origin main
```

**Note:** You may be prompted for GitHub credentials. Use:
- Personal Access Token (recommended) instead of password
- Or use GitHub CLI: `gh auth login`

## Step 5: Ensure Results Are Visible

### On GitHub, notebooks render with outputs automatically if:
1. ✅ Notebook has outputs saved (cells show `[1]:`, `[2]:`, etc.)
2. ✅ Outputs are included in the `.ipynb` file
3. ✅ Repository is public (private repos may have limited rendering)

### Viewing Your Notebook on GitHub:
- Navigate to: `https://github.com/YOUR_USERNAME/REPO_NAME/blob/main/main_analysis.ipynb`
- GitHub will render the notebook with outputs visible
- First page will show markdown and any code outputs from early cells

### Alternative: Use nbviewer for Better Rendering
1. Go to: https://nbviewer.org/
2. Paste your GitHub notebook URL
3. View with enhanced formatting

## Step 6: Add a README.md (Optional but Recommended)

Create a proper `README.md` for better project visibility:

```markdown
# EEG Memory Recognition Study

## Overview
This project contains the analysis pipeline for [describe your study].

## Key Results
View the main results in [main_analysis.ipynb](main_analysis.ipynb).

## Project Structure
- `main_analysis.ipynb` - Main results and visualizations
- `notebooks/` - Analysis pipeline notebooks
- `src/` - Python source code
- `results/` - Generated results and figures
- `data/` - Preprocessed data (not in repo)

## Requirements
See `requirements.txt`

## Usage
[Add usage instructions]
```

## Troubleshooting

### Notebook outputs not showing?
1. Make sure you saved the notebook after running cells
2. Check that cells have `"execution_count"` and `"outputs"` in the JSON
3. Re-run cells and save: `jupyter nbconvert --to notebook --execute --inplace main_analysis.ipynb`

### Large files not uploading?
- Check `.gitignore` - large data files (`.fif`, `.set`) are excluded
- If you need to include large files, consider Git LFS:
  ```bash
  git lfs install
  git lfs track "*.fif"
  git lfs track "*.set"
  ```

### Authentication issues?
- Use Personal Access Token instead of password
- Or install GitHub CLI: `brew install gh` (Mac) then `gh auth login`

## File Size Considerations

Your project may have large files. The `.gitignore` excludes:
- Large EEG data files (`.fif`, `.set`)
- Preprocessed data directories
- Temporary files

If repository is still too large:
1. Use Git LFS for specific large files
2. Consider storing data separately (Zenodo, OSF, etc.)
3. Only commit essential results and figures

## Next Steps

1. ✅ Ensure notebook has outputs saved
2. ✅ Initialize git and commit
3. ✅ Create GitHub repository
4. ✅ Push to GitHub
5. ✅ Share the repository URL

For questions or issues, refer to:
- [GitHub Documentation](https://docs.github.com)
- [Jupyter Notebook Documentation](https://jupyter-notebook.readthedocs.io/)

