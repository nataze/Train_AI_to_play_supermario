# Deployment Guide: Streamlit Cloud

## Prerequisites
- GitHub account
- Streamlit Cloud account (free at [share.streamlit.io](https://share.streamlit.io))

## Step-by-Step Deployment

### 1. Create GitHub Repository

Go to [github.com/new](https://github.com/new) and create a new repository:
- **Repository name:** `RL_Super_Mario` (or your preferred name)
- **Description:** "A3C Reinforcement Learning for Super Mario Bros with live training visualization"
- **Visibility:** Public (required for free Streamlit Cloud deployment)
- **DO NOT** initialize with README, .gitignore, or license (we already have these)

### 2. Push Code to GitHub

In your terminal, run these commands from the project directory:

```bash
# Initialize git repository
git init

# Add all files
git add .

# Create initial commit
git commit -m "Initial commit: A3C Super Mario Bros RL Training with Streamlit"

# Rename branch to main
git branch -M main

# Add your GitHub repository as remote (replace YOUR_USERNAME)
git remote add origin https://github.com/YOUR_USERNAME/RL_Super_Mario.git

# Push to GitHub
git push -u origin main
```

### 3. Deploy on Streamlit Cloud

1. **Go to Streamlit Cloud:**
   - Visit [share.streamlit.io](https://share.streamlit.io)
   - Sign in with your GitHub account

2. **Create New App:**
   - Click "New app" button
   - **Repository:** Select `YOUR_USERNAME/RL_Super_Mario`
   - **Branch:** `main`
   - **Main file path:** `streamlit_app.py`
   - Click "Deploy!"

3. **Wait for Deployment:**
   - First deployment takes 3-5 minutes
   - Streamlit will install all dependencies from `requirements.txt` and `packages.txt`
   - You'll see build logs in real-time

4. **Your App is Live!**
   - You'll get a URL like `https://your-app-name.streamlit.app`
   - Share this URL with anyone to showcase your RL project

## What Gets Deployed

✅ **Included Files:**
- All Python code (`streamlit_app.py`, `train.py`, `env.py`, `model.py`, `optimizer.py`, `process.py`, `test.py`)
- Pre-trained model (`trained_models/A3CSuperMarioBros_1_1`)
- Configuration files (`.streamlit/config.toml`, `packages.txt`, `requirements.txt`)
- Documentation (`README.md`)

❌ **Excluded Files (.gitignore):**
- Virtual environments (`.venv/`, `.rl_super_mario_env/`)
- Python cache files (`__pycache__/`)
- Tensorboard logs (too large)
- Output videos (generated files)
- IDE files (`.vscode/`, `.idea/`)

## Features Available on Streamlit Cloud

All three modes work perfectly on Streamlit Cloud:

### ✅ Live Training Monitor
- Train the neural network from scratch
- Watch observer windows update in real-time
- See metrics improve as the AI learns
- Visitors can start their own training sessions

### ✅ Multiple Training Windows
- Show 8-10 animated gameplay clips
- Each window explores differently
- Uses the pre-trained model

### ✅ Single Video Playback  
- Traditional frame snapshots
- Full MP4 video playback

## Troubleshooting

### Build Fails
- Check that `packages.txt` includes system dependencies:
  ```
  libgl1-mesa-glx
  libglib2.0-0
  ```
- Ensure `requirements.txt` has all Python packages

### App Crashes on Start
- Streamlit Cloud automatically sets `RENDER_MODE=rgb_array` via the config
- Check logs for specific error messages

### Slow Performance
- Live Training mode may be slower on Streamlit Cloud's free tier
- Consider reducing observer windows from 6 to 4
- Reduce frames per window from 80 to 50

## Updating Your Deployment

To update your live app after making changes:

```bash
git add .
git commit -m "Description of your changes"
git push origin main
```

Streamlit Cloud will automatically redeploy within 1-2 minutes.

## Custom Domain (Optional)

Streamlit Cloud free tier gives you:
- Default URL: `https://your-app-name.streamlit.app`
- To use a custom domain, upgrade to Streamlit Cloud Teams

## Support

- Streamlit Docs: [docs.streamlit.io](https://docs.streamlit.io)
- Community Forum: [discuss.streamlit.io](https://discuss.streamlit.io)
- GitHub Issues: Open an issue in your repository
