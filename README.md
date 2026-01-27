# Super Mario Bros RL Training

This project trains an asynchronous A3C agent to play **Super Mario Bros** and exposes the gameplay through a Streamlit experience that can run locally or on Streamlit Community Cloud.

## Getting Started

```bash
python3 -m venv .rl_super_mario_env
source .venv/bin/activate
pip install -r requirements.txt
```

## Training & Evaluation

```bash
# Train with multiple live observer windows on macOS
export OMP_NUM_THREADS=1
export SDL_VIDEODRIVER=metal
export RENDER_MODE=human
python train.py --world 1 --stage 1 --action_type complex --num_processes 6 --num_observers 20

# Train headless (no windows, faster)
python train.py --world 1 --stage 1 --action_type complex --num_processes 6 --num_observers 0

# Export a gameplay video using a trained checkpoint
python test.py --world 1 --stage 1 --action_type complex --saved_path trained_models --output_path output
```

## Streamlit Experience

```bash
streamlit run streamlit_app.py
```

The Streamlit app offers three visualization modes:

**Live Training Monitor** - Watch the AI actually learn in real-time:
- Trains the neural network from scratch (no pre-trained model needed)
- Performs gradient updates every few seconds
- Shows 6-12 observer windows that update with the improving policy
- Displays training metrics (loss, reward) as learning progresses
- Perfect for understanding how reinforcement learning works!

**Multiple Training Windows** - Watch Mario play with a pre-trained model:
- Shows 8-10 animated gameplay clips using stochastic sampling
- Each window explores differently with unique random seeds
- Great for visualizing policy behavior


### Deploy on Streamlit Cloud

**Quick Deploy:**

1. **Initialize Git & Push to GitHub:**
   ```bash
   git init
   git add .
   git commit -m "Initial commit: A3C Super Mario Bros RL Training"
   git branch -M main
   git remote add origin https://github.com/YOUR_USERNAME/RL_Super_Mario.git
   git push -u origin main
   ```

2. **Deploy on Streamlit Cloud:**
   - Go to [share.streamlit.io](https://share.streamlit.io)
   - Click "New app"
   - Select your GitHub repository
   - Set main file path: `streamlit_app.py`
   - Click "Deploy"

3. **Required Files (already included):**
   - ✅ `.streamlit/config.toml` - Streamlit configuration
   - ✅ `packages.txt` - System dependencies (libgl1-mesa-glx, libglib2.0-0)
   - ✅ `requirements.txt` - Python packages
   - ✅ `trained_models/A3CSuperMarioBros_1_1` - Pre-trained model

**Notes:**
- The app will work in all three modes (Live Training, Multiple Windows, Single Video)
- Live Training mode works great on Streamlit Cloud and lets visitors train from scratch
- System packages in `packages.txt` are required for OpenCV and Gym rendering
- Environment variables are automatically set for headless rendering

## Project Structure

- `train.py` – multiprocess A3C trainer.
- `test.py` – loads checkpoints and records a video.
- `env.py` – wrappers, rewards, preprocessing, and monitor utilities.
- `streamlit_app.py` – interactive UI for showcasing agents.

## Notes

- Set `OMP_NUM_THREADS=1` to prevent thread thrashing on macOS/Linux.
- The Gym Nintendo environments require `SDL_VIDEODRIVER=dummy` when running headless (Streamlit already sets this).
- **Multiple Observer Windows**: Use `--num_observers N` to spawn N evaluation windows that show the current policy in action. Each window explores differently due to stochastic action sampling. Set to 0 for headless training.

