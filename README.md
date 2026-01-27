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


## 🌐 Live Demo

**Try the app online:** [YOUR_APP_URL_HERE]

The web app offers three interactive modes to explore reinforcement learning:

### 🔴 Live Training Monitor
**Watch AI learn from scratch in real-time:**
1. Select "Live Training Monitor" from the sidebar
2. Configure settings (or use defaults: 4 training workers, 6 observer windows)
3. Click "▶️ Start Training"
4. Watch the observer windows update every 5 seconds as the neural network learns
5. See metrics improve: loss decreases, reward increases
6. Mario's behavior evolves from random to purposeful movement
7. Click "⏹️ Stop Training" to save the trained model

**What you'll see:** Initially, Mario moves randomly in all directions. After a few minutes, he starts moving right consistently. After 10-15 minutes, he learns to jump over obstacles!

### 🎮 Multiple Training Windows (Pre-trained Model)
**Compare different strategies side-by-side:**
1. Select "Multiple Training Windows" from the sidebar
2. Choose number of windows (4-12, default: 8)
3. Click "🚀 Launch N Training Windows"
4. Watch 8 animated gameplay clips, each exploring differently
5. Each window uses stochastic sampling with unique random seeds

### 📹 Single Video Playback
**Traditional video playback experience:**
1. Select "Single Video Playback" from the sidebar
2. Adjust max rollout steps
3. Click "🚀 Generate Training Screens"
4. View frame snapshots and full MP4 playback

## Project Structure

- `train.py` – multiprocess A3C trainer.
- `test.py` – loads checkpoints and records a video.
- `env.py` – wrappers, rewards, preprocessing, and monitor utilities.
- `streamlit_app.py` – interactive UI for showcasing agents.

## Notes

- Set `OMP_NUM_THREADS=1` to prevent thread thrashing on macOS/Linux.
- The Gym Nintendo environments require `SDL_VIDEODRIVER=dummy` when running headless (Streamlit already sets this).
- **Multiple Observer Windows**: Use `--num_observers N` to spawn N evaluation windows that show the current policy in action. Each window explores differently due to stochastic action sampling. Set to 0 for headless training.

