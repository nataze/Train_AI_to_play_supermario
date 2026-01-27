import os
from collections import deque
from tempfile import NamedTemporaryFile

import cv2
import imageio.v2 as imageio
import numpy as np
import streamlit as st
import torch
import torch.nn.functional as F
from PIL import Image

from env import create_train_env
from model import ActorCritic
from optimizer import GlobalAdam
import torch.multiprocessing as mp
import threading
import time


os.environ["OMP_NUM_THREADS"] = "1"
os.environ["SDL_VIDEODRIVER"] = "dummy"
os.environ["RENDER_MODE"] = "rgb_array"  # Force rgb_array mode for Streamlit (no windows)


st.set_page_config(
    page_title="Super Mario RL Training",
    page_icon="🎮",
    layout="wide",
    initial_sidebar_state="expanded",
)


def render_frame(env):
    """Try to get an RGB frame from the (possibly wrapped) environment."""
    candidate = env
    while candidate is not None:
        try:
            frame = candidate.render()
            if frame is not None:
                frame = np.array(frame)
                if frame.ndim == 3:
                    if frame.shape[2] == 4:
                        frame = cv2.cvtColor(frame, cv2.COLOR_RGBA2RGB)
                elif frame.ndim == 2:
                    frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2RGB)
                return frame
        except Exception:
            pass
        candidate = getattr(candidate, "env", None)

    return np.zeros((240, 256, 3), dtype=np.uint8)


def load_model(world, stage, action_type):
    model_path = f"trained_models/A3CSuperMarioBros_{world}_{stage}"
    if not os.path.exists(model_path):
        return None

    # Ensure rgb_array mode for Streamlit (thread-safe, no window rendering)
    os.environ["RENDER_MODE"] = "rgb_array"
    env, num_states, num_actions = create_train_env(world, stage, action_type)
    model = ActorCritic(num_states, num_actions)

    if torch.cuda.is_available():
        model.load_state_dict(torch.load(model_path))
        model.cuda()
    else:
        model.load_state_dict(
            torch.load(model_path, map_location=torch.device("cpu"))
        )

    model.eval()
    return model, env


def generate_frames(model, env, max_steps=200, stuck_tolerance=120, seed=None, use_sampling=False):
    """Generate frames from a single rollout.
    
    Args:
        model: The neural network model
        env: The game environment
        max_steps: Maximum number of steps to run
        stuck_tolerance: Stop if same action repeated this many times
        seed: Random seed for reproducibility
        use_sampling: If True, sample from policy; if False, use argmax
    """
    import random
    import numpy as np
    from torch.distributions import Categorical
    
    if seed is not None:
        torch.manual_seed(seed)
        random.seed(seed)
        np.random.seed(seed)
    
    frames = []
    state = torch.from_numpy(env.reset())
    done = True
    step_count = 0
    actions = deque(maxlen=stuck_tolerance)

    while step_count < max_steps:
        step_count += 1

        if done:
            h_0 = torch.zeros((1, 512), dtype=torch.float32)
            c_0 = torch.zeros((1, 512), dtype=torch.float32)
        else:
            h_0 = h_0.detach()
            c_0 = c_0.detach()

        if torch.cuda.is_available():
            h_0 = h_0.cuda()
            c_0 = c_0.cuda()
            state = state.cuda()

        with torch.no_grad():
            logits, _, h_0, c_0 = model(state, h_0, c_0)
            policy = F.softmax(logits, dim=1)
            
            if use_sampling:
                # Sample from policy for exploration variety
                m = Categorical(policy)
                action = m.sample().item()
            else:
                # Deterministic action selection
                action = torch.argmax(policy).item()

        np_state, _, done, info = env.step(action)
        frame = render_frame(env)
        frames.append(frame)

        actions.append(action)
        if done:
            state = torch.from_numpy(env.reset())
        else:
            state = torch.from_numpy(np_state)

        if (
            actions.maxlen
            and len(actions) == actions.maxlen
            and actions.count(actions[-1]) == actions.maxlen
        ):
            break

        if info.get("flag_get"):
            break

    return frames


def generate_multiple_rollouts(model, world, stage, action_type, num_windows=8, steps_per_window=200):
    """Generate frames from multiple parallel rollouts, each with different randomness.
    
    Returns a list of (window_id, frames) tuples.
    """
    import random
    
    # Ensure we're in rgb_array mode (no window rendering)
    original_render_mode = os.environ.get("RENDER_MODE")
    os.environ["RENDER_MODE"] = "rgb_array"
    
    rollouts = []
    try:
        for i in range(num_windows):
            # Create fresh env for each rollout to ensure independence
            env, _, _ = create_train_env(world, stage, action_type)
            seed = 42 + i * 1000 + random.randint(0, 999)
            frames = generate_frames(
                model, 
                env, 
                max_steps=steps_per_window,
                stuck_tolerance=steps_per_window,  # Don't abort on repeated actions
                seed=seed,
                use_sampling=True  # Use stochastic sampling like the observer windows
            )
            rollouts.append((i, frames))
            try:
                env.close()
            except:
                pass
    finally:
        # Restore original render mode if it existed
        if original_render_mode:
            os.environ["RENDER_MODE"] = original_render_mode
    
    return rollouts


def perform_training_steps(model, env, optimizer, num_steps=50):
    """Perform a few training steps and return metrics."""
    from torch.distributions import Categorical
    
    model.train()
    total_loss = 0
    total_reward = 0
    
    for _ in range(num_steps):
        state = torch.from_numpy(env.reset())
        done = False
        h_0 = torch.zeros((1, 512), dtype=torch.float)
        c_0 = torch.zeros((1, 512), dtype=torch.float)
        
        log_policies = []
        values = []
        rewards = []
        entropies = []
        
        # Collect experience
        for step in range(20):  # Short episodes
            logits, value, h_0, c_0 = model(state, h_0, c_0)
            policy = F.softmax(logits, dim=1)
            log_policy = F.log_softmax(logits, dim=1)
            entropy = -(policy * log_policy).sum(1, keepdim=True)
            
            m = Categorical(policy)
            action = m.sample().item()
            action_ix = torch.tensor([[action]])
            selected_log_prob = log_policy.gather(1, action_ix)
            
            np_state, reward, done, _ = env.step(action)
            state = torch.from_numpy(np_state)
            
            values.append(value)
            log_policies.append(selected_log_prob)
            rewards.append(torch.tensor([[reward]], dtype=torch.float))
            entropies.append(entropy)
            total_reward += reward
            
            if done:
                break
        
        # Compute loss and update
        R = torch.zeros((1, 1), dtype=torch.float)
        if not done:
            _, R, _, _ = model(state, h_0, c_0)
        
        actor_loss = 0
        critic_loss = 0
        entropy_loss = 0
        next_value = R
        
        for value, log_policy, reward, entropy in list(zip(values, log_policies, rewards, entropies))[::-1]:
            R = R * 0.9 + reward
            advantage = R - value
            actor_loss = actor_loss + log_policy * advantage.detach()
            critic_loss = critic_loss + advantage.pow(2) / 2
            entropy_loss = entropy_loss + entropy
        
        loss = -actor_loss + critic_loss - 0.01 * entropy_loss
        total_loss += loss.item()
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    
    model.eval()
    return total_loss / num_steps, total_reward / num_steps


def capture_live_frames(model, world, stage, action_type, num_windows=6, steps=100):
    """Capture frames from multiple live rollouts using current model."""
    import random
    from torch.distributions import Categorical
    
    os.environ["RENDER_MODE"] = "rgb_array"
    rollouts = []
    
    for i in range(num_windows):
        env, _, _ = create_train_env(world, stage, action_type)
        # Ensure seed is within valid range (0 to 2^32 - 1)
        seed = (int(time.time() * 1000) + i) % (2**32)
        
        torch.manual_seed(seed)
        random.seed(seed)
        np.random.seed(seed)
        
        frames = []
        state = torch.from_numpy(env.reset())
        done = True
        h_0 = torch.zeros((1, 512), dtype=torch.float32)
        c_0 = torch.zeros((1, 512), dtype=torch.float32)
        
        for step in range(steps):
            if done:
                h_0 = torch.zeros((1, 512), dtype=torch.float32)
                c_0 = torch.zeros((1, 512), dtype=torch.float32)
                state = torch.from_numpy(env.reset())
            
            with torch.no_grad():
                logits, _, h_0, c_0 = model(state, h_0, c_0)
                policy = F.softmax(logits, dim=1)
                m = Categorical(policy)
                action = m.sample().item()
            
            np_state, _, done, info = env.step(action)
            frame = render_frame(env)
            frames.append(frame)
            
            if not done:
                state = torch.from_numpy(np_state)
            
            if info.get("flag_get"):
                break
        
        rollouts.append((i, frames))
        env.close()
    
    return rollouts


def frames_to_gif_bytes(frames, fps=15):
    """Convert frames to an animated GIF in memory."""
    if not frames:
        return None
    
    safe_frames = []
    for frame in frames:
        if frame.dtype != np.uint8:
            frame = np.clip(frame * 255, 0, 255).astype(np.uint8)
        if len(frame.shape) == 3 and frame.shape[2] == 4:
            frame = cv2.cvtColor(frame, cv2.COLOR_RGBA2RGB)
        safe_frames.append(frame)
    
    with NamedTemporaryFile(delete=False, suffix=".gif") as tmp_file:
        tmp_path = tmp_file.name
    
    try:
        # Create animated GIF with imageio
        imageio.mimsave(
            tmp_path, 
            safe_frames, 
            fps=fps, 
            loop=0  # 0 = loop forever
        )
        with open(tmp_path, "rb") as file:
            data = file.read()
    finally:
        os.remove(tmp_path)
    
    return data


def frames_to_video_bytes(frames, fps=15):
    if not frames:
        return None

    safe_frames = []
    for frame in frames:
        if frame.dtype != np.uint8:
            frame = np.clip(frame * 255, 0, 255).astype(np.uint8)
        if frame.shape[2] == 4:
            frame = cv2.cvtColor(frame, cv2.COLOR_RGBA2RGB)
        safe_frames.append(frame)

    with NamedTemporaryFile(delete=False, suffix=".mp4") as tmp_file:
        tmp_path = tmp_file.name

    try:
        imageio.mimsave(tmp_path, safe_frames, fps=fps, codec="libx264")
        with open(tmp_path, "rb") as file:
            data = file.read()
    finally:
        os.remove(tmp_path)

    return data


# ==================== UI LAYOUT ==================== #

st.markdown(
    """
    <style>
    .main-header {
        font-size: 3rem;
        font-weight: 800;
        color: #E52521;
        text-align: center;
        margin-bottom: 0.5rem;
        text-shadow: 2px 2px 6px rgba(0,0,0,0.35);
    }
    .sub-header {
        font-size: 1.4rem;
        color: #ffd700;
        text-align: center;
        margin-bottom: 2rem;
    }
    .description-box {
        background: rgba(255,255,255,0.05);
        border-left: 6px solid #E52521;
        padding: 1.5rem;
        border-radius: 10px;
        margin-bottom: 2rem;
    }
    /* Compact styling for multi-window grid */
    [data-testid="column"] img {
        border: 2px solid rgba(229, 37, 33, 0.3);
        border-radius: 4px;
        image-rendering: pixelated;
    }
    [data-testid="column"] [data-testid="caption"] {
        font-size: 0.75rem;
        text-align: center;
        color: #888;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown('<div class="main-header">🎮 Super Mario Bros Neural Training</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="sub-header">A reincarnation of the classic—now powered by reinforcement learning</div>',
    unsafe_allow_html=True,
)

st.markdown(
    """
    <div class="description-box">
        <p>
            Welcome! This project resurrects the legendary Super Mario Bros adventure by teaching a neural network
            how to play. Using the Async Advantage Actor-Critic (A3C) algorithm, multiple agents explore alternate
            realities of the Mushroom Kingdom in parallel and share their discoveries.
        </p>
        <p>
            Choose from two visualization modes: <strong>Live Training Monitor</strong> trains the network from scratch 
            and updates observer windows every few seconds so you watch Mario learn in real-time. 
            <strong>View Training Windows</strong> shows 8-10 animated gameplay clips using a pre-trained model.
        </p>
    </div>
    """,
    unsafe_allow_html=True,
)

st.sidebar.header("⚙️ Configuration")
world = st.sidebar.selectbox("World", list(range(1, 9)), index=0)
stage = st.sidebar.selectbox("Stage", [1, 2, 3, 4], index=0)
action_type = st.sidebar.selectbox("Action Type", ["complex", "simple", "right"], index=0)

display_mode = st.sidebar.radio(
    "Display Mode",
    ["Live Training Monitor", "View Training Windows"],
    help="Live Training: Watch the AI learn in real-time | Training Windows: Pre-recorded simulations"
)

if display_mode == "Live Training Monitor":
    st.sidebar.markdown("### 🔴 Live Training Settings")
    num_train_processes = st.sidebar.slider(
        "Training workers",
        min_value=2,
        max_value=8,
        value=4,
        step=1,
        help="Number of parallel A3C training processes"
    )
    num_observer_windows = st.sidebar.slider(
        "Observer windows",
        min_value=4,
        max_value=12,
        value=6,
        step=1,
        help="Number of live gameplay windows to display"
    )
    refresh_interval = st.sidebar.slider(
        "Refresh interval (sec)",
        min_value=3,
        max_value=15,
        value=5,
        step=1,
        help="How often to update the observer windows"
    )
    frames_per_window = st.sidebar.slider(
        "Frames per capture",
        min_value=50,
        max_value=150,
        value=80,
        step=10,
        help="Number of frames to capture for each window update"
    )
elif display_mode == "View Training Windows":
    num_windows = st.sidebar.slider(
        "Number of windows",
        min_value=4,
        max_value=12,
        value=8,
        step=1,
        help="Number of parallel simulation windows to display (each shows animated gameplay)"
    )
    steps_per_window = st.sidebar.slider(
        "Steps per window",
        min_value=100,
        max_value=400,
        value=200,
        step=25,
        help="How many frames each simulation captures (more = longer video)"
    )
    fps = st.sidebar.slider(
        "Animation FPS",
        min_value=10,
        max_value=30,
        value=15,
        step=5,
        help="Frames per second for the animated windows"
    )
# Initialize session state for live training
if 'training_active' not in st.session_state:
    st.session_state.training_active = False
if 'global_model' not in st.session_state:
    st.session_state.global_model = None
if 'training_step' not in st.session_state:
    st.session_state.training_step = 0

# Handle Live Training Mode separately (doesn't need pre-trained model)
if display_mode == "Live Training Monitor":
    st.markdown("### 🔴 Live Training Mode")
    st.info("⚡ Train the neural network from scratch and watch it learn in real-time!")
    
    col1, col2, col3 = st.columns([1, 1, 2])
    
    with col1:
        if st.button("▶️ Start Training", type="primary", disabled=st.session_state.training_active):
            # Initialize training
            os.environ["RENDER_MODE"] = "rgb_array"
            env, num_states, num_actions = create_train_env(world, stage, action_type)
            st.session_state.global_model = ActorCritic(num_states, num_actions)
            st.session_state.training_env = env
            st.session_state.optimizer = torch.optim.Adam(st.session_state.global_model.parameters(), lr=1e-4)
            st.session_state.training_active = True
            st.session_state.training_step = 0
            st.session_state.total_reward = 0
            st.session_state.avg_loss = 0
            st.rerun()
    
    with col2:
        if st.button("⏹️ Stop Training", disabled=not st.session_state.training_active):
            st.session_state.training_active = False
            # Save current model
            if st.session_state.global_model:
                save_path = f"trained_models/A3CSuperMarioBros_{world}_{stage}"
                torch.save(st.session_state.global_model.state_dict(), save_path)
                st.success(f"✅ Model saved to {save_path}")
            st.rerun()
    
    with col3:
        if st.session_state.training_active:
            st.success("🟢 Training Active")
        else:
            st.error("🔴 Training Stopped")
    
    # Display live observer windows
    if st.session_state.training_active and st.session_state.global_model:
        st.markdown("---")
        
        # Perform training steps
        with st.spinner("⚡ Performing training updates..."):
            avg_loss, avg_reward = perform_training_steps(
                st.session_state.global_model,
                st.session_state.training_env,
                st.session_state.optimizer,
                num_steps=10  # Do 10 training steps between observer updates
            )
            st.session_state.avg_loss = avg_loss
            st.session_state.total_reward += avg_reward
        
        # Show metrics
        metric_cols = st.columns(3)
        with metric_cols[0]:
            st.metric("Training Step", st.session_state.training_step)
        with metric_cols[1]:
            st.metric("Avg Loss", f"{st.session_state.avg_loss:.4f}")
        with metric_cols[2]:
            st.metric("Cumulative Reward", f"{st.session_state.total_reward:.1f}")
        
        st.markdown("---")
        st.markdown("### 🎮 Live Observer Windows")
        st.caption(f"Showing current policy behavior. Updates every {refresh_interval} seconds.")
        
        # Capture and display observer windows
        with st.spinner(f"Capturing {num_observer_windows} live rollouts..."):
            rollouts = capture_live_frames(
                st.session_state.global_model,
                world, stage, action_type,
                num_windows=num_observer_windows,
                steps=frames_per_window
            )
        
        cols_per_row = 3
        for row_start in range(0, len(rollouts), cols_per_row):
            cols = st.columns(cols_per_row)
            for col_idx, (window_id, frames) in enumerate(rollouts[row_start:row_start + cols_per_row]):
                if frames and len(frames) > 0:
                    with cols[col_idx]:
                        gif_bytes = frames_to_gif_bytes(frames, fps=12)
                        if gif_bytes:
                            st.image(
                                gif_bytes,
                                caption=f"🎮 Observer {window_id + 1}",
                                use_container_width=True
                            )
        
        st.session_state.training_step += 1
        
        # Auto-refresh
        st.info(f"🔄 Training continues... Next update in {refresh_interval} seconds")
        time.sleep(refresh_interval)
        st.rerun()

# For other modes, require pre-trained model
else:
    model_env = load_model(world, stage, action_type)

    if not model_env:
        st.warning(
            f"No trained checkpoint found in `trained_models/A3CSuperMarioBros_{world}_{stage}`. "
            f"Train the agent locally with `python train.py --world {world} --stage {stage}` "
            "and refresh this page."
        )
    else:
        model, env = model_env

        if display_mode == "View Training Windows":
            button_label = f"🚀 Launch {num_windows} Training Windows"
        else:
            button_label = "🚀 Generate Training Screens"

        if st.button(button_label, type="primary"):
            if display_mode == "View Training Windows":
                with st.spinner(f"Generating {num_windows} parallel simulations..."):
                    rollouts = generate_multiple_rollouts(
                        model, world, stage, action_type, 
                        num_windows=num_windows, 
                        steps_per_window=steps_per_window
                    )

                if not rollouts:
                    st.error("Failed to generate rollouts.")
                else:
                    total_frames = sum(len(frames) for _, frames in rollouts)
                    st.success(f"Generated {num_windows} simulations with {total_frames} total frames!")

                    st.markdown("### 🎮 Multi-Window Training Monitor")
                    st.caption(
                        f"Each window shows animated gameplay from a different stochastic rollout. "
                        f"Like the Python command with {num_windows} live observer windows. "
                        f"Watch Mario move differently in each window due to random action sampling!"
                    )
                    
                    # Display in grid with 4 columns for better sizing
                    cols_per_row = 4
                    for row_start in range(0, len(rollouts), cols_per_row):
                        cols = st.columns(cols_per_row)
                        for col_idx, (window_id, frames) in enumerate(rollouts[row_start:row_start + cols_per_row]):
                            if frames and len(frames) > 0:
                                with cols[col_idx]:
                                    # Convert frames to animated GIF
                                    with st.spinner(f"Rendering window {window_id + 1}..."):
                                        gif_bytes = frames_to_gif_bytes(frames, fps=fps)
                                        if gif_bytes:
                                            st.image(
                                                gif_bytes,
                                                caption=f"🎮 Window {window_id + 1} ({len(frames)} frames)",
                                                use_container_width=True
                                            )
                                        else:
                                            st.error(f"Failed to render window {window_id + 1}")
                    
                    # Show summary stats
                    avg_steps = total_frames / num_windows
                    duration_seconds = avg_steps / fps
                    st.info(
                        f"📊 **Stats:** {num_windows} windows • "
                        f"{total_frames} total frames • "
                        f"~{duration_seconds:.1f}s per window @ {fps} FPS"
                    )

st.markdown("---")
st.caption("Built with PyTorch, Gym Super Mario Bros, and Streamlit • A3C Agent Showcase")

