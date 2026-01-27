import os
import gym_super_mario_bros
import warnings
warnings.filterwarnings(
    "ignore",
    message=".*Gym has been unmaintained since 2022.*",
    category=UserWarning,
)
import gym
from gym.spaces import Box
from gym import Wrapper
from nes_py.wrappers import JoypadSpace
from gym_super_mario_bros.actions import SIMPLE_MOVEMENT, COMPLEX_MOVEMENT, RIGHT_ONLY
import cv2
import numpy as np
import subprocess as sp

# Record a sequence of image frames and encode them in to a video file using ffmpeg
class Monitor:
  def __init__(self, width, height, saved_path):

    self.command = ["ffmpeg", "-y", "-f", "rawvideo", "-vcodec", "rawvideo", "-s", "{}X{}".format(width, height),
                    "-pix_fmt", "rgb24", "-r", "80", "-i", "-", "-an", "-vcodec", "mpeg4", saved_path]
    self.pipe = None
    try:
      self.pipe = sp.Popen(self.command, stdin=sp.PIPE, stderr=sp.PIPE)
    except FileNotFoundError:
      warnings.warn(
          "ffmpeg not found. Video recording is disabled. "
          "Install ffmpeg or ensure it is on PATH to export gameplay MP4s.",
          RuntimeWarning,
      )

  def record(self, image_array):
    # coverts numpy array (e.g an image) to raw bytes which are written directly to ffmpeg's input stream
    if not self.pipe or self.pipe.stdin.closed:
      return
    try:
      self.pipe.stdin.write(image_array.tobytes())
    except (BrokenPipeError, OSError):
      self.pipe = None

# preprocess frame before it is analyzed
def process_frame(frame):
  if isinstance(frame, np.ndarray) and frame.size > 0:
    frame = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY) # convert frame to grayscale
    frame = cv2.resize(frame, (84, 84))[None, :, :] / 255. # rezie frame, then add channel dimension, the normalize pixel values to btw 0 and 1
    return frame
  else:
    return np.zeros((1, 84, 84)) # else frame is invalid, return blank 84x84 zero array
  
class CustomReward(Wrapper):
  def __init__(self, env=None, monitor=None):
    super(CustomReward, self).__init__(env)
    self.observation_space = Box(low=0, high=255, shape=(1, 84, 84), dtype=np.float32)
    self.curr_score = 0
    self.curr_x_pos = 0
    if monitor:
      self.monitor = monitor
    else:
      self.monitor = None

  def step(self, action):
    state, reward, terminated, truncated, info = self.env.step(action)
    done = False
    if terminated or truncated:
      done = True
    if self.monitor:
      self.monitor.record(state)
    state = process_frame(state)
    
    # Reward for score increase
    reward += (info["score"] - self.curr_score) / 40
    self.curr_score = info["score"]
    
    # Reward for moving right (encourage forward progress)
    x_pos = info.get("x_pos", 0)
    reward += (x_pos - self.curr_x_pos) / 10
    self.curr_x_pos = x_pos
    
    if done:
      if info["flag_get"]:
        reward += 50
      else:
        reward -= 50
    
    return state, reward / 10., done, info
  
  def reset(self):
    self.curr_score = 0
    self.curr_x_pos = 0
    return process_frame(self.env.reset())
  

class CustomSkipFrame(Wrapper):
  def __init__(self, env, skip=4):
    super(CustomSkipFrame, self).__init__(env)
    self.observation_space = Box(low=0, high=255, shape=(4, 84, 84))
    self.skip = skip
  
  def step(self, action):
    total_reward = 0
    states = []
    state, reward, done, info = self.env.step(action)
    for i in range(self.skip):
      if not done:
        state, reward, done, info = self.env.step(action)
        total_reward += reward
        states.append(state)
      else:
        states.append(state)
    states = np.concatenate(states, 0)[None, :, :, :]
    return states.astype(np.float32), reward, done, info
  
  def reset(self):
    state = self.env.reset()
    states = np.concatenate([state for _ in range(self.skip)], 0)[None, :, :, :]
    return states.astype(np.float32)
  

def create_train_env(world, stage, action_type, output_path=None):
  render_mode = os.environ.get("RENDER_MODE", "rgb_array")
  env = gym.make(
      "SuperMarioBros-{}-{}-v0".format(world, stage),
      apply_api_compatibility=True,
      render_mode=render_mode)
  if output_path:
    monitor = Monitor(256, 240, output_path)
  else:
    monitor = None
  
  if action_type == "right":
    actions = RIGHT_ONLY
  elif action_type == "simple":
    actions = SIMPLE_MOVEMENT
  else:
    actions = COMPLEX_MOVEMENT

  env = JoypadSpace(env, actions)
  env = CustomReward(env, monitor)
  env = CustomSkipFrame(env)
  return env, env.observation_space.shape[0], len(actions)