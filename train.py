import os
os.environ["OMP_NUM_THREADS"] = "1"
import argparse
import torch
from env import create_train_env
from model import ActorCritic
from optimizer import GlobalAdam
from process import local_train, local_test
import torch.multiprocessing as _mp
import shutil


def get_args():
  parser = argparse.ArgumentParser("""Implementation of model described in the paper: Asynchronous Methods for Deep Reinforcement Learning for Super Mario Bros""")
  parser.add_argument("--world", type=int, default=1)
  parser.add_argument("--stage", type=int, default=1)
  parser.add_argument("--action_type", type=str, default="complex")
  parser.add_argument("--lr", type=float, default=1e-4)
  parser.add_argument("--gamma", type=float, default=0.9, help="discount factor for rewards")
  parser.add_argument("--tau", type=float, default=0.1, help="parameter for GAE")
  parser.add_argument("--beta", type=float, default=0.01, help="entropy coefficient")
  parser.add_argument("--num_local_steps", type=int, default=50)
  parser.add_argument("--num_global_steps", type=int, default=int(5e6))
  parser.add_argument("--num_processes", type=int, default=6)
  parser.add_argument("--save_interval", type=int, default=500, help="Number of steps between savings")
  parser.add_argument("--max_actions", type=int, default=200, help="Maximum repetition steps in test phase")
  parser.add_argument("--log_path", type=str, default="tensorboard/A3CSuperMarioLogs")
  parser.add_argument("--saved_path", type=str, default="trained_models")
  parser.add_argument("--num_observers", type=int, default=20,
                      help="Number of evaluation windows (local_test processes) to spawn")
  parser.add_argument("--load_from_previous_stage", action="store_true",
                      help="Load weight from previous trained stage")
  parser.add_argument("--use_gpu", action="store_true")
  args = parser.parse_args()
  return args


def train(opt):
  torch.manual_seed(123)
  if os.path.isdir(opt.log_path):
    shutil.rmtree(opt.log_path)
  os.makedirs(opt.log_path)
  if not os.path.isdir(opt.saved_path):
    os.makedirs(opt.saved_path)
  mp = _mp.get_context("spawn")
  env, num_states, num_actions = create_train_env(opt.world, opt.stage, opt.action_type)
  global_model = ActorCritic(num_states, num_actions)
  if opt.use_gpu:
    global_model.cuda()
  global_model.share_memory()
  if opt.load_from_previous_stage:
    if opt.stage == 1:
      previous_world = opt.world - 1
      previous_stage = 4
    else:
      previous_world = opt.world
      previous_stage = opt.stage - 1
    file_ = "{}/A3CSuperMarioBros_{}_{}".format(opt.saved_path, previous_world, previous_stage)
    if os.path.isfile(file_):
      global_model.load_state_dict(torch.load(file_))

  optimizer = GlobalAdam(global_model.parameters(), lr=opt.lr)
  processes = []
  print(f"Spawning {opt.num_processes} training workers...")
  for index in range(opt.num_processes):
    if index == 0:
      process = mp.Process(target=local_train, args=(index, opt, global_model, optimizer, True))
    else:
      process = mp.Process(target=local_train, args=(index, opt, global_model, optimizer))
    process.start()
    processes.append(process)
  print(f"Spawning {opt.num_observers} observer windows...")
  for observer_idx in range(opt.num_observers):
    observer_process = mp.Process(
        target=local_test,
        args=(opt.num_processes + observer_idx, opt, global_model)
    )
    observer_process.start()
    processes.append(observer_process)
  for process in processes:
    process.join()


if __name__ == "__main__":
  opt = get_args()
  train(opt)
  print("training finished")


  # python train.py --world 1 --stage 1 --action_type complex --saved_path trained_models --log_path tensorboard/A3CSuperMarioLogs