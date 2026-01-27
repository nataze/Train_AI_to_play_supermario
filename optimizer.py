import torch

# A custom optimizer class called GlobalAdam, which is a modified version of PyTorch’s built-in Adam optimizer
class GlobalAdam(torch.optim.Adam):
  def __init__(self, params, lr):
    # Ensure optimizer state is created immediately
    super(GlobalAdam, self).__init__(params, lr=lr)
    # for each parameter p, this code modifies optimizer state
    for group in self.param_groups:
      for p in group["params"]:
        state = self.state[p]
        state["step"] = 0
        state["exp_avg"] = torch.zeros_like(p.data)
        state["exp_avg_sq"] = torch.zeros_like(p.data)

        # Allows multiple processes (workers) to access and modify the same tensors in shared memory.
        state["exp_avg"].share_memory_()
        state["exp_avg_sq"].share_memory_()