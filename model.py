import torch.nn as nn
import torch.nn.functional as F

"""
| Component      | Purpose                                  |
| -------------- | ---------------------------------------- |
| Convs (Conv2d) | Extract spatial features from images     |
| LSTMCell       | Adds memory across time                  |
| Actor head     | Outputs action probabilities             |
| Critic head    | Outputs estimated value of current state |
| Custom init    | Improves training stability              |

"""

class ActorCritic(nn.Module):
  def __init__(self, num_inputs, num_actions):
    super(ActorCritic, self).__init__()
    # convolutional layers used to extract features from image-based observations. Each one performs a 2D convolution
    self.conv1 = nn.Conv2d(num_inputs, 32, 3, stride=2, padding=1)
    self.conv2 = nn.Conv2d(32, 32, 3, stride=2, padding=1)
    self.conv3 = nn.Conv2d(32, 32, 3, stride=2, padding=1)
    self.conv4 = nn.Conv2d(32, 32, 3, stride=2, padding=1)

    # After the convolutions, the model uses an LSTMCell to process temporal information (i.e., memory or recurrent state)
    # 32 * 6 * 6: input size after flattening conv outputs (assuming input is something like 84×84)
    # 512: size of hidden and cell states
    self.lstm = nn.LSTMCell(32 * 6 * 6, 512)

    self.critic_linear = nn.Linear(512, 1) # critic outputs a single value
    self.actor_linear = nn.Linear(512, num_actions) # Outputs a vector of size num_actions—unnormalized probabilities for each action.
    self._initialize_weights()

  def _initialize_weights(self):
    for module in self.modules():
      if isinstance(module, nn.Conv2d) or isinstance(module, nn.Linear):
        # Uses Xavier initialization for conv and linear layers, which helps maintain good signal propagation during training.
        nn.init.xavier_uniform_(module.weight)
        # Sets all biases to zero.
        nn.init.constant_(module.bias, 0)
      elif isinstance(module, nn.LSTMCell):
        # For LSTM, zeros out both input-to-hidden (bias_ih) and hidden-to-hidden (bias_hh) biases
        nn.init.constant_(module.bias_ih, 0)
        nn.init.constant_(module.bias_hh, 0)

  def forward(self, x, hx, cx):
    # Input x (batch of observations) goes through conv layers with ReLU activations.
    x = F.relu(self.conv1(x))
    x = F.relu(self.conv2(x))
    x = F.relu(self.conv3(x))
    x = F.relu(self.conv4(x))
    # flatten and feed to LSTM input
    x = x.view(x.size(0), -1) 
    hx, cx = self.lstm(x, (hx, cx))

    # Returns: policy_logits, value, next_hx, next_cx
    return self.actor_linear(hx), self.critic_linear(hx), hx, cx