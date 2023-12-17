import torch
import torch.nn as nn

class keypoint_estimator_angle(nn.Module):
  def __init__(self):
    super().__init__()
    self.l = nn.Linear(1, 1)
  def forward(self, emo):
    x = self.l(emo)
    x = nn.functional.tanh(x)
    return x