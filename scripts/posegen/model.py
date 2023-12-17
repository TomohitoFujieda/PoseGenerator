import torch
import torch.nn as nn

class keypoint_estimator(nn.Module):
  def __init__(self):
    super().__init__()
    self.l = nn.Linear(10, 2)

  def forward(self, frm, emo):
    x = torch.cat([frm, emo], dim=1)
    x = self.l(x)
    return x