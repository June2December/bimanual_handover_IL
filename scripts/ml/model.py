import torch.nn as nn

class BCPolicy(nn.Module):
    def __init__(self, state_dim, arm_dim, grip_dim, hidden_dim=256):
        super().__init__()

        self.backbone = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )

        self.arm_head = nn.Linear(hidden_dim, arm_dim)
        self.grip_head = nn.Linear(hidden_dim, grip_dim)

    def forward(self, x):
        feat = self.backbone(x)
        return self.arm_head(feat), self.grip_head(feat)