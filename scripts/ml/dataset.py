import numpy as np
import torch
from torch.utils.data import Dataset

class BCDataset(Dataset):
    def __init__(
        self,
        df,
        state_cont_cols,
        state_bin_cols,
        phase_cols,
        arm_action_cont_cols,
        grip_action_cols,
        state_mean,
        state_std,
        action_mean,
        action_std,
    ):
        # state continuous
        state_cont = df[state_cont_cols].values.astype(np.float32)
        state_cont = (state_cont - state_mean) / state_std

        # state binary
        state_bin = df[state_bin_cols].values.astype(np.float32)

        # phase one-hot
        phase = df[phase_cols].values.astype(np.float32)

        # final state
        self.state = np.concatenate([state_cont, state_bin, phase], axis=1)

        # arm action
        arm_action = df[arm_action_cont_cols].values.astype(np.float32)
        arm_action = (arm_action - action_mean) / action_std
        self.arm_action = arm_action

        # grip action
        self.grip_action = df[grip_action_cols].values.astype(np.float32)

    def __len__(self):
        return len(self.state)

    def __getitem__(self, idx):
        return {
            "state": torch.tensor(self.state[idx], dtype=torch.float32),
            "arm_action": torch.tensor(self.arm_action[idx], dtype=torch.float32),
            "grip_action": torch.tensor(self.grip_action[idx], dtype=torch.float32),
        }