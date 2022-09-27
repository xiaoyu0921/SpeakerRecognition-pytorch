import torch
import torch.nn as nn

from utils_ import myconfig


class BaseSpeakerEncoder(nn.Module):
    # 从硬盘中去加载一个已经训练好的模型
    def _load_from(self, saved_model):
        var_dict = torch.load(saved_model, map_location=myconfig.DEVICE)
        self.load_state_dict(var_dict["encoder_state_dict"])