import torch.nn as nn
import torch

from models.PPL_KeyModule.AR import AttentionRegularization

class RegionActivator(nn.Module):

    def __init__(self, in_dim=2048, part_dim=512, part_num=4):
        super(RegionActivator, self).__init__()

        self.in_dim = in_dim
        self.part_dim = part_dim
        self.part_num = part_num

        self.feat_conv = nn.Sequential(
            nn.Conv2d(in_channels=self.in_dim, out_channels=self.part_dim, kernel_size=(1, 1), stride=(1, 1),
                      padding=0, bias=False),
            nn.BatchNorm2d(self.part_dim),
            nn.ReLU(inplace=True)
        )

        self.atte_conv = nn.Sequential(
            nn.Conv2d(in_channels=self.in_dim, out_channels=self.part_num, kernel_size=(1, 1), stride=(1, 1),
                      padding=0, bias=False),
            nn.BatchNorm2d(self.part_num),
            nn.Sigmoid()
        )

        self.attention_regularization = AttentionRegularization(num_part=self.part_num)

    def forward(self, x_base):
        # The complete implementation will be provided after paper acceptance.
        raise NotImplementedError(
            "Forward implementation withheld during anonymous review; "
            "the complete code will be released after acceptance."
        )
