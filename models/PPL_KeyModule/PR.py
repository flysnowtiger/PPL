import torch.nn as nn
import torch
import torch.nn.functional as F

class ContextualRefiner(nn.Module):

    def __init__(self, in_dim=512, out_dim=512, part_num=4, seq_len=8):
        super(ContextualRefiner, self).__init__()

        self.in_dim = in_dim
        self.out_dim = out_dim
        self.part_num = part_num
        self.seq_len = seq_len
        # self.map_project = nn.Sequential(nn.Linear(2048, in_dim, bias=False),
        #                                  )



        self.part_linears = nn.ModuleList()
        self.part_encoders = nn.ModuleList()
        for i in range(self.part_num):
            linear = nn.Linear(in_dim, in_dim, bias=False)

            self.part_linears.append(linear)
            encoder = nn.Sequential(nn.Linear(in_dim*3, in_dim, bias=False),  ## change here
                                    nn.ReLU(),
                                    nn.Linear(in_dim, out_dim, bias=False))
            self.part_encoders.append(encoder)

    def forward(self, x_part, x_map):
        # The complete implementation will be provided after paper acceptance.
        raise NotImplementedError(
            "Forward implementation withheld during anonymous review; "
            "the complete code will be released after acceptance."
        )
