# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import torch
from torch import nn
import os.path as osp
import os
import torch.nn.functional as F
from functools import partial

from models.backbone.resnet import *
from models.backbone.vision_transformer import VisionTransformer

from models.PPL_KeyModule.PG import RegionActivator
from models.PPL_KeyModule.PR import ContextualRefiner
from models.PPL_KeyModule.PE import TemporalEnhancedTransformer
from models.PPL_KeyModule.STR import SpatialTemporalArgularization

# ===================
#   Initialization
# ===================

def weight_init_classifier(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        nn.init.normal_(m.weight,std=0.001)
        if m.bias is not None:
            nn.init.constant_(m.bias,0.0)

def weight_init_kaiming(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        nn.init.kaiming_normal_(m.weight,a=0,mode='fan_out')
        nn.init.constant_(m.bias,0.0)
    elif classname.find('Conv') != -1:
        nn.init.kaiming_normal_(m.weight,a=0,mode='fan_in')
        if m.bias is not None:
            nn.init.constant_(m.bias,0.0)
    elif classname.find('BatchNorm') != -1:
        if m.affine:
            nn.init.constant_(m.weight,1.0)
            nn.init.constant_(m.bias,0.0)

def init_pretrained_weight(model,  pretrained_path):
    """Initializes model with pretrained weight

    Layers that don't match with pretrained layers in name or size are kept unchanged
    """
    # pretrain_dict = model_zoo.load_url(model_url, model_dir = './')
    pretrain_model = os.path.join(pretrained_path, 'resnet50-19c8e357.pth')
    pretrain_dict = torch.load(pretrain_model)

    model_dict = model.state_dict()
    pretrain_dict = {k: v for k,v in pretrain_dict.items() if k in model_dict and model_dict[k].size() == v.size()}
    model_dict.update(pretrain_dict)
    model.load_state_dict(model_dict)

class ProgressivePartialLearning(nn.Module):

    def __init__(self, num_classes, pretrained_model_dir, seq_len, model_mode, str_mode, part_num, layer, lambda_1, lambda_2, visual, **kwargs):
        super(ProgressivePartialLearning, self).__init__()

        self.model_mode = model_mode
        self.str_mode = str_mode
        self.layer = layer
        self.lambda_1 = lambda_1
        self.lambda_2 = lambda_2

        self.visual = visual
        self.num_classes = num_classes
        self.base_dim = 2048
        resnet50 = ResNet()

        init_pretrained_weight(resnet50, pretrained_path=pretrained_model_dir)
        print('Loading pretrained ImageNet model .........')

        self.cnn_backbone = nn.Sequential(resnet50.conv1,
                                    resnet50.bn1,
                                    resnet50.relu,
                                    resnet50.maxpool,
                                    resnet50.layer1,
                                    resnet50.layer2,
                                    resnet50.layer3,
                                    resnet50.layer4,
                                    )

        self.bottleneck_base = nn.BatchNorm1d(self.base_dim)
        self.classifier_base = nn.Linear(self.base_dim, self.num_classes, bias=False)
        self.bottleneck_base.apply(weight_init_kaiming)
        self.classifier_base.apply(weight_init_classifier)

        self.part_dim = 512
        self.part_num = part_num
        self.seq_len = seq_len

        if 'pg' in self.model_mode:
            self.part_generation = RegionActivator(in_dim=self.base_dim, part_dim=self.part_dim,
                                                          part_num=self.part_num)
        if 'pr' in self.model_mode:
            self.part_refinement = ContextualRefiner(in_dim=self.part_dim, out_dim=self.part_dim,
                                                     part_num=self.part_num, seq_len=self.seq_len)
        if 'pe' in self.model_mode:
            self.part_enhancement = TemporalEnhancedTransformer(
                    seq_len=self.seq_len, token_num=0, part_num= part_num, in_dim=self.part_dim, embed_dim=self.part_dim,
                    depth=2, num_heads=8, mlp_ratio=1, qkv_bias=True, norm_layer=partial(nn.LayerNorm, eps=1e-6))

            # self.lstm_block = nn.ModuleList()
            # for i in range(self.part_num):
            #     self.lstm_block.append(nn.LSTM(input_size=self.part_dim, hidden_size=self.part_dim, num_layers=2, bidirectional=False, batch_first=True))
            # self.cg_block = nn.ModuleList()
            # for i in range(self.part_num):
            #     context_gating = nn.Sequential(
            #     nn.Linear(self.part_dim, self.part_dim, bias=True),
            #     nn.Sigmoid(),
            #     )
            #     self.cg_block.append(context_gating)

            # self.transformer_block = nn.ModuleList()
            # for i in range(self.part_num):
            #     temporal_transformer = VisionTransformer(
            #                     token_num=self.seq_len, patch_size= self.seq_len, in_dim=self.part_dim, embed_dim=self.part_dim,
            #                     depth=2, num_heads=8, mlp_ratio=4, qkv_bias=True, norm_layer=partial(nn.LayerNorm, eps=1e-6)
            #                     )
            #     self.transformer_block.append(temporal_transformer)



        if 'str' in self.model_mode:
            self.part_regularization = SpatialTemporalArgularization(margin='soft', distance='cosine',
                                                                      mode=str_mode)

        self.bottleneck_part = nn.BatchNorm1d(self.part_dim * self.part_num)
        self.classifier_part = nn.Linear(self.part_dim * self.part_num, self.num_classes, bias=False)
        self.bottleneck_part.apply(weight_init_kaiming)
        self.classifier_part.apply(weight_init_classifier)


    def forward(self, inputs):
        # The complete forward implementation will be released after acceptance.
        raise NotImplementedError(
            "The forward implementation is withheld during anonymous review."
        )
