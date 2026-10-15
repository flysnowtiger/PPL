import torch
from torch import nn
import torch.utils.model_zoo as model_zoo
from torch.nn import functional as F

from models.backbone.resnet import *

from models.backbone.resnets1 import resnet50_s1
from models.PPL_KeyModule.STR import SpatialTemporalArgularization


from functools import partial

def weights_init_kaiming(m):
	classname = m.__class__.__name__
	if classname.find('Linear') != -1:
		nn.init.kaiming_normal_(m.weight, a=0, mode='fan_out')
		nn.init.constant_(m.bias, 0.0)
	elif classname.find('Conv') != -1:
		nn.init.kaiming_normal_(m.weight, a=0, mode='fan_in')
		if m.bias is not None:
			nn.init.constant_(m.bias, 0.0)
	elif classname.find('BatchNorm') != -1:
		if m.affine:
			nn.init.constant_(m.weight, 1.0)
			nn.init.constant_(m.bias, 0.0)


def weight_init_classifier(m):
	classname = m.__class__.__name__
	if classname.find('Linear') != -1:
		nn.init.normal_(m.weight, std=0.001)
		if m.bias is not None:
			nn.init.constant_(m.bias, 0.0)


class ResNet50_PCB(nn.Module):

	def __init__(self, num_classes, pretrained_model_dir, seq_len, model_mode, str_mode, part_num, layer, visual,**kwargs):
		super(ResNet50_PCB, self).__init__()

		self.in_planes = 2048

		resnet2d = resnet50_s1(pretrained=True)
		self.base_layer0 = nn.Sequential(resnet2d.conv1,
				                         resnet2d.bn1,
				                         nn.ReLU(inplace=True),
				                         resnet2d.maxpool,
				                         )

		self.base_layer1 = resnet2d.layer1
		self.base_layer2 = resnet2d.layer2
		self.base_layer3 = resnet2d.layer3
		self.base_layer4 = resnet2d.layer4

		self.seq_len = seq_len
		self.num_classes = num_classes
		self.base_dim = 2048

		self.avg_2d = nn.AdaptiveAvgPool2d((1, 1))
		self.avg_3d = nn.AdaptiveAvgPool3d((1, 1, 1))

		self.bottleneck_base = nn.BatchNorm1d(self.base_dim)
		self.bottleneck_base.bias.requires_grad_(False)
		self.bottleneck_base.apply(weights_init_kaiming)
		self.classifier_base = nn.Linear(self.base_dim, num_classes)
		self.classifier_base.apply(weight_init_classifier)

		########################### New Spatial-Temporal Transformer
		self.token_num = 4
		self.reduce_dim = 512
		self.spa_dim = self.reduce_dim * self.token_num
		self.patch_num = 16*8

		self.part_linears = nn.ModuleList()
		for i in range(self.token_num):
			self.part_linears.append(
				nn.Sequential(nn.Linear(self.base_dim, self.reduce_dim, bias=False),
								nn.ReLU(inplace=True))
			    )

		self.bottleneck_token = nn.BatchNorm1d(self.reduce_dim*self.token_num)
		self.bottleneck_token.bias.requires_grad_(False)
		self.bottleneck_token.apply(weights_init_kaiming)
		self.classifier_token = nn.Linear(self.reduce_dim*self.token_num, num_classes)
		self.classifier_token.apply(weight_init_classifier)

		self.part_regularization = SpatialTemporalArgularization(margin='soft', distance='cosine',
																 mode=str_mode)


	def forward(self, x):
		b, t, c, w, h = x.size()
		x = x.view(b * t, c, w, h)

		x0 = self.base_layer0(x)
		x1 = self.base_layer1(x0)
		x2 = self.base_layer2(x1)
		x3 = self.base_layer3(x2)
		x4 = self.base_layer4(x3) # (b * t, c, 16, 8)

		h = x4.size(2)
		w = x4.size(3)

		feat_list = []
		BN_feat_list = []
		cls_score_list = []

		f_base = self.avg_2d(x4).reshape(b,t,-1).mean(1)
		feat_list.append(f_base)
		BN_f_base = self.bottleneck_base(f_base)
		BN_feat_list.append(BN_f_base)
		cls_score_base = self.classifier_base(BN_f_base)
		cls_score_list.append(cls_score_base)

		#### PCB
		part_feat = F.adaptive_avg_pool2d(x4, (self.token_num, 1)).reshape(b*t, self.base_dim, self.token_num)

		feat_token = []
		for i in range(self.token_num):
			part_pcb = self.part_linears[i](part_feat[:,:,i])
			feat_token.append(part_pcb)

		feat_token = torch.stack(feat_token, dim=1) ### bt*token_num*reduce_dim
		feat_token = feat_token.reshape(b,t,self.token_num, self.reduce_dim)

		regular_loss = 0.1*self.part_regularization(feat_token)

		f_token = feat_token.mean(1).flatten(1)
		feat_list.append(f_token)
		BN_f_token = self.bottleneck_token(f_token)
		BN_feat_list.append(BN_f_token)
		cls_score_token = self.classifier_token(BN_f_token)
		cls_score_list.append(cls_score_token)

		# torch.cuda.empty_cache()
		if self.training:
			return cls_score_list, BN_feat_list, regular_loss
		else:
			f_test = []
			f_base = BN_feat_list[0]
			f_test.append(f_base)
			f_final = BN_feat_list[-1]
			f_test.append(f_final)
			f_cat = torch.cat((f_base, f_final), dim=1)
			f_test.append(f_cat)
			return f_test

