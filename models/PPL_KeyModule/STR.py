import numpy
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
from torch.autograd import Variable
import numpy as np
import torchvision


class SpatialTemporalArgularization(nn.Module):
	"""Triplet loss with hard positive/negative mining.

	Reference:
	Hermans et al. In Defense of the Triplet Loss for Person Re-Identification. arXiv:1703.07737.

	Code imported from https://github.com/Cysu/open-reid/blob/master/reid/loss/triplet.py.

	Args:
		margin (float): margin for triplet.
	"""

	def __init__(self, margin=0.3, distance='cosine',  mode='maxmin'):
		super(SpatialTemporalArgularization, self).__init__()
		if distance not in ['euclidean', 'cosine']:
			raise KeyError("Unsupported distance: {}".format(distance))
		self.distance = distance
		self.margin = margin
		self.mode = mode
		self.ranking_loss = nn.MarginRankingLoss(margin=margin)


	def forward(self, x_tensor):
		# The complete implementation will be provided after paper acceptance.
		raise NotImplementedError(
			"Forward implementation withheld during anonymous review; "
			"the complete code will be released after acceptance."
		)
