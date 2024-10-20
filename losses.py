"""
Utility methods for constructing loss functions
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

from torch import Tensor
from typing import Optional


def create_loss(config):
	"""
	Parse configuration file and return a relevant loss function
	"""
	if config['model'] == 'black-box' and config['num_classes'] == 2:
		return nn.BCELoss(reduction='mean')
	elif config['model'] == 'black-box' and config['num_classes'] > 2:
		if config.get('weight_loss'):
			return nn.CrossEntropyLoss(weight=config['weight'], reduction='mean')
		else:
			return nn.CrossEntropyLoss(reduction='mean')
	elif config['model'] == 'cbm':
		return CBLoss(num_classes=config['num_classes'], reduction='mean', alpha=config['alpha'], config=config)


class CBLoss(nn.Module):
	"""
	Loss function for the concept bottleneck model
	"""

	def __init__(
			self,
			num_classes: Optional[int] = 2,
			reduction: str = 'mean',
			alpha: float = 1,
			config: dict = {}) -> None:
		"""
		Initializes the loss object

		@param num_classes: the number of the classes of the target variable
		@param reduction: reduction to apply to the output of the CE loss
		@param alpha: parameter controlling the trade-off between the target and concept prediction during the joint
						optimization. The higher the @alpha, the higher the weight of the concept prediction loss
		"""
		super(CBLoss, self).__init__()
		self.num_classes = num_classes
		self.reduction = reduction
		self.alpha = alpha
		self.config = config


	def forward(self, concepts_pred: Tensor, concepts_true: Tensor,
				target_pred_probs: Tensor, target_pred_logits: Tensor, target_true: Tensor) -> Tensor:
		"""
		Computes the loss for the given predictions

		@param concepts_pred: predicted concept values
		@param concepts_true: ground-truth concept values
		@param target_pred_probs: predicted probabilities, aka normalized logits, for the target variable
		@param target_pred_logits: predicted logits for the target variable
		@param target_true: ground-truth target variable values
		@return: target prediction loss, a tensor of prediction losses for each of the concepts, summed concept
					prediction loss and the total loss
		"""

		summed_concepts_loss = 0
		concepts_loss = []

		# NOTE: all concepts are assumed to be binary
		for concept_idx in range(concepts_true.shape[1]):
			c_loss = F.binary_cross_entropy(
				concepts_pred[:, concept_idx], concepts_true[:, concept_idx].float(), reduction=self.reduction)
			concepts_loss.append(c_loss)
			summed_concepts_loss += c_loss

		if self.num_classes == 2:
			if self.config.get('weight_loss'):
				loss_pos = self.config['weight'][1] * target_true * torch.log(target_pred_probs)
				loss_neg = (1 - target_true) * torch.log(1 - target_pred_probs)
				target_loss = torch.mean(-(loss_pos + loss_neg))
			else:
				target_loss = F.binary_cross_entropy(
					target_pred_probs, target_true, reduction=self.reduction)
		else:
			if self.config.get('weight_loss'):
				target_loss = F.cross_entropy(
					target_pred_logits, target_true.long(), weight = self.config['weight'], reduction=self.reduction)
			else:
				target_loss = F.cross_entropy(
					target_pred_logits, target_true.long(), reduction=self.reduction)

		total_loss = target_loss + self.alpha * summed_concepts_loss

		return target_loss, concepts_loss, summed_concepts_loss, total_loss
