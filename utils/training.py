"""
PyTorch utility functions
"""
from torch import nn


def set_bn_to_eval(m):
	if isinstance(m, nn.BatchNorm2d):
		m.eval()


def freeze_module(m):
	m.eval()
	for param in m.parameters():
		param.requires_grad = False


def unfreeze_module(m):
	m.train()
	for param in m.parameters():
		param.requires_grad = True