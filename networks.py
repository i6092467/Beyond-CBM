"""
Neural network architectures
"""
import torch.nn.functional as F
from torch import load, nn


class FCNNEncoder(nn.Module):
	"""
	Extracts a vectorial representation using a simple fully connected network
	"""

	def __init__(self, num_inputs: int, num_outputs: int, num_hidden: int, num_deep: int):
		super(FCNNEncoder, self).__init__()

		self.fc0 = nn.Linear(num_inputs, num_hidden)
		self.bn0 = nn.BatchNorm1d(num_hidden)
		self.fcs = nn.ModuleList([nn.Linear(num_hidden, num_hidden) for _ in range(num_deep)])
		self.bns = nn.ModuleList([nn.BatchNorm1d(num_hidden) for _ in range(num_deep)])
		self.dp = nn.Dropout(0.05)
		self.out = nn.Linear(num_hidden, num_outputs)

		self.embedding_dim = num_outputs

	def forward(self, x):
		z = self.bn0(self.dp(F.relu(self.fc0(x))))
		for bn, fc in zip(self.bns, self.fcs):
			z = bn(self.dp(F.relu(fc(z))))
		return self.out(z)


class FCNNProbe(nn.Module):
	"""
	An MLP probe to predict concepts from representations
	"""
	def __init__(self, num_inputs: int, num_outputs: int, num_hidden: int, num_deep: int, activation: str = 'sigmoid'):
		super(FCNNProbe, self).__init__()

		self.fc0 = nn.Linear(num_inputs, num_hidden)
		self.fcs = nn.ModuleList([nn.Linear(num_hidden, num_hidden) for _ in range(num_deep)])
		self.out = nn.Linear(num_hidden, num_outputs)
		if activation == 'sigmoid':
			self.act_out = nn.Sigmoid()
		else:
			NotImplementedError('Probe activation function not supported!')

		self.out_dim = num_outputs

	def forward(self, z):
		c = F.relu(self.fc0(z))
		for fc in self.fcs:
			c = F.relu(fc(c))
		c = self.out(c)
		return c, self.act_out(c)


class LinearProbe(nn.Module):
	"""
	A linear probe to predict concepts from representations
	"""
	def __init__(self, num_inputs: int, num_outputs: int, activation: str = 'sigmoid'):
		super(LinearProbe, self).__init__()
		self.out = nn.Linear(num_inputs, num_outputs)
		if activation == 'sigmoid':
			self.act_out = nn.Sigmoid()
		else:
			NotImplementedError('Probe activation function not supported!')

		self.out_dim = num_outputs

	def forward(self, z):
		c = self.out(z)
		return c, self.act_out(c)
