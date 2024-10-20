"""
Models
"""
import os

import torch
import torch.nn.functional as F
from torch import load, nn
from torchvision import models
from networks import FCNNEncoder


def create_model(config):
	"""
	Parse the configuration file and return a relevant model
	"""
	if config['model'] == 'black-box':
		return BlackBox(config)
	elif config['model'] == 'cbm':
		return CBM(config)
	else:
		print("Could not create model with name ", config["model"], "!")
		quit()


class Identity(nn.Module):
	"""
	Generates identity block as layer of a model
	"""
	def __init__(self):
		super(Identity, self).__init__()

	def forward(self, x):
		return x


class BlackBox(nn.Module):
	"""
	Standard black-box neural network classifier
	"""

	def __init__(self, config):
		super(BlackBox, self).__init__()

		# Configuration arguments
		self.name = config['model']
		self.device = config['device']
		self.num_covariates = config['num_covariates']
		self.num_hidden_z = config['num_hidden_z']
		self.num_hidden_y = config['num_hidden_y']
		self.num_classes = config['num_classes']
		self.encoder_arch = config['encoder_arch']
		self.activation_z = config['act_z']
		self.head_arch = config['head_arch']

		# Architectures
		# Encoder h(.)
		if self.dataset in ('SD_ImageNet','CIFAR10'):
			self.encoder = nn.Sequential(nn.Linear(config['emb_size'], int(config['emb_size']/2), bias=False), nn.ReLU(),
										 nn.Linear(int(config['emb_size']/2), self.num_hidden_z, bias=False), ).to(
				config['device'])
		elif config['encoder_arch'] == 'FCNN':
			self.encoder = FCNNEncoder(num_inputs=self.num_covariates, num_outputs=self.num_hidden_z, num_hidden=256,
									   num_deep=2).to(config['device'])
		elif config['encoder_arch'] == 'resnet18':
			self.encoder_res = models.resnet18(pretrained=False)
			self.encoder_res.load_state_dict(
				torch.load(os.path.join(config['model_directory'], 'resnet18-5c106cde.pth')))
			self.encoder_res.to(config['device'])
			n_features = self.encoder_res.fc.in_features
			self.projector = nn.Sequential(nn.Linear(n_features, n_features, bias=False), nn.ReLU(),
										   nn.Linear(n_features, self.num_hidden_z, bias=False), ).to(config['device'])
			self.encoder_res.fc = Identity()
			self.encoder = nn.Sequential(self.encoder_res, self.projector)
		elif config['encoder_arch'] == 'inceptionv3':
			self.encoder_res = models.inception_v3(pretrained=False)
			self.encoder_res.load_state_dict(
				torch.load(os.path.join(config['model_directory'], 'inception_v3_google-0cc3c7bd.pth')))
			self.encoder_res.to(config['device'])
			n_features = self.encoder_res.fc.in_features
			self.projector = nn.Sequential(nn.Linear(n_features, int(n_features/2), bias=False), nn.ReLU(),
										   nn.Linear(int(n_features/2), self.num_hidden_z, bias=False), ).to(config['device'])
			self.encoder_res.fc = Identity()
			self.encoder = nn.Sequential(self.encoder_res, self.projector)

		else:
			NotImplementedError('ERROR: architecture not supported!')

		if config['act_z'] == 'ReLU':
			self.act_z = torch.relu
		else:
			NotImplementedError('ERROR: activation function not supported!')

		# Link function g(.)
		if config['head_arch'] == 'FCNN':
			if config['concatenation']:
				self.fc1_y = nn.Linear(self.num_hidden_z + config['num_concepts'], self.num_hidden_y)
			else:
				self.fc1_y = nn.Linear(self.num_hidden_z, self.num_hidden_y)
			if self.num_classes == 2:
				self.fc2_y = nn.Linear(self.num_hidden_y, 1)
				self.act_y = nn.Sigmoid()
			elif self.num_classes > 2:
				self.fc2_y = nn.Linear(self.num_hidden_y, self.num_classes)
				self.act_y = nn.Softmax(dim=1)
		else:
			NotImplementedError('ERROR: architecture not supported!')

	def forward(self, x, z_=None, conc =None):
		"""
		Forward pass

		:param x: covariates
		:param z_: intervened representations (optional, overrides the network's representations if given)
		:return: representations, predicted probabilities and logits for the target variable
		"""

		# Get intermediate representations
		if z_ is None:
			if self.encoder_arch == 'inceptionv3':
				if self.training:
					interm = self.encoder_res(x)[0]
				else:
					interm = self.encoder_res(x)
				z = self.act_z(self.projector(interm))
			else:
				z = self.act_z(self.encoder(x))
		# Intervene if necessary
		else:
			z = z_
		if conc is None:
			input = z

		else:
			# Concatenate concepts to the embedding
			input = torch.cat((z, conc), dim=1)
		# Get predicted targets
		y_pred_logits = F.relu(self.fc1_y(input))
		y_pred_logits = self.fc2_y(y_pred_logits)
		y_pred_probs = self.act_y(y_pred_logits)

		return z, y_pred_probs, y_pred_logits


class CBM(nn.Module):
	"""
	Vanilla CBM
	"""

	def __init__(self, config):
		super(CBM, self).__init__()

		# Configuration arguments
		self.name = config['model']
		self.device = config['device']
		self.num_covariates = config['num_covariates']
		self.num_concepts = config['num_concepts']
		self.num_hidden_y = config['num_hidden_y']
		self.num_classes = config['num_classes']
		self.encoder_arch = config['encoder_arch']
		self.head_arch = config['head_arch']

		# Architectures
		# Encoder h(.)
		if config['encoder_arch'] == 'FCNN':
			self.encoder = FCNNEncoder(num_inputs=self.num_covariates, num_outputs=self.num_concepts, num_hidden=256,
									   num_deep=2).to(config['device'])
		elif config['encoder_arch'] == 'resnet18':
			self.encoder_res = models.resnet18(pretrained=False)
			self.encoder_res.load_state_dict(
				torch.load(os.path.join(config['model_directory'], 'resnet18-5c106cde.pth')))
			self.encoder_res.to(config['device'])

			n_features = self.encoder_res.fc.in_features
			self.projector = nn.Sequential(nn.Linear(n_features, n_features, bias=False), nn.ReLU(),
										   nn.Linear(n_features, self.num_concepts, bias=False), ).to(config['device'])
			self.encoder_res.fc = Identity()
			self.encoder = nn.Sequential(self.encoder_res, self.projector)
		elif config['encoder_arch'] == 'inceptionv3':
			self.encoder_res = models.inception_v3(pretrained=False)
			self.encoder_res.load_state_dict(
				torch.load(os.path.join(config['model_directory'], 'inception_v3_google-0cc3c7bd.pth')))
			self.encoder_res.to(config['device'])
			n_features = self.encoder_res.fc.in_features
			self.projector = nn.Sequential(nn.Linear(n_features, int(n_features/2), bias=False), nn.ReLU(),
										   nn.Linear(int(n_features/2), self.num_concepts, bias=False), ).to(config['device'])
			self.encoder_res.fc = Identity()
			self.encoder = nn.Sequential(self.encoder_res, self.projector)

		else:
			NotImplementedError('ERROR: architecture not supported!')

		# Assume binary concepts
		self.act_c = nn.Sigmoid()

		# Link function g(.)
		if config['head_arch'] == 'FCNN':
			bn_y = nn.BatchNorm1d(self.num_concepts)
			fc1_y = nn.Linear(self.num_concepts, self.num_hidden_y)
			if self.num_classes == 2:
				fc2_y = nn.Linear(self.num_hidden_y, 1)
				self.act_y = nn.Sigmoid()
			elif self.num_classes > 2:
				fc2_y = nn.Linear(self.num_hidden_y, self.num_classes)
				self.act_y = nn.Softmax(dim=1)
			self.head = nn.Sequential(fc1_y, nn.ReLU(), fc2_y)
		else:
			NotImplementedError('ERROR: architecture not supported!')

	def forward(self, x, c_=None):
		"""
		Forward pass

		:param x: covariates
		:param z_: intrervened concepts c'
		:return: predicted concepts, probabilities and logits for the target variable
		"""

		# Get intermediate representations
		if c_ is None:
			c = self.act_c(self.encoder(x))
		# Intervene if necessary
		else:
			c = c_

		# Get predicted targets
		y_pred_logits = self.head(c)
		y_pred_probs = self.act_y(y_pred_logits)

		return c, y_pred_probs, y_pred_logits


class pCBM(CBM):
	"""
	Adjusted version of the post hoc CBM model (directly uses representations instead of concept activation vectors)
	"""

	def __init__(self, config, backbone):
		super(CBM, self).__init__()

		# Configuration arguments
		self.name = 'pCBM'
		self.device = config['device']
		self.num_covariates = config['num_covariates']
		self.num_hidden_z = config['num_hidden_z']
		self.num_hidden_y = config['num_hidden_y']
		self.num_classes = config['num_classes']
		self.encoder_arch = config['encoder_arch']
		self.activation_z = config['act_z']
		self.head_arch = config['head_arch']
		self.num_concepts = config['num_concepts']

		# Architectures
		if config['encoder_arch'] == 'inceptionv3':
			self.encoder_res = models.inception_v3(pretrained=False)
			self.encoder_res.load_state_dict(
				torch.load(os.path.join(config['model_directory'], 'inception_v3_google-0cc3c7bd.pth')))
			self.encoder_res.to(config['device'])
			n_features = self.encoder_res.fc.in_features
			self.projector = nn.Sequential(nn.Linear(n_features, int(n_features / 2), bias=False), nn.ReLU(),
										   nn.Linear(int(n_features / 2), self.num_hidden_z, bias=False), ).to(
				config['device'])
			self.encoder_res.fc = Identity()
			self.encoder = nn.Sequential(self.encoder_res, self.projector)
		else:
			self.encoder = backbone
		self.encoder.to(config['device'])
		self.probe = nn.Linear(self.num_hidden_z, self.num_concepts)
		self.probe.to(config['device'])
		self.act_c = nn.Sigmoid()

		# Link function g(.)
		if config['head_arch'] == 'FCNN':
			self.fc1_y = nn.Linear(self.num_concepts, self.num_hidden_y)
			self.fc1_y.to(config['device'])
			if self.num_classes == 2:
				self.fc2_y = nn.Linear(self.num_hidden_y, 1)
				self.act_y = nn.Sigmoid()
			elif self.num_classes > 2:
				self.fc2_y = nn.Linear(self.num_hidden_y, self.num_classes)
				self.act_y = nn.Softmax(dim=1)
			self.fc2_y.to(config['device'])
		else:
			NotImplementedError('ERROR: architecture not supported!')

		# Residual
		self.residual_layer = None

	def forward(self, x, c_=None):
		"""
		Forward pass

		:param x: covariates
		:param z_: intrervened concepts c'
		:return: predicted concepts, probabilities and logits for the target variable
		"""

		# Get intermediate representations
		if c_ is None:
			if self.encoder_arch == 'inceptionv3':
				if self.encoder.training:
					interm = self.encoder_res(x)[0]
					c = self.act_c(self.probe(torch.relu(self.projector(interm))))
				else:
					interm = self.encoder_res(x)
					c = self.act_c(self.probe(torch.relu(self.projector(interm))))
			else:
				c = self.act_c(self.probe(torch.relu(self.encoder(x))))
		# Intervene if necessary
		else:
			c = c_

		# Get predicted targets
		y_pred_logits = torch.relu(self.fc1_y(c))
		y_pred_logits = self.fc2_y(y_pred_logits)
		if self.residual_layer is not None:
			r = self.residual_layer(torch.relu(self.encoder(x)))
			y_pred_logits = y_pred_logits + r
		y_pred_probs = self.act_y(y_pred_logits)

		return c, y_pred_probs, y_pred_logits
