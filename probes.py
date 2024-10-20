"""
Utility functions for probing
"""
import numpy as np
import torch

from sklearnex import patch_sklearn

patch_sklearn()

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from torch import nn
from networks import FCNNProbe, LinearProbe
from tqdm import tqdm
import os


def create_probe(model, config):
	"""
	Parse the configuration file and return a differentiable concept probe for the given model
	"""
	if config['model'] == 'black-box' and config['probe']:
		if config['probe_type'] == 'linear':
			return LinearProbe(num_inputs=model.num_hidden_z, num_outputs=config['num_concepts'])
		else:
			return FCNNProbe(num_inputs=model.num_hidden_z, num_outputs=config['num_concepts'], num_hidden=256,
							 num_deep=1, activation='sigmoid')
	else:
		return None


def train_torch_lin_probe_post_hoc(model, train_loader, config, num_epochs_probe, learning_rate_probe,
								   weight_decay_probe, optimizer_probe='sgd'):
	"""
	Train a linear probing function post hoc using PyTorch
	"""
	lin_probe = LinearProbe(num_inputs=model.num_hidden_z, num_outputs=config['num_concepts'])
	lin_probe.to(config['device'])
	lin_probe.train()

	# NOTE: assumes that concepts are binary-valued
	loss_fn = nn.BCELoss()

	assert optimizer_probe in ['sgd', 'adam'], 'Only SGD and Adam optimizers are available!'
	optim_params = [
		{'params': filter(lambda p: p.requires_grad, lin_probe.parameters()), 'lr': learning_rate_probe,
		 'weight_decay': weight_decay_probe}
	]
	if optimizer_probe == 'sgd':
		optimizer = torch.optim.SGD(optim_params)
	elif optimizer_probe == 'adam':
		optimizer = torch.optim.Adam(optim_params)

	print()
	print('TRAINING LINEAR PROBE POST HOC IN PYTORCH')
	print()

	for epoch in range(0, num_epochs_probe):
		running_len = 0
		running_total_loss = 0

		with tqdm(total=len(train_loader) * config['train_batch_size'], desc=f'Epoch {epoch + 1}/{num_epochs_probe}',
				  unit='data points', position=0, leave=True) as pbar:

			for k, batch in enumerate(train_loader):
				batch_features, target_true = batch['features'].float().to(config['device']), \
					batch['labels'].float().to(config['device'])  # put the data on the device
				concepts_true = batch['concepts'].float().to(config['device'])

				# Forward pass
				z, target_pred_probs, target_pred_logits = model(batch_features)
				target_pred_probs = target_pred_probs.squeeze(1)
				target_pred_logits = target_pred_logits.squeeze(1)

				concepts_pred_logits, concepts_pred_probs = lin_probe(z)

				# Backward pass depends on the training mode of the model
				optimizer.zero_grad()
				# Compute the loss
				total_loss = loss_fn(concepts_pred_probs, concepts_true)

				running_total_loss += total_loss.item() * batch_features.size(0)

				running_len += batch_features.size(0)
				total_loss.backward()
				optimizer.step()  # perform an update

				# Update the progress bar
				pbar.set_postfix(
					**{'Total loss': running_total_loss / running_len, 'lr': optimizer.param_groups[0]['lr']})

				pbar.update(config['train_batch_size'])

	lin_probe.eval()
	torch.save(lin_probe.state_dict(), os.path.join(os.path.join(config['log_directory'], 'checkpoints'),
													'linear_probe' +  config['run_name'] + '_' +
													config['experiment_name'] + '_' + str(config['seed']) + '.pth'))
	# Disable gradients
	for param in lin_probe.parameters():
		param.requires_grad = False
	return lin_probe


def train_torch_nlin_probe_post_hoc(model, train_loader, config, num_epochs_probe, learning_rate_probe,
									weight_decay_probe, optimizer_probe='sgd'):
	"""
	Train a nonlinear probing function post hoc using PyTorch
	"""
	nlin_probe = FCNNProbe(num_inputs=model.num_hidden_z, num_outputs=config['num_concepts'], num_deep=1,
						   num_hidden=int((model.num_hidden_z + config['num_concepts']) / 2))
	nlin_probe.to(config['device'])
	nlin_probe.train()

	# NOTE: assumes that concepts are binary-valued
	loss_fn = nn.BCELoss()

	assert optimizer_probe in ['sgd', 'adam'], 'Only SGD and Adam optimizers are available!'
	optim_params = [
		{'params': filter(lambda p: p.requires_grad, nlin_probe.parameters()), 'lr': learning_rate_probe,
		 'weight_decay': weight_decay_probe}
	]
	if optimizer_probe == 'sgd':
		optimizer = torch.optim.SGD(optim_params)
	elif optimizer_probe == 'adam':
		optimizer = torch.optim.Adam(optim_params)

	print()
	print('TRAINING NONLINEAR PROBE POST HOC IN PYTORCH')
	print()

	for epoch in range(0, num_epochs_probe):
		running_len = 0
		running_total_loss = 0

		with tqdm(total=len(train_loader) * config['train_batch_size'], desc=f'Epoch {epoch + 1}/{num_epochs_probe}',
				  unit='data points', position=0, leave=True) as pbar:

			for k, batch in enumerate(train_loader):
				batch_features, target_true = batch['features'].float().to(config['device']), \
					batch['labels'].float().to(config['device'])  # put the data on the device
				concepts_true = batch['concepts'].float().to(config['device'])

				# Forward pass
				z, target_pred_probs, target_pred_logits = model(batch_features)
				target_pred_probs = target_pred_probs.squeeze(1)
				target_pred_logits = target_pred_logits.squeeze(1)

				concepts_pred_logits, concepts_pred_probs = nlin_probe(z)

				# Backward pass depends on the training mode of the model
				optimizer.zero_grad()
				# Compute the loss
				total_loss = loss_fn(concepts_pred_probs, concepts_true)

				running_total_loss += total_loss.item() * batch_features.size(0)

				running_len += batch_features.size(0)
				total_loss.backward()
				optimizer.step()  # perform an update

				# Update the progress bar
				pbar.set_postfix(
					**{'Total loss': running_total_loss / running_len, 'lr': optimizer.param_groups[0]['lr']})

				pbar.update(config['train_batch_size'])

	torch.save(nlin_probe.state_dict(), os.path.join(config['log_directory'], 'checkpoints',
													 'nonlinear_probe' + config['run_name'] + '_' +
													 config['experiment_name'] + '_' + str(config['seed']) + '.pth'))

	nlin_probe.eval()
	# Disable gradients
	for param in nlin_probe.parameters():
		param.requires_grad = False
	return nlin_probe


def train_lr_probes_post_hoc(model, train_loader, config):
	"""
	Train and test linear concept probes for the given model, using logistic regression from scikit-learn

	:param model: model to probe
	:param train_loader: train data loader
	:param config: configuration file with further arguments
	:return:
	"""
	print()
	print('Training linear probes...')

	zs_train = np.zeros((0, model.num_hidden_z))
	cs_train = np.zeros((0, config['num_concepts']))
	for k, batch in enumerate(train_loader):
		z, y_pred_probs, y_pred_logits = model(batch['features'].float().to(config['device']))
		zs_train = np.vstack((zs_train, z.cpu().numpy()))
		cs_train = np.vstack((cs_train, batch['concepts'].float().to(config['device']).cpu().numpy()))

	# Fit a logistic regression model for each concept variable
	probes = []
	for j in range(config['num_concepts']):
		probe_j = LogisticRegression(max_iter=500)
		probe_j.fit(zs_train, cs_train[:, j])
		probes.append(probe_j)

	return probes


def train_rf_probes_post_hoc(model, train_loader, config):
	"""
	Train and test nonlinear concept probes for the given model, using Random Forest from scikit-learn

	:param model: model to probe
	:param train_loader: train data loader
	:param config: configuration file with further arguments
	:return:
	"""
	print()
	print('Training nonlinear probes...')

	zs_train = np.zeros((0, model.num_hidden_z))
	cs_train = np.zeros((0, config['num_concepts']))
	for k, batch in enumerate(train_loader):
		z, y_pred_probs, y_pred_logits = model(batch['features'].float().to(config['device']))
		zs_train = np.vstack((zs_train, z.cpu().numpy()))
		cs_train = np.vstack((cs_train, batch['concepts'].float().to(config['device']).cpu().numpy()))

	# Fit an RF probe for each concept variable
	probes = []
	for j in range(config['num_concepts']):
		probe_j = RandomForestClassifier()
		probe_j.fit(zs_train, cs_train[:, j])
		probes.append(probe_j)

	return probes
