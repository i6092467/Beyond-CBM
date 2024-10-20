"""
Functions for performing concept-based interventions
"""
import os
import torch
import copy
import numpy as np

from torch import nn
from tqdm import tqdm
from utils.metrics import calc_concept_metrics, calc_target_metrics
from utils.training import freeze_module, unfreeze_module

from models import pCBM
from train import _train_one_epoch_cbm, create_optimizer
from losses import CBLoss


class InterventionPolicy(object):
	"""Intervention strategy"""
	def generate_intervention(self, x, c_pred, c, y_pred, y, perc):
		NotImplementedError()


class RandomSubsetInterventionPolicy(InterventionPolicy):
	"""Random subset intervention strategy"""
	def generate_intervention(self, x, c_pred, c, y_pred, y, perc):
		c_mask = generate_random_subset_intervention_mask(x.shape[0], c.shape[1],  int(perc * c.shape[1]))
		c_= c * c_mask + c_pred * (1 - c_mask)
		return c_, c_mask


class UncertaintyInterventionPolicy(InterventionPolicy):
	"""Uncertainty-based intervention strategy"""
	def generate_intervention(self, x, c_pred, c, y_pred, y, perc):
		c_mask = generate_uncertain_subset_intervention_mask(x.shape[0], c.shape[1], int(perc * c.shape[1]), c_pred)
		c_= c * c_mask + c_pred * (1 - c_mask)
		return c_, c_mask


def intervene_on_representations(concept_probe, zs, c_ , intervention_mask, lmbd, config, step_size_inter,
								 weight_decay_inter, num_epochs_inter=1000, optimizer_inter='sgd', eps=1e-6,
								 verbose=1):
	"""Intervention procedure for concept-based instance-specific intervention on black-box neural networks"""
	# Setup
	num_samples = zs.size(0)		# number of data points
	z_dim = zs.size(1)				# dimensionality of the representation
	num_concepts = c_.shape[1]		# number of concepts

	# Original representations, z
	zs = torch.tensor(zs).to(config['device']).float()

	# Intervened representations, z'
	zs_ = nn.Parameter(zs.clone().detach(), requires_grad=True).to(config['device'])

	# Intervened concepts, c'
	c_ = torch.tensor(c_).to(config['device']).float()

	# Mask indicating which concepts to intervene on for which data points
	intervention_mask = torch.tensor(intervention_mask).to(config['device']).float()

	# Concept probe, q(.)
	concept_probe.to(config['device'])

	# Optimisation algorithm
	assert optimizer_inter in ['sgd', 'adam'], 'Only SGD and Adam optimizers are available!'
	optim_params = [{'params': zs_, 'lr': step_size_inter, 'weight_decay': weight_decay_inter}]
	if optimizer_inter == 'sgd':
		optimizer = torch.optim.SGD(optim_params)
	elif optimizer_inter == 'adam':
		optimizer = torch.optim.Adam(optim_params)

	# Concept prediction loss
	loss_fn_concept = nn.BCELoss(reduction='none')

	updateable = torch.ones(num_samples)

	loss_last_epoch = 1000 * torch.ones(num_samples).to(config['device'])
	epoch = 0
	epsilon = eps * lmbd

	while updateable.any() and epoch < num_epochs_inter:
		concepts_pred_logits, concepts_pred_proba = concept_probe(zs_)

		# Intervenability loss with Euclidean distance criterion
		loss_indiv = torch.linalg.norm(zs - zs_, ord=2, dim=1) / z_dim * (1 / 2) + \
					 lmbd * (intervention_mask * loss_fn_concept(concepts_pred_proba, c_)).mean(1)

		# Convergence criterion
		updateable = (loss_last_epoch - loss_indiv) > epsilon

		# NOTE: only update those data points which have not converged
		loss = (loss_indiv * updateable).sum()

		optimizer.zero_grad()
		loss.backward()
		optimizer.step()
		loss_last_epoch = loss_indiv
		epoch += 1

	zs_.requires_grad = False

	if verbose:
		print('Concept loss before intervention  : {:.4f}'.format(loss_fn_concept(concept_probe(zs)[1], c_).mean(1).mean().item()))
		print('Concept loss after intervention   : {:.4f}'.format(loss_fn_concept(concept_probe(zs_)[1], c_).mean(1).mean().item()))
		print('Mean z difference                 : {:.4f}'.format((zs - zs_).abs().mean().item()))
		print('Median z difference               : {:.4f}'.format((zs - zs_).abs().median().item()))
		print('Not converged                     :', updateable.sum().item())

	return zs_


def generate_random_subset_intervention_mask(batch_size, num_concepts, num_concepts_inter):
	"""Masking operation for random subset intervention strategy"""
	a1 = np.ones((1, num_concepts_inter))
	a2 = np.zeros((1, num_concepts - num_concepts_inter))
	m = np.tile(np.hstack((a1, a2)), (batch_size, 1))
	for i in range(batch_size):
		col_inds = np.arange(0, num_concepts)
		np.random.shuffle(col_inds)
		m[i] = m[i, col_inds]
	return m


def generate_uncertain_subset_intervention_mask(batch_size, num_concepts, num_concepts_inter, c_pred_probs):
	"""Masking operation for uncertainty-based intervention strategy"""
	m = np.ones((batch_size, num_concepts))
	for i in range(batch_size):
		uncs = 1 / np.abs(c_pred_probs[i, :] - 0.5)
		uncs_sorted = copy.deepcopy(uncs)
		uncs_sorted.sort()
		t = uncs_sorted[-num_concepts_inter]
		m[i] = (uncs >= t) * 1.
	return m


def evaluate_representation_interventions(model, probe: nn.Module, dataloader, loss_fn_y,
										  intervention_policy: InterventionPolicy, num_steps, num_batches,
										  intervention_params: dict, config: dict):
	"""Utility function for evaluating post hoc interventions on the black-box model"""
	intervened_perc = np.linspace(0, 1, num_steps)

	aurocs = np.zeros((num_batches, len(intervened_perc)))
	auprs = np.zeros((num_batches, len(intervened_perc)))
	losses = np.zeros((num_batches, len(intervened_perc)))

	for i, perc in enumerate(intervened_perc):
		# Generates intervention on an increasing percentage of concepts
		it = iter(dataloader)
		for b in range(num_batches):
			batch = next(it)
			z, y_pred_probs, y_pred_logits = model(batch['features'].float().to(config['device']))
			if config['num_classes'] == 2:
				y_pred_probs = y_pred_probs.squeeze(1)
				y_pred_logits = y_pred_logits.squeeze(1)
			cs = batch['concepts'].float().to(config['device'])
			concepts_pred_logits, concepts_pred_probs = probe(z)
			concepts_pred_logits = concepts_pred_logits.cpu().numpy()
			concepts_pred_probs = concepts_pred_probs.cpu().numpy()
			cs_pred_lin = []
			cs_pred_lin_probs = []

			for j in range(config['num_concepts']):
				cs_pred_lin.append(np.stack((concepts_pred_probs[:, j] > 0.5, concepts_pred_probs[:, j] > 0.5)).T)
				cs_pred_lin_probs.append(np.stack((concepts_pred_probs[:, j], concepts_pred_probs[:, j])).T)

			# Evaluate pre-intervention
			y_metrics = calc_target_metrics(batch['labels'].float().to(config['device']).cpu().numpy(),
											y_pred_probs.cpu().numpy(), config)

			if int(perc * config['num_concepts']) >= 1:
				# Intervene on a percentage of the concepts per data point
				cs_, c_mask = intervention_policy.generate_intervention(
					x=batch['features'].float().to(config['device']), c_pred=concepts_pred_probs, c=cs.cpu().numpy(),
					y_pred=y_pred_probs.cpu().numpy(), y=batch['labels'].float().to(config['device']).cpu().numpy(),
					perc=perc)

				# Intervene via the probe
				zs_ = intervene_on_representations(probe, z, cs, c_mask, lmbd=intervention_params['lmbd'], config=config,
												   step_size_inter=intervention_params['step_size'],
												   weight_decay_inter=intervention_params['weight_decay'],
												   num_epochs_inter=intervention_params['num_epochs'],
												   optimizer_inter=intervention_params['optimizer'],
												   eps=intervention_params['eps'], verbose=0)
			else:
				zs_ = z

			# Evaluate post-intervention
			_, y_pred_probs_, y_pred_logits_ = model(batch['features'].float().to(config['device']), zs_)
			if config['num_classes'] == 2:
				y_pred_probs_ = y_pred_probs_.squeeze(1)
				y_pred_logits_ = y_pred_logits_.squeeze(1)
			y_metrics_ = calc_target_metrics(batch['labels'].float().to(config['device']).cpu().numpy(),
											 y_pred_probs_.cpu().numpy(), config)

			aurocs[b, i] = y_metrics_['AUROC']
			auprs[b, i] = y_metrics_['AUPR']
			if config['num_classes'] ==2:
				losses[b, i] = loss_fn_y(y_pred_probs_, batch['labels'].float().to(config['device'])).cpu().numpy()
			elif config['num_classes'] > 2:
				losses[b, i] = loss_fn_y(y_pred_probs_, batch['labels'].to(config['device'])).cpu().numpy()

	return losses, aurocs, auprs


def evaluate_cbm_interventions(model, dataloader, loss_fn_y, intervention_policy: InterventionPolicy, num_steps,
							   num_batches, config: dict):
	"""Utility function for evaluating interventions on the CBM"""
	intervened_perc = np.linspace(0, 1, num_steps)

	aurocs = np.zeros((num_batches, len(intervened_perc)))
	auprs = np.zeros((num_batches, len(intervened_perc)))
	losses = np.zeros((num_batches, len(intervened_perc)))

	for i, perc in enumerate(intervened_perc):
		it = iter(dataloader)
		# Generates intervention on an increasing percentage of concepts
		for b in range(num_batches):
			batch = next(it)
			cs_pred, y_pred_probs, y_pred_logits = model(batch['features'].float().to(config['device']))
			if config['num_classes'] == 2:
				y_pred_probs = y_pred_probs.squeeze(1)
				y_pred_logits = y_pred_logits.squeeze(1)
			cs = batch['concepts'].float().to(config['device'])
			cs_pred = cs_pred.cpu().numpy()

			if int(perc * config['num_concepts']) >= 1:
				# Intervene on a percentage of the concepts per data point adhering to the given policy
				cs_, c_mask = intervention_policy.generate_intervention(
					x=batch['features'].float().to(config['device']), c_pred=cs_pred, c=cs.cpu().numpy(),
					y_pred=y_pred_probs.cpu().numpy(), y=batch['labels'].float().to(config['device']).cpu().numpy(),
					perc=perc)
			else:
				cs_ = cs_pred

			cs_ = torch.tensor(cs_).float().to(config['device'])

			# Evaluate post-intervention
			_, y_pred_probs_, y_pred_logits_ = model(batch['features'].float().to(config['device']), cs_)
			if config['num_classes'] == 2:
				y_pred_probs_ = y_pred_probs_.squeeze(1)
				y_pred_logits_ = y_pred_logits_.squeeze(1)
			y_metrics_ = calc_target_metrics(batch['labels'].float().to(config['device']).cpu().numpy(),
											 y_pred_probs_.cpu().numpy(), config)

			aurocs[b, i] = y_metrics_['AUROC']
			auprs[b, i] = y_metrics_['AUPR']
			if config['num_classes'] ==2:
				losses[b, i] = loss_fn_y(y_pred_probs_, batch['labels'].float().to(config['device'])).cpu().numpy()
			elif config['num_classes'] > 2:
				losses[b, i] = loss_fn_y(y_pred_probs_, batch['labels'].to(config['device'])).cpu().numpy()

	return losses, aurocs, auprs


def finetune_intervenability_black_box(config, model, probe, loss_fn_y, intervention_policy, data_loader, max_iter=None):
	"""Procedure to fine-tune a black-box neural network for intervenability"""
	unfreeze_module(model)
	model.train()

	optim_params = [
		{'params': filter(lambda p: p.requires_grad, model.parameters()),
		 'lr': config['learning_rate'],
		 'weight_decay': config['weight_decay']}
	]

	if config['optimizer'] == 'sgd':
		optimizer = torch.optim.SGD(optim_params)
	elif config['optimizer'] == 'adam':
		optimizer = torch.optim.Adam(optim_params)

	# ---------------------------------
	# Create a directory for model checkpoints
	# ---------------------------------
	checkpoint_dir = os.path.join(config['log_directory'], 'checkpoints')
	if not os.path.exists(checkpoint_dir):
		os.makedirs(checkpoint_dir)

	# Numbers of training epochs
	n_epochs = config['num_epochs']

	print()
	print('FINE-TUNING FOR INTERVENABILITY')
	print()

	running_len = 0
	running_total_loss = 0

	cnt = 0

	for epoch in range(0, n_epochs):
		with tqdm(total=len(data_loader) * config['train_batch_size'], desc=f'Epoch {epoch + 1}/{n_epochs}',
				  unit='data points', position=0, leave=True) as pbar:
			for k, batch in enumerate(data_loader):

				if max_iter is not None and cnt > max_iter:
					model.eval()
					freeze_module(probe)
					freeze_module(model)
					return model

				cnt += 1

				if config['num_classes'] ==2:
					batch_features, target_true = batch['features'].float().to(config['device']), \
						batch['labels'].float().to(config['device'])  # put the data on the device
				elif config['num_classes'] > 2:
					batch_features, target_true = batch['features'].to(config['device']), \
						batch['labels'].to(config['device'])  # put the data on the device
				concepts_true = batch['concepts'].float().to(config['device'])

				# Forward pass
				z, target_pred_probs, target_pred_logits = model(batch_features)

				concepts_pred_logits__, concepts_pred_probs__ = probe(z)
				concepts_pred_logits = concepts_pred_logits__.detach().cpu().numpy()
				concepts_pred_probs = concepts_pred_probs__.detach().cpu().numpy()

				# Generating interventions
				cs_, c_mask = intervention_policy.generate_intervention(
					x=batch_features, c_pred=concepts_pred_probs,
					c=concepts_true.cpu().numpy(), y_pred = target_pred_probs.detach().cpu().numpy(),
					y=target_true.detach().cpu().numpy(), perc=config['perc'])

				zs_ = intervene_on_representations(
					probe, z, concepts_true, c_mask, lmbd=config['lmbd'], config=config,
					step_size_inter=config['learning_rate_inter'],
					weight_decay_inter=config['weight_decay_inter'],
					num_epochs_inter=config['num_epochs_inter'],
					optimizer_inter=config['optimizer_inter'], eps=config['eps'], verbose=0)
				zs_.requires_grad = True

				# Evaluating interventions
				_, target_pred_probs_, target_pred_logits_ = model(batch_features, zs_)
				target_pred_probs = target_pred_probs.squeeze()
				target_pred_probs_ = target_pred_probs_.squeeze()

				# Backward pass
				optimizer.zero_grad()
				intervenability = loss_fn_y(target_pred_probs_, target_true)
				intervenability.backward()
				optimizer.step()  # perform an update

				running_total_loss += intervenability.item() * batch_features.size(0)
				running_len += batch_features.size(0)

				pbar.set_postfix(
					**{'Total loss': running_total_loss / running_len, 'lr': optimizer.param_groups[0]['lr']})
				pbar.update(config['train_batch_size'])
	if 'run_name' in config:
		torch.save(model.state_dict(), os.path.join(checkpoint_dir, 'final_ft_model_' + config['run_name'] + '_' +
											config['experiment_name'] + '_' + str(config['seed']) + '.pth') )
	freeze_module(model)
	model.eval()
	freeze_module(probe)

	return model, probe


def finetune_with_probe_black_box(config, model, probe, loss_fn_y, loss_fn_c, data_loader, max_iter=None):
	"""Procedure to fine-tune a black-box neural network using the multitask loss"""
	unfreeze_module(model)
	model.train()
	unfreeze_module(probe)
	probe.train()

	optim_params = [
		{'params': list(filter(lambda p: p.requires_grad, model.parameters())) + \
				   list(filter(lambda p: p.requires_grad, probe.parameters())),
		 'lr': config['learning_rate'],
		 'weight_decay': config['weight_decay']}
	]

	if config['optimizer'] == 'sgd':
		optimizer = torch.optim.SGD(optim_params)
	elif config['optimizer'] == 'adam':
		optimizer = torch.optim.Adam(optim_params)

	# ---------------------------------
	# Create a directory for model checkpoints
	# ---------------------------------
	checkpoint_dir = os.path.join(config['log_directory'], 'checkpoints')
	if not os.path.exists(checkpoint_dir):
		os.makedirs(checkpoint_dir)

	# Numbers of training epochs
	n_epochs = config['num_epochs']

	print()
	print('FINE-TUNING MULTITASK')
	print()

	running_len = 0
	running_total_loss = 0
	running_target_loss = 0
	running_concept_loss = 0

	cnt = 0

	for epoch in range(0, n_epochs):
		with tqdm(total=len(data_loader) * config['train_batch_size'], desc=f'Epoch {epoch + 1}/{n_epochs}',
				  unit='data points', position=0, leave=True) as pbar:
			for k, batch in enumerate(data_loader):

				if max_iter is not None and cnt > max_iter:
					model.eval()
					freeze_module(probe)
					freeze_module(model)
					return model

				cnt += 1

				if config['num_classes'] ==2:
					batch_features, target_true = batch['features'].float().to(config['device']), \
						batch['labels'].float().to(config['device'])  # put the data on the device
				elif config['num_classes'] > 2:
					batch_features, target_true = batch['features'].to(config['device']), \
						batch['labels'].to(config['device'])  # put the data on the device
				concepts_true = batch['concepts'].float().to(config['device'])

				# Forward pass
				z, target_pred_probs, target_pred_logits = model(batch_features)
				target_pred_probs = target_pred_probs.squeeze()
				target_pred_logits = target_pred_logits.squeeze()

				concepts_pred_logits, concepts_pred_probs = probe(z)

				# Backward pass
				optimizer.zero_grad()

				concept_loss = loss_fn_c(concepts_pred_probs, concepts_true)
				target_loss = loss_fn_y(target_pred_probs, target_true)
				loss = target_loss + config['alpha'] * concept_loss
				loss.backward()
				optimizer.step()  # perform an update

				running_total_loss += loss.item() * batch_features.size(0)
				running_target_loss += target_loss.item() * batch_features.size(0)
				running_concept_loss += concept_loss.item() * batch_features.size(0)
				running_len += batch_features.size(0)

				pbar.set_postfix(
					**{'Total loss': running_total_loss / running_len, 'Concept loss': running_concept_loss / running_len,
					   'Target loss': running_target_loss / running_len, 'lr': optimizer.param_groups[0]['lr']})
				pbar.update(config['train_batch_size'])

	if 'run_name' in config:
		torch.save(model.state_dict(), os.path.join(checkpoint_dir, 'final_ft_mt_model_' + config['run_name'] + '_' +
													config['experiment_name'] + '_' + str(config['seed']) + '.pth'))
		torch.save(probe.state_dict(),
				   os.path.join(checkpoint_dir, 'probe_final_ft_mt_model_' + config['run_name'] + '_' +
								config['experiment_name'] + '_' + str(config['seed']) + '.pth'))

	freeze_module(model)
	model.eval()
	freeze_module(probe)
	probe.eval()

	return model, probe


def finetune_concatenate_black_box(config, model, loss_fn_y, data_loader, max_iter=None):
	"""Procedure to fine-tune a black-box neural network by appending concepts to its representations"""
	unfreeze_module(model)
	model.train()
	model.encoder.requires_grad = False

	optim_params = [
		{'params': filter(lambda p: p.requires_grad, model.parameters()),
		 'lr': config['learning_rate'],
		 'weight_decay': config['weight_decay']}
	]

	if config['optimizer'] == 'sgd':
		optimizer = torch.optim.SGD(optim_params)
	elif config['optimizer'] == 'adam':
		optimizer = torch.optim.Adam(optim_params)

	# ---------------------------------
	# Create a directory for model checkpoints
	# ---------------------------------
	checkpoint_dir = os.path.join(config['log_directory'], 'checkpoints')
	if not os.path.exists(checkpoint_dir):
		os.makedirs(checkpoint_dir)

	# Numbers of training epochs
	n_epochs = config['num_epochs']

	print()
	print('FINE-TUNING BY CONCATENATING CONCEPTS')
	print()

	running_len = 0
	running_total_loss = 0
	cnt = 0

	for epoch in range(0, n_epochs):
		with tqdm(total=len(data_loader) * config['train_batch_size'], desc=f'Epoch {epoch + 1}/{n_epochs}',
				  unit='data points', position=0, leave=True) as pbar:
			for k, batch in enumerate(data_loader):

				if max_iter is not None and cnt > max_iter:
					model.eval()
					freeze_module(model)
					return model

				cnt += 1

				if config['num_classes'] ==2:
					batch_features, target_true = batch['features'].float().to(config['device']), \
						batch['labels'].float().to(config['device'])  # put the data on the device
				elif config['num_classes'] > 2:
					batch_features, target_true = batch['features'].to(config['device']), \
						batch['labels'].to(config['device'])  # put the data on the device
				concepts_true = batch['concepts'].float().to(config['device'])

				# Intervention by appending concepts
				c_mask = generate_random_subset_intervention_mask(batch_features.shape[0], concepts_true.shape[1],
																  int(config['perc'] * concepts_true.shape[1]))
				c_mask = torch.tensor(c_mask).to(config['device'])
				concepts_true = (concepts_true * c_mask + (1 - c_mask) * 0.5).float().to(config['device'])

				# Forward pass
				z, target_pred_probs, target_pred_logits = model(batch_features, conc=concepts_true)
				target_pred_probs = target_pred_probs.squeeze(1)
				target_pred_logits = target_pred_logits.squeeze(1)

				# Backward pass
				optimizer.zero_grad()

				total_loss = loss_fn_y(target_pred_probs, target_true)
				total_loss.backward()
				running_total_loss += total_loss.item() * batch_features.size(0)

				optimizer.step()  # perform an update

				running_len += batch_features.size(0)

				pbar.set_postfix(
					**{'Total loss': running_total_loss / running_len, 'lr': optimizer.param_groups[0]['lr']})
				pbar.update(config['train_batch_size'])

	if 'run_name' in config:
		torch.save(model.state_dict(), os.path.join(checkpoint_dir, 'final_ft_concat_model_' + config['run_name'] + '_' +
													config['experiment_name'] + '_' + str(config['seed']) + '.pth'))

	freeze_module(model)
	model.eval()

	return model


def evaluate_representation_interventions_concat(model, dataloader, loss_fn_y, num_steps, num_batches, config: dict):
	"""Evaluate interventions on the black-box models fine-tuned by appending concepts to representations"""
	intervened_perc = np.linspace(0, 1, num_steps)

	aurocs = np.zeros((num_batches, len(intervened_perc)))
	auprs = np.zeros((num_batches, len(intervened_perc)))
	losses = np.zeros((num_batches, len(intervened_perc)))

	for i, perc in enumerate(intervened_perc):
		it = iter(dataloader)

		for b in range(num_batches):
			batch = next(it)
			batch_features = batch['features'].to(config['device'])
			concepts_true = batch['concepts'].float().to(config['device'])
			c_mask = generate_random_subset_intervention_mask(batch_features.shape[0], concepts_true.shape[1],
															  int(perc * concepts_true.shape[1]))
			c_mask = torch.tensor(c_mask).to(config['device'])
			concepts_true = (concepts_true * c_mask + (1 - c_mask) * 0.5).float().to(config['device'])

			if config['num_classes'] ==2:
				batch_features, target_true = batch['features'].float().to(config['device']), \
					batch['labels'].float().to(config['device'])  # put the data on the device
			elif config['num_classes'] > 2:
				batch_features, target_true = batch['features'].to(config['device']), \
					batch['labels'].to(config['device'])  # put the data on the device

			# Forward pass
			z, target_pred_probs, target_pred_logits = model(batch_features, conc=concepts_true)
			target_pred_probs = target_pred_probs.squeeze(1)
			target_pred_logits = target_pred_logits.squeeze(1)

			y_metrics = calc_target_metrics(batch['labels'].float().to(config['device']).cpu().numpy(),
											target_pred_probs.cpu().numpy(), config)
			aurocs[b, i] = y_metrics['AUROC']
			auprs[b, i] = y_metrics['AUPR']
			if config['num_classes'] ==2:
				losses[b, i] = loss_fn_y(target_pred_probs, batch['labels'].float().to(config['device'])).cpu().numpy()
			elif config['num_classes'] > 2:
				losses[b, i] = loss_fn_y(target_pred_probs, batch['labels'].to(config['device'])).cpu().numpy()
	return losses, aurocs, auprs


def finetune_post_hoc_CBM(config, model, data_loader):
	"""Trains a CBM model post hoc using sequential optimisation. Optionally, includes a residual model."""
	post_hoc_CBM = pCBM(config, copy.deepcopy(model.encoder))

	post_hoc_CBM.train()
	post_hoc_CBM.encoder.requires_grad = False

	optim_params = [
		{'params': list(filter(lambda p: p.requires_grad, post_hoc_CBM.parameters())),
		 'lr': config['j_learning_rate'],
		 'weight_decay': config['weight_decay']}
	]

	if config['optimizer'] == 'sgd':
		optimizer = torch.optim.SGD(optim_params)
	elif config['optimizer'] == 'adam':
		optimizer = torch.optim.Adam(optim_params)

	# ---------------------------------
	# Create a directory for model checkpoints
	# ---------------------------------
	checkpoint_dir = os.path.join(config['log_directory'], 'checkpoints')
	if not os.path.exists(checkpoint_dir):
		os.makedirs(checkpoint_dir)

	loss_fn = CBLoss(num_classes=config['num_classes'], reduction='mean', alpha=1.0, config=config)

	print()
	print('TRAINING POST HOC CBM')
	print()

	# Numbers of training epochs
	c_epochs = config['c_epochs']
	t_epochs = config['t_epochs']

	print('\nStarting concepts training!\n')
	mode = 'c'

	# Freeze the target prediction part
	freeze_module(post_hoc_CBM.fc1_y)
	freeze_module(post_hoc_CBM.fc2_y)

	c_optimizer = create_optimizer(config, post_hoc_CBM, mode)

	for epoch in range(c_epochs):
		_train_one_epoch_cbm(mode, epoch, config, post_hoc_CBM, c_optimizer, loss_fn, data_loader,
							 writer=None)

	# Prepare parameters for target training
	unfreeze_module(post_hoc_CBM.fc1_y)
	unfreeze_module(post_hoc_CBM.fc2_y)
	freeze_module(post_hoc_CBM.probe)

	print('\nStarting target training!\n')
	mode = 't'
	t_optimizer = create_optimizer(config, post_hoc_CBM, mode)

	for epoch in range(0, t_epochs):
		_train_one_epoch_cbm(mode, epoch, config, post_hoc_CBM, t_optimizer, loss_fn, data_loader,
							 writer=None)

	if config['residual']:
		print('\nStarting residual training!\n')

		# Initialise the residual layer
		if post_hoc_CBM.num_classes == 2:
			post_hoc_CBM.residual_layer = nn.Linear(post_hoc_CBM.num_hidden_z, 1)
		elif post_hoc_CBM.num_classes > 2:
			post_hoc_CBM.residual_layer = nn.Linear(post_hoc_CBM.num_hidden_z, post_hoc_CBM.num_classes)

		post_hoc_CBM.residual_layer.to(torch.device(config['device']))

		# Prepare parameters for residual training
		freeze_module(post_hoc_CBM.fc1_y)
		freeze_module(post_hoc_CBM.fc2_y)
		unfreeze_module(post_hoc_CBM.residual_layer)

		mode = 't'
		t_optimizer = create_optimizer(config, post_hoc_CBM, mode)

		for epoch in range(0, t_epochs):
			_train_one_epoch_cbm(mode, epoch, config, post_hoc_CBM, t_optimizer, loss_fn, data_loader,
								 writer=None)


	if 'run_name' in config:
		if config['residual']:
			torch.save(post_hoc_CBM.state_dict(), os.path.join(
				checkpoint_dir, 'final_ft_pCBM_model_res_' + config['run_name'] + '_' + config['experiment_name'] + '_' +
								str(config['seed']) + '.pth'))
		else:
			torch.save(post_hoc_CBM.state_dict(), os.path.join(
				checkpoint_dir, 'final_ft_pCBM_model_' + config['run_name'] + '_' + config['experiment_name'] + '_' +
								str(config['seed']) + '.pth'))

	freeze_module(post_hoc_CBM)
	post_hoc_CBM.eval()

	return post_hoc_CBM