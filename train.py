"""
Run this file, giving a configuration file as input, to train models, e.g.:
	python train.py --config configfile.yaml
"""

import argparse
import os
import random
import sys
from collections import Counter
from os.path import join

import numpy as np
import torch
import yaml

from torch.utils.tensorboard import SummaryWriter

from datasets.synthetic_dataset import get_synthetic_datasets
from datasets.awa_dataset import get_AwA_dataloaders
from datasets.CXR_dataset import get_CXR_dataloaders
from datasets.CLIP_dataset import get_ImageNetCLIP_dataloaders
from datasets.CUB_dataset import get_CUB_dataloaders

from torch.utils.data import DataLoader, SubsetRandomSampler
from tqdm import tqdm

from losses import create_loss
from models import create_model
from validate import validate_epoch_black_box, validate_epoch_cbm

from utils.training import freeze_module, unfreeze_module, set_bn_to_eval

os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'


def create_optimizer(config, model, mode):
	"""
	Parse the configuration file and return a relevant optimizer object
	"""
	assert config['optimizer'] in ['sgd', 'adam'], 'Only SGD and Adam optimizers are available!'

	optim_params = [
		{'params': filter(lambda p: p.requires_grad, model.parameters()), 'lr': config[mode + '_learning_rate'],
		 'weight_decay': config['weight_decay']}
	]

	if config['optimizer'] == 'sgd':
		return torch.optim.SGD(optim_params)
	elif config['optimizer'] == 'adam':
		return torch.optim.Adam(optim_params)


def _create_data_loaders(config, gen, trainset, train_ids, validset=None, val_ids=None):
	"""
	Construct dataloaders based on the given datasets and config arguments
	"""
	train_subsampler = SubsetRandomSampler(train_ids, gen)
	if val_ids is not None:
		val_subsampler = SubsetRandomSampler(val_ids, gen)

	pm = config['device'] == 'cuda'
	train_loader = DataLoader(trainset, batch_size=config['train_batch_size'], sampler=train_subsampler,
							  num_workers=config["workers"], pin_memory=pm, generator=gen, drop_last=True)
	if validset is not None and val_ids is not None:
		val_loader = DataLoader(validset, batch_size=config['val_batch_size'], sampler=val_subsampler,
								num_workers=config['workers'], pin_memory=pm, generator=gen)
	else:
		val_loader = None
	return train_loader, val_loader


def _get_data(config):
	"""
	Parse the configuration file and return a relevant dataset
	"""
	if config['dataset'] == 'synthetic':
		print('SYNTHETIC DATASET')
		type = None
		if 'sim_type' in config:
			type = config['sim_type']
			print('SIMULTAION TYPE: ' + str(type))
		else:
			print('SIMULTAION TYPE: default')
		trainset, validset, testset = get_synthetic_datasets(
			num_vars=config['num_covariates'], num_points=config['num_points'], num_predicates=config['num_concepts'],
			train_ratio=0.6, val_ratio=float(config['val_ratio']), use_val_train = config['use_val_train'], type=type, seed=config['seed'])
	elif config['dataset'] == 'awa2':
		print('AWA2 DATASET')
		trainset, validset, testset = get_AwA_dataloaders(
			"all_classes.txt", config['train_batch_size'], config["workers"], config["data_path"],
			train_ratio=0.6, val_ratio=0.2, seed=config['seed'])
	elif config['dataset'] == 'CUB':
		print('CUB DATASET')
		trainset, validset, testset = get_CUB_dataloaders(config["data_path"], train_val_split=0.6, test_val_split=0.5, seed=config['seed'])

	elif config['dataset'] == 'CXR':
		print('CXR DATASET')
		trainset, validset, testset = get_CXR_dataloaders(
			config['dataset'], config["data_path"], train_val_split=0.8, test_val_split=0.5, seed=config['seed'])
	elif config['dataset'] == 'cheXpert':
		print('cheXpert DATASET')
		trainset, validset, testset = get_CXR_dataloaders(
			config['dataset'], config["data_path"], train_val_split=0.8, test_val_split=0.5, seed=config['seed'])
	elif config['dataset'] in ('SD_ImageNet','CIFAR10'):
		trainset, validset, testset = get_ImageNetCLIP_dataloaders(config['dataset'], config["data_path"], test_val_split=0.5, seed=config['seed'])

	else:
		NotImplementedError('ERROR: Dataset not supported!')

	return trainset, validset, testset


def _train_one_epoch_black_box(mode, epoch, config, model, optimizer, loss_fn, train_loader, writer):
	"""
	Train a black-box model for one epoch
	"""
	running_len = 0
	running_total_loss = 0

	# Training mode and the number of epochs
	num_epochs = config['num_epochs']

	# Decrease the learning rate, if applicable
	if epoch >= 0 and config['decrease_every'] > 0 and (epoch + 1) % config['decrease_every'] == 0:
		for g in optimizer.param_groups:
			g['lr'] = g['lr'] / config['lr_divisor']

	with tqdm(total=len(train_loader) * config['train_batch_size'], desc=f'Epoch {epoch + 1}/{num_epochs}',
			  unit='data points', position=0, leave=True) as pbar:
		model.train()

		for k, batch in enumerate(train_loader):
			if config['num_classes'] ==2:
				batch_features, target_true = batch['features'].float().to(config['device']), \
					batch['labels'].float().to(config['device'])  # put the data on the device
			elif config['num_classes'] > 2:
				batch_features, target_true = batch['features'].to(config['device']), \
					batch['labels'].to(config['device'])
			concepts_true = batch['concepts'].to(config['device'])

			# Forward pass
			z, target_pred_probs, target_pred_logits = model(batch_features)
			target_pred_probs = target_pred_probs.squeeze(1)
			target_pred_logits = target_pred_logits.squeeze(1)

			# Backward pass depends on the training mode of the model
			optimizer.zero_grad()
			# Compute the loss
			# Weighted binary cross entropy loss
			if config.get('weight_loss') and config['num_classes'] == 2:
				loss_pos = config['weight'][1] * target_true * torch.log(target_pred_probs)
				loss_neg = (1 - target_true) * torch.log(1 - target_pred_probs)
				total_loss = torch.mean(-(loss_pos + loss_neg))
			else:
				total_loss = loss_fn(target_pred_probs, target_true)

			running_total_loss += total_loss.item() * batch_features.size(0)

			running_len += batch_features.size(0)
			if mode == 'j':
				total_loss.backward()
			optimizer.step()  # perform an update
			writer.add_scalar("Loss/total_loss", running_total_loss / running_len, epoch)
			# Update the progress bar
			pbar.set_postfix(**{'Total loss': running_total_loss / running_len, 'lr': optimizer.param_groups[0]['lr']})
			pbar.update(config['train_batch_size'])

	return None


def _train_one_epoch_cbm(mode, epoch, config, model, optimizer, loss_fn, train_loader, writer):
	"""
	Train a CBM for one epoch
	"""
	running_len = 0
	running_target_loss = 0
	running_concepts_loss = [0] * config['num_concepts']
	running_summed_concepts_loss = 0
	running_total_loss = 0
	# Training mode and the number of epochs
	if mode == 'j':
		num_epochs = config['j_epochs']
	elif mode == 'c':
		num_epochs = config['c_epochs']
	elif mode == 't':
		num_epochs = config['t_epochs']
	else:
		raise ValueError('Training mode unknown!')

	# Decrease the learning rate, if applicable
	if epoch >= 0 and config['decrease_every'] > 0 and (epoch + 1) % config['decrease_every'] == 0:
		for g in optimizer.param_groups:
			g['lr'] = g['lr'] / config['lr_divisor']

	with tqdm(total=len(train_loader) * config['train_batch_size'], desc=f'Epoch {epoch + 1}/{num_epochs}',
			  unit='data points', position=0, leave=True) as pbar:
		model.train()
		if config['training_mode'] == 'sequential' and mode == 't':
			model.apply(set_bn_to_eval)

		for k, batch in enumerate(train_loader):
			batch_features, target_true = batch['features'].float().to(config['device']), \
				batch['labels'].float().to(config['device'])  # put the data on the device
			concepts_true = batch['concepts'].to(config['device'])

			# Forward pass
			concepts_pred, target_pred_probs, target_pred_logits = model(batch_features)
			target_pred_probs = target_pred_probs.squeeze(1)
			target_pred_logits = target_pred_logits.squeeze(1)

			# Backward pass depends on the training mode of the model
			optimizer.zero_grad()
			# Compute the loss
			target_loss, concepts_loss, summed_concepts_loss, total_loss = loss_fn(
				concepts_pred, concepts_true, target_pred_probs, target_pred_logits, target_true)

			running_target_loss += target_loss.item() * batch_features.size(0)
			for concept_idx in range(len(concepts_loss)):
				running_concepts_loss[concept_idx] += concepts_loss[concept_idx].item() * batch_features.size(0)
			running_summed_concepts_loss += summed_concepts_loss.item() * batch_features.size(0)
			running_total_loss += total_loss.item() * batch_features.size(0)

			running_len += batch_features.size(0)
			if mode == 'j':
				total_loss.backward()
			elif mode == 'c':
				summed_concepts_loss.backward()
			else:
				target_loss.backward()
			optimizer.step()  # perform an update

			# Update the progress bar
			pbar.set_postfix(**{'Target loss': running_target_loss / running_len,
								'Concepts loss': running_summed_concepts_loss / running_len,
								'Total loss': running_total_loss / running_len, 'lr': optimizer.param_groups[0]['lr']})

			pbar.update(config['train_batch_size'])
		writer.add_scalar("Loss/target_loss", running_target_loss / running_len, epoch)
		writer.add_scalar("Loss/concept_loss", running_summed_concepts_loss / running_len, epoch)
		writer.add_scalar("Loss/total_loss", running_total_loss / running_len, epoch)


def train_black_box(config, gen):
	"""
	Train and test a black-box model
	"""

	# Log the print-outs
	old_stdout = sys.stdout
	log_file = open(
		os.path.join(config['log_directory'], config['experiment_name'] + '_' + config['run_name'] + '_' +
					 str(config['seed']) + '.log'), 'w')
	sys.stdout = log_file

	# ---------------------------------
	#       Prepare data
	# ---------------------------------
	trainset, validset, testset = _get_data(config=config)
	if config.get('binary'):
		config['num_concepts'] = len(testset.concepts[0])
	# Retrieve labels
	train_labels = []
	valid_labels = []
	test_labels = []
	all_c = [[] for _ in range(config['num_concepts'])]

	train_labels = [x for x in range(len(trainset))]
	valid_labels = [x for x in range(len(validset))]
	test_labels = [x for x in range(len(testset))]
	train_labels = np.array(train_labels)
	valid_labels = np.array(valid_labels)
	test_labels = np.array(test_labels)

	# Generating weights for class imbalance
	id = np.unique(trainset.img_index)
	weight = []
	for num in id:
		weight.append(sum([x == num for x in trainset.img_index]))
	config['weight'] = torch.tensor(1 - weight/sum(weight)).to('cuda').float()

	print('Number of training data points', len(train_labels))
	print('Number of test data points', len(test_labels))
	print('Concept class distributions: ')
	for concept_idx in range(len(all_c)):
		print("...", Counter(all_c[concept_idx]))

	# ---------------------------------
	# Create a directory for model checkpoints and tb_logger
	# ---------------------------------
	checkpoint_dir = os.path.join(config['log_directory'], 'checkpoints')
	tb_dir = os.path.join(config['log_directory'], 'tb_logger')
	tb_logger_dir = os.path.join(tb_dir,config['run_name'] + '_' + config['experiment_name'] + '_' + str(config['seed']))
	if not os.path.exists(checkpoint_dir):
		os.makedirs(checkpoint_dir)
	if not os.path.exists(tb_dir):
		os.makedirs(tb_dir)
	if not os.path.exists(tb_logger_dir):
		os.makedirs(tb_logger_dir)
	writer = SummaryWriter(log_dir=tb_logger_dir)

	# Numbers of training epochs
	n_epochs = config['num_epochs']

	# Instantiate dataloaders
	train_loader, valid_loader = _create_data_loaders(config, gen, trainset, train_ids=np.arange(len(train_labels)),
													  validset=validset, val_ids=np.arange(len(valid_labels)))
	test_loader = DataLoader(testset, batch_size=config['val_batch_size'], num_workers=config['workers'], generator=gen)

	# Initialize model and training objects
	model = create_model(config)
	init = 0
	model.to(config['device'])
	loss_fn = create_loss(config)

	# Evaluate the randomly initialised model before training
	print("\nEVALUATION ON THE VALIDATION SET:\n")
	loss_valid, y_metrics, c_lin_metrics, _, c_nlin_metrics, _ = \
		validate_epoch_black_box(-1, config, model, train_loader, valid_loader, loss_fn)

	print()
	print('TRAINING ' + str(config['model']))
	print()

	mode = 'j'
	optimizer = create_optimizer(config, model, mode)

	for epoch in range(init, n_epochs):
		# Training the model
		_train_one_epoch_black_box(mode, epoch, config, model, optimizer, loss_fn, train_loader,writer)

		# Validate the model periodically (except after the first epoch)
		if epoch > 0 and epoch % config['validate_per_epoch'] == 0:
			print("\nEVALUATION ON THE VALIDATION SET:\n")
			loss_valid, y_metrics, c_lin_metrics, _, c_nlin_metrics, _ = \
				validate_epoch_black_box(epoch, config, model, train_loader, valid_loader, loss_fn)

		# Saving model checkpoints
		if config['pth_store_frequency']:
			if epoch % config['pth_store_frequency'] == 0:
				model_name = 'model_' + config['run_name'] + '_' + config['experiment_name'] + '_' + str(
					config['seed']) + '_' + str(epoch) + '.pth'
				torch.save(model.state_dict(), join(checkpoint_dir, model_name))
	torch.save(model.state_dict(), join(checkpoint_dir, 'final_model_' + config['run_name'] + '_' +
										config['experiment_name'] + '_' + str(config['seed']) + '.pth'))
	print("\nTRAINING FINISHED, MODEL SAVED!", flush=True)

	print("\nEVALUATION ON THE TEST SET:\n")
	loss_test, y_metrics, c_lin_metrics, _, c_nlin_metrics, _ = validate_epoch_black_box(
		n_epochs + 1, config, model, train_loader, test_loader, loss_fn)

	# Stop logging print-outs
	sys.stdout = old_stdout
	log_file.close()
	writer.close()
	return None


def train_cbm(config, gen):
	"""
	Train and test a CBM
	"""
	# Log the print-outs
	old_stdout = sys.stdout
	log_file = open(
		os.path.join(config['log_directory'], config['experiment_name'] + '_' + config['run_name'] + '_' +
					 str(config['seed']) + '.log'), 'w')
	sys.stdout = log_file

	# ---------------------------------
	#       Prepare data
	# ---------------------------------
	trainset, validset, testset = _get_data(config=config)
	if config.get('binary'):
		config['num_concepts'] = len(testset.concepts[0])
	# Retrieve labels
	all_c = [[] for _ in range(config['num_concepts'])]

	train_labels = [x for x in range(len(trainset))]
	valid_labels = [x for x in range(len(validset))]
	test_labels = [x for x in range(len(testset))]
	train_labels = np.array(train_labels)
	valid_labels = np.array(valid_labels)
	test_labels = np.array(test_labels)

	# Generating weights for class imbalance
	id = np.unique(trainset.img_index)
	weight = []
	for num in id:
		weight.append(sum([x == num for x in trainset.img_index]))
	config['weight'] = torch.tensor(1 - weight/sum(weight)).to('cuda').float()

	print('Number of training data points', len(train_labels))
	print('Number of test data points', len(test_labels))
	print('Target class distribution: ', Counter(train_labels))
	print('Concept class distributions: ')
	for concept_idx in range(len(all_c)):
		print("...", Counter(all_c[concept_idx]))
	print()

	# ---------------------------------
	# Create a directory for model checkpoints and loggers
	# ---------------------------------
	checkpoint_dir = os.path.join(config['log_directory'], 'checkpoints')
	tb_dir = os.path.join(config['log_directory'], 'tb_logger')
	tb_logger_dir = os.path.join(tb_dir,
								 config['run_name'] + '_' + config['experiment_name'] + '_' + str(config['seed']))
	if not os.path.exists(checkpoint_dir):
		os.makedirs(checkpoint_dir)
	if not os.path.exists(tb_dir):
		os.makedirs(tb_dir)
	if not os.path.exists(tb_logger_dir):
		os.makedirs(tb_logger_dir)
	writer = SummaryWriter(log_dir=tb_logger_dir)

	# Numbers of training epochs
	if config['training_mode'] == 'joint':
		c_epochs = config['j_epochs']
		t_epochs = config['j_epochs']
	elif config['training_mode'] == 'sequential':
		c_epochs = config['c_epochs']
		t_epochs = config['t_epochs']

	# Instantiate dataloaders
	train_loader, valid_loader = _create_data_loaders(config, gen, trainset, train_ids=np.arange(len(train_labels)),
													  validset=validset, val_ids=np.arange(len(valid_labels)))
	test_loader = DataLoader(testset, batch_size=config['val_batch_size'], num_workers=config['workers'], generator=gen)

	# Initialize model and training objects
	model = create_model(config)
	model.to(config['device'])
	loss_fn = create_loss(config)

	# Evaluate the randomly initialised model before training
	print("\nEVALUATION ON THE VALIDATION SET:\n")
	loss_valid, y_metrics, c_metrics, _ = validate_epoch_cbm(-1, config, model, train_loader, valid_loader, loss_fn)

	print()
	print('TRAINING ' + str(config['model']))
	print()

	# Concept learning
	if config['training_mode'] == 'sequential':
		print('\nStarting concepts training!\n')
		mode = 'c'

		# Freeze the target prediction part
		model.head.apply(freeze_module)

		c_optimizer = create_optimizer(config, model, mode)

		for epoch in range(c_epochs):
			_train_one_epoch_cbm(mode, epoch, config, model, c_optimizer, loss_fn, train_loader, writer)

			# Validate the model periodically (except after the first epoch)
			if epoch > 0 and epoch % config['validate_per_epoch'] == 0:
				print("\nEVALUATION ON THE VALIDATION SET:\n")
				loss_valid, y_metrics, c_metrics, _ = \
					validate_epoch_cbm(epoch, config, model, train_loader, valid_loader, loss_fn)

		# Prepare parameters for target training
		model.head.apply(unfreeze_module)
		model.encoder.apply(freeze_module)

	# Sequential vs. joint optimisation
	if config['training_mode'] == 'sequential':
		print('\nStarting target training!\n')
		mode = 't'
		optimizer = create_optimizer(config, model, mode)
	else:
		print('\nStarting joint training!\n')
		mode = 'j'
		optimizer = create_optimizer(config, model, mode)

	for epoch in range(0, t_epochs):
		_train_one_epoch_cbm(mode, epoch, config, model, optimizer, loss_fn, train_loader, writer)

		# Validate the model periodically (except after the first epoch)
		if epoch > 0 and epoch % config['validate_per_epoch'] == 0:
			print("\nEVALUATION ON THE VALIDATION SET:\n")
			loss_valid, y_metrics, c_metrics, _ = \
				validate_epoch_cbm(epoch, config, model, train_loader, valid_loader, loss_fn)
		if config['pth_store_frequency']:
			if epoch % config['pth_store_frequency'] == 0:
				model_name = 'model_' + config['run_name'] + '_' + config['experiment_name'] + '_' + str(
					config['seed']) + '_' + str(epoch) + '.pth'
				torch.save(model.state_dict(), join(checkpoint_dir, model_name))
	torch.save(model.state_dict(), join(checkpoint_dir, 'final_model_' + config['run_name'] + '_' +
										config['experiment_name'] + '_' + str(config['val_ratio']) + '_' + str(config['seed']) + '.pth'))
	print('\nTRAINING FINISHED, MODEL SAVED!', flush=True)

	print("\nEVALUATION ON THE TEST SET:\n")
	loss_test, y_metrics, c_metrics, _ = validate_epoch_cbm(
		t_epochs + 1, config, model, train_loader, test_loader, loss_fn)

	# Stop logging print-outs
	sys.stdout = old_stdout
	log_file.close()
	writer.close()

	return None


def main():
	parser = argparse.ArgumentParser()
	parser.add_argument('-c', '--config')
	# Overrides config arguments if necessary
	parser.add_argument('-s', '--seed', type=int)
	parser.add_argument('-d_p', '--data_path')
	parser.add_argument('-rn', '--run_name')
	args = parser.parse_args()
	argsdict = vars(args)

	with open(argsdict['config'], 'r') as f:
		config = yaml.safe_load(f)
	config['filename'] = argsdict['config']

	if argsdict['seed'] is not None:
		config['seed'] = argsdict['seed']
	if argsdict['run_name'] is not None:
		config['run_name'] = argsdict['run_name']
	if argsdict['data_path'] is not None:
		config['data_path'] = argsdict['data_path']
	if argsdict['val_ratio'] is not None:
		config['val_ratio'] = argsdict['val_ratio']
	if config.get('sparse'):
		if config['sparse'] == True:
			config['num_concepts'] = 22

	# Ensure reproducibility
	random.seed(config['seed'])
	np.random.seed(config['seed'])
	gen = torch.manual_seed(config['seed'])
	torch.backends.cudnn.benchmark = False
	torch.use_deterministic_algorithms(True)

	# Choose the relevant model training routine
	if config['model'] == 'black-box':
		train = train_black_box
	elif config['model'] == 'cbm':
		train = train_cbm

	if config['suppress_warnings']:
		import warnings
		warnings.filterwarnings('ignore')

	train(config, gen)


if __name__ == "__main__":
	main()
