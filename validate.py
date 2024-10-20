"""
Utility functions to evaluate models
"""
import numpy as np
import torch

from probes import train_lr_probes_post_hoc, train_rf_probes_post_hoc

from utils.metrics import calc_target_metrics, calc_concept_metrics


def validate_epoch_black_box(epoch, config, model, train_dataloader, valid_dataloader, loss_fn):
	"""
	Run a single validation epoch for the black-box model
	"""
	model.eval()

	with torch.no_grad():
		probes_lin = train_lr_probes_post_hoc(model, train_dataloader, config)
		probes_nlin = train_rf_probes_post_hoc(model, train_dataloader, config)

		zs_valid = np.zeros((0, model.num_hidden_z))
		cs_valid = np.zeros((0, config['num_concepts']))
		cs_pred_lin = []
		cs_pred_lin_probs = []
		cs_pred_nlin = []
		cs_pred_nlin_probs = []
		ys_valid = np.zeros((0, 1))
		if config['num_classes'] == 2:
			ys_pred_probs = np.zeros((0, 1))
		elif config['num_classes'] > 2:
			ys_pred_probs = np.zeros((0, config['num_classes']))
		loss_valid = 0
		total = 0

		for k, batch in enumerate(valid_dataloader):
			z, y_pred_probs, y_pred_logits = model(batch['features'].float().to(config['device']))
			if config['num_classes'] == 2:
				y_pred_probs = y_pred_probs.squeeze(1)
				y_pred_logits = y_pred_logits.squeeze(1)
			if config['num_classes'] == 2:
				loss_k = loss_fn(y_pred_probs, batch['labels'].float().to(config['device']))
			elif config['num_classes'] > 2:
				loss_k = loss_fn(y_pred_probs, batch['labels'].to(config['device']))
			loss_valid += loss_k * batch['features'].float().to(config['device']).size(0)

			zs_valid = np.vstack((zs_valid, z.cpu().numpy()))
			cs_valid = np.vstack((cs_valid, batch['concepts'].float().to(config['device']).cpu().numpy()))
			if config['num_classes'] == 2:
				ys_valid = np.vstack((ys_valid, batch['labels'].float().to(config['device']).unsqueeze(1).cpu().numpy()))
				ys_pred_probs = np.vstack((ys_pred_probs, y_pred_probs.unsqueeze(1).cpu().numpy()))
			elif config['num_classes'] > 2:
				ys_valid = np.vstack((ys_valid, batch['labels'].unsqueeze(1).to(config['device']).cpu().numpy()))
				ys_pred_probs = np.vstack((ys_pred_probs, y_pred_probs.cpu().numpy()))

			total += batch['features'].size(0)

		for j in range(config['num_concepts']):
			cs_pred_lin.append(probes_lin[j].predict(zs_valid))
			cs_pred_lin_probs.append(probes_lin[j].predict_proba(zs_valid))
			cs_pred_nlin.append(probes_nlin[j].predict(zs_valid))
			cs_pred_nlin_probs.append(probes_nlin[j].predict_proba(zs_valid))

		loss_valid /= total

		y_metrics = calc_target_metrics(ys_valid, ys_pred_probs, config)
		c_lin_metrics, c_lin_metrics_per_concept = calc_concept_metrics(cs_valid, cs_pred_lin_probs, config)
		c_nlin_metrics, c_nlin_metrics_per_concept = calc_concept_metrics(cs_valid, cs_pred_nlin_probs, config)

		print('')
		print('--------------------------------------')
		print('Concepts (lin. probe)   : ' + str(c_lin_metrics))
		print('Concepts (nonlin. probe): ' + str(c_nlin_metrics))
		print('Target                  : ' + str(y_metrics))

	model.train()

	return loss_valid, y_metrics, c_lin_metrics, c_lin_metrics_per_concept, c_nlin_metrics, c_nlin_metrics_per_concept


def validate_epoch_cbm(epoch, config, model, train_dataloader, valid_dataloader, loss_fn):
	"""
	Run a single validation epoch for the CBM
	"""
	model.eval()

	with torch.no_grad():

		cs_pred = np.zeros((0, config['num_concepts']))
		cs_pred_probs = []
		cs_valid = np.zeros((0, config['num_concepts']))
		ys_valid = np.zeros((0, 1))
		if config['num_classes'] == 2:
			ys_pred_probs = np.zeros((0, 1))
		elif config['num_classes'] > 2:
			ys_pred_probs = np.zeros((0, config['num_classes']))
		loss_valid = 0
		total = 0

		for k, batch in enumerate(valid_dataloader):
			cs, y_pred_probs, y_pred_logits = model(batch['features'].float().to(config['device']))
			if config['num_classes'] == 2:
				y_pred_probs = y_pred_probs.squeeze(1)
				y_pred_logits = y_pred_logits.squeeze(1)

			target_loss, concepts_loss, summed_concepts_loss, loss_k = \
				loss_fn(concepts_pred=cs, concepts_true=batch['concepts'].float().to(config['device']),
						target_pred_probs=y_pred_probs, target_pred_logits=y_pred_logits,
						target_true=batch['labels'].float().to(config['device']))
			loss_valid += loss_k * batch['features'].float().to(config['device']).size(0)

			cs_pred = np.vstack((cs_pred, cs.cpu().numpy()))
			cs_valid = np.vstack((cs_valid, batch['concepts'].float().to(config['device']).cpu().numpy()))
			if config['num_classes'] == 2:
				ys_valid = np.vstack((ys_valid, batch['labels'].float().to(config['device']).unsqueeze(1).cpu().numpy()))
				ys_pred_probs = np.vstack((ys_pred_probs, y_pred_probs.unsqueeze(1).cpu().numpy()))
			elif config['num_classes'] > 2:
				ys_valid = np.vstack((ys_valid, batch['labels'].unsqueeze(1).to(config['device']).cpu().numpy()))
				ys_pred_probs = np.vstack((ys_pred_probs, y_pred_probs.cpu().numpy()))

			total += batch['features'].size(0)

		loss_valid /= total

		for j in range(config['num_concepts']):
			cs_pred_probs.append(
				np.hstack((np.expand_dims(1 - cs_pred[:, j], 1), np.expand_dims(cs_pred[:, j], 1))))

		y_metrics = calc_target_metrics(ys_valid, ys_pred_probs, config)
		c_metrics, c_metrics_per_concept = calc_concept_metrics(cs_valid, cs_pred_probs, config)

		print('')
		print('--------------------------------------')
		print('Concepts   : ' + str(c_metrics))
		print('Target     : ' + str(y_metrics))

	model.train()

	return loss_valid, y_metrics, c_metrics, c_metrics_per_concept


def validate_epoch_concat_black_box(epoch, config, model, train_dataloader, valid_dataloader, loss_fn):
	"""
	Run a single validation epoch for the model fine-tuned by appending concept to the model's representations
	"""
	model.eval()

	with torch.no_grad():

		zs_valid = np.zeros((0, model.num_hidden_z))
		cs_valid = np.zeros((0, config['num_concepts']))
		ys_valid = np.zeros((0, 1))
		if config['num_classes'] == 2:
			ys_pred_probs = np.zeros((0, 1))
		elif config['num_classes'] > 2:
			ys_pred_probs = np.zeros((0, config['num_classes']))
		loss_valid = 0
		total = 0

		for k, batch in enumerate(valid_dataloader):
			z, y_pred_probs, y_pred_logits = model(
				batch['features'].float().to(config['device']),
				conc=0.5*torch.ones_like(batch['concepts'].float().to(config['device'])).float().to(config['device']))
			if config['num_classes'] == 2:
				y_pred_probs = y_pred_probs.squeeze(1)
				y_pred_logits = y_pred_logits.squeeze(1)
				loss_k = loss_fn(y_pred_probs, batch['labels'].float().to(config['device']))
			elif config['num_classes'] > 2:
				loss_k = loss_fn(y_pred_probs, batch['labels'].to(config['device']))
			loss_valid += loss_k * batch['features'].float().to(config['device']).size(0)

			zs_valid = np.vstack((zs_valid, z.cpu().numpy()))
			cs_valid = np.vstack((cs_valid, batch['concepts'].float().to(config['device']).cpu().numpy()))
			if config['num_classes'] == 2:
				ys_valid = np.vstack((ys_valid, batch['labels'].float().to(config['device']).unsqueeze(1).cpu().numpy()))
				ys_pred_probs = np.vstack((ys_pred_probs, y_pred_probs.unsqueeze(1).cpu().numpy()))
			elif config['num_classes'] > 2:
				ys_valid = np.vstack((ys_valid, batch['labels'].unsqueeze(1).to(config['device']).cpu().numpy()))
				ys_pred_probs = np.vstack((ys_pred_probs, y_pred_probs.cpu().numpy()))

			total += batch['features'].size(0)

		loss_valid /= total

		y_metrics = calc_target_metrics(ys_valid, ys_pred_probs, config)

		print('')
		print('--------------------------------------')
		print('Target                  : ' + str(y_metrics))

	model.train()

	return loss_valid, y_metrics
