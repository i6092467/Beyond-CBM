"""
Utility functions for evaluation metrics
"""
import numpy as np
import torch

from sklearnex import patch_sklearn

patch_sklearn()

from sklearn.metrics import average_precision_score, roc_auc_score
from torchmetrics import AveragePrecision


def _roc_auc_score_with_missing(labels, scores):
	""" Computes OVR macro-averaged AUROC under missing classes	"""

	aurocs = np.zeros((scores.shape[1],))
	weights = np.zeros((scores.shape[1],))
	for c in range(scores.shape[1]):
		if len(labels[labels == c]) > 0:
			labels_tmp = (labels == c) * 1.0
			aurocs[c] = roc_auc_score(labels_tmp, scores[:, c], average='weighted', multi_class='ovr')
			weights[c] = len(labels[labels == c])
		else:
			aurocs[c] = np.NaN
			weights[c] = np.NaN

	# Computing weighted average
	mask = ~np.isnan(aurocs)
	weighted_sum = np.sum(aurocs[mask] * weights[mask])
	average = weighted_sum / len(labels)
	return average


def calc_target_metrics(ys, scores_pred, config, n_decimals=3, n_bins_cal=10):
	"""
	Calculates AUROC, AUPR, Brier score and Expected Calibration Error (ECE) of two numpy arrays: ys and scores_pred
	Computed on the target of the classification problem
	"""

	# AUROC
	if config['num_classes'] == 2:
		auroc = roc_auc_score(ys, scores_pred, average='macro', multi_class='ovr')
	elif config['num_classes'] > 2:
		auroc = _roc_auc_score_with_missing(ys, scores_pred)

	# AUPR
	if config['num_classes'] == 2:
		aupr = average_precision_score(ys, scores_pred, average='macro')
	elif config['num_classes'] > 2:
		ap = AveragePrecision(task='multiclass', num_classes=config['num_classes'], average='weighted')
		aupr = float(ap(torch.tensor(scores_pred), torch.tensor(ys.squeeze()).type(torch.int64)).cpu().numpy())

	# Brier score
	if config['num_classes'] == 2:
		brier = brier_score(ys, np.squeeze(scores_pred))
	else:
		brier = brier_score(ys, scores_pred)

	# ECE
	if config['num_classes'] == 2:
		if len(scores_pred.shape) == 1:
			scores_pred_ = np.expand_dims(scores_pred, 1)
		else:
			scores_pred_ = scores_pred
		ece = expected_calibration_error(ys, np.hstack((1 - scores_pred_, scores_pred_)), num_bins=n_bins_cal)
	else:
		ece = expected_calibration_error(ys, scores_pred, num_bins=n_bins_cal)

	return {'AUROC': np.round(auroc, n_decimals), 'AUPR': np.round(aupr, n_decimals),
			'Brier': np.round(brier, n_decimals), 'ECE': np.round(ece, n_decimals)}


def calc_concept_metrics(cs, concepts_pred, config, n_decimals=3, n_bins_cal=10):
	"""
	Calculates the average AUROC, AUPR, Brier score and Expected Calibration Error (ECE) of inputs cs and concepts_pred
	Computed on the binary concept set
	"""
	num_concepts = cs.shape[1]
	metrics_per_concept = []

	for j in range(num_concepts):
		# AUROC
		auroc = 0.0
		if len(np.unique(cs[:, j])) == 2:
			auroc = roc_auc_score(cs[:, j], concepts_pred[j][:, 1], average='macro', multi_class='ovr')
		elif len(np.unique(cs[:, j])) > 2:
			auroc = roc_auc_score(cs[:, j], concepts_pred[j], average='macro', multi_class='ovr')

		# AUPR
		aupr = 0.0
		if len(np.unique(cs[:, j])) == 2:
			aupr = average_precision_score(cs[:, j], concepts_pred[j][:, 1], average='macro')
		elif len(np.unique(cs[:, j])) > 2:
			ap = AveragePrecision(task='multiclass', num_classes=config['num_classes'], average='macro')
			aupr = float(ap(torch.tensor(cs[:, j]), torch.tensor(concepts_pred[j])).cpu().numpy())

		# Brier score
		if len(np.unique(cs[:, j])) == 2:
			brier = brier_score(cs[:, j], concepts_pred[j][:, 1])
		else:
			brier = brier_score(cs[:, j], concepts_pred[j])

		# ECE
		if len(concepts_pred[j].shape) == 1:
			tmp = np.expand_dims(concepts_pred[j], 1)
			ece = expected_calibration_error(
				cs[:, j], np.hstack((1 - tmp, tmp)), n_bins_cal)
		else:
			ece = expected_calibration_error(cs[:, j], concepts_pred[j], n_bins_cal)

		metrics_per_concept.append({'AUROC': np.round(auroc, n_decimals), 'AUPR': np.round(aupr, n_decimals),
									'Brier': np.round(brier, n_decimals), 'ECE': np.round(ece, n_decimals)})
	# Averaging results per concept
	auroc, aupr, brier, ece = 0.0, 0.0, 0.0, 0.0
	for j in range(num_concepts):
		auroc += metrics_per_concept[j]['AUROC']
		aupr += metrics_per_concept[j]['AUPR']
		brier += metrics_per_concept[j]['Brier']
		ece += metrics_per_concept[j]['ECE']
	auroc /= num_concepts
	aupr /= num_concepts
	brier /= num_concepts
	ece /= num_concepts
	metrics_overall = {'AUROC': np.round(auroc, n_decimals), 'AUPR': np.round(aupr, n_decimals),
					   'Brier': np.round(brier, n_decimals), 'ECE': np.round(ece, n_decimals)}

	return metrics_overall, metrics_per_concept


def brier_score(y_true, y_prob):
	"""
	Computation of the Brier score
	- for multiclass, @y_true must be of dimensionality (n_samples, ) and @y_prob must be (n_samples, n_classes)
    - for binary, @y_true must be of dimensionality (n_samples, ) and @y_prob must be (n_samples, )
	"""

	if len(y_prob.shape) == 2:
		# NOTE: we use the original definition by Brier for categorical variables
		# See the original paper by Brier https://doi.org/10.1175/1520-0493(1950)078<0001:VOFEIT>2.0.CO;2
		sc = 0
		for j in range(y_prob.shape[1]):
			# Correction to multiclass
			sc += np.sum((np.squeeze((y_true == j) * 1.) - y_prob[:, j]) ** 2)
		sc /= y_true.shape[0]
		return sc
	elif len(y_prob.shape) == 1:
		return np.mean((y_prob - y_true) ** 2)


def expected_calibration_error(y_true, y_prob, num_bins=10):
	"""
	Compute the Expected Calibration Error (ECE) for multiclass classification.

	Args:
		y_true (numpy.ndarray): 1D array of true labels.
		y_prob (numpy.ndarray): 2D array of predicted probabilities for each class.
		num_bins (int): Number of bins to divide the predicted probabilities range into.

	Returns:
		float: The Expected Calibration Error.
	See: https://arxiv.org/pdf/1706.04599.pdf
	Smaller calibration is better
	"""
	n_samples, n_classes = y_prob.shape
	bin_boundaries = np.linspace(0, 1, num_bins + 1)  # Bins boundaries for predicted probabilities
	bin_indices = np.digitize(y_prob.max(axis=1), bin_boundaries) - 1  # Assign each sample to a bin
	bin_confidences = np.zeros(num_bins)  # Confidence (mean predicted probability) for each bin
	bin_accuracies = np.zeros(num_bins)  # Accuracy for each bin
	bin_counts = np.zeros(num_bins)  # Count of samples in each bin

	for bin_idx in range(num_bins):
		# Fixing the case where none of the confidences are in a range of some of the bins
		if np.sum(bin_indices == bin_idx) > 0:
			bin_samples = y_prob[bin_indices == bin_idx, :]
			bin_true_labels = y_true[bin_indices == bin_idx]
			bin_confidences[bin_idx] = np.mean(np.max(bin_samples, axis=1))
			bin_accuracies[bin_idx] = np.mean(np.argmax(bin_samples, axis=1) == bin_true_labels)
			bin_counts[bin_idx] = bin_samples.shape[0]

	total_samples = np.sum(bin_counts)
	ece = np.sum(bin_counts / total_samples * np.abs(bin_accuracies - bin_confidences))

	return ece


def tl_expected_calibration_error(y_true, y_prob, num_bins=10):
	"""
	Adapted from https://github.com/AIgen/df-posthoc-calibration/blob/main/assessment.py
    Compute the Top-Level Expected Calibration Error (ECE) for multiclass classification.

    Args:
        y_true (numpy.ndarray): 1D array of true labels.
        y_prob (numpy.ndarray): 2D array of predicted probabilities for each class.
        num_bins (int): Number of bins to divide the predicted probabilities range into.

    Returns:
        float: The Top-Level Expected Calibration Error.
    See: https://arxiv.org/pdf/2107.08353.pdf
    Smaller calibration is better
    """
	n_samples, n_classes = y_prob.shape
	ece = 0
	for l in range(n_classes):
		l_inds = np.argwhere(y_true.squeeze() == l).squeeze()
		bin_boundaries = np.linspace(0, 1, num_bins + 1)  # Bins boundaries for predicted probabilities
		y_prob_new = y_prob[l_inds].squeeze()
		y_true_new = y_true[l_inds].squeeze()
		bin_indices = np.digitize(y_prob_new.max(axis=1), bin_boundaries) - 1  # Assign each sample to a bin
		bin_confidences = np.zeros(num_bins)  # Confidence (mean predicted probability) for each bin
		bin_accuracies = np.zeros(num_bins)  # Accuracy for each bin
		bin_counts = np.zeros(num_bins)  # Count of samples in each bin

		for bin_idx in range(num_bins):
			if np.sum(bin_indices == bin_idx) > 0:
				# Fixing the case where none of the confidences are in a range of some of the bins
				bin_samples = y_prob_new[bin_indices == bin_idx, :]
				bin_true_labels = y_true_new[bin_indices == bin_idx]
				bin_confidences[bin_idx] = np.mean(np.max(bin_samples, axis=1))
				bin_accuracies[bin_idx] = np.mean(np.argmax(bin_samples, axis=1) == bin_true_labels)
				bin_counts[bin_idx] = bin_samples.shape[0]

		total_samples = np.sum(bin_counts)
		ece += l_inds.size * np.sum(bin_counts / total_samples * np.abs(bin_accuracies - bin_confidences))
	tl_ece = ece / n_samples
	return tl_ece
