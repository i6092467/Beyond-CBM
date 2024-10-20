"""
Functions for generating nonlinear synthetic data
"""
import numpy as np
from numpy.random import multivariate_normal, uniform

import torch
from torch.utils import data

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.datasets import make_spd_matrix, make_low_rank_matrix


def random_nonlin_map(n_in, n_out, n_hidden, rank=100):
	"""
	Reaturn a random nonlinear function parameterized by an MLP
	"""
	# Random MLP mapping
	W_0 = make_low_rank_matrix(n_in, n_hidden, effective_rank=rank)
	W_1 = make_low_rank_matrix(n_hidden, n_hidden, effective_rank=rank)
	W_2 = make_low_rank_matrix(n_hidden, n_out, effective_rank=rank)
	# No biases
	b_0 = np.random.uniform(0, 0, (1, n_hidden))
	b_1 = np.random.uniform(0, 0, (1, n_hidden))
	b_2 = np.random.uniform(0, 0, (1, n_out))

	nlin_map = lambda x: np.matmul(
		ReLU(np.matmul(ReLU(np.matmul(x, W_0) + np.tile(b_0, (x.shape[0], 1))), W_1) +
			 np.tile(b_1, (x.shape[0], 1))), W_2) + np.tile(b_2, (x.shape[0], 1))

	return nlin_map


def ReLU(x):
	return x * (x > 0)


def sigmoid(x):
	return 1 / (1 + np.exp(-x))


def generate_synthetic_data_default(p: int, n: int, k: int, seed: int):
	# Generative process: x --> c --> y

	"""
	Generate a nonlinear synthetic dataset

	@param p: number of covariates
	@param n: number of data points
	@param k: number of concepts
	@param seed: random generator seed
	@return: a design matrix of dimensions (@n, @p), concept values and labels
	"""
	# Replicability
	np.random.seed(seed)

	# Generate covariates
	mu = uniform(-5, 5, p)
	sigma = make_spd_matrix(p, random_state=seed)
	X = multivariate_normal(mean=mu, cov=sigma, size=n)
	ss = StandardScaler()
	X = ss.fit_transform(X)

	# Nonlinear maps
	g = random_nonlin_map(n_in=p, n_out=k, n_hidden=int((p + k) / 2))
	f = random_nonlin_map(n_in=k, n_out=1, n_hidden=int(k / 2))

	# Generate concepts
	c = g(X)
	tmp = np.tile(np.median(c, 0), (X.shape[0], 1))
	c = (c >= tmp) * 1.0

	# Generate labels
	y = f(c)
	tmp = np.tile(np.median(y, 0), (X.shape[0], 1))
	y = (y >= tmp) * 1.0

	return X, c, y

def generate_synthetic_data_unobserved_c(p: int, n: int, k: int, seed: int, j: int = 90):
	# Generative process: x --> c --> y, some concepts are not observed

	"""
	Generate a nonlinear synthetic dataset

	@param p: number of covariates
	@param n: number of data points
	@param k: number of concepts
	@param seed: random generator seed
	@return: a design matrix of dimensions (@n, @p), concept values and labels
	"""
	# Replicability
	np.random.seed(seed)

	# Generate covariates
	mu = uniform(-5, 5, p)
	sigma = make_spd_matrix(p, random_state=seed)
	X = multivariate_normal(mean=mu, cov=sigma, size=n)
	ss = StandardScaler()
	X = ss.fit_transform(X)

	# Nonlinear maps
	g = random_nonlin_map(n_in=p, n_out=k + j, n_hidden=int((p + k + j) / 2), rank=1000)
	f = random_nonlin_map(n_in=k + j, n_out=1, n_hidden=int((k + j)/ 2))

	# Generate concepts
	c = g(X)
	tmp = np.tile(np.median(c, 0), (X.shape[0], 1))
	c = (c >= tmp) * 1.0

	# Generate labels
	y = f(c)
	tmp = np.tile(np.median(y, 0), (X.shape[0], 1))
	y = (y >= tmp) * 1.0

	return X, c[:, 0:k], y


class SyntheticDataset(data.dataset.Dataset):
	"""
	Dataset class for the nonlinear synthetic data
	"""
	def __init__(self, num_vars: int, num_points: int, num_predicates: int, type: str = None,
				 indices: np.ndarray = None, seed: int = 42):
		"""
		Initializes the dataset.

		@param num_vars: number of covariates
		@param num_points: number of data points
		@param num_predicates: number of concepts
		@param indices: indices of the data points to be kept; the rest of the data points will be discarded
		@param seed: random generator seed
		"""
		# Shall a partial predicate set be used?
		self.predicate_idx = np.arange(0, num_predicates)

		generate_synthetic_data = None
		if type == 'default' or type is None:
			generate_synthetic_data = generate_synthetic_data_default
		elif type == 'unobserved_c':
			generate_synthetic_data = generate_synthetic_data_unobserved_c
		else:
			ValueError('Simulation type not implemented!')

		self.X, self.c, self.y = generate_synthetic_data(p=num_vars, n=num_points, k=num_predicates, seed=seed)

		if indices is not None:
			self.X = self.X[indices]
			self.c = self.c[indices]
			self.y = self.y[indices]

	def __getitem__(self, index):
		"""
		Returns points from the dataset

		@param index: index
		@return: a dictionary with the data; dict['features'] contains features, dict['label'] contains
		target labels, dict['concepts'] contains concepts
		"""
		labels = self.y[index, 0]
		concepts = self.c[index, self.predicate_idx]
		features = self.X[index]

		return {'features': features, 'labels': labels, 'concepts': concepts}

	def __len__(self):
		return self.X.shape[0]


def get_synthetic_dataloaders(num_vars: int, num_points: int, num_predicates: int, batch_size: int,
							  num_workers: int, train_ratio: float = 0.6, val_ratio: float = 0.2, seed: int = 42,
							  type: str = None):
	"""
	Constructs data loaders for the synthetic data

	@param num_vars: number of covariates
	@param num_points: number of data points
	@param num_predicates: number of concepts
	@param batch_size: batch size
	@param num_workers: number of worker processes
	@param train_ratio: the ratio specifying the training set size in the train-validation-test split
	@param val_ratio: the ratio specifying the validation set size in the train-validation-test split
	@param seed: random generator seed
	@return: a dictionary with data loaders
	"""

	# Train-validation-test split
	indices_train, indices_valtest = train_test_split(np.arange(0, num_points), train_size=train_ratio,
													  random_state=seed)
	indices_val, indices_test = train_test_split(indices_valtest, train_size=val_ratio / (1. - train_ratio),
												 random_state=2 * seed)

	# Datasets
	synthetic_datasets = {'train': SyntheticDataset(num_vars=num_vars, num_points=num_points,
													num_predicates=num_predicates, indices=indices_train,
													seed=seed, type=type),
						  'val': SyntheticDataset(num_vars=num_vars, num_points=num_points,
												  num_predicates=num_predicates, indices=indices_val,
												  seed=seed, type=type),
						  'test': SyntheticDataset(num_vars=num_vars, num_points=num_points,
												   num_predicates=num_predicates, indices=indices_test,
												   seed=seed, type=type)}
	# Data loaders
	synthetic_loaders = {x: torch.utils.data.DataLoader(synthetic_datasets[x], batch_size=batch_size, shuffle=True,
														num_workers=num_workers) for x in ['train', 'val', 'test']}

	return synthetic_loaders


def get_synthetic_datasets(num_vars: int, num_points: int, num_predicates: int,
						   train_ratio: float = 0.6, val_ratio: float = 0.2, use_val_train: bool = False, seed: int = 42, type: str = None):
	"""
	Constructs dataset objects for the synthetic data

	@param num_vars: number of covariates
	@param num_points: number of data points
	@param num_predicates: number of concepts
	@param train_ratio: the ratio specifying the training set size in the train-validation-test split
	@param val_ratio: the ratio specifying the validation set size in the train-validation-test split
	@param seed: random generator seed
	@return: dataset objects for the training, validation and test sets
	"""
	# Train-validation-test split
	indices_train, indices_valtest = train_test_split(np.arange(0, num_points), train_size=train_ratio,
													  random_state=seed)
	indices_val, indices_test = train_test_split(indices_valtest, train_size=val_ratio / (1. - train_ratio),
												 random_state=2 * seed)

	# Check if training with the validation set is required (in CBM val)
	if use_val_train:
		indices_train_new = indices_val
	else:
		indices_train_new = indices_train

	# Datasets
	synthetic_datasets = {'train': SyntheticDataset(num_vars=num_vars, num_points=num_points,
													num_predicates=num_predicates, indices=indices_train_new,
													seed=seed, type=type),
						  'val': SyntheticDataset(num_vars=num_vars, num_points=num_points,
												  num_predicates=num_predicates, indices=indices_val,
												  seed=seed, type=type),
						  'test': SyntheticDataset(num_vars=num_vars, num_points=num_points,
												   num_predicates=num_predicates, indices=indices_test,
												   seed=seed, type=type)}

	return synthetic_datasets['train'], synthetic_datasets['val'], synthetic_datasets['test']
