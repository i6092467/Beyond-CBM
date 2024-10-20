"""
Run this file, giving a configuration file as input, to finetune models, e.g.:
	python finetune.py --config configfile.yaml
"""

import argparse
import yaml
import os
import random
import torch
import numpy as np
import matplotlib.pyplot as plt

from copy import deepcopy
from models import BlackBox, CBM
from datasets.synthetic_dataset import get_synthetic_datasets
from datasets.awa_dataset import get_AwA_dataloaders
from datasets.SkinCon_dataset import get_SC_dataloaders
from datasets.CXR_dataset import get_CXR_dataloaders
from datasets.CUB_dataset import get_CUB_dataloaders
from datasets.CLIP_dataset import get_ImageNetCLIP_dataloaders
from torch import nn
from torch.utils.data import DataLoader
from validate import validate_epoch_black_box, validate_epoch_cbm,validate_epoch_concat_black_box
from probes import (train_lr_probes_post_hoc, train_rf_probes_post_hoc, train_torch_lin_probe_post_hoc,
                    train_torch_nlin_probe_post_hoc)
from utils.metrics import calc_target_metrics, calc_concept_metrics
from intervene import (intervene_on_representations, generate_random_subset_intervention_mask,
                       evaluate_representation_interventions, evaluate_cbm_interventions,evaluate_representation_interventions_concat,
                       RandomSubsetInterventionPolicy, UncertaintyInterventionPolicy,
                       finetune_intervenability_black_box,
                       finetune_concatenate_black_box, finetune_with_probe_black_box, finetune_post_hoc_CBM)
from sklearn.decomposition import PCA
from utils.plotting import (plot_curves_with_ci_lu, plot_calibration_curves,plot_calibration_curves_multiclass)
from sklearn.metrics import balanced_accuracy_score
from networks import FCNNProbe, LinearProbe


os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
BATCH_SIZE = 512

CB_COLOR_CYCLE = ['#377eb8', '#ff7f00', '#4daf4a', '#f781bf', '#a65628', '#984ea3',
                  '#999999', '#e41a1c', '#dede00', '#377eb8', '#ff7f00', '#4daf4a',
                  '#f781bf', '#a65628', '#984ea3', '#999999', '#e41a1c', '#dede00']

parser = argparse.ArgumentParser()
parser.add_argument('-c', '--config')
parser.add_argument('-s', '--seed', type=int)
parser.add_argument('-d_p', '--data_path')
parser.add_argument('-v_r', '--val_ratio')
parser.add_argument('-rn', '--run_name')
parser.add_argument('-ft', '--type_ft')
args = parser.parse_args()
argsdict = vars(args)

with open(argsdict['config'], 'r') as f:
    config = yaml.safe_load(f)
config['filename'] = argsdict['config']
print('Experiment: ', config['filename'])

if argsdict['seed'] is not None:
    config['seed'] = argsdict['seed']
if argsdict['run_name'] is not None:
    config['run_name'] = argsdict['run_name']
if argsdict['data_path'] is not None:
    config['data_path'] = argsdict['data_path']
if argsdict['val_ratio'] is not None:
    config['val_ratio'] = float(argsdict['val_ratio'])

print('EXPERIMENT: ', config['filename'], 'Seed: ', config['seed'], 'Run name: ', config['run_name'], ' Batch size: ', BATCH_SIZE)

# Ensure reproducibility
random.seed(config['seed'])
np.random.seed(config['seed'])
gen = torch.manual_seed(config['seed'])
torch.backends.cudnn.benchmark = False
torch.use_deterministic_algorithms(True)


# Load trained model
MODEL_PATH = os.path.join(config['log_directory'], 'checkpoints/final_model_' + config['run_name'] + '_' +
                           config['experiment_name'] + '_' + str(config['seed']) + '.pth'
                           )
print('Loading pre-trained model: ',MODEL_PATH)
# Instantiate the architecture
if config['model'] == 'black-box':
    model = BlackBox(config)

# Load the trained model
model.load_state_dict(torch.load(MODEL_PATH))
model.eval()
model.to(torch.device(config['device']))

# Disable gradients
for param in model.parameters():
    param.requires_grad = False

# Generate the dataset
gen = torch.manual_seed(config['seed'])

if config['dataset'] == 'synthetic':
    print('Synthetic DATASET')
    trainset, validset, testset = get_synthetic_datasets(
        num_vars=config['num_covariates'], num_points=config['num_points'],
        num_predicates=config['num_concepts'], train_ratio=0.6, val_ratio=config['val_ratio'], seed=config['seed'], # 0.2
        type=config['sim_type'])
elif config['dataset'] == 'awa2':
    print('AWA2 DATASET')
    trainset, validset, testset = get_AwA_dataloaders("all_classes.txt", config['train_batch_size'], config["workers"],
                                                      config["data_path"], train_ratio=0.6, val_ratio=0.2, seed=config['seed'])
elif config['dataset'] == 'CUB':
    print('CUB DATASET')
    trainset, validset, testset = get_CUB_dataloaders(config["data_path"], train_val_split=0.6, test_val_split=0.5,
                                                      seed=config['seed'])
elif config['dataset'] == 'CXR':
    print('CXR DATASET')
    trainset, validset, testset = get_CXR_dataloaders(config['dataset'], config["data_path"], train_val_split=0.8,
                                                      test_val_split=0.5, seed=config['seed'])
elif config['dataset'] == 'cheXpert':
    print('cheXpert DATASET')
    trainset, validset, testset = get_CXR_dataloaders(config['dataset'], config["data_path"], train_val_split=0.8,
                                                      test_val_split=0.5, seed=config['seed'])
elif config['dataset'] in ('SD_ImageNet', 'CIFAR10'):
    trainset, validset, testset = get_ImageNetCLIP_dataloaders(config['dataset'], config["data_path"],
                                                               test_val_split=0.5, seed=config['seed'])
else:
    NotImplementedError('ERROR: Dataset not supported!')

pm = config['device'] == 'cuda'
train_loader = DataLoader(trainset, batch_size=config['train_batch_size'], num_workers=config['workers'],
                          pin_memory=pm, generator=gen)
# NOTE: use training batch size for the validation set since probes are trained on it
val_loader = DataLoader(validset, batch_size=config['train_batch_size'], num_workers=config['workers'],
                        pin_memory=pm, generator=gen)
test_loader = DataLoader(testset, batch_size=config['val_batch_size'], num_workers=config['workers'],
                         pin_memory=pm, generator=gen)

# Target prediction loss
if config['num_classes'] == 2:
    loss_fn_y = nn.BCELoss()
elif config['num_classes'] > 2:
    loss_fn_y = nn.CrossEntropyLoss()


print('Training linear probe')
# Train a linear concept probe using PyTorch on the validation set
lin_probe = train_torch_lin_probe_post_hoc(
    model, val_loader, config, num_epochs_probe=100, learning_rate_probe=0.01, weight_decay_probe=0, #150
    optimizer_probe='sgd')
for param in lin_probe.parameters():
    param.requires_grad = False
print("Post hoc CBM")
config_pCBM = {'learning_rate': 0.0001, 'weight_decay': 0.0, 'optimizer': 'adam',
              'log_directory': config['log_directory'], 'num_epochs': 100, 'train_batch_size': BATCH_SIZE,
              'device': 'cuda', 'dataset': config['dataset'], 'alpha': 1.0, 'run_name': config['run_name'],
              'num_classes': config['num_classes'], 'experiment_name': config['experiment_name'], 'seed': config['seed']}
config['intervenability'] = False
post_hoc_CBM = finetune_post_hoc_CBM(config, model, val_loader)


print("Fine-tuning for intervenability")
model_ft = deepcopy(model)
config_ft = {'learning_rate': 0.0001, 'weight_decay': 0.0, 'optimizer': 'adam',
             'log_directory': config['log_directory'], 'num_epochs': 100, 'train_batch_size': BATCH_SIZE,
             'lmbd': 0.8, 'learning_rate_inter': 0.01, 'weight_decay_inter': 0.0, 'num_epochs_inter': 10000,
             'optimizer_inter': 'adam', 'eps': 1e-4, 'device': 'cuda', 'perc': 0.5,
             'dataset': config['dataset'], 'num_classes': config['num_classes'],
             'run_name': config['run_name'], 'experiment_name': config['experiment_name'], 'seed': config['seed'], 'val_ratio': config['val_ratio']}
intervention_policy = RandomSubsetInterventionPolicy()
model_ft, probe_ft = finetune_intervenability_black_box(
    config=config_ft, model=model_ft, probe=lin_probe, loss_fn_y=loss_fn_y,
    intervention_policy=intervention_policy, data_loader=val_loader, max_iter=None)

print("Fine-tuning for multitask")
model_ft_ = deepcopy(model)
probe_ft_ = deepcopy(lin_probe)
config_ft_ = {'learning_rate': 0.0001, 'weight_decay': 0.0, 'optimizer': 'adam',
              'log_directory': config['log_directory'], 'num_epochs': 100, 'train_batch_size': BATCH_SIZE,
              'device': 'cuda', 'dataset': config['dataset'], 'alpha': 1.0,'run_name':config['run_name'],
              'num_classes': config['num_classes'], 'experiment_name':config['experiment_name'],'seed':config['seed'],'val_ratio': config['val_ratio']}
model_ft_, probe_ft_ = finetune_with_probe_black_box(config_ft_, model_ft_,  probe_ft_, loss_fn_y, nn.BCELoss(),
                                                    val_loader, max_iter=None)
prev = config['experiment_name']
config['experiment_name'] = prev + '_mtFT'
probe_ft_ = train_torch_lin_probe_post_hoc(
    model_ft_, val_loader, config, num_epochs_probe=100, learning_rate_probe=0.01, weight_decay_probe=0,
    optimizer_probe='sgd')
config['experiment_name'] = prev

print("Fine-tuning for concatenation")
config_concat = {'learning_rate': 0.0001, 'weight_decay': 0.0, 'optimizer': 'adam',
                 'log_directory': config['log_directory'], 'num_epochs': 100, 'train_batch_size': BATCH_SIZE,
                 'lmbd': 0.8, 'learning_rate_inter': 0.01, 'weight_decay_inter': 0.0, 'num_epochs_inter': 10000,
                 'num_classes': config['num_classes'], 'optimizer_inter': 'adam', 'eps': 1e-4, 'device': 'cuda',
                 'perc': 0.5, 'dataset': config['dataset'], 'concatenation': True, 'run_name': config['run_name'],
                 'experiment_name': config['experiment_name'], 'seed': config['seed'], 'val_ratio': config['val_ratio']}
config['concatenation'] = True
model_concat = BlackBox(config)
model_concat.encoder.load_state_dict(model.encoder.state_dict())
model_concat.to(torch.device(config['device']))
model_concat = finetune_concatenate_black_box(
    config=config_concat, model=model_concat, loss_fn_y=loss_fn_y, data_loader=val_loader, max_iter=None)
config['concatenation'] = False


print("Finished finetuning")