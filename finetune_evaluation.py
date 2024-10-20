"""
Run this file, giving two configuration files as input, to evaluate models, e.g.:
	python finetune_evaluation.py --config configfile_blackbox.yaml --config_CBM configfile_cbm.yaml
"""
import argparse
import yaml
import os
import random
import torch
import numpy as np
import matplotlib.pyplot as plt

from copy import deepcopy
from models import BlackBox, CBM, pCBM
from datasets.synthetic_dataset import get_synthetic_datasets
from datasets.awa_dataset import get_AwA_dataloaders
from datasets.SkinCon_dataset import get_SC_dataloaders
from datasets.CXR_dataset import get_CXR_dataloaders
from datasets.CUB_dataset import get_CUB_dataloaders

from torch import nn
from losses import create_loss
from torch.utils.data import DataLoader
from validate import validate_epoch_black_box, validate_epoch_cbm,validate_epoch_concat_black_box
from probes import (train_lr_probes_post_hoc, train_rf_probes_post_hoc, train_torch_lin_probe_post_hoc,
                    train_torch_nlin_probe_post_hoc)
from utils.metrics import calc_target_metrics, calc_concept_metrics
from intervene import (intervene_on_representations,
                       evaluate_representation_interventions, evaluate_cbm_interventions,evaluate_representation_interventions_concat,
                       finetune_intervenability_black_box,
                       finetune_concatenate_black_box, finetune_with_probe_black_box)
from policies import (RandomSubsetInterventionPolicy, UncertaintyInterventionPolicy,
                      generate_random_subset_intervention_mask)
from sklearn.decomposition import PCA
from utils.plotting import (plot_curves_with_ci_lu, plot_calibration_curves,plot_calibration_curves_multiclass)
from sklearn.metrics import balanced_accuracy_score
from networks import FCNNProbe, LinearProbe
from datasets.CLIP_dataset import get_ImageNetCLIP_dataloaders

os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
BATCH_SIZE = 512

CB_COLOR_CYCLE = ['#377eb8', '#ff7f00', '#4daf4a', '#f781bf', '#a65628', '#984ea3',
                  '#999999', '#e41a1c', '#dede00', '#377eb8', '#ff7f00', '#4daf4a',
                  '#f781bf', '#a65628', '#984ea3', '#999999', '#e41a1c', '#dede00']

parser = argparse.ArgumentParser()
parser.add_argument('-c', '--config')
parser.add_argument('-c_cbm', '--config_CBM')
parser.add_argument('-b', '--batch_size')
parser.add_argument('-i_p', '--inter_policy')
parser.add_argument('-s', '--seed', type=int)
parser.add_argument('-d_p', '--data_path')
parser.add_argument('-rn', '--run_name')
parser.add_argument('-ld', '--log_directory')
parser.add_argument('-ft', '--type_ft')
args = parser.parse_args()
argsdict = vars(args)

with open(argsdict['config'], 'r') as f:
    config = yaml.safe_load(f)
config['filename'] = argsdict['config']
config['inter_policy'] = 'random'
print('Experiment: ', config['filename'])

if argsdict['seed'] is not None:
    config['seed'] = argsdict['seed']
if argsdict['run_name'] is not None:
    config['run_name'] = argsdict['run_name']
if argsdict['data_path'] is not None:
    config['data_path'] = argsdict['data_path']
if argsdict['config_CBM'] is not None:
    config['config_CBM'] = argsdict['config_CBM']
if argsdict['batch_size'] is not None:
    config['batch_size'] = argsdict['batch_size']
if argsdict['inter_policy'] is not None:
    config['inter_policy'] = argsdict['inter_policy']

print('EXPERIMENT: ', config['filename'], 'Seed: ', config['seed'], 'Run name: ', config['run_name'],
      ' Batch size: ', BATCH_SIZE)

# Ensure reproducibility
random.seed(config['seed'])
np.random.seed(config['seed'])
gen = torch.manual_seed(config['seed'])
torch.backends.cudnn.benchmark = False
torch.use_deterministic_algorithms(True)
print('SEED: ', config['seed'])

# Generate the dataset
gen = torch.manual_seed(config['seed'])

if config['dataset'] == 'synthetic':
    print('Synthetic DATASET')
    trainset, validset, testset = get_synthetic_datasets(
        num_vars=config['num_covariates'], num_points=config['num_points'],
        num_predicates=config['num_concepts'], train_ratio=0.6, val_ratio=0.2, seed=config['seed'], type=config['sim_type'])
elif config['dataset'] == 'awa2':
    print('AWA2 DATASET')
    trainset, validset, testset = get_AwA_dataloaders("all_classes.txt", config['train_batch_size'], config["workers"],
                                                      config["data_path"], train_ratio=0.6, val_ratio=0.2, seed=config['seed'])
elif config['dataset'] == 'CXR':
    print('CXR DATASET')
    trainset, validset, testset = get_CXR_dataloaders(config['dataset'], config["data_path"], train_val_split=0.8,
                                                      test_val_split=0.5, seed=config['seed'])
elif config['dataset'] == 'CUB':
    print('CUB DATASET')
    trainset, validset, testset = get_CUB_dataloaders(config["data_path"], train_val_split=0.6, test_val_split=0.5,
                                                      seed=config['seed'])
elif config['dataset'] == 'cheXpert':
    print('cheXpert DATASET')
    trainset, validset, testset = get_CXR_dataloaders(config['dataset'], config["data_path"], train_val_split=0.8,
                                                      test_val_split=0.5, seed=config['seed'])
elif config['dataset'] in ('SD_ImageNet', 'CIFAR10'):
    trainset, validset, testset = get_ImageNetCLIP_dataloaders(config['dataset'], config["data_path"],
                                                               test_val_split=0.5, seed=config['seed'])

pm = config['device'] == 'cuda'
train_loader = DataLoader(trainset, batch_size=config['train_batch_size'], num_workers=config['workers'],
                          pin_memory=pm, generator=gen)
# NOTE: use training batch size for the validation set since we will train probes on it
val_loader = DataLoader(validset, batch_size=config['train_batch_size'], num_workers=config['workers'],
                        pin_memory=pm, generator=gen)
test_loader = DataLoader(testset, batch_size=BATCH_SIZE, num_workers=config['workers'],
                         pin_memory=pm, generator=gen)

# Load trained model
MODEL_PATH = os.path.join(config['log_directory'], 'checkpoints/final_model_' + config['run_name'] + '_' +
                           config['experiment_name'] + '_' + str(config['seed']) + '.pth'
                           )

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

# Target prediction loss
if config['num_classes'] ==2:
    loss_fn_y = nn.BCELoss()
elif config['num_classes'] >2:
    loss_fn_y = nn.CrossEntropyLoss()

# Train a linear concept probe using PyTorch on the validation set
lin_probe = LinearProbe(num_inputs=model.num_hidden_z, num_outputs=config['num_concepts'])
MODEL_PATH_probe = os.path.join(os.path.join(config['log_directory'], 'checkpoints'), 'linear_probe' +
                                config['run_name'] + '_' + config['experiment_name'] + '_' + str(config['seed']) + '.pth')
lin_probe.load_state_dict(torch.load(MODEL_PATH_probe))
lin_probe.eval()
lin_probe.to(torch.device(config['device']))
for param in lin_probe.parameters():
    param.requires_grad = False

print("Evaluation and intervenability of BB-FTI")
# Fine-tune for intervenability
model_ft = deepcopy(model)
model_ft.load_state_dict(torch.load(os.path.join(os.path.join(config['log_directory'], 'checkpoints'), 'final_ft_model_' +  config['run_name'] + '_' +
											config['experiment_name'] + '_' + str(config['seed']) + '.pth')))
model_ft.eval()
model_ft.to(torch.device(config['device']))
# Disable gradients
for param in model_ft.parameters():
    param.requires_grad = False

print("Evaluation of BB-FTA")
# Fine-tune concatenating concepts and zs
config_concat = {'learning_rate': 0.0001, 'weight_decay': 0.0, 'optimizer': 'adam',
             'log_directory': config['log_directory'], 'num_epochs': 150, 'train_batch_size': BATCH_SIZE,
             'lmbd': 0.8, 'learning_rate_inter': 0.01, 'weight_decay_inter': 0.0, 'num_epochs_inter': 10000,
             'num_classes': config['num_classes'], 'optimizer_inter': 'adam', 'eps': 1e-4, 'device': 'cuda',
             'perc': 0.5,'dataset':config['dataset'],'concatenation':True,'run_name':config['run_name'],
            'experiment_name':config['experiment_name'],'seed':config['seed']}
config['concatenation']=True
model_concat = BlackBox(config)
model_concat.encoder.load_state_dict(model.encoder.state_dict())
model_concat.load_state_dict(torch.load(os.path.join(os.path.join(config['log_directory'], 'checkpoints'), 'final_ft_concat_model_' + config['run_name'] + '_' +
													config['experiment_name'] + '_' + str(config['seed']) + '.pth')))
model_concat.eval()
model_concat.to(torch.device(config['device']))

# Disable gradients
for param in model_concat.parameters():
    param.requires_grad = False
config['concatenation']=False

print("Evaluation of BB-FTMT")
model_ft_ = deepcopy(model)
model_ft_.load_state_dict(torch.load(os.path.join(os.path.join(config['log_directory'], 'checkpoints'), 'final_ft_mt_model_' + config['run_name'] + '_' +
													config['experiment_name'] + '_' + str(config['seed']) + '.pth')))
model_ft_.eval()
model_ft_.to(torch.device(config['device']))
for param in model_ft_.parameters():
    param.requires_grad = False
probe_ft_ = deepcopy(lin_probe)
conf_old = config['experiment_name']
config['experiment_name'] = conf_old + '_mtFT'
MODEL_PATH_probe = os.path.join(os.path.join(config['log_directory'], 'checkpoints'), 'linear_probe' +  config['run_name'] + '_' +
										config['experiment_name'] + '_' + str(config['seed']) + '.pth')
probe_ft_.load_state_dict(torch.load(MODEL_PATH_probe))
probe_ft_.eval()
probe_ft_.to(torch.device(config['device']))
for param in probe_ft_.parameters():
    param.requires_grad = False
config_ft_ = {'learning_rate': 0.0001, 'weight_decay': 0.0, 'optimizer': 'adam',
              'log_directory': config['log_directory'], 'num_epochs': 50, 'train_batch_size': BATCH_SIZE,
              'device': 'cuda', 'dataset': config['dataset'], 'alpha': 1.0,'num_concepts': config['num_concepts'],'num_classes': config['num_classes']}
config['experiment_name'] = conf_old
print("Evaluation of CBM")

with open(config['config_CBM'], 'r') as f:
    config_cbm = yaml.safe_load(f)
if config_cbm['suppress_warnings']:
    import warnings
    warnings.filterwarnings('ignore')
config_cbm['seed'] = config['seed']
MODEL_PATH_CBM = os.path.join(config_cbm['log_directory'], 'checkpoints/final_model_' +
                              config_cbm['run_name'] + '_' + config_cbm['experiment_name'] + '_' + str(config_cbm['seed']) + '.pth')
if config_cbm.get('binary'):
    config_cbm['num_concepts'] = len(testset.concepts[0])
model_cbm = CBM(config_cbm)
# Load the trained model
model_cbm.load_state_dict(torch.load(MODEL_PATH_CBM))
model_cbm.eval()
model_cbm.to(torch.device(config_cbm['device']))
# Disable gradients
for param in model_cbm.parameters():
    param.requires_grad = False


print("Evaluation of pCBM")

MODEL_PATH_PCBM = os.path.join(os.path.join(config['log_directory'], 'checkpoints'), 'final_ft_pCBM_model_' + \
                               config['run_name'] + '_' + config['experiment_name'] + '_' + str(
    config['seed']) + '.pth')

print('This is loading: ' + MODEL_PATH_PCBM)

model_pcbm = pCBM(config, deepcopy(model.encoder))
# Load the trained model
model_pcbm.load_state_dict(torch.load(MODEL_PATH_PCBM))
model_pcbm.eval()
model_pcbm.to(torch.device(config['device']))

# Disable gradients
for param in model_pcbm.parameters():
    param.requires_grad = False


print("Studying interventions")
if config['inter_policy'] == 'random':
    print('Random interventions')
    intervention_policy = RandomSubsetInterventionPolicy()
elif config['inter_policy'] == 'unc':
    print('Uncertainty based interventions')
    intervention_policy = UncertaintyInterventionPolicy()
intervention_params = {'lmbd': 0.8, 'step_size': 0.01, 'weight_decay': 0.0, 'num_epochs': 100000,
                       'optimizer': 'adam', 'eps': 1e-6}

# BB
losses, aurocs, auprs = evaluate_representation_interventions(
    model=model, probe=lin_probe, dataloader=test_loader, loss_fn_y=loss_fn_y,
    intervention_policy=intervention_policy, num_steps=10, num_batches=len(test_loader),
    intervention_params=intervention_params,
    config=config)
# BB - FTI
losses_ft, aurocs_ft, auprs_ft = evaluate_representation_interventions(
    model=model_ft, probe=lin_probe, dataloader=test_loader, loss_fn_y=loss_fn_y,
    intervention_policy=intervention_policy, num_steps=10, num_batches=len(test_loader),
    intervention_params=intervention_params,
    config=config)
# BB - FTA
losses_concat, aurocs_concat, auprs_concat = evaluate_representation_interventions_concat(
    model=model_concat, dataloader=test_loader, loss_fn_y=loss_fn_y,
    num_steps=10, num_batches=len(test_loader), config=config)
# BB - FTMT
losses_ft_, aurocs_ft_, auprs_ft_ = evaluate_representation_interventions(
    model=model_ft_, probe=probe_ft_, dataloader=test_loader, loss_fn_y=loss_fn_y,
    intervention_policy=intervention_policy, num_steps=10, num_batches=len(test_loader),
    intervention_params=intervention_params,
    config=config)
# BB - CBM
losses_cbm, aurocs_cbm, auprs_cbm = evaluate_cbm_interventions(
    model=model_cbm, dataloader=test_loader, loss_fn_y=loss_fn_y,
    intervention_policy=intervention_policy, num_steps=10, num_batches=len(test_loader),
    config=config_cbm)

# BB - pCBM
print('Intervening on pCBM')
losses_pcbm, aurocs_pcbm, auprs_pcbm = evaluate_cbm_interventions(
    model=model_pcbm, dataloader=test_loader, loss_fn_y=loss_fn_y,
    intervention_policy=intervention_policy, num_steps=10, num_batches=len(test_loader),
    config=config)

print('Generating plots')

xs = [100 * np.linspace(0, 1, 10),
      100 * np.linspace(0, 1, 10),
      100 * np.linspace(0, 1, 10),
      100 * np.linspace(0, 1, 10),
      100 * np.linspace(0, 1, 10),
      100 * np.linspace(0, 1, 10)]
avgs = [np.median(aurocs, 0),
        np.median(aurocs_cbm, 0),
        np.median(aurocs_ft, 0),
        np.median(aurocs_concat, 0),
        np.median(aurocs_ft_, 0),
        np.median(aurocs_pcbm, 0)]
lower = [np.quantile(aurocs, 0.25, 0),
         np.quantile(aurocs_cbm, 0.25, 0),
         np.quantile(aurocs_ft, 0.25, 0),
         np.quantile(aurocs_concat, 0.25, 0),
         np.quantile(aurocs_ft_, 0.25, 0),
         np.quantile(aurocs_pcbm, 0.25, 0)]
upper = [np.quantile(aurocs, 0.75, 0),
         np.quantile(aurocs_cbm, 0.75, 0),
         np.quantile(aurocs_ft, 0.75, 0),
         np.quantile(aurocs_concat, 0.75, 0),
         np.quantile(aurocs_ft_, 0.75, 0),
         np.quantile(aurocs_pcbm, 0.75, 0)]
labels = ['Black box', 'CBM', 'Fine-tuned, I', 'Fine-tuned, A', 'Fine-tuned, MT', 'pCBM']
xlab = '% concepts intervened on'
ylab = 'AUROC'
plot_curves_with_ci_lu(xs, avgs, lower, upper, labels, xlab, ylab, font_size=20, title=None, baseline=None,
                       baseline_lab=None, baseline_cl=None,
                       dir='./AUROC_evaluation_' + config[
                           'experiment_name'] + "_" + str(config['seed']) + '_' + config['inter_policy'] + '.png',
                       legend=False, legend_outside=True, cls=[CB_COLOR_CYCLE[0], CB_COLOR_CYCLE[7],
                                                               CB_COLOR_CYCLE[3], CB_COLOR_CYCLE[1], CB_COLOR_CYCLE[2],
                                                               CB_COLOR_CYCLE[4]],
                       ms=None, markersize=8, linewidth=3, figsize=(7, 5), ylim=None, tick_step=None, grid=False)

xs = [100 * np.linspace(0, 1, 10),
      100 * np.linspace(0, 1, 10),
      100 * np.linspace(0, 1, 10),
      100 * np.linspace(0, 1, 10),
      100 * np.linspace(0, 1, 10),
      100 * np.linspace(0, 1, 10)]
avgs = [np.median(auprs, 0),
        np.median(auprs_cbm, 0),
        np.median(auprs_ft, 0),
        np.median(auprs_concat, 0),
        np.median(auprs_ft_, 0),
        np.median(auprs_pcbm, 0)]
lower = [np.quantile(auprs, 0.25, 0),
         np.quantile(auprs_cbm, 0.25, 0),
         np.quantile(auprs_ft, 0.25, 0),
         np.quantile(auprs_concat, 0.25, 0),
         np.quantile(auprs_ft_, 0.25, 0),
         np.quantile(auprs_pcbm, 0.25, 0)]
upper = [np.quantile(auprs, 0.75, 0),
         np.quantile(auprs_cbm, 0.75, 0),
         np.quantile(auprs_ft, 0.75, 0),
         np.quantile(auprs_concat, 0.75, 0),
         np.quantile(auprs_ft_, 0.75, 0),
         np.quantile(auprs_pcbm, 0.75, 0)]
labels = ['Black box', 'CBM', 'Fine-tuned, I', 'Fine-tuned, A', 'Fie-tuned, MT', 'pCBM']
xlab = '% concepts intervened on'
ylab = 'AUPR'
plot_curves_with_ci_lu(xs, avgs, lower, upper, labels, xlab, ylab, font_size=20, title=None, baseline=None,
                       baseline_lab=None, baseline_cl=None,
                       dir='./AUPR_evaluation_' + config[
                           'experiment_name'] + "_" + str(config['seed']) + '_' + config['inter_policy'] + '.png',
                       legend=False, legend_outside=True, cls=[CB_COLOR_CYCLE[0], CB_COLOR_CYCLE[7],
                                                               CB_COLOR_CYCLE[3], CB_COLOR_CYCLE[1], CB_COLOR_CYCLE[2],
                                                               CB_COLOR_CYCLE[4]],
                       ms=None, markersize=8, linewidth=3, figsize=(7, 5), ylim=None, tick_step=None, grid=False)

print('Finished')