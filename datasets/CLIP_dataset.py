import torch

import numpy as np

import pandas as pd

from PIL import Image

from torch.utils.data import DataLoader, Dataset

from torchvision import transforms

# from utils.misc_utils import set_seeds

from sklearn.model_selection import train_test_split

class CLIP_DatasetGenerator(Dataset):
    """ImageNet and CIFAR-10 SD embeddings Dataset object"""
    def __init__(self,features, class_labels, concept_labels):
        self.embs = features
        self.img_label = class_labels
        self.concept_label = concept_labels


    def __getitem__(self, index):
        # Gets an element of the dataset

        imageData = self.embs[index]

        image_label = self.img_label[index]

        image_attr = self.concept_label[index]

        # Return a tuple of images, labels, and protected attributes
        return {'img_code': index, 'labels': image_label,
                'features': imageData, 'concepts': image_attr}

    def __len__(self):

        return len(self.img_label)

def train_test_split_CLIP(dataset, root_dir, test_val_split = 0.5, seed=42):
    """Performs train-validation-test split for the SD embeddings of the ImageNet and CIFAR-10 datasets"""

    np.random.seed(seed)
    if dataset == 'SD_ImageNet':
        train_feat = torch.load(root_dir + '/StableDiffusionEmbeddings/train_embeddings_SD.pt')
        train_concepts = torch.load(root_dir + 'train_concept_labels.pt')
        train_class = torch.load(root_dir + 'train_class_labels.pt')
        train_feat = train_feat.reshape(train_feat.shape[0], -1)

        val_orig_feat = torch.load(root_dir + '/StableDiffusionEmbeddings/val_embeddings_SD.pt')
        val_orig_concepts = torch.load(root_dir + 'val_concept_labels.pt')
        val_orig_class = torch.load(root_dir + 'val_class_labels.pt')
        val_orig_feat = val_orig_feat.reshape(val_orig_feat.shape[0], -1)
        val_feat, test_feat, val_concepts, test_concepts, val_class, test_class = train_test_split(val_orig_feat,
                                                                                                   val_orig_concepts,
                                                                                                   val_orig_class,
                                                                                                   test_size=test_val_split, # test_val_split
                                                                                                   shuffle=True,
                                                                                                   random_state=seed)
        conc_idx = np.load(root_dir + f"/100_final_concepts_val.npy")

        test_concepts = test_concepts[:, conc_idx]
        val_concepts = val_concepts[:, conc_idx]
        train_concepts = train_concepts[:, conc_idx]

    elif dataset == 'CIFAR10':
        train_orig_feat = torch.load(root_dir + 'train_cifar10_embeddings_SD.pt')
        train_orig_concepts = torch.load(root_dir + 'cifar10_train_concept_labels.pt')
        train_orig_class = torch.load(root_dir + 'train_cifar10_class_labels_SD.pt')
        train_orig_feat = train_orig_feat.reshape(train_orig_feat.shape[0], -1)

        train_feat, val_feat, train_concepts, val_concepts, train_class, val_class = train_test_split(train_orig_feat,
                                                                                                   train_orig_concepts,
                                                                                                   train_orig_class,
                                                                                                   test_size=0.25,
                                                                                                   shuffle=True,
                                                                                                   random_state=seed)
        test_feat = torch.load(root_dir + 'test_cifar10_embeddings_SD.pt')
        test_concepts = torch.load(root_dir + 'cifar10_test_concept_labels.pt')
        test_class = torch.load(root_dir + 'test_cifar10_class_labels_SD.pt')
        test_feat = test_feat.reshape(test_feat.shape[0], -1)

    return train_feat, train_concepts, train_class, val_feat, val_concepts, val_class,  \
        test_feat, test_concepts, test_class

def get_ImageNetCLIP_dataloaders(dataset,root_dir, test_val_split = 0.5, seed=42):
    """Returns a dictionary of data loaders for the SD embeddings of the ImageNet and CIFAR-10 datasets, for the training, validation, and test sets."""

    train_feat, train_concepts, train_class, val_feat, val_concepts, val_class, \
        test_feat, test_concepts, test_class = \
        train_test_split_CLIP(dataset= dataset, root_dir=root_dir, test_val_split = test_val_split, seed=seed)


    # Datasets
    image_datasets = {'train': CLIP_DatasetGenerator(train_feat, train_class, train_concepts),
                      'val': CLIP_DatasetGenerator(val_feat, val_class, val_concepts),
                      'test': CLIP_DatasetGenerator(test_feat, test_class, test_concepts)}

    return image_datasets['train'], image_datasets['val'], image_datasets['test']
