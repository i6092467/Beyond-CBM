import pickle
import numpy as np
import os
import torchvision.transforms as transforms
from PIL import Image
from torch.utils.data import Dataset

class CUB_DatasetGenerator(Dataset):
    """CUB Dataset object"""
    def __init__(self, data_pkl, transform=None):
        """
        Arguments:
        pkl_file_paths: list of full path to all the pkl data
        transform: whether to apply any special transformation. Default = None, i.e. use standard ImageNet preprocessing
        """
        self.data = data_pkl
        self.transform = transform
        self.img_index = [0,1,2,3]
    def __getitem__(self, index):
        # Gets an element of the dataset
        img_data = self.data.data[index]
        img_path = img_data['img_path']
        imageData = Image.open(img_path).convert('RGB')
        imageData = imageData.resize((224, 224))
        image_label = img_data['class_label']

        image_attr = np.array(img_data['attribute_label'])

        if self.transform != None: imageData = self.transform(imageData)

        # Return a tuple of images, labels, and protected attributes
        return {'img_code': index, 'labels': image_label,
                'features': imageData, 'concepts': image_attr}

    def __len__(self):
        return len(self.data)

def train_test_split_CUB(root_dir, train_val_split = 0.6, test_val_split = 0.5, seed=42):
    """Performs train-validation-test split for the CUB dataset"""

    np.random.seed(seed)
    data_train = []
    data_train.extend(pickle.load(open(os.path.join(root_dir,'train.pkl'), 'rb')))
    CUB_complete_train = CUB_DatasetGenerator(data_train, transform=None)
    data_test = []
    data_test.extend(pickle.load(open(os.path.join(root_dir,'test.pkl'),'rb')))
    CUB_complete_test = CUB_DatasetGenerator(data_test, transform=None)
    data_val = []
    data_val.extend(pickle.load(open(os.path.join(root_dir,'val.pkl'), 'rb')))
    CUB_complete_val = CUB_DatasetGenerator(data_val, transform=None)

    return CUB_complete_train, CUB_complete_val, CUB_complete_test


def get_CUB_dataloaders(root_dir, train_val_split = 0.6, test_val_split = 0.5, seed=42):
    """Returns a dictionary of data loaders for the CUB, for the training, validation, and test sets."""

    train_imgs, val_imgs, test_imgs = \
        train_test_split_CUB(root_dir=root_dir, train_val_split = train_val_split, test_val_split = test_val_split, seed=seed)

    # Following the transformations from CBM paper
    resol = 299

    train_transform = transforms.Compose([
        transforms.ColorJitter(brightness=32 / 255, saturation=(0.5, 1.5)),
        transforms.RandomResizedCrop(resol),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),  # implicitly divides by 255
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[2, 2, 2])
    ])

    test_transform = transforms.Compose([
        transforms.CenterCrop(resol),
        transforms.ToTensor(),  # implicitly divides by 255
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[2, 2, 2])
    ])

    # Datasets
    image_datasets = {'train': CUB_DatasetGenerator(train_imgs,
                                                                transform=train_transform),
                      'val': CUB_DatasetGenerator(val_imgs,
                                                              transform=test_transform),
                      'test': CUB_DatasetGenerator(test_imgs,
                                                               transform=test_transform)}

    return image_datasets['train'], image_datasets['val'], image_datasets['test']
