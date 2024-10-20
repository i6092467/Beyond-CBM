"""
Chest X-ray datasets
"""
import numpy as np

import pandas as pd

from PIL import Image

from torch.utils.data import DataLoader, Dataset

from torchvision import transforms

from sklearn.model_selection import train_test_split


class ChestXRay_mimic_DatasetGenerator(Dataset):
    """Chest X-ray Dataset object for CheXpert and MIMIC-CXR"""
    def __init__(self, imgs, img_list, label_list, transform):

        self.img_index = []
        self.listImageAttributes = []
        self.transform = transform
        self.imgs = imgs

        # Iterate over images and retrieve labels and protected attributes
        for i in range(len(img_list)):

            row = label_list.iloc[i]
            imageLabel = row['No Finding']
            imageAttr = np.array(row[row.index != 'No Finding'])
            imageAttr[np.isnan(imageAttr)] = 0
            # We convert the concept labels to binary by mapping the uncertain cases, if any, to 0
            imageAttr[imageAttr == - 1] = 0
            if imageLabel == 1:
                imgLabel = 0
            else:
                imgLabel = 1

            self.img_index.append(imgLabel)
            self.listImageAttributes.append(imageAttr)

    def __getitem__(self, index):
        # Gets an element of the dataset

        imageData = Image.fromarray(self.imgs[index]).convert('RGB')
        image_label = self.img_index[index]
        image_attr = self.listImageAttributes[index]
        if self.transform != None:
            imageData = self.transform(imageData)

        # Return a tuple of images, labels, and attributes
        return {'img_code': index, 'labels': image_label,
                'features': imageData, 'concepts': image_attr}

    def __len__(self):
        return len(self.img_index)


def train_test_split_CXR(dataset, root_dir, train_val_split = 0.6, test_val_split = 0.5, seed=42):
    """Performs train-validation-test split for the CheXpert and MIMIC-CXR dataset"""

    np.random.seed(seed)
    if dataset == 'CXR':
        # Loading class labels for MIMIC-CXR
        class_names = ['Atelectasis', 'Cardiomegaly',
                       'Consolidation', 'Edema', 'Enlarged Cardiomediastinum', 'Fracture',
                       'Lung Lesion', 'Lung Opacity', 'No Finding', 'Pleural Effusion',
                       'Pleural Other', 'Pneumonia', 'Pneumothorax', 'Support Devices']
        # Data and metadata generated with "./preprocess_CXR.py"
        file = 'files_224_60000_nounc.npy'
        meta = 'meta_data_60000_nounc.csv'

    elif dataset == 'cheXpert':
        # Loading class labels for CheXpert
        class_names = ['Enlarged Cardiomediastinum', 'Cardiomegaly', 'Lung Opacity', 'Lung Lesion', 'Edema',
                       'Consolidation', 'Pneumonia', 'Atelectasis', 'Pneumothorax', 'Pleural Effusion', 'Pleural Other',
                       'Fracture', 'Support Devices', 'No Finding']
        # Data and metadata generated with "./preprocess_cheXpert.py"
        file = 'cheXpert_files_224_totalnounc.npy'
        meta = 'cheXpert_meta_data_totalnounc.csv'

    img_mat = np.load(root_dir + file)
    df = pd.read_csv(root_dir + meta)
    print('Using data: ' + file + meta)
    print('Number of images total: ', len(df))

    # patient id split
    patient_id = sorted(list(set(df['subject_id'])))
    train_idx, test_val_idx = train_test_split(patient_id, train_size=train_val_split, shuffle=True,
                                               random_state=seed)
    test_idx, val_idx = train_test_split(test_val_idx, test_size=test_val_split, shuffle=True,
                                         random_state=seed)
    print('Number of patients in the training set: ', len(train_idx))
    print('Number of patients in the val set: ', len(val_idx))
    print('Number of patients in the test set: ', len(test_idx))

    # get the train dataframe and sample
    df_train = df[df['subject_id'].isin(train_idx)]
    df_train = df_train.sort_values(by=['subject_id'])
    if dataset == 'CXR':
        train_list = sorted(df.index[df['dicom_id'].isin(df_train['dicom_id'])].tolist())
    elif dataset == 'cheXpert':
        train_list = sorted(df.index[df['Path'].isin(df_train['Path'])].tolist())
    train_label = df_train[class_names]
    train_imgs = img_mat[train_list, :, :]
    print('Number of images in train set: ', len(df_train))

    # get the val dataframe and sample
    df_val = df[df['subject_id'].isin(val_idx)]
    df_val = df_val.sort_values(by=['subject_id'])
    if dataset == 'CXR':
        val_list = sorted(df.index[df['dicom_id'].isin(df_val['dicom_id'])].tolist())
    elif dataset == 'cheXpert':
        val_list = sorted(df.index[df['Path'].isin(df_val['Path'])].tolist())
    val_label = df_val[class_names]
    val_imgs = img_mat[val_list, :, :]
    print('Number of images in val set: ', len(df_val))

    # get the test dataframe and sample
    df_test = df[df['subject_id'].isin(test_idx)]
    df_test = df_test.sort_values(by=['subject_id'])
    if dataset == 'CXR':
        test_list = sorted(df.index[df['dicom_id'].isin(df_test['dicom_id'])].tolist())
    elif dataset == 'cheXpert':
        test_list = sorted(df.index[df['Path'].isin(df_test['Path'])].tolist())
    test_label = df_test[class_names]
    test_imgs = img_mat[test_list, :, :]
    print('Number of images in test set: ', len(df_test))

    return train_list, val_list, test_list, train_label, val_label, test_label,  \
        train_imgs, val_imgs, test_imgs


def get_CXR_dataloaders(dataset,root_dir, train_val_split = 0.6, test_val_split = 0.5, seed=42):
    """Returns a dictionary of data loaders for the CheXpert and MIMIC-CXR dataset,
    for the training, validation, and test sets."""

    train_list, val_list, test_list, train_label, val_label, test_label, \
    train_imgs, val_imgs, test_imgs = \
        train_test_split_CXR(dataset=dataset, root_dir=root_dir, train_val_split=train_val_split,
                             test_val_split=test_val_split, seed=seed)

    # Transformations on the train set: Resize, affine and horizontal flip
    transResize = 224
    transformList = []
    transformList.append(transforms.RandomAffine(degrees=(0, 5), translate=(0.05, 0.05), shear=(5)))
    transformList.append(transforms.RandomHorizontalFlip())
    transformList.append(transforms.Resize(size=transResize))
    transformList.append(transforms.ToTensor())
    train_transform = transforms.Compose(transformList)

    # Transformations on the test and validation set: Resize
    transformList = []
    transformList.append(transforms.Resize(transResize))
    transformList.append(transforms.ToTensor())
    test_transform = transforms.Compose(transformList)

    # Datasets
    image_datasets = {'train': ChestXRay_mimic_DatasetGenerator(imgs=train_imgs,
                                                                img_list=train_list,
                                                                label_list=train_label,
                                                                transform=train_transform),
                      'val': ChestXRay_mimic_DatasetGenerator(imgs=val_imgs,
                                                              img_list=val_list,
                                                              label_list=val_label,
                                                              transform=test_transform),
                      'test': ChestXRay_mimic_DatasetGenerator(imgs=test_imgs,
                                                               img_list=test_list,
                                                               label_list=test_label,
                                                               transform=test_transform)}

    return image_datasets['train'], image_datasets['val'], image_datasets['test']
