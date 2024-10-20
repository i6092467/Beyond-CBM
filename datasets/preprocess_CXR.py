"""
Preprocessing script for the MIMIC-CXR dataset
"""
# import libraries
import numpy as np
import pandas as pd
import PIL
from PIL import Image
from tqdm import tqdm
import zipfile

# data paths
# root dir: root path where necessary files are stored
# data dir: where image files are stored (change the path if it is stored somewhere else)
# res_dir: where resulting annotations and image matrix should be saved
# TODO: fill in relevant directories
root_dir = ...  # to be filled
data_dir = ...  # to be filled
res_dir = ...   # to be filled
class_names = ['Atelectasis', 'Cardiomegaly', 'Consolidation', 'Edema', 'Enlarged Cardiomediastinum', 'Fracture',
               'Lung Lesion', 'Lung Opacity', 'No Finding', 'Pleural Effusion', 'Pleural Other', 'Pneumonia',
               'Pneumothorax', 'Support Devices']

# MIMIC-CXR labels
labels = pd.read_csv(root_dir + 'mimic-cxr-2.0.0-chexpert.csv')

# Dropping readings with uncertain values
conditions = [(labels[col] == -1) for col in class_names]
combined_condition = pd.concat(conditions, axis=1).any(axis=1)
labels.drop(labels[combined_condition].index, inplace = True)

# meta data
df = pd.read_csv(root_dir + 'mimic-cxr-2.0.0-metadata.csv')
df = df[['dicom_id','subject_id','study_id']]
# Droping duplicate patient recordings except for the last visit
df = df.drop_duplicates(subset=['subject_id'], keep='last')
df = df.sort_values(by=['subject_id'])

# merge with tables for patient and label info
df = pd.merge(df, labels)

# add a path column for dataloader
df['path'] = ('files/p' + df['subject_id'].astype(str).str[:2] + '/p' + df['subject_id'].astype(str) + '/s' +
              df['study_id'].astype(str) + '/' + df['dicom_id'].astype(str) + '.jpg')
# resize parameter
transResize = 224
# create img_mat
size = len(df)
size = 10000
img_mat = np.zeros((size,transResize,transResize))
df2 = df[:size].copy()
    
# initialize cnt
cnt = 0
with zipfile.ZipFile(data_dir, 'r') as z:
    # iterate through files
    for filename in tqdm(df['path'][:size]):
        # read image
        try: 
            img = PIL.Image.open(z.open(filename)).convert('RGB') 
        except:
            # drop the row if problematic sample
            print(filename)
            df2.drop(df2[df2['path'] == filename].index, inplace=True)
            continue
            
        # crop depending on the size to a square ratio
        width, height = img.size
        r_min = max(0,(height-width)/2)
        r_max = min(height,(height+width)/2)
        c_min = max(0,(width-height)/2)
        c_max = min(width,(height+width)/2)
        img = img.crop((c_min,r_min,c_max,r_max))
            
        # hist equalize and reshape
        img = img.resize((transResize,transResize))
        img = PIL.ImageOps.equalize(img)
        img = img.convert('L')
            
        # assign
        img_mat[cnt,:,:] = np.array(img)   
            
        # increment
        cnt = cnt + 1
# save
img_mat = img_mat[0:len(df2),:,:]
np.save(res_dir + 'files_' + str(transResize) + '_nounc.npy', img_mat)

# save dataframe as csv
df2.to_csv(res_dir + 'meta_data_nounc.csv', index=False)
