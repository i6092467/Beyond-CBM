"""
Preprocessing script for the CheXpert dataset
"""
# import libraries
import numpy as np
import pandas as pd
import PIL
from PIL import Image
from tqdm import tqdm

# data paths
# root_dir: root path where necessary files are stored
# res_dir: where resulting annotations and image matrix should be saved
# TODO: fill in relevant directories
root_dir = ...  # to be filled
res_dir = ...   # to be filled

class_names = ['Enlarged Cardiomediastinum', 'Cardiomegaly', 'Lung Opacity', 'Lung Lesion', 'Edema', 'Consolidation',
               'Pneumonia', 'Atelectasis', 'Pneumothorax', 'Pleural Effusion', 'Pleural Other', 'Fracture',
               'Support Devices', 'No Finding']

# CheXpert labels
labels = pd.read_csv(root_dir + 'train.csv')

# Dropping readings with uncertain values
conditions = [(labels[col] == -1) for col in class_names]
combined_condition = pd.concat(conditions, axis=1).any(axis=1)
labels.drop(labels[combined_condition].index, inplace = True)

# meta data
subject_ids = []
for index, row in labels.iterrows():
    path_parts = row['Path'].split("/")
    patient_part = next(part for part in path_parts if "patient" in part)
    subject_ids.append(patient_part[len("patient"):])
labels['subject_id'] = subject_ids

# Droping duplicate patient recordings except for the last visit
labels = labels.drop_duplicates(subset=['subject_id'], keep='last')
labels = labels.sort_values(by=['subject_id'])

# resize parameter
transResize = 224
# create img_mat
size = len(labels)
img_mat = np.zeros((size,transResize,transResize))
df2 = labels[:size].copy()
    
# initialize cnt
cnt = 0
total_iterations = len(root_dir + labels['Path'][:size])
# Create a tqdm progress bar
for idx, filename in tqdm(enumerate(root_dir + labels['Path'][:size]), total=total_iterations, desc='Processing'):
    # read image
    try:
        img = PIL.Image.open(filename).convert('RGB')
    except:
        # drop the row if problematic sample
        print(filename)
        df2.drop(df2[df2['Path'] == filename].index, inplace=True)
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
np.save(res_dir + 'cheXpert_files_' + str(transResize) + '_totalnounc.npy', img_mat)

# save dataframe as csv
df2.to_csv(res_dir + 'cheXpert_meta_data_totalnounc.csv',index=False)
