"""
Functions to filter the most correlated and relevant concepts in ImageNet data annotated using VLMs
"""
import numpy as np
import torch
from sklearn.cluster import KMeans
from scipy.stats import chi2_contingency
import pandas as pd
import os
import matplotlib.pyplot as plt


def cramers_v(x, y):
    # Create a contingency table
    contingency_table = pd.crosstab(x, y)
    chi2, p, dof, _ = chi2_contingency(contingency_table)
    n = contingency_table.sum().sum()
    return np.sqrt(chi2 / (n * (min(contingency_table.shape) - 1)))


def calculate_cramers_v_with_target(data, target):
    """
    Calculate Cramer's V coefficient between each feature and a target variable.

    Parameters:
    data : numpy array
        The dataset where columns represent features.
    target : numpy array
        The target variable (y).

    Returns:
    list :
        List of Cramer's V coefficients for each feature with the target variable.
    """
    num_features = data.shape[1]
    cramers_vs = []
    for i in range(num_features):
        # Calculate Cramer's V coefficient
        cramers_vs.append(cramers_v(data[:, i], target))

    return cramers_vs

# TODO: Fill in corresponding directories with concept and class labels
data = torch.load( "...")
y = torch.load("...")
indices = np.random.randint(0, data.shape[0], size=50000)
data = data[indices, :] * 1
y = y[indices] * 1
# Computing distance matrix
corr = torch.corrcoef(data.transpose(0, 1))
dist = 1 - corr
kmeans = KMeans(n_clusters=75, random_state=0)
kmeans.fit(dist)
labels = kmeans.labels_

feature_importance = calculate_cramers_v_with_target(data.numpy(), y.numpy())


indices_list = [[] for _ in range(75)]
final_concepts = []
for i in range(75):  # 25 for 100 and 75 for 300
    indices_list[i] = np.where(labels == i)[0].tolist()
    sorted_indices = np.argsort(np.array(feature_importance)[indices_list[i]])[-4:]
    final_concepts.append(np.array(indices_list[i])[sorted_indices])
final_concepts = np.sort(np.array(final_concepts).flatten())

# TODO: Fill in corresponding directories with concept class names
with open(os.path.join("..."), "r") as file:
    # Read the contents of the file
    concept_names_graph = [line.strip() for line in file]

final_concepts_list = [concept_names_graph[i] for i in final_concepts]

# TODO: Fill in corresponding directories with output file for new concept class names
# Open the new file in write mode
with open(os.path.join("..."), "w",) as file:
    # Write each concept in the final_concepts_list to the file, each on a new line
    for concept in final_concepts_list:
        file.write(f"{concept}\n")
# TODO: Fill in corresponding directories with output file for features importances and new selected concepts
np.save("...", feature_importance)
np.save("...",final_concepts)

print(final_concepts_list)
