# Beyond Concept Bottleneck Models: How to Make Black Boxes Intervenable?

This repository contains the code for the paper "*Beyond Concept Bottleneck Models: 
How to Make Black Boxes Intervenable?*". (https://neurips.cc/virtual/2024/poster/95656)

**Abstract**: Recently, interpretable machine learning has re-explored concept bottleneck models (CBM). An advantage of this model class is the user's ability to intervene on predicted concept values, affecting the downstream output. In this work, we introduce a method to perform such concept-based interventions on *pretrained* neural networks, which are not interpretable by design, only given a small validation set with concept labels. Furthermore, we formalise the notion of *intervenability* as a measure of the effectiveness of concept-based interventions and leverage this definition to fine-tune black boxes. Empirically, we explore the intervenability of black-box classifiers on synthetic tabular and natural image benchmarks. We focus on backbone architectures of varying complexity, from simple, fully connected neural nets to Stable Diffusion. We demonstrate that the proposed fine-tuning improves intervention effectiveness and often yields better-calibrated predictions. To showcase the practical utility of our techniques, we apply them to deep chest X-ray classifiers and show that fine-tuned black boxes are more intervenable than CBMs. Lastly, we establish that our methods are still effective under vision-language-model-based concept annotations, alleviating the need for a human-annotated validation set.

### Usage 

All the libraries required are in the conda environment `environment.yml`. 
To install it, follow the instructions below:
```
conda env create -f environment.yml   # install dependencies
conda activate intervenable-models    # activate environment
```

- Scripts `train.py` and `validate.py` can be used to train and validate models
- `intervene.py` implements intervention and fine-tuning procedures
- `finetune.py` and `finetune_evaluation.py` implement the finetuning of black box models, train the baselines and evaluate the resulting models
- `models.py` and `losses.py` define models and loss functions
- `networks.py` provides neural network architectures
- `probes.py` contains the utility functions for probing
- `./utils` contains complimentary functions used to evaluate and train the models
- `./datasets` contains the data loaders for all datasets and the processing scripts to generate synthetic data, modified chest X-rays files, compute CLIP and Stable Diffusion embeddings for CIFAR-10, ImageNet and Animals with Attributes and extract VLM-based concept annotations
- `./configs` contains example configuration files for training black-box and CBM classifiers, the data paths need to be modified accordingly
Further details are documented in code. 
-`./bin` contains sample shell scripts to run experiments on a GPU and reproduce our results

## Citing

To cite "*Beyond Concept Bottleneck Models: 
How to Make Black Boxes Intervenable?*" please use the following BibTEX entry:

```
@inproceedings{
laguna2024beyond,
title={Beyond Concept Bottleneck Models: 
How to Make Black Boxes Intervenable?},
author={Laguna, Sonia and Marcinkevi{\v{c}}s, Ri{\v{c}}ards and Vandenhirtz, Moritz and Vogt, Julia E},
booktitle={Thirty-eighth Conference on Neural Information Processing Systems},
year={2024}
url={https://openreview.net/forum?id=KYHma7hzjr}
}
```
