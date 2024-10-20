"""
Functions for generating CLIP Animals with Attributes 2 embeddings
"""
from PIL import Image
import requests
import torchvision
import torch
from transformers import CLIPProcessor, CLIPModel
from torchvision import transforms
from datasets.awa_dataset import get_AwA_dataloaders
import torch.multiprocessing
torch.multiprocessing.set_sharing_strategy('file_system')

model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

awa_path = "Instert AwA data path"
# Load dataset
dataset, _, _ = get_AwA_dataloaders("all_classes.txt",  64, 2, awa_path, train_ratio=0.6, val_ratio=0.2, seed=42)
data_loader = torch.utils.data.DataLoader(dataset, batch_size=64, num_workers=4, generator=torch.manual_seed(42))

model.to("cuda")
data_storage=[]
conc_storage=[]
class_storage=[]
with torch.no_grad():
    for i, (batch) in enumerate(data_loader):
        inputs = processor(text=[''], images=batch['features'], return_tensors="pt", padding=True)
        inputs.to("cuda")
        output = model(**inputs)
        data_storage.append(output.image_embeds.cpu())
        conc_storage.append(batch['concepts'].cpu())
        class_storage.append(batch['labels'].cpu())
data_storage = torch.cat(data_storage)
conc_storage = torch.cat(conc_storage)
class_storage = torch.cat(class_storage)
torch.save(data_storage, awa_path + '/CLIP_embeddings.pt')
torch.save(conc_storage, awa_path + '/conc_labels.pt')
torch.save(class_storage, awa_path + '/class_labels.pt')