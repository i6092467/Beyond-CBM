"""
Functions for generating ImageNet concept labels based on
"Oikarinen, T., Das, S., Nguyen, L. M., & Weng, T. W. (2023). Label-free concept bottleneck models. arXiv preprint arXiv:2304.06129."
"""
import torchvision
import torch
from transformers import CLIPProcessor, CLIPModel


model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
with open('../datasets/imagenet/imagenet_filtered.txt', 'r') as file:
    # Read the contents of the file
    concept_list = [line.strip() for line in file]
neg_concept_list = ['not ' + line for line in concept_list]
pos_and_neg_concepts = concept_list + neg_concept_list


### GETTING TEXT EMBEDDINGS

class CustomTransform:
    def __init__(self, processor):
        self.processor = processor

    def __call__(self, image):
        # Convert the image to a tensor
        # Use the processor to prepare the inputs
        inputs = self.processor(text=pos_and_neg_concepts, images=image, return_tensors="pt", padding=True)
        return inputs


transform = CustomTransform(processor)

# Load imagenet from folder
split = 'train'
imagenet_data = torchvision.datasets.ImageNet('../datasets/imagenet', split=split, transform=transform)
data_loader = torch.utils.data.DataLoader(imagenet_data,
                                          batch_size=128,
                                          shuffle=False,
                                          num_workers=20, pin_memory=False, )
model.to("cuda")

with torch.no_grad():
    for i, (inputs, target) in enumerate(data_loader):
        inputs.to("cuda")
        inputs['pixel_values'] = inputs['pixel_values'].squeeze(1)
        inputs['input_ids'] = inputs['input_ids'][0]
        inputs['attention_mask'] = inputs['attention_mask'][0]
        output = model(**inputs)
        text_embed = output.text_embeds.cpu()
        break


### GETTING IMAGE EMBEDDINGS


class CustomTransform2:
    def __init__(self, processor):
        self.processor = processor

    def __call__(self, image):
        # Convert the image to a tensor
        # Use the processor to prepare the inputs
        inputs = self.processor(text=[''], images=image, return_tensors="pt", padding=True)
        return inputs


transform = CustomTransform2(processor)

# Load imagenet from folder
imagenet_data = torchvision.datasets.ImageNet('../datasets/imagenet', split=split, transform=transform)
data_loader = torch.utils.data.DataLoader(imagenet_data,
                                          batch_size=128,
                                          shuffle=False,
                                          num_workers=15, pin_memory=True, )
data_storage = [[], [], []]
with torch.no_grad():
    for i, (inputs, target) in enumerate(data_loader):
        if i % 100 == 0:
            print(i / len(data_loader))
            if i != 0 and i % 1000 == 0:
                data_storage = [torch.cat(sublist) for sublist in data_storage]
                # torch.save(data_storage[0], f'{split}_{i // 1000}_embeddings.pt')
                torch.save(data_storage[1], f'{split}_{i // 1000}_concept_labels.pt')
                # torch.save(data_storage[2], f'{split}_{i // 1000}_class_labels.pt')
                data_storage = [[], [], []]
        inputs.to("cuda")
        inputs['pixel_values'] = inputs['pixel_values'].squeeze(1)
        inputs['input_ids'] = inputs['input_ids'][0]
        inputs['attention_mask'] = inputs['attention_mask'][0]
        output = model(**inputs)
        similarity = torch.nn.functional.cosine_similarity(
            text_embed.unsqueeze(0).expand(output.image_embeds.shape[0], -1, -1),
            output.image_embeds.unsqueeze(1).expand(-1, text_embed.shape[0], -1).cpu(), dim=-1, eps=1e-8)
        similarity_pos_neg = similarity.reshape(target.shape[0], 2, -1)
        concept_label = 1 - similarity_pos_neg.argmax(dim=1)  # 0 if negated, 1 if positive
        data_storage[0].append(output.image_embeds.cpu())
        data_storage[1].append(concept_label > 0)
        data_storage[2].append(target)
data_storage = [torch.cat(sublist) for sublist in data_storage]
# torch.save(data_storage[0], f'{split}_{(i // 1000) + 1}_embeddings.pt')
torch.save(data_storage[1], f'{split}_{(i // 1000) + 1}_concept_labels.pt')
# torch.save(data_storage[2], f'{split}_{(i // 1000) + 1}_class_labels.pt')

# Merge the files
if split == 'val':
    for name in ['embeddings', 'concept_labels', 'class_labels']:
        data_storage = []
        i = 1
        data_storage.append(torch.load(f'{split}_{i}_{name}.pt'))
        data_storage = data_storage[0]
        torch.save(data_storage, f'{split}_{name}.pt')
else:
    for name in ['embeddings', 'concept_labels', 'class_labels']:
        data_storage = []
        for i in range(1, 11):
            data_storage.append(torch.load(f'{split}_{i}_{name}.pt'))
        data_storage = torch.cat(data_storage)
        torch.save(data_storage, f'{split}_{name}.pt')
