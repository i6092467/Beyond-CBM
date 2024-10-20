"""
Functions for generating Stable Diffusion ImageNet embeddings
"""
import torchvision
import torch

from torchvision import transforms

from diffusers import AutoencoderKL
from diffusers.image_processor import VaeImageProcessor

vae = AutoencoderKL.from_pretrained("CompVis/stable-diffusion-v1-4", subfolder="vae")
processor = VaeImageProcessor()
torch_device = "cuda"
vae.to(torch_device)


### GETTING TEXT EMBEDDINGS

class CustomTransform:
    def __init__(self, processor):
        self.processor = processor

    def __call__(self, image):
        # Convert the image to a tensor
        # Use the processor to prepare the inputs
        inputs = self.processor.preprocess(image, resize_mode='crop', width=128, height=128)
        return inputs


transform = transforms.Compose([
    CustomTransform(processor)  # Your custom transform
])

# Load imagenet from folder
split = 'train'
imagenet_data = torchvision.datasets.ImageNet('../datasets/imagenet', split=split, transform=transform)
data_loader = torch.utils.data.DataLoader(imagenet_data,
                                          batch_size=64,
                                          shuffle=False,
                                          num_workers=15, pin_memory=False, )
data_storage = [[], []]
### GETTING IMAGE EMBEDDINGS
with torch.no_grad():
    for i, (inputs, target) in enumerate(data_loader):
        if i % 100 == 0:
            print(i / len(data_loader))
            if i != 0 and i % 1000 == 0:
                data_storage = [torch.cat(sublist) for sublist in data_storage]
                torch.save(data_storage[0], f'{split}_{i // 1000}_embeddings_SD.pt')
                torch.save(data_storage[1], f'{split}_{i // 1000}_class_labels_SD.pt')
                data_storage = [[], []]
        inputs = inputs.to(torch_device)
        output = vae.encode(inputs.squeeze(1)).latent_dist.sample()
        data_storage[0].append(output.cpu())
        data_storage[1].append(target)
data_storage = [torch.cat(sublist) for sublist in data_storage]
torch.save(data_storage[0], f'{split}_{(i // 1000) + 1}_embeddings_SD.pt')
torch.save(data_storage[1], f'{split}_{(i // 1000) + 1}_class_labels_SD.pt')

# Merge the files
if split == 'val':
    for name in ['embeddings', 'class_labels']:
        data_storage = []
        i = 1
        data_storage.append(torch.load(f'{split}_{i}_{name}_SD.pt'))
        data_storage = data_storage[0]
        torch.save(data_storage, f'{split}_{name}_SD.pt')
else:
    for name in ['embeddings', 'class_labels']:
        data_storage = []
        for i in range(1, 21):
            data_storage.append(torch.load(f'{split}_{i}_{name}_SD.pt'))
        data_storage = torch.cat(data_storage)
        torch.save(data_storage, f'{split}_{name}_SD.pt')
