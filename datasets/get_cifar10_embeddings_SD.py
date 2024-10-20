"""
Functions for generating Stable Diffusion CIFAR-10 embeddings
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
        inputs = self.processor.preprocess(
            image, resize_mode="crop", width=128, height=128
        )
        return inputs


transform = transforms.Compose([CustomTransform(processor)])  # Your custom transform

# Load imagenet from folder
for split in ["train", "test"]:
    if split == "train":
        train = True
    else:
        train = False
    imagenet_data = torchvision.datasets.CIFAR10(
        train=train, transform=transform, root="../datasets/cifar10"
    )
    data_loader = torch.utils.data.DataLoader(
        imagenet_data,
        batch_size=64,
        shuffle=False,
        num_workers=15,
        pin_memory=False,
    )
    data_storage = [[], []]
    ### GETTING IMAGE EMBEDDINGS
    with torch.no_grad():
        for i, (inputs, target) in enumerate(data_loader):
            if i % 100 == 0:
                print(i / len(data_loader))
            inputs = inputs.to(torch_device)
            output = vae.encode(inputs.squeeze(1)).latent_dist.sample()
            data_storage[0].append(output.cpu())
            data_storage[1].append(target)
    data_storage = [torch.cat(sublist) for sublist in data_storage]
    torch.save(data_storage[0], f"{split}_cifar10_embeddings_SD.pt")
    torch.save(data_storage[1], f"{split}_cifar10_class_labels_SD.pt")