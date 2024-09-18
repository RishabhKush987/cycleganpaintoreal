import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
import torch.nn as nn
import time
from torchvision.transforms.autoaugment import InterpolationMode
from torchvision import transforms
import numpy as np
from torch.autograd import Variable
import torchvision
from torch.utils.data import Subset
import math
from torchvision.io import read_image 
from PIL import Image
import cv2
import os
import torchvision.transforms as trans
from torch.utils.data import Dataset
from DataLoader.Landscape import LandscapeDataset
from Model.cyclegan import Generator
from Model.cyclegan import Discriminator
from transformers import SamModel, SamProcessor
from segment_anything import sam_model_registry, SamAutomaticMaskGenerator, SamPredictor
from diffusers import ControlNetModel, StableDiffusionControlNetPipeline


def show_output(result_dict):
    #  if axes:
    #     ax = axes
    #  else:
    #     ax = plt.gca()
    #     ax.set_autoscale_on(False)
     sorted_result = sorted(result_dict, key=(lambda x: x['area']), reverse=True)
     # Plot for each segment area
     for val in sorted_result:
        mask = val['segmentation']
        img = np.ones((mask.shape[0], mask.shape[1], 3))
        color_mask = np.random.random((1, 3)).tolist()[0]
        for i in range(3):
            img[:,:,i] = color_mask[i]
            plt.imshow(np.dstack((img, mask*0.5)))

CHECKPOINT_PATH='/content/cycleganpaintoreal/sam_vit_h_4b8939.pth'
MODEL_TYPE = "vit_h"

batch_size_train = 16
transform_ = transforms.Compose([
    transforms.ToTensor()
])
dataset_landscape_train = LandscapeDataset('/content/cycleganpaintoreal/datasets/landscape2photo/',transforms_ = transform_, datatype = 'train')
test_loader = torch.utils.data.DataLoader(dataset = dataset_landscape_train, batch_size=batch_size_train, shuffle=False)


count_train = len(test_loader.dataset)
print(count_train)
# netG_A2B  = torch.load('./saved_models/netG_A2B.pt')
# netG_B2A  = torch.load('./saved_models/netG_B2A.pt')
device = "cuda" if torch.cuda.is_available() else "cpu"



sam = sam_model_registry[MODEL_TYPE](checkpoint=CHECKPOINT_PATH).to(device)


mask_generator = SamAutomaticMaskGenerator(sam)

controlnet = ControlNetModel.from_pretrained("lllyasviel/sd-controlnet-seg")
pipeline = StableDiffusionControlNetPipeline.from_pretrained(
  "runwayml/stable-diffusion-v1-5", controlnet=controlnet
)

for i, (data) in enumerate(test_loader, 0):
    # recon = netG_A2B(data['A'])
    # print(recon.size())
    # im = Image.fromarray(cv2.resize((recon[0].permute(1,2,0).detach().cpu().numpy()*255).astype(np.uint8),(256,256)))
    # im.save("./recon.png")
    output_mask = mask_generator.generate(data['A'][0].permute(1,2,0).cpu().numpy())
    # plot = plt.figure(figsize=(20,20))
    # axes[0].imshow(data['A'][0].permute(1,2,0))
    show_output(output_mask)
    plt.savefig('foo.png')
    img = Image.open("foo.png")
    image = pipeline("landscape", img, num_inference_steps=20).images[0]

    image.save('./realimage.png')


    break;    

