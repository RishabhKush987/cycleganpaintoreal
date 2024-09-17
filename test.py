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


batch_size_train = 16
transform_ = transforms.Compose([
    transforms.ToTensor()
])
dataset_landscape_train = LandscapeDataset('/content/cycleganpaintoreal/datasets/landscape2photo/',transforms_ = transform_, datatype = 'train')
test_loader = torch.utils.data.DataLoader(dataset = dataset_landscape_train, batch_size=batch_size_train, shuffle=False)


count_train = len(test_loader.dataset)
print(count_train)
netG_A2B  = torch.load('./saved_models/netG_A2B.pt')
# netG_B2A  = torch.load('./saved_models/netG_B2A.pt')
device = "cuda" if torch.cuda.is_available() else "cpu"

model = SamModel.from_pretrained("facebook/sam-vit-huge").to(device)
processor = SamProcessor.from_pretrained("facebook/sam-vit-huge")

input_points = [[[256, 256]]]  # 2D location of a window in the image


for i, (data) in enumerate(test_loader, 0):
    recon = netG_A2B(data['A'])
    print(recon.size())
    im = Image.fromarray(cv2.resize((recon[0].permute(1,2,0).detach().cpu().numpy()*255).astype(np.uint8),(256,256)))
    im.save("./recon.png")
    inputs = processor(data['A'][0], input_points=input_points, return_tensors="pt").to(device)
    with torch.no_grad():
        outputs = model(**inputs)

    masks = processor.image_processor.post_process_masks(
        outputs.pred_masks.cpu(), inputs["original_sizes"].cpu(), inputs["reshaped_input_sizes"].cpu()
    )
    scores = outputs.iou_scores
    print(scores)
    break;    

