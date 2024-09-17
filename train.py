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
from transformers import ViTModel, CvtModel
from torchvision.io import read_image 
from PIL import Image
import cv2
import os
import torchvision.transforms as trans
from torch.utils.data import Dataset
from DatasetLoader.Landscape import LandscapeDataset
from Model.cyclegan import Generator
from Model.cyclegan import Discriminator
from Metrics.PerceptualLoss import VGGPerceptualLoss

    
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
batch_size_train = 16
batch_size_test = 16
torch.manual_seed(0)
                                      

dataset_flower_train = LandscapeDataset('./datasets/landscape2photo/', datatype = 'train')
train_loader = torch.utils.data.DataLoader(dataset = dataset, batch_size=batch_size_train, shuffle=False)



netG_A2B = Generator(3, 3)
netG_B2A = Generator(3, 3)
netD_A = Discriminator(3)
netD_B = Discriminator(3)


params = list(netG_A2B.parameters()) + list(netG_B2A.parameters())

optimizer_G = torch.optim.Adam(params, lr=0.0002, betas=(0.5, 0.999))
optimizer_D_A = torch.optim.Adam(netD_A.parameters(), lr=0.0002, betas=(0.5, 0.999))
optimizer_D_B = torch.optim.Adam(netD_B.parameters(), lr=0.0002, betas=(0.5, 0.999))

criterion_GAN = torch.nn.MSELoss()
criterion_cycle = torch.nn.L1Loss()
criterion_identity = torch.nn.L1Loss()

target_real = torch.ones(batch_size_train, requires_grad=False)
target_fake = torch.zeros(batch_size_train, requires_grad=False)

lr_scheduler_G = torch.optim.lr_scheduler.LambdaLR(optimizer_G, lr_lambda=LambdaLR(200, 0, 100).step)
lr_scheduler_D_A = torch.optim.lr_scheduler.LambdaLR(optimizer_D_A, lr_lambda=LambdaLR(200, 0, 100).step)
lr_scheduler_D_B = torch.optim.lr_scheduler.LambdaLR(optimizer_D_B, lr_lambda=LambdaLR(200, 0, 100).step)

epochs = 100

count_train = len(train_loader.dataset)
print(count_train)
print('training started')
# best = 999999999
for epoch in range(epochs):
    start_time = time.time()

    print('epochs {}'.format(epoch+1))
    for data in train_loader: 

        real_A = data['A']
        real_B = data['B']

        optimizer_G.zero_grad()

        same_B = netG_A2B(real_B)
        loss_identity_B = criterion_identity(same_B, real_B)*5.0
        # G_B2A(A) should equal A if real A is fed
        same_A = netG_B2A(real_A)
        loss_identity_A = criterion_identity(same_A, real_A)*5.0

        # GAN loss
        fake_B = netG_A2B(real_A)
        pred_fake = netD_B(fake_B)
        loss_GAN_A2B = criterion_GAN(pred_fake, target_real)

        fake_A = netG_B2A(real_B)
        pred_fake = netD_A(fake_A)
        loss_GAN_B2A = criterion_GAN(pred_fake, target_real)

        # Cycle loss
        recovered_A = netG_B2A(fake_B)
        loss_cycle_ABA = criterion_cycle(recovered_A, real_A)*10.0

        recovered_B = netG_A2B(fake_A)
        loss_cycle_BAB = criterion_cycle(recovered_B, real_B)*10.0

        # Total loss
        loss_G = loss_identity_A + loss_identity_B + loss_GAN_A2B + loss_GAN_B2A + loss_cycle_ABA + loss_cycle_BAB
        loss_G.backward()
        
        optimizer_G.step()

        optimizer_D_A.zero_grad()

        # Real loss
        pred_real = netD_A(real_A)
        loss_D_real = criterion_GAN(pred_real, target_real)

        # Fake loss
        fake_A = fake_A_buffer.push_and_pop(fake_A)
        pred_fake = netD_A(fake_A.detach())
        loss_D_fake = criterion_GAN(pred_fake, target_fake)

        # Total loss
        loss_D_A = (loss_D_real + loss_D_fake)*0.5
        loss_D_A.backward()

        optimizer_D_A.step()
        ###################################

        ###### Discriminator B ######
        optimizer_D_B.zero_grad()

        # Real loss
        pred_real = netD_B(real_B)
        loss_D_real = criterion_GAN(pred_real, target_real)
        
        # Fake loss
        fake_B = fake_B_buffer.push_and_pop(fake_B)
        pred_fake = netD_B(fake_B.detach())
        loss_D_fake = criterion_GAN(pred_fake, target_fake)

        # Total loss
        loss_D_B = (loss_D_real + loss_D_fake)*0.5
        loss_D_B.backward()

        optimizer_D_B.step()
        ########################

    print('loss_G :{:.5f}, loss_G_identity :{:.5f}, loss_G_GAN : {:.5f}, loss_G_cycle : {:.5f}, loss_D : {:.5f}'.format(loss_G, (loss_identity_A + loss_identity_B), (loss_GAN_A2B + loss_GAN_B2A), (loss_cycle_ABA + loss_cycle_BAB), (loss_D_A + loss_D_B)))
    end_time = time.time()
    print('Time taken:{:.4f} minutes'.format((end_time - start_time)/60))
    torch.save(netG_A2B, './saved_models/netG_A2B.pt')
    torch.save(netG_B2A, './saved_models/netG_B2A.pt')
    torch.save(netD_A, './saved_models/netD_A.pt')
    torch.save(netD_B, './saved_models/netD_B.pt')



