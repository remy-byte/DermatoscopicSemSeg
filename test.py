import os
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor, v2
import matplotlib.pyplot as plts
from torchvision.io import read_image
import pandas as pd
from torchvision import transforms
import sys
from skimage.transform import resize
import numpy as np
from skimage.io import imread
import matplotlib.pyplot as plt
from prepare_dataset import ImageDataset
from tqdm import tqdm



df = pd.read_csv('prepared_dataset.csv')


transform_method = v2.Compose([
    v2.Resize((256,256), antialias=True),
    lambda x : x / 255.,
    v2.Normalize(mean=[0.7532, 0.5764, 0.4884],
                             std = [0.1991, 0.1957, 0.2009])
    ])


mask_method = v2.Resize((256,256))

image_dataset = ImageDataset('prepared_dataset.csv', transform=transform_method, target_transform=mask_method)

image_loader = DataLoader(image_dataset, batch_size = 8)

cnt = 0

psum = torch.tensor([0.0,0.0,0.0])
psum_sq = torch.tensor([0.0, 0.0, 0.0])


for batch_idx, (image,mask) in tqdm(enumerate(image_loader)):
    fig = plt.figure(figsize = (14, 7))
    for i in range(8):
        ax = fig.add_subplot(2, 4, i + 1, xticks = [], yticks = [])   
        im = image[i].squeeze()
        ma = mask[i].squeeze()
        #mask = mask.permute()
        print(ma.shape)
        im = im.permute(1,2,0)
        plt.imshow(im)
        plt.imshow(ma, cmap='gray', alpha=0.3)
        cnt+=1
        
    break
    print(image.shape)

#plt.savefig('imagine_slideshow.png')
    psum += image.sum(axis = [0,2,3])
    psum_sq += (image ** 2).sum(axis = [0, 2, 3])
    
#print(psum, psum_sq)
    

count = len(df) * 256 * 256

# mean and std
total_mean = psum / count
total_var  = (psum_sq / count) - (total_mean ** 2)
total_std  = torch.sqrt(total_var)

# output
print('mean: '  + str(total_mean))
print('std:  '  + str(total_std))

# mean = [0.7532, 0.5764, 0.4884]
# std = [0.1991, 0.1957, 0.2009]

#mean = [192.0766, 146.9906, 124.5532]
# std = [50.7700, 49.9057, 51.2340]