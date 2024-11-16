import os
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor
import matplotlib.pyplot as plts
from torchvision.io import read_image
import pandas as pd
from torchvision import transforms
import sys
from skimage.transform import resize
import numpy as np
from skimage.io import imread
import matplotlib.pyplot as plt
path = os.path.abspath("..")
sys.path.append(path)
from utils.utils import open_image, prepare_dataset

prepare_dataset()
dataframe = pd.read_csv('prepared_dataset.csv')

class ImageDataset(Dataset):
    def __init__(self, csv_file, transform=None, target_transform=None):
        self.__image_masks = pd.read_csv(csv_file)
        self.__transform = transform
        self.__target_transform = target_transform


    def __len__(self):
        return len(self.__image_masks)
    
    def __getitem__(self, index):
        
        image = imread(self.__image_masks.iloc[index, 0])
        mask = imread(self.__image_masks.iloc[index, 1])

        image = torch.FloatTensor(image)
        mask = torch.FloatTensor(mask)

        image = torch.permute(image,(2,0,1))
        mask  = torch.permute(mask.unsqueeze(1),(1,0,2))

        if self.__transform:
            image = self.__transform(image)

        if self.__target_transform:
            mask = self.__target_transform(mask)
 
        return image,mask
    

transform_method = transforms.Compose([
    transforms.Resize((256,256)),
    ])



data = ImageDataset('prepared_dataset.csv',transform=transform_method,target_transform=transform_method)

train_dataset = DataLoader(data, batch_size=4)

def plotn(train_dataset, only_mask = False):
     for i , (image, mask) in enumerate(train_dataset):
        if i == 2:
            return
        #print(image.shape)
        image = image[1].squeeze()
        
        mask = mask[1].squeeze()
        image = image.permute(1,2,0)
        #print(mask.unique())
        plt.imshow(image / 255.)
        plt.imshow(mask,cmap='gray', alpha=0.3)
        plt.savefig('imagine.png')
        plt.show()  



plotn(train_dataset)


    
        


        
        
        









