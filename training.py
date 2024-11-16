import os
from typing import Any
from pytorch_lightning.utilities.types import EVAL_DATALOADERS, STEP_OUTPUT, TRAIN_DATALOADERS, OptimizerLRScheduler
import torch
from torch import nn
import torch.nn.functional as F
from torchvision import transforms
from torch.utils.data import DataLoader
import pytorch_lightning as L
from pytorch_lightning.callbacks import ModelCheckpoint
from dataset.prepare_dataset import ImageDataset
from torchmetrics import JaccardIndex
import matplotlib.pyplot as plt



class SegNetModel(L.LightningModule):
    def __init__(self):
        super(SegNetModel, self).__init__()

        train_loss = 0.0
        test_loss = 0.0

        self.enc_conv0 = nn.Conv2d(in_channels=3, out_channels=16, kernel_size=(3,3), padding=1)
        self.act0 = nn.ReLU()
        self.bn0 = nn.BatchNorm2d(16)
        self.pool0 = nn.MaxPool2d(kernel_size=(2,2))

        self.enc_conv1 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=(3,3), padding=1)
        self.act1 = nn.ReLU()
        self.bn1 = nn.BatchNorm2d(32)
        self.pool1 = nn.MaxPool2d(kernel_size=(2,2))

        self.enc_conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=(3,3), padding=1)
        self.act2 = nn.ReLU()
        self.bn2 = nn.BatchNorm2d(64)
        self.pool2 =  nn.MaxPool2d(kernel_size=(2,2))

        self.enc_conv3 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=(3,3), padding=1)
        self.act3 = nn.ReLU()
        self.bn3 = nn.BatchNorm2d(128)
        self.pool3 =  nn.MaxPool2d(kernel_size=(2,2))

        self.bottleneck_conv = nn.Conv2d(in_channels=128, out_channels=256, kernel_size=(3,3), padding=1)
        
        self.upsample0 =  nn.UpsamplingBilinear2d(scale_factor=2)
        self.dec_conv0 = nn.Conv2d(in_channels=256, out_channels=128, kernel_size=(3,3), padding=1)
        self.dec_act0 = nn.ReLU()
        self.dec_bn0 = nn.BatchNorm2d(128)

        self.upsample1 =  nn.UpsamplingBilinear2d(scale_factor=2)
        self.dec_conv1 =  nn.Conv2d(in_channels=128, out_channels=64, kernel_size=(3,3), padding=1)
        self.dec_act1 = nn.ReLU()
        self.dec_bn1 = nn.BatchNorm2d(64)

        self.upsample2 = nn.UpsamplingBilinear2d(scale_factor=2)
        
        self.dec_conv2 = nn.Conv2d(in_channels=64, out_channels=32, kernel_size=(3,3), padding=1)
        self.dec_act2 = nn.ReLU()
        self.dec_bn2 = nn.BatchNorm2d(32)

        self.upsample3 = nn.UpsamplingBilinear2d(scale_factor=2)
        self.dec_conv3 = nn.Conv2d(in_channels=32, out_channels=1, kernel_size=(1,1))


    def forward(self, x):
        e0 = self.pool0(self.bn0(self.act0(self.enc_conv0(x))))
        e1 = self.pool1(self.bn1(self.act1(self.enc_conv1(e0))))
        e2 = self.pool2(self.bn2(self.act2(self.enc_conv2(e1))))
        e3 = self.pool3(self.bn3(self.act3(self.enc_conv3(e2))))

        b = self.bottleneck_conv(e3)

        d0 = self.dec_bn0(self.dec_act0(self.dec_conv0(self.upsample0(b))))
        d1 = self.dec_bn1(self.dec_act1(self.dec_conv1(self.upsample1(d0))))
        d2 = self.dec_bn2(self.dec_act2(self.dec_conv2(self.upsample2(d1))))
        d3 = self.dec_conv3(self.upsample3(d2))
        return d3
    
    def configure_optimizers(self) -> OptimizerLRScheduler:
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
        return optimizer
    
    def training_step(self, batch, batch_idx):
        
        image, mask =  batch

        preds = self(image)
        loss_class = nn.BCEWithLogitsLoss(reduction='mean')
        loss = loss_class(preds, mask)
        jaccard = JaccardIndex(task='binary').to(self.device)
        jaccard_IOU  = jaccard(preds,mask).to(self.device)

        self.log("train_loss", loss,
                 prog_bar=True,
                 on_epoch=True,
        )
        self.log("Iou", jaccard_IOU,
                 prog_bar=True,
                 on_epoch=True,
        )


        return loss
    
    def validation_step(self, batch, batch_idx):
        image, mask =  batch
        image, mask =  batch

        preds = self(image)
        loss_class = nn.BCEWithLogitsLoss(reduction='mean')
        loss = loss_class(preds, mask)
        jaccard = JaccardIndex(task='binary').to(self.device)
        jaccard_IOU  = jaccard(preds,mask).to(self.device)

        
        self.log("val_loss", loss,
                    prog_bar=True,
                    on_epoch=True,
            )
        self.log("val_Iou", jaccard_IOU,
                    prog_bar=True,
                    on_epoch=True,
            )


        return loss
            

class Dermioscopic_DatasetModule(L.LightningDataModule):
        
    def __init__(self) -> None:
        super().__init__()
        self.ds_map = {}
        self.transform_method = transforms.Compose([
        transforms.Resize((256,256), antialias=True),
        lambda x: x / 255.,
        transforms.Normalize(mean=[0.7532, 0.5764, 0.4884],
                             std = [0.1991, 0.1957, 0.2009])
    ])
        
        self.mask_transform = transforms.Resize((256,256), antialias=True, interpolation=transforms.InterpolationMode.NEAREST)
    
    def _get_dataset(self,stage: str):
        if stage == 'validation':
            csv_path =  '/root/Dermioscopic_Semantic_Segmentation/dataset/valid_dataset.csv'
            return ImageDataset(csv_file=csv_path, 
                            transform=self.transform_method,
                             target_transform=self.mask_transform)
        else:
            csv_path =  '/root/Dermioscopic_Semantic_Segmentation/dataset/prepared_dataaset.csv'
            return ImageDataset(csv_file=csv_path, 
                            transform=self.transform_method,
                             target_transform=self.mask_transform)
    

    def setup(self, stage: str):
        if (stage == 'fit'):
            self.ds_map['validation'] = self._get_dataset('validation')
        self.ds_map[stage] = self._get_dataset(stage)
        # if (stage == 'test'):
        #     self.ds_map['test'] = self._get_dataset('/root/Dermioscopic_Semantic_Segmentation/dataset/test_dataset.csv') 

    
    def train_dataloader(self) -> TRAIN_DATALOADERS:
        return DataLoader(self.ds_map['fit'],
                         batch_size = 32,
                         num_workers=8,
                         shuffle=True)
    
    
    def val_dataloader(self):
        return DataLoader(self.ds_map['validation'],
                         batch_size = 8,
                         num_workers=8)
    
    def test_dataloader(self) -> TRAIN_DATALOADERS:
        return DataLoader(self.ds_map['test'],
                         batch_size = 8,
                         num_workers=8)

checkpoint_callback = ModelCheckpoint(
    dirpath = 'checkpoints_model/',
    save_top_k= 1,
    verbose = True, 
    monitor = 'val_Iou',
    mode = 'max',
)
    

model = SegNetModel()
data = Dermioscopic_DatasetModule()
trainer = L.Trainer(max_epochs=25,
                    log_every_n_steps=1, callbacks=checkpoint_callback)

#trainer.fit(model,data)
checkpoint = torch.load('/root/Dermioscopic_Semantic_Segmentation/checkpoints_model/epoch=22-step=138.ckpt', map_location = lambda storage, loc : storage)
model.load_state_dict(checkpoint['state_dict'])
csv_path = '/root/Dermioscopic_Semantic_Segmentation/dataset/test_dataset.csv'
transforms_method =  transforms.Compose([transforms.Resize((256,256), antialias=True),
        lambda x: x /255.,
        transforms.Normalize(mean=[0.7532, 0.5764, 0.4884],
                             std = [0.1991, 0.1957, 0.2009])
    ])
        
mask_transform = transforms.Resize((256,256), antialias=True, interpolation=transforms.InterpolationMode.NEAREST)

model.eval()
predictions = []
image_mask = []
plots = 5
device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
dataset = ImageDataset(csv_path, transform=transforms_method, target_transform=mask_transform)
plots = 5


mean = [0.7532, 0.5764, 0.4884]
std = [0.1991, 0.1957, 0.2009]

dataloader = DataLoader(dataset, num_workers=8, batch_size=8)
for i, (image, mask) in enumerate(dataloader):
    if i == plots:
        break
   # image2 = image * [0.1991, 0.1957, 0.2009] - [0.7532, 0.5764, 0.4884]
    pred = model(image)
    print(pred.shape)
    img = image[2].squeeze()
    img = img.clone()  # clone the tensor to not change it in-place
    for t, m, s in zip(img, mean, std):
        t.mul_(s).add_(m)  # unnormalize

    pred = pred[2].squeeze()
    pred = torch.where(pred >= 0.5, 1, 0)
    plt.imshow(img.squeeze(0).permute(1, 2, 0))
    plt.imshow(pred.detach().numpy(), alpha=0.5, cmap='gray')
    plt.show()  
    plt.savefig('imagine_pred.png')
    plt.show()





        


