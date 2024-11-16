# Dermatoscopic Semantic Segmentation
Fun little project using Lightning module for training an U-Net architecture, [PH2 dataset](https://www.fc.up.pt/addi/ph2%20database.html) a binary type dataset, which is used in the classification of melanoma.
## Installation

Use the package manager [pip](https://pip.pypa.io/en/stable/) to install pytorchligthning module.

```bash
pip install pytorch-lightning
```

## Few things to note
PH2 Dataset represents a good start for trying out semnatic segmenation architectures as it represents a per pixel binary classification.
![imagine_pred](https://github.com/user-attachments/assets/7cde2901-1a4f-4e7e-8e4e-d06a9f224165)

For the main architecture I used [U-Net] (https://arxiv.org/pdf/1505.04597) as a starting point for semantic segmentation while the main focus was on trying out the lightning modularization that is done over pytorch.
The main components I used are the Dataset module from pytorch for preproccesinng the data, and lightningDataModule for the creation of the instance of the train/test/val datasets.
As for the hyperparams used I experimented with Adam, BCELoss, and JaccardIndex, on 25 epochs as the main metric to see the performance of the model.
![image](https://github.com/user-attachments/assets/c86449b2-f979-476b-8a0d-d5f3e135fd59)

As for the metrics, every experiments that I tried out, they were all logged with tensorboard. One of my highest results were 76% mean IoU
![image](https://github.com/user-attachments/assets/2494c2f7-10f0-4c28-bd7c-0ce1f52da12c)
