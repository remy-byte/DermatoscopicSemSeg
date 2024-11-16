import cv2
import numpy as np
import os
import pandas as pd


def open_image(image_ref: str)-> np.array:

    image = cv2.imread(image_ref)

    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    return image


def prepare_dataset(root = 'PH2Dataset'):
    images = []
    masks = []

    for root, dirs, files in os.walk(os.path.join(root, 'PH2 Dataset Images')):
        if root.endswith('_Dermoscopic_Image'):
            images.append(os.path.abspath(os.path.join(root,files[0])))
        if root.endswith('_lesion'):
            masks.append(os.path.abspath(os.path.join(root, files[0])))

    images_dataframe = pd.DataFrame({"Image":images, "Mask": masks},)

    images_dataframe.to_csv('prepared_dataset.csv', index=False)


prepare_dataset()