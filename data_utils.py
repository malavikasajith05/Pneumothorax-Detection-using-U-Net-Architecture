# data_utils.py

import os
import numpy as np
from skimage.io import imread
from skimage.transform import resize
from tqdm import tqdm

def extract_data(data_df, IMG_HEIGHT, IMG_WIDTH, IMG_PATH, MASK_PATH):
    X = np.zeros((len(data_df), IMG_HEIGHT, IMG_WIDTH, 1), dtype=np.float32)
    Y = np.zeros((len(data_df), IMG_HEIGHT, IMG_WIDTH, 1), dtype=np.float32)
    Label_seg = np.zeros(len(data_df), dtype=np.uint8)
    
    img_data = list(data_df.T.to_dict().values())
    for i, data_row in tqdm(enumerate(img_data), total=len(img_data)):
        patientImage = data_row['new_filename']
        imageLabel = data_row['has_pneumo']

        imagePath = os.path.join(IMG_PATH, patientImage)
        lungImage = imread(imagePath)
        lungImage = np.expand_dims(resize(lungImage, (IMG_HEIGHT, IMG_WIDTH), mode='constant', preserve_range=True), axis=-1)
        X[i] = lungImage
        Label_seg[i] = imageLabel

        maskPath = os.path.join(MASK_PATH, patientImage)
        maskImage = imread(maskPath)
        maskImage = np.expand_dims(resize(maskImage, (IMG_HEIGHT, IMG_WIDTH), mode='constant', preserve_range=True), axis=-1)
        Y[i] = maskImage

    return X, Y, Label_seg
