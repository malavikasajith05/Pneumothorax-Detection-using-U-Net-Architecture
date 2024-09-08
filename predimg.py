import os
import numpy as np
import cv2
from tensorflow.keras.models import load_model
from data_utils import extract_data
from model_utils import UNetPlusPlus
from metrics import dice_coef, bce_dice_loss, iou_loss_score, my_iou_metric
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt



model = load_model(r'model\unetplusplus_final_model.hdf5', custom_objects={'bce_dice_loss': bce_dice_loss, 'dice_coef': dice_coef, 'iou_loss_score': iou_loss_score})


def preprocess_image(image_path, img_height=256, img_width=256):
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    img = cv2.resize(img, (img_height, img_width))
    img = img.astype(np.float32) / 255.0
    img = np.expand_dims(img, axis=-1)
    return img


def predict_mask(image_path):
   
    new_image = preprocess_image(image_path)
    
    
    prediction = model.predict(np.array([new_image]))
    
    
    threshold = 0.5
    binary_prediction = (prediction[0] > threshold).astype(np.uint8)
    
    
    binary_prediction_rgb = cv2.cvtColor(binary_prediction * 255, cv2.COLOR_GRAY2RGB)
    
    return binary_prediction_rgb

image_path = r'val\img\19_test_1_.png'

predicted_mask = predict_mask(image_path)


original_image = cv2.imread(image_path)


predicted_mask_resized = cv2.resize(predicted_mask, (original_image.shape[1], original_image.shape[0]))


result_image = np.concatenate((original_image, predicted_mask_resized), axis=1)


scale_percent = 50  
width = int(result_image.shape[1] * scale_percent / 100)
height = int(result_image.shape[0] * scale_percent / 100)
resized_image = cv2.resize(result_image, (width, height))

cv2.imshow('Original Image vs. Predicted Mask', resized_image)
cv2.waitKey(0)
cv2.destroyAllWindows()