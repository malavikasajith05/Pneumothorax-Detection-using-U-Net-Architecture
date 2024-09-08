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


def load_true_mask(true_mask_path, img_height=256, img_width=256):
    mask = cv2.imread(true_mask_path, cv2.IMREAD_GRAYSCALE)
    mask = cv2.resize(mask, (img_height, img_width))
    mask = mask.astype(np.float32) / 255.0
    mask = np.expand_dims(mask, axis=-1)
    return mask


def predict_and_save_masks(image_folder, mask_folder, result_folder):
    image_files = os.listdir(image_folder)
    for image_file in image_files:
        image_path = os.path.join(image_folder, image_file)
        true_mask_path = os.path.join(mask_folder, image_file)
        
        
        new_image = preprocess_image(image_path)
        true_mask = load_true_mask(true_mask_path)
        
        
        prediction = model.predict(np.array([new_image]))
        
        
        threshold = 0.5
        binary_prediction = (prediction[0] > threshold).astype(np.uint8)
        
       
        true_mask_resized = cv2.resize(true_mask, (new_image.shape[1], new_image.shape[0]))
        
        
        binary_prediction_rgb = cv2.cvtColor(binary_prediction * 255, cv2.COLOR_GRAY2RGB)
        true_mask_rgb = cv2.cvtColor(true_mask_resized * 255, cv2.COLOR_GRAY2RGB)
        
        
        line = np.ones((true_mask_rgb.shape[0], 10, 3), dtype=np.uint8) * 255  # 10 is the width of the line
        
      
        result_image = np.concatenate((true_mask_rgb, line, binary_prediction_rgb), axis=1)
        
        
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(result_image, 'True Mask', (10, 30), font, 1, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.putText(result_image, 'Predicted Mask', (true_mask_rgb.shape[1] + 30, 30), font, 1, (255, 255, 255), 2, cv2.LINE_AA)
        
       
        result_path = os.path.join(result_folder, image_file)
        cv2.imwrite(result_path, result_image)


image_folder = r'val\img'

mask_folder = r'val\mask'

result_folder = r'result'


predict_and_save_masks(image_folder, mask_folder, result_folder)

y_pred = model.predict(X_test_seg)  


y_pred_proba = y_pred 

y_pred_proba = y_pred[:, 0]  


fpr, tpr, thresholds = roc_curve(Label_test, y_pred_proba)
roc_auc = auc(fpr, tpr)


plt.figure()
plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (area = %0.2f)' % roc_auc)
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic')
plt.legend(loc="lower right")
plt.savefig('ROCvsAUC2.png')  
plt.show()  