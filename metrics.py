# metrics.py

import tensorflow as tf
import numpy as np
from tensorflow.keras.losses import binary_crossentropy
from tensorflow.keras import backend as K

def dice_loss(y_true, y_pred):
    smooth = 1.
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = y_true_f * y_pred_f
    score = (2. * K.sum(intersection) + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)
    return 1. - score

def bce_dice_loss(y_true, y_pred):
    return binary_crossentropy(y_true, y_pred) + dice_loss(y_true, y_pred)

def iou_loss_score(y_pred, y_true, smooth=1):
    intersection = K.sum(K.abs(y_true * y_pred), axis=-1)
    union = K.sum(y_true, -1) + K.sum(y_pred, -1) - intersection
    iou = (intersection + smooth)/(union + smooth)
    return iou

def dice_coef(y_true, y_pred):
    y_true_f = K.flatten(y_true) 
    y_pred_f = K.flatten(y_pred)
#     y_pred = K.cast(y_pred, 'float32')
#     y_pred_f = K.cast(K.greater(K.flatten(y_pred), 0.05), 'float32')
    intersection = y_true_f * y_pred_f
    score = 2. * K.sum(intersection) / (K.sum(y_true_f) + K.sum(y_pred_f))
    return score

def get_iou_vector(A, B):
    batch_size = A.shape[0]
    metric = 0.0
    for batch in range(batch_size):
        t, p = A[batch], B[batch]
        true = np.sum(t)
        pred = np.sum(p)
        
       
        if true == 0:
            metric += (pred == 0)
            continue
        
        
        intersection = np.sum(t * p)
        union = true + pred - intersection
        iou = intersection / union
        
       
        iou = np.floor(max(0, (iou - 0.45)*20)) / 10
        
        metric += iou
        
    
    metric /= batch_size
    return metric


def my_iou_metric(label, pred):
    
    return tf.py_function(get_iou_vector, [label, pred > 0.5], tf.float64)