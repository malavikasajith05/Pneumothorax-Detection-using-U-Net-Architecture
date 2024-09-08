import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
import os
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report
import itertools
from data_utils import extract_data
from model_utils import UNetPlusPlus
from metrics import dice_coef, bce_dice_loss, iou_loss_score, my_iou_metric

IMG_HEIGHT = 256
IMG_WIDTH = 256
IMG_CHANNEL = 1
IMG_PATH = r'C:\Users\aniru\OneDrive\Desktop\malu\siim-acr-pneumothorax\png_images'
MASK_PATH = r'C:\Users\aniru\OneDrive\Desktop\malu\siim-acr-pneumothorax\png_masks'


train_df = pd.read_csv(r'C:\Users\aniru\OneDrive\Desktop\malu\siim-acr-pneumothorax\stage_1_train_images.csv')
test_df = pd.read_csv(r'C:\Users\aniru\OneDrive\Desktop\malu\siim-acr-pneumothorax\stage_1_test_images.csv')


train_d, val_d = train_test_split(train_df, test_size=0.1, random_state=42)


X_train_seg, Y_train_seg, Label_train = extract_data(train_d, IMG_HEIGHT, IMG_WIDTH, IMG_PATH, MASK_PATH)
X_val_seg, Y_val_seg, Label_val = extract_data(val_d, IMG_HEIGHT, IMG_WIDTH, IMG_PATH, MASK_PATH)
X_test_seg, Y_test_seg, Label_test = extract_data(test_df, IMG_HEIGHT, IMG_WIDTH, IMG_PATH, MASK_PATH)


seed = 42 
train_gen = ImageDataGenerator(rescale=1.0/255.0)
val_gen = ImageDataGenerator(rescale=1.0/255.0)
train_it = train_gen.flow(X_train_seg, Y_train_seg / 255.0, batch_size=16, shuffle=True, seed=seed)
val_it = train_gen.flow(X_val_seg, Y_val_seg / 255.0, batch_size=16, shuffle=True, seed=seed)


model = UNetPlusPlus()


model.compile(loss=bce_dice_loss, optimizer=Adam(learning_rate=1e-4), metrics=[dice_coef, iou_loss_score])


checkpoint = ModelCheckpoint("unetplusplus_best_model.hdf5", monitor='val_loss', verbose=1, save_best_only=True, mode='min')
early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)


history = model.fit(train_it, validation_data=val_it, epochs=20, callbacks=[checkpoint, early_stopping], verbose=1)


model.save('unetplusplus_final_model.hdf5')
epochs = range(1, len(history.history['loss']) + 1)  


train_dice_coeff = history.history['dice_coef']
val_dice_coeff = history.history['val_dice_coef']
train_dice_loss = history.history['loss']
val_dice_loss = history.history['val_loss']


binary_predictions = model.predict(X_val_seg)


plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
plt.plot(epochs, train_dice_coeff, 'b-o')
plt.title('Train Dice Coefficient')
plt.xlabel('Epochs')
plt.ylabel('Dice Coefficient')
plt.grid(True)
plt.subplot(1, 2, 2)
plt.plot(epochs, val_dice_coeff, 'r-o')
plt.title('Validation Dice Coefficient')
plt.xlabel('Epochs')
plt.ylabel('Dice Coefficient')
plt.grid(True)
plt.tight_layout()
plt.show()


plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
plt.plot(epochs, train_dice_loss, 'b-o')
plt.title('Train Dice Loss')
plt.xlabel('Epochs')
plt.ylabel('Dice Loss')
plt.grid(True)
plt.subplot(1, 2, 2)
plt.plot(epochs, val_dice_loss, 'r-o')
plt.title('Validation Dice Loss')
plt.xlabel('Epochs')
plt.ylabel('Dice Loss')
plt.grid(True)
plt.tight_layout()
plt.show()


y_true = np.squeeze(Y_val_seg).astype(int)
y_pred = np.squeeze(binary_predictions).astype(int)


accuracy = accuracy_score(y_true.flatten(), y_pred.flatten())


precision = precision_score(y_true.flatten(), y_pred.flatten(), average='weighted', zero_division=1)


recall = recall_score(y_true.flatten(), y_pred.flatten(), average='weighted')


f1 = f1_score(y_true.flatten(), y_pred.flatten(), average='weighted')


conf_matrix = confusion_matrix(y_true.flatten(), y_pred.flatten())


class_report = classification_report(y_true.flatten(), y_pred.flatten())


metrics_names = ['Accuracy', 'Precision', 'Recall', 'F1 Score']
metrics_values = [accuracy, precision, recall, f1]

plt.figure(figsize=(10, 5))
plt.bar(metrics_names, metrics_values, color='skyblue')
plt.xlabel('Metrics')
plt.ylabel('Values')
plt.title('Model Evaluation Metrics')
plt.ylim(0, 1.0)


for i, v in enumerate(metrics_values):
    plt.text(i, v + 0.01, str(round(v, 2)), ha='center', va='bottom')

plt.savefig('model_metrics2.png')
plt.close()


plt.figure(figsize=(8, 6))
plt.imshow(conf_matrix, interpolation='nearest', cmap=plt.cm.Blues)
plt.title('Confusion Matrix')
plt.colorbar()

classes = ['Non Pneumothorax', 'Pneumothorax']  
tick_marks = np.arange(len(classes))

plt.xticks(tick_marks, classes, rotation=45)
plt.yticks(tick_marks, classes)

thresh = conf_matrix.max() / 2.
for i, j in itertools.product(range(conf_matrix.shape[0]), range(conf_matrix.shape[1])):
    plt.text(j, i, format(conf_matrix[i, j], 'd'),
             horizontalalignment="center",
             color="white" if conf_matrix[i, j] > thresh else "black")

plt.tight_layout()
plt.ylabel('True label')
plt.xlabel('Predicted label')

plt.savefig('confusion_matrix2.png')
plt.close()


with open('classification_report2.txt', 'w') as f:
    f.write(class_report)

print("Plots and Classification Report saved successfully.")