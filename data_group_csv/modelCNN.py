import numpy as np
import pandas as pd
import os
import gc
import warnings;
warnings.filterwarnings('ignore')

root = "./images_dir_gray_128"
import sys
import os
from math import log
import numpy as np
import scipy as sp
from PIL import Image
import matplotlib.pyplot as plt

from keras.preprocessing.image import ImageDataGenerator
batches = ImageDataGenerator().flow_from_directory(directory=root, target_size=(64,64), batch_size=700)
print(batches.class_indices)

imgs, labels = next(batches)

print("imgs.shape: ", imgs.shape)
print("labels.shape: ", labels.shape)

from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split( imgs, labels , test_size=0.2) #    random_state=2

num_classes = 12;

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from keras.layers import Dropout, Conv2D, Flatten, MaxPool1D, MaxPooling2D
from keras import layers

# Create the model CNN
model = Sequential()
model.add(Conv2D(30, kernel_size=(3, 3), activation='relu', input_shape=(64,64,3)))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Flatten())
model.add(Dense(256, input_shape=(64,64,3), activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(256, input_shape=(64,64,3), activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(num_classes, activation='softmax'))
model.compile(loss='categorical_crossentropy', optimizer = 'adam', metrics=['accuracy'])
model.summary()

model.fit(X_train, Y_train, epochs=10, verbose=1, validation_split=0.1)


# save the model
from tensorflow import keras
model.save('./ketqua/model_CNN')

# Test the model after training(Kiểm tra mô hình sau khi đào tạo)
test_results = model.evaluate(X_test, Y_test, verbose=1)
print(f'Test results - Loss: {test_results[0]} - Accuracy: {test_results[1]}%')

train_results = model.evaluate(X_train, Y_train , verbose=1)
print(f'Train results - Loss: {train_results[0]} - Accuracy: {train_results[1]}%')

# predict crisp classes for test set(dự đoán các lớp nhãn cho bộ thử nghiệm)
y_pred = model.predict_classes(X_test)

Y_test2 = np.argmax(Y_test, axis=1)

# Accuracy: (tp + tn) / (p + n)
from sklearn.metrics import accuracy_score
accuracy = accuracy_score(Y_test2, y_pred)
print('Accuracy: %.4f' % accuracy)

# Precision tp / (tp + fp)
from sklearn.metrics import precision_score
precision = precision_score(Y_test2, y_pred, pos_label='positive', average='micro')
print('Precision: %.4f' % precision)

# Recall: tp / (tp + fn)
from sklearn.metrics import recall_score
recall = recall_score(Y_test2, y_pred, pos_label='positive', average='micro')
print('Recall: %.4f' % recall)

# F1: 2 tp / (2 tp + fp + fn)
from sklearn.metrics import f1_score
f1 = f1_score(Y_test2, y_pred, pos_label='positive', average='micro')
print('F1 score: %.4f' % f1)

#   tên nhãn
#   nhan = ['Ramnit', 'Lollipop', 'Kelihos_ver3', 'Vundo', 'Simda', 'Tracur', 'Kelihos_ver1' , 'Obfuscator.ACY', 'Gatak']

# Classification report for precision, recall, f1-score and accuracy (báo cáo phân loại)
from sklearn.metrics import classification_report
matrixXXX = classification_report(Y_test2, y_pred, digits=4)
print('\n Classification report:')
print(matrixXXX)

# create Confusion matrix
from sklearn.metrics import confusion_matrix
#Cmatrix = confusion_matrix(Y_test2, y_pred, normalize='true')

# Tạo biểu đồ cho Confusion matrix
from sklearn.metrics import ConfusionMatrixDisplay
from matplotlib import pyplot as plt
#cmd = ConfusionMatrixDisplay(Cmatrix, display_labels=nhan)
#cmd = ConfusionMatrixDisplay(Cmatrix)
#cmd.plot(cmap='Blues', xticks_rotation='vertical')
#plt.savefig('./ketqua/model_CNN.png')



print("--------------- [The end] ------------------")


