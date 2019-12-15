import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn
from tensorflow.python.tools import module_util as _module_util
import keras
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Dense, Conv2D, MaxPooling2D, Flatten, Dropout

img_width, img_height = 64, 64
# data path
train_data_dir = 'dataset/train'
validation_data_dir = 'dataset/validation'

# rescaling data
datagen = ImageDataGenerator(rescale=1./255)

# import train data and validation data
train_generator = datagen.flow_from_directory(
        train_data_dir,
        target_size=(img_width, img_height),
        class_mode='binary')

validation_generator = datagen.flow_from_directory(
        validation_data_dir,
        target_size=(img_width, img_height),
        class_mode='binary')

# create machine learning model
model = Sequential()
model.add(Conv2D(32, kernel_size=(5, 5), activation = 'relu', input_shape=(img_width, img_height, 3) ))
model.add(MaxPooling2D(pool_size = (2, 2)))
model.add(Conv2D(16, kernel_size = (3, 3), activation = 'relu'))
model.add(MaxPooling2D(pool_size = (2, 2)))
model.add(Flatten())
model.add(Dense(64, activation = 'relu'))
model.add(Dense(1, activation = 'sigmoid'))
model.compile(loss = keras.losses.binary_crossentropy,
              optimizer=keras.optimizers.Adam(lr=0.00005, beta_2=0.9),
              metrics=['accuracy'])
model_hist = model.fit_generator(train_generator, 
                                 epochs=20,
                                 validation_data=validation_generator)

# Save the weights
model.save_weights('model_weights.h5')

# Save the model architecture
with open('model_architecture.json', 'w') as f:
    f.write(model.to_json())

model.save('modelx.h5')

plt.figure(figsize=(10, 6))
plt.plot(model_hist.history['loss'], label='loss')
plt.plot(model_hist.history['val_loss'], label='val_loss')
plt.title('Grafik Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.grid()
plt.legend()
plt.show()

plt.figure(figsize=(10, 6))
plt.plot(model_hist.history['val_accuracy'], label='val_accuracy')
plt.plot(model_hist.history['accuracy'], label='accuracy')
plt.title('Grafik Akurasi')
plt.xlabel('Epoch')
plt.ylabel('Akurasi')
plt.grid()
plt.legend()
plt.show()