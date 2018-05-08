
# coding: utf-8

# In[1]:

import cv2
import glob
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from keras.backend.tensorflow_backend import set_session
from keras.models import Sequential
from keras.layers import Conv2D, Flatten, Dense, MaxPooling2D, Dropout
from keras.utils.np_utils import to_categorical
from keras import optimizers, regularizers
from keras import losses
import h5py


# In[2]:

config = tf.ConfigProto()
set_session(tf.Session(config=config))


# In[ ]:




# In[3]:

X_train = []
x_label = []
for img_class, directory in enumerate(['red', 'yellow', 'green', 'none']):
    for i, file_name in enumerate(glob.glob("{}/*.jpg".format(directory))):
        file = cv2.imread(file_name)

        file = cv2.cvtColor(file, cv2.COLOR_BGR2RGB);
        resized = cv2.resize(file, (32,64))

        X_train.append(resized/255.)
        x_label.append(img_class)
        
X_train = np.array(X_train)
x_label = np.array(x_label)


# In[ ]:




# In[4]:

labels = to_categorical(x_label)


# In[5]:

num_classes = 4
model = Sequential()
model.add(Conv2D(32, (3, 3), input_shape=(64, 32, 3), padding='same', activation='relu', kernel_initializer='random_uniform', kernel_regularizer=regularizers.l2(0.01)))
model.add(MaxPooling2D(2,2))
Dropout(0.8)
model.add(Conv2D(32, (3, 3), padding='same', activation='relu', kernel_initializer='random_uniform', kernel_regularizer=regularizers.l2(0.01)))
model.add(MaxPooling2D(2,2))
Dropout(0.8)
model.add(Flatten())

model.add(Dense(8, activation='relu', kernel_initializer='random_uniform', kernel_regularizer=regularizers.l2(0.01)))
model.add(Dense(num_classes, activation='softmax'))


# In[ ]:




# In[6]:

loss = losses.categorical_crossentropy
optimizer = optimizers.Adam()

model.compile(loss=loss, optimizer=optimizer, metrics=['accuracy'])

model.fit(X_train, labels, batch_size=32, epochs=20, verbose=True, validation_split=0.1, shuffle=True)

score = model.evaluate(X_train, labels, verbose=0)

print(score)


# In[7]:

model.save('tl_classifier_simulator_one.h5')


# In[ ]:




# In[ ]:




# In[ ]:




# In[ ]:




# In[ ]:




# In[ ]:




# In[ ]:




# In[ ]:




# In[ ]:




# In[ ]:




# In[ ]:



