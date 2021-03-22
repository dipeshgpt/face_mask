#!/usr/bin/env python
# coding: utf-8

# In[ ]:


from keras.applications import VGG16

# VGG16 was designed to work on 224 x 224 pixel input images sizes
img_rows = 224
img_cols = 224 

#Loads the VGG16 model 
model = VGG16(weights = 'imagenet', 
                 include_top = False, 
                 input_shape = (img_rows, img_cols, 3))


# In[ ]:


# Let's print our layers 
for (i,layer) in enumerate(model.layers):
    print(str(i) + " "+ layer.__class__.__name__, layer.trainable)


# In[ ]:


### Let's freeze all layers except the top 4 
from keras.applications import VGG16

# VGG16 was designed to work on 224 x 224 pixel input images sizes
img_rows = 224
img_cols = 224 

# Re-loads the VGG16 model without the top or FC layers
model = VGG16(weights = 'imagenet', 
                 include_top = False, 
                 input_shape = (img_rows, img_cols, 3))

# Here we freeze the last 4 layers 
# Layers are set to trainable as True by default
for layer in model.layers:
    layer.trainable = False
    
# Let's print our layers 
for (i,layer) in enumerate(model.layers):
    print(str(i) + " "+ layer.__class__.__name__, layer.trainable)


# In[ ]:


### Let's make a function that returns our FC Head
def addTopModel(bottom_model, num_classes, D=256):
    """creates the top or head of the model that will be 
    placed ontop of the bottom layers"""
    top_model = bottom_model.output
    top_model = Flatten(name = "flatten")(top_model)
    top_model = Dense(D, activation = "relu")(top_model)
    top_model = Dropout(0.3)(top_model)
    top_model = Dense(num_classes, activation = "softmax")(top_model)
    return top_model


# In[ ]:


model.input


# In[ ]:


model.layers


# In[ ]:


### Let's add our FC Head back onto VGG
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Conv2D, MaxPooling2D, ZeroPadding2D
from keras.layers.normalization import BatchNormalization
from keras.models import Model

num_classes = 2

FC_Head = addTopModel(model, num_classes)

modelnew = Model(inputs=model.input, outputs=FC_Head)

print(modelnew.summary())


# In[ ]:


### Loading our Mask's Dataset
from keras.preprocessing.image import ImageDataGenerator

train_data_dir = 'experiments/test_folder/train/'
validation_data_dir = 'experiments/test_folder/test/'

train_datagen = ImageDataGenerator(
      rescale=1./255,
      rotation_range=20,
      width_shift_range=0.2,
      height_shift_range=0.2,
      horizontal_flip=True,
      fill_mode='nearest')
 
validation_datagen = ImageDataGenerator(rescale=1./255)
 
# Change the batchsize according to your system RAM
train_batchsize = 16
val_batchsize = 10
 
train_generator = train_datagen.flow_from_directory(
        train_data_dir,
        target_size=(img_rows, img_cols),
        batch_size=train_batchsize,
        class_mode='categorical')
 
validation_generator = validation_datagen.flow_from_directory(
        validation_data_dir,
        target_size=(img_rows, img_cols),
        batch_size=val_batchsize,
        class_mode='categorical',
        shuffle=False)


# In[ ]:


### Training our top layers
from keras.optimizers import RMSprop

# Note we use a very small learning rate 
modelnew.compile(loss = 'categorical_crossentropy',
              optimizer = RMSprop(lr = 0.001),
              metrics = ['accuracy'])

nb_train_samples = 1190
nb_validation_samples = 170
epochs = 3
batch_size = 16

history = modelnew.fit_generator(
    train_generator,
    steps_per_epoch = nb_train_samples // batch_size,
    epochs = epochs,
    validation_data = validation_generator,
    validation_steps = nb_validation_samples // batch_size)

modelnew.save("face_mask_vgg.h5")


# In[5]:
"""
#To detect the mask live using webcam
from keras.preprocessing import image
from keras.models import load_model
import numpy as np
import cv2
modelnew = load_model('face_mask_vgg.h5')

cap = cv2.VideoCapture(0)
while True:
    status , photo = cap.read()
    photo_new = cv2.resize(photo, (224,224))
    #photo = image.img_to_array(photo)
    photo_new = np.expand_dims(photo_new, axis = 0)
    res = modelnew.predict(photo_new)
    classes = np.argmax(res, axis = 1)
    cv2.imshow('hi',photo)
    if cv2.waitKey(10) == 13:
            break
    if classes[0] == 0:
        #photo = cv2.rectangle(photo, (100,50), (150,100), [0,255,0], 5  )
        photo = cv2.putText(photo, 'With Mask', (200,30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2, cv2.LINE_AA)
        cv2.imshow('hi' , photo)
        if cv2.waitKey(10) == 13:
            break
    elif classes[0] == 1:
        #photo = cv2.rectangle(photo, (100,50), (150,100), [0,255,0], 5  )
        photo = cv2.putText(photo, 'Without Mask', (200,30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2, cv2.LINE_AA)
        cv2.imshow('hi' , photo)
        if cv2.waitKey(10) == 13:
            break
cap.release()        
cv2.destroyAllWindows() """

