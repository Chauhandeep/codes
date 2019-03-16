# -*- coding: utf-8 -*-

import keras
from keras.applications.inception_resnet_v2 import InceptionResNetV2
from keras.preprocessing import image
from keras.engine import Layer
from keras.applications.inception_resnet_v2 import preprocess_input
from keras.layers import Conv2D, UpSampling2D, InputLayer, Conv2DTranspose, Input, Reshape, merge, concatenate
from keras.layers import Activation, Dense, Dropout, Flatten
from keras.layers.normalization import BatchNormalization
from keras.callbacks import TensorBoard 
from keras.models import Sequential, Model
from keras.layers.core import RepeatVector, Permute
from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img
import numpy as np
import os
import random
import tensorflow as tf
from skimage.color import rgb2lab, lab2rgb, rgb2gray, gray2rgb
from skimage.transform import resize
from skimage.io import imsave

# Get images
X = []
for filename in os.listdir('images/Train/'):
    X.append(img_to_array(load_img('images/Train/'+filename)))
X = np.array(X, dtype=float)
Xtrain = 1.0/255*X


#Load weights
inception = InceptionResNetV2(weights='imagenet', include_top=True)
inception.graph = tf.get_default_graph()


embed_input = Input(shape=(1000,))

#Encoder
encoder_input = Input(shape=(256, 256, 1,))
encoder_output = Conv2D(64, (3,3), activation='relu', padding='same', strides=2)(encoder_input)
encoder_output = Conv2D(128, (3,3), activation='relu', padding='same')(encoder_output)
encoder_output = Conv2D(128, (3,3), activation='relu', padding='same', strides=2)(encoder_output)
encoder_output = Conv2D(256, (3,3), activation='relu', padding='same')(encoder_output)
encoder_output = Conv2D(256, (3,3), activation='relu', padding='same', strides=2)(encoder_output)
encoder_output = Conv2D(512, (3,3), activation='relu', padding='same')(encoder_output)
encoder_output = Conv2D(512, (3,3), activation='relu', padding='same')(encoder_output)
encoder_output = Conv2D(256, (3,3), activation='relu', padding='same')(encoder_output)

#Fusion
fusion_output = RepeatVector(32 * 32)(embed_input) 
fusion_output = Reshape(([32, 32, 1000]))(fusion_output)
fusion_output = concatenate([encoder_output, fusion_output], axis=3) 
fusion_output = Conv2D(256, (1, 1), activation='relu', padding='same')(fusion_output) 

#Decoder
decoder_output = Conv2D(128, (3,3), activation='relu', padding='same')(fusion_output)
decoder_output = UpSampling2D((2, 2))(decoder_output)
decoder_output = Conv2D(64, (3,3), activation='relu', padding='same')(decoder_output)
decoder_output = UpSampling2D((2, 2))(decoder_output)
decoder_output = Conv2D(32, (3,3), activation='relu', padding='same')(decoder_output)
decoder_output = Conv2D(16, (3,3), activation='relu', padding='same')(decoder_output)
decoder_output = Conv2D(2, (3, 3), activation='tanh', padding='same')(decoder_output)
decoder_output = UpSampling2D((2, 2))(decoder_output)

model = Model(inputs=[encoder_input, embed_input], outputs=decoder_output)

def create_inception_embedding(grayscaled_rgb):
    grayscaled_rgb_resized = []
    for i in grayscaled_rgb:
        i = resize(i, (299, 299, 3), mode='constant')
        grayscaled_rgb_resized.append(i)
    grayscaled_rgb_resized = np.array(grayscaled_rgb_resized)
    grayscaled_rgb_resized = preprocess_input(grayscaled_rgb_resized)
    with inception.graph.as_default():
        embed = inception.predict(grayscaled_rgb_resized)
    return embed

# Image transformer
datagen = ImageDataGenerator(
        shear_range=0.2,
        zoom_range=0.2,
        rotation_range=20,
        horizontal_flip=True)

#Generate training data
batch_size = 10

def image_a_b_gen(batch_size):
    for batch in datagen.flow(Xtrain, batch_size=batch_size):
        grayscaled_rgb = gray2rgb(rgb2gray(batch))
        embed = create_inception_embedding(grayscaled_rgb)
        lab_batch = rgb2lab(batch)
        X_batch = lab_batch[:,:,:,0]
        X_batch = X_batch.reshape(X_batch.shape+(1,))
        Y_batch = lab_batch[:,:,:,1:] / 128
        yield ([X_batch, create_inception_embedding(grayscaled_rgb)], Y_batch)


#Train model      
model.compile(optimizer='rmsprop', loss='mse')

checkpoint_path = "/home/deepanshu/Desktop/colorTraining/cp-{epoch:}.ckpt"
checkpoint_dir = os.path.dirname(checkpoint_path)
    
cp_callback = tf.keras.callbacks.ModelCheckpoint(checkpoint_path, 
                                                 save_weights_only=True,
                                                 verbose=1)

checkpoint_files = os.listdir('/home/deepanshu/Desktop/colorTraining')      

numEpochs = 25

def cmp_to_key(mycmp):
    'Convert a cmp= function into a key= function'
    class K:
        def __init__(self, obj, *args):
            self.obj = obj
        def __lt__(self, other):
            return mycmp(self.obj, other.obj) < 0
        def __gt__(self, other):
            return mycmp(self.obj, other.obj) > 0
        def __eq__(self, other):
            return mycmp(self.obj, other.obj) == 0
        def __le__(self, other):
            return mycmp(self.obj, other.obj) <= 0
        def __ge__(self, other):
            return mycmp(self.obj, other.obj) >= 0
        def __ne__(self, other):
            return mycmp(self.obj, other.obj) != 0
    return K

def compare(x, y):
    num1 = int(re.findall('cp-([0-9]*).ckpt',x)[0])
    num2 = int(re.findall('cp-([0-9]*).ckpt',y)[0])
    if num1<num2:
        return -1
    elif num2<num1:
        return 1
    else:
        return 0


if len(checkpoint_files)==0:
    model.fit_generator(image_a_b_gen(batch_size), epochs=numEpochs, steps_per_epoch=1000,callbacks = [cp_callback])
else:
    try:
        checkpoint_files = sorted(checkpoint_files,key=cmp_to_key(compare))
        number  = re.findall('cp-([0-9]*).ckpt',checkpoint_files[len(checkpoint_files)-1]) 
        print(checkpoint_files[len(checkpoint_files)-1])
        print(number[0])
        classifier.load_weights(checkpoint_dir + '/' + checkpoint_files[len(checkpoint_files)-1])
        for checkpoint in checkpoint_files:
            os.rename(checkpoint_dir + '/' + checkpoint,'/home/deepanshu/Desktop/colorTrainingTemp' + checkpoint)
        model.fit_generator(image_a_b_gen(batch_size),
                                 steps_per_epoch = 100,
                                 epochs = numEpochs-int(number[0]),
                                 callbacks = [cp_callback])
        #Renaming Checkpoints
        new_checkpoints = os.listdir('/home/deepanshu/Desktop/colorTraining')
        new_checkpoints = sorted(new_checkpoints,key=cmp_to_key(compare))
        i=0
        for checkpoint in new_checkpoints:
            i=i+1
            os.rename(checkpoint_dir + '/' + checkpoint,checkpoint_dir + '/cp-' + str(int(number[0])+i) + '.ckpt')
    except KeyboardInterrupt:
        new_checkpoints = os.listdir('/home/deepanshu/Desktop/colorTraining')
        new_checkpoints = sorted(new_checkpoints,key=cmp_to_key(compare))
        i=0
        for checkpoint in new_checkpoints:
            i=i+1
            os.rename(checkpoint_dir + '/' + checkpoint,checkpoint_dir + '/cp-' + str(int(number[0])+i) + '.ckpt')


color_me = []
for filename in os.listdir('images/test/'):
    color_me.append(img_to_array(load_img('images/test/'+filename)))
color_me = np.array(color_me, dtype=float)
gray_me = gray2rgb(rgb2gray(1.0/255*color_me))
color_me_embed = create_inception_embedding(gray_me)
color_me = rgb2lab(1.0/255*color_me)[:,:,:,0]
color_me = color_me.reshape(color_me.shape+(1,))


# Test model
output = model.predict([color_me, color_me_embed])
output = output * 128

# Output colorizations
for i in range(len(output)):
    cur = np.zeros((256, 256, 3))
    cur[:,:,0] = color_me[i][:,:,0]
    cur[:,:,1:] = output[i]
    imsave("result/img_"+str(i)+".png", lab2rgb(cur))