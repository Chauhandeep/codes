# Convolutional Neural Network

# Installing Theano
# pip install --upgrade --no-deps git+git://github.com/Theano/Theano.git

# Installing Tensorflow
# pip install tensorflow

# Installing Keras
# pip install --upgrade keras

# Part 1 - Building the CNN

# Importing the Keras libraries and packages
import tensorflow as tf
from keras.models import Sequential
from keras.layers import Conv2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.layers import Dense
import os
import re

# Initialising the CNN
classifier = Sequential()

# Step 1 - Convolution
classifier.add(Conv2D(32, (3, 3), input_shape = (64, 64, 3), activation = 'relu'))

# Step 2 - Pooling
classifier.add(MaxPooling2D(pool_size = (2, 2)))

# Step 3 - Flattening
classifier.add(Flatten())

# Step 4 - Full connection
classifier.add(Dense(units = 128, activation = 'relu'))
classifier.add(Dense(units = 1, activation = 'sigmoid'))

# Compiling the CNN
classifier.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])

# Part 2 - Fitting the CNN to the images

from keras.preprocessing.image import ImageDataGenerator

train_datagen = ImageDataGenerator(rescale = 1./255,
                                   shear_range = 0.2,
                                   zoom_range = 0.2,
                                   horizontal_flip = True)

test_datagen = ImageDataGenerator(rescale = 1./255)

training_set = train_datagen.flow_from_directory('dataset/training_set',
                                                 target_size = (64, 64),
                                                 batch_size = 32,
                                                 class_mode = 'binary')

test_set = test_datagen.flow_from_directory('dataset/test_set',
                                            target_size = (64, 64),
                                            batch_size = 32,
                                            class_mode = 'binary')

checkpoint_path = "/home/deepanshu/Desktop/training_1/cp-{epoch:}.ckpt"
checkpoint_dir = os.path.dirname(checkpoint_path)
    
cp_callback = tf.keras.callbacks.ModelCheckpoint(checkpoint_path, 
                                                 save_weights_only=True,
                                                 verbose=1)

checkpoint_files = os.listdir('/home/deepanshu/Desktop/training_1')      

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
    classifier.fit_generator(training_set,
                             steps_per_epoch = 8000,
                             epochs = numEpochs,
                             validation_data = test_set,
                             validation_steps = 2000,
                             callbacks = [cp_callback])
else:
    try:
        checkpoint_files = sorted(checkpoint_files,key=cmp_to_key(compare))
        number  = re.findall('cp-([0-9]*).ckpt',checkpoint_files[len(checkpoint_files)-1]) 
        print(checkpoint_files[len(checkpoint_files)-1])
        print(number[0])
        classifier.load_weights(checkpoint_dir + '/' + checkpoint_files[len(checkpoint_files)-1])
        for checkpoint in checkpoint_files:
            os.rename(checkpoint_dir + '/' + checkpoint,'/tmp/training_1/' + checkpoint)
        classifier.fit_generator(training_set,
                                 steps_per_epoch = 8000,
                                 epochs = numEpochs-int(number[0]),
                                 validation_data = test_set,
                                 validation_steps = 2000,
                                 callbacks = [cp_callback])
        #Renaming Checkpoints
        new_checkpoints = os.listdir('/home/deepanshu/Desktop/training_1')
        new_checkpoints = sorted(new_checkpoints,key=cmp_to_key(compare))
        i=0
        for checkpoint in new_checkpoints:
            i=i+1
            os.rename(checkpoint_dir + '/' + checkpoint,checkpoint_dir + '/cp-' + str(int(number[0])+i) + '.ckpt')
    except KeyboardInterrupt:
        new_checkpoints = os.listdir('/home/deepanshu/Desktop/training_1')
        new_checkpoints = sorted(new_checkpoints,key=cmp_to_key(compare))
        i=0
        for checkpoint in new_checkpoints:
            i=i+1
            os.rename(checkpoint_dir + '/' + checkpoint,checkpoint_dir + '/cp-' + str(int(number[0])+i) + '.ckpt')
       

    
        


