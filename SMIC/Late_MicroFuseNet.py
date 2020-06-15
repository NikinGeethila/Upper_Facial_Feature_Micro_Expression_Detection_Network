import os
import cv2
from keras import regularizers
import dlib
import numpy
import imageio
from sklearn.metrics import confusion_matrix
from keras.models import Sequential, Model
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.layers.convolutional import Convolution3D, MaxPooling3D
from keras.utils import multi_gpu_model
from keras.optimizers import SGD, RMSprop
from keras.layers import Concatenate, Input, concatenate, add, multiply, maximum
from keras.callbacks import ModelCheckpoint
from keras.utils import np_utils, generic_utils
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from keras import backend as K
K.set_image_dim_ordering('th')

# DLib Face Detection
predictor_path = "shape_predictor_68_face_landmarks.dat"
predictor = dlib.shape_predictor(predictor_path)
detector = dlib.get_frontal_face_detector()
"""
class TooManyFaces(Exception):
    pass

class NoFaces(Exception):
    pass

def get_landmark(img):
    rects = detector(img, 1)
    if len(rects) > 1:
        pass
    if len(rects) == 0:
        pass
    ans = numpy.matrix([[p.x, p.y] for p in predictor(img, rects[0]).parts()])
    return ans

def annotate_landmarks(img, landmarks, font_scale = 0.4):
    for idx, point in enumerate(landmarks):
        pos = (point[0, 0], point[0, 1])
        cv2.putText(img, str(idx), pos, fontFace=cv2.FONT_HERSHEY_SCRIPT_SIMPLEX, fontScale=font_scale, color=(0, 0, 255))
        cv2.circle(img, pos, 3, color=(0, 255, 255))
    return img

negativepath = '../../../Datasets/SIMC_E_categorical/Negative/'
positivepath = '../../../Datasets/SIMC_E_categorical/Positive/'
surprisepath = '../../../Datasets/SIMC_E_categorical/Surprise/'

eye_training_list = []
nose_training_list = []

for typepath in (negativepath,positivepath,surprisepath):
    directorylisting = os.listdir(typepath)
    print(typepath)
    for video in directorylisting:
        videopath = typepath + video
        eye_frames = []
        nose_mouth_frames = []
        framelisting = os.listdir(videopath)
        framerange = [x for x in range(18)]
        for frame in framerange:
               imagepath = videopath + "/" + framelisting[frame]
               image = cv2.imread(imagepath)
               landmarks = get_landmark(image)
               numpylandmarks = numpy.asarray(landmarks)
               up = min(numpylandmarks[18][1], numpylandmarks[19][1], numpylandmarks[23][1], numpylandmarks[24][1]) - 20
               down = max(numpylandmarks[31][1], numpylandmarks[32][1], numpylandmarks[33][1], numpylandmarks[34][1],
                          numpylandmarks[35][1]) + 5
               left = min(numpylandmarks[17][0], numpylandmarks[18][0], numpylandmarks[36][0])
               right = max(numpylandmarks[26][0], numpylandmarks[25][0], numpylandmarks[45][0])
               eye_image = image[up:down, left:right]
               eye_image = cv2.resize(eye_image, (64, 64), interpolation = cv2.INTER_AREA)
               eye_image = cv2.cvtColor(eye_image, cv2.COLOR_BGR2GRAY)
               nose_mouth_image = image[numpylandmarks[2][1]:numpylandmarks[6][1], numpylandmarks[2][0]:numpylandmarks[14][0]]
               nose_mouth_image = cv2.resize(nose_mouth_image, (32, 32), interpolation = cv2.INTER_AREA)
               nose_mouth_image = cv2.cvtColor(nose_mouth_image, cv2.COLOR_BGR2GRAY)
               eye_frames.append(eye_image)
               nose_mouth_frames.append(nose_mouth_image)
        eye_frames = numpy.asarray(eye_frames)
        nose_mouth_frames = numpy.asarray(nose_mouth_frames)
        eye_videoarray = numpy.rollaxis(numpy.rollaxis(eye_frames, 2, 0), 2, 0)
        nose_mouth_videoarray = numpy.rollaxis(numpy.rollaxis(nose_mouth_frames, 2, 0), 2, 0)
        eye_training_list.append(eye_videoarray)
        nose_training_list.append(nose_mouth_videoarray)
        if typepath==surprisepath:
            eye_training_list.append(eye_videoarray)
            nose_training_list.append(nose_mouth_videoarray)
print(len(eye_videoarray))
eye_training_list = numpy.asarray(eye_training_list)
nose_training_list = numpy.asarray(nose_training_list)

eye_trainingsamples = len(eye_training_list)
nose_trainingsamples = len(nose_training_list)

eye_traininglabels = numpy.zeros((eye_trainingsamples, ), dtype = int)
nose_traininglabels = numpy.zeros((nose_trainingsamples, ), dtype = int)

eye_traininglabels[0:66] = 0
eye_traininglabels[66:113] = 1
eye_traininglabels[113:156] = 2

nose_traininglabels[0:66] = 0
nose_traininglabels[66:113] = 1
nose_traininglabels[113:156] = 2

eye_traininglabels = np_utils.to_categorical(eye_traininglabels, 3)
nose_traininglabels = np_utils.to_categorical(nose_traininglabels, 3)

etraining_data = [eye_training_list, eye_traininglabels]
(etrainingframes, etraininglabels) = (etraining_data[0], etraining_data[1])
etraining_set = numpy.zeros((eye_trainingsamples, 1, 64, 64, 18))
for h in range(eye_trainingsamples):
    etraining_set[h][0][:][:][:] = etrainingframes[h,:,:,:]

etraining_set = etraining_set.astype('float32')
etraining_set -= numpy.mean(etraining_set)
etraining_set /= numpy.max(etraining_set)

ntraining_data = [nose_training_list, nose_traininglabels]
(ntrainingframes, ntraininglabels) = (ntraining_data[0], ntraining_data[1])
ntraining_set = numpy.zeros((nose_trainingsamples, 1, 32, 32, 18))
for h in range(nose_trainingsamples):
        ntraining_set[h][0][:][:][:] = ntrainingframes[h,:,:,:]

ntraining_set = ntraining_set.astype('float32')
ntraining_set -= numpy.mean(ntraining_set)
ntraining_set /= numpy.max(ntraining_set)

numpy.save('numpy_training_datasets/late_microexpfusenetnoseimages.npy', ntraining_set)
numpy.save('numpy_training_datasets/late_microexpfuseneteyeimages.npy', etraining_set)
numpy.save('numpy_training_datasets/late_microexpfusenetnoselabels.npy', nose_traininglabels)
numpy.save('numpy_training_datasets/late_microexpfuseneteyelabels.npy', eye_traininglabels)

# Load training images and labels that are stored in numpy array
"""
etraining_set = numpy.load('numpy_training_datasets/late_microexpfuseneteyeimages.npy')
eye_traininglabels = numpy.load('numpy_training_datasets/late_microexpfusenetnoselabels.npy')

image_rows, image_columns, image_depth = 64, 64, 18
# Late MicroExpFuseNet Model
model = Sequential()
model.add(Convolution3D(32, (3, 3, 15), input_shape=(1, image_rows, image_columns, image_depth)))
model.add(Activation('LeakyReLU'))
model.add(MaxPooling3D(pool_size=(3, 3, 3)))
model.add(Activation('LeakyReLU'))
model.add(Dropout(0.5))
model.add(Flatten())
model.add(Dense(1024, init='normal'))
model.add(Dropout(0.5))
model.add(Dense(128, init='normal'))
model.add(Dropout(0.5))
model.add(Dense(3, init='normal'))
model.add(Activation('softmax'))
model.compile(loss = 'categorical_crossentropy', optimizer = 'SGD', metrics = ['accuracy'])

model.summary()

filepath="weights_late_microexpfusenet/weights-improvement-{epoch:02d}-{val_acc:.2f}.hdf5"
checkpoint = ModelCheckpoint(filepath, monitor='val_acc', verbose=1, save_best_only=True, mode='max')
callbacks_list = [checkpoint]



# Spliting the dataset into training and validation sets
etrain_images, evalidation_images, etrain_labels, evalidation_labels =  train_test_split(etraining_set, eye_traininglabels, test_size=0.2, random_state=42)

# Save validation set in a numpy array
numpy.save('numpy_validation_datasets/late_microexpfusenet_eval_images.npy', evalidation_images)
numpy.save('numpy_validation_datasets/late_microexpfusenet_eval_labels.npy', evalidation_labels)

# Training the model
history = model.fit(etrain_images, etrain_labels, validation_data = (evalidation_images, evalidation_labels), callbacks=callbacks_list, batch_size = 16, nb_epoch = 100, shuffle=True)

# Loading Load validation set from numpy array

eimg = numpy.load('numpy_validation_datasets/late_microexpfusenet_eval_images.npy')
labels = numpy.load('numpy_validation_datasets/late_microexpfusenet_eval_labels.npy')


# Finding Confusion Matrix using pretrained weights

predictions = model.predict([eimg])
predictions_labels = numpy.argmax(predictions, axis=1)
validation_labels = numpy.argmax(labels, axis=1)
cfm = confusion_matrix(validation_labels, predictions_labels)
print (cfm)

