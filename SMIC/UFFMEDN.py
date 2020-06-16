import numpy
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from keras.models import Sequential, Model
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.layers.convolutional import Convolution3D, MaxPooling3D, ZeroPadding3D
from keras.layers import LeakyReLU ,PReLU
from keras.callbacks import ModelCheckpoint
from sklearn.model_selection import train_test_split
from keras import backend as K



def evaluate(segment_train_images, segment_validation_images, segment_train_labels, segment_validation_labels,test_index ):

    model = Sequential()
    #model.add(ZeroPadding3D((1,1,0),input_shape=(1, 32, 32, 18)))
    model.add(Convolution3D(32, (3, 3, 15),input_shape=(1, 32, 32, 18)))
    model.add( PReLU(alpha_initializer="zeros"))
    model.add(Dropout(0.5))
    model.add(MaxPooling3D(pool_size=(3, 3, 3)))
    model.add(PReLU(alpha_initializer="zeros"))
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

    filepath="weights_late_microexpfusenet/weights-improvement"+str(test_index)+"-{epoch:02d}-{val_acc:.2f}.hdf5"
    checkpoint = ModelCheckpoint(filepath, monitor='val_acc', verbose=1, save_best_only=True, mode='max')
    callbacks_list = [checkpoint]






    # Training the model

    history = model.fit(segment_train_images, segment_train_labels, validation_data = (segment_validation_images, segment_validation_labels), callbacks=callbacks_list, batch_size = 16, nb_epoch = 100, shuffle=True)








    # Finding Confusion Matrix using pretrained weights

    predictions = model.predict([segment_validation_images])
    predictions_labels = numpy.argmax(predictions, axis=1)
    validation_labels = numpy.argmax(segment_validation_labels, axis=1)
    cfm = confusion_matrix(validation_labels, predictions_labels)
    print (cfm)
    print("accuracy: ",accuracy_score(validation_labels, predictions_labels))

    return accuracy_score(validation_labels, predictions_labels)




K.set_image_dim_ordering('th')

segmentName='UpperFace'

# Load training images and labels that are stored in numpy array

segment_training_set = numpy.load('numpy_training_datasets/{0}_images.npy'.format(segmentName))
segment_traininglabels = numpy.load('numpy_training_datasets/{0}_labels.npy'.format(segmentName))

'''
#-----------------------------------------------------------------------------------------------------------------
#LOOCV
loo = LeaveOneOut()
loo.get_n_splits(segment_training_set)
tot=0
count=0
for train_index, test_index in loo.split(segment_training_set):

    print(segment_traininglabels[train_index])
    print(segment_traininglabels[test_index])

    val_acc = evaluate(segment_training_set[train_index], segment_training_set[test_index],segment_traininglabels[train_index], segment_traininglabels[test_index] ,test_index)
    tot+=val_acc
    count+=1
    print("------------------------------------------------------------------------")
    print("validation acc:",val_acc)
    print("------------------------------------------------------------------------")
print(tot/count)

'''
#-----------------------------------------------------------------------------------------------------------------
#Test train split


# Spliting the dataset into training and validation sets
segment_train_images, segment_validation_images, segment_train_labels, segment_validation_labels = train_test_split(segment_training_set,
                                                                                            segment_traininglabels,
                                                                                            test_size=0.2, random_state=42)

# Save validation set in a numpy array
numpy.save('numpy_validation_datasets/{0}_images.npy'.format(segmentName), segment_validation_images)
numpy.save('numpy_validation_datasets/{0}_images.npy'.format(segmentName), segment_validation_labels)

# Loading Load validation set from numpy array
#
# eimg = numpy.load('numpy_validation_datasets/{0}_images.npy'.format(segmentName))
# labels = numpy.load('numpy_validation_datasets/{0}_images.npy'.format(segmentName))

evaluate(segment_train_images, segment_validation_images,segment_train_labels, segment_validation_labels ,0)