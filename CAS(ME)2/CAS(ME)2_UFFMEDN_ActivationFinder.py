import numpy
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score,f1_score
from keras.models import Sequential, Model
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.layers.convolutional import Convolution3D, MaxPooling3D, ZeroPadding3D
from keras.layers import LeakyReLU ,PReLU
from keras.callbacks import ModelCheckpoint,EarlyStopping,ReduceLROnPlateau,Callback
from sklearn.model_selection import train_test_split,LeaveOneOut,KFold
from keras import backend as K
from keras.optimizers import Adam,SGD
import os
from matplotlib import pyplot

class myCallback(Callback):
    def on_epoch_end(self, epoch, logs={}):
        if(logs.get('val_acc') >= 1.0):
            print("\nReached %2.2f%% accuracy, so stopping training!!" %(1.0*100))
            self.model.stop_training = True

def evaluate(segment_train_images, segment_validation_images, segment_train_labels, segment_validation_labels,test_index ):

    model = Sequential()
    #model.add(ZeroPadding3D((2,2,0)))
    model.add(
        Convolution3D(32, (20, 20,9), strides=(10, 10, 3), input_shape=(1, sizeH, sizeV, sizeD), padding='Same'))

    model.add(PReLU())
    # model.add(Dropout(0.5))
    model.add(
        Convolution3D(32, (3, 3, 3), strides=1, padding='Same'))
    model.add(PReLU())
    # model.add(Dropout(0.5))
    # model.add(MaxPooling3D(pool_size=(3, 3, 3)))
    # model.add( PReLU())
    # model.add(Dropout(0.5))
    model.add(Flatten())
    # model.add(Dense(1024, init='normal'))
    # model.add(Dropout(0.5))
    # model.add(Dense(128, init='normal'))
    # model.add(Dropout(0.5))
    model.add(Dense(3, init='normal'))
    model.add(Dropout(0.5))
    model.add(Activation('softmax'))
    opt = SGD(lr=0.01)
    model.compile(loss='categorical_crossentropy', optimizer=opt, metrics=['accuracy'])

    model.summary()

    filepath="weights_CAS(ME)2/weights-improvement"+str(test_index)+"-{epoch:02d}-{val_acc:.2f}.hdf5"
    checkpoint = ModelCheckpoint(filepath, monitor='val_acc', verbose=1, save_best_only=True, mode='max')
    EarlyStop = EarlyStopping(monitor='val_acc', min_delta=0, patience=20, restore_best_weights=True, verbose=1, mode='max')
    reduce = ReduceLROnPlateau(monitor='val_acc', factor=0.5, patience=30,cooldown=10, verbose=1,min_delta=0, mode='max',min_lr=0.0005)
    callbacks_list = [ EarlyStop, reduce,myCallback()]






    # Training the model

    history = model.fit(segment_train_images, segment_train_labels, validation_data = (segment_validation_images, segment_validation_labels), callbacks=callbacks_list, batch_size = 16, nb_epoch = 500, shuffle=True,verbose=1)








    # Finding Confusion Matrix using pretrained weights

    predictions = model.predict([segment_validation_images])
    predictions_labels = numpy.argmax(predictions, axis=1)
    validation_labels = numpy.argmax(segment_validation_labels, axis=1)
    cfm = confusion_matrix(validation_labels, predictions_labels)
    print (cfm)
    print("accuracy: ",accuracy_score(validation_labels, predictions_labels))


    # summarize feature map shapes
    for i in range(len(model.layers)):
        layer = model.layers[i]
        # check for convolutional layer
        if 'conv' not in layer.name:
            continue
        # summarize output shape
        print(i, layer.name, layer.output.shape)

    # redefine model to output right after the first hidden layer
    model = Model(inputs=model.inputs, outputs=model.layers[3].output)
    model.summary()


    feature_maps = model.predict(segment_validation_images)
    # print(feature_maps)
    feature_maps=feature_maps[0]
    # plot all 64 maps in an 8x8 squares
    ix = 1
    for _ in range(3):
        for _ in range(1):
            # specify subplot and turn of axis
            ax = pyplot.subplot(3, 1, ix)
            ax.set_xticks([])
            ax.set_yticks([])
            # plot filter channel in grayscale
            pyplot.imshow(feature_maps[0, :, :, ix - 1])
            ix += 1
    # show the figure
    pyplot.show()

    return accuracy_score(validation_labels, predictions_labels),validation_labels,predictions_labels




K.set_image_dim_ordering('th')

segmentName='FullFAce'
sizeH=32
sizeV=32
sizeD=9

# Load training images and labels that are stored in numpy array

segment_training_set = numpy.load('numpy_training_datasets/{0}_images_{1}x{2}x{3}.npy'.format(segmentName,sizeH, sizeV,sizeD))
segment_traininglabels = numpy.load('numpy_training_datasets/{0}_labels_{1}x{2}x{3}.npy'.format(segmentName,sizeH, sizeV,sizeD))




'''
#-----------------------------------------------------------------------------------------------------------------
#LOOCV
loo = LeaveOneOut()
loo.get_n_splits(segment_training_set)
tot=0
count=0
accs=[]
accs2=[]

val_labels=[]
pred_labels=[]
for train_index, test_index in loo.split(segment_training_set):

    # print(segment_traininglabels[train_index])
    # print(segment_traininglabels[test_index])
    print(test_index)
    val_acc, val_label, pred_label = evaluate(segment_training_set[train_index], segment_training_set[test_index],
                                              segment_traininglabels[train_index], segment_traininglabels[test_index],
                                              test_index)
    tot += val_acc
    val_labels.extend(val_label)
    pred_labels.extend(pred_label)
    accs.append(val_acc)
    accs2.append(segment_traininglabels[test_index])
    count+=1
    print("------------------------------------------------------------------------")
    print("validation acc:",val_acc)
    print("------------------------------------------------------------------------")
print(tot/count)
cfm = confusion_matrix(val_labels, pred_labels)
tp_and_fn = sum(cfm.sum(1))
tp_and_fp = sum(cfm.sum(0))
tp = sum(cfm.diagonal())
print("cfm: \n",cfm)
print("tp_and_fn: ",tp_and_fn)
print("tp_and_fp: ",tp_and_fp)
print("tp: ",tp)

precision = tp / tp_and_fp
recall = tp / tp_and_fn
print("precision: ",precision)
print("recall: ",recall)
print("F1-score: ",f1_score(val_labels,pred_labels,average=None))
print("F1-score: ",f1_score(val_labels,pred_labels,average="macro"))
print("F1-score: ",f1_score(val_labels,pred_labels,average="weighted"))
print("F1-score: ",f1_score(val_labels,pred_labels,average="samples"))
'''

#-----------------------------------------------------------------------------------------------------------------
#Test train split


# Spliting the dataset into training and validation sets
segment_train_images, segment_validation_images, segment_train_labels, segment_validation_labels = train_test_split(segment_training_set,
                                                                                            segment_traininglabels,
                                                                                            test_size=0.2,random_state=42)

# Save validation set in a numpy array
# numpy.save('numpy_validation_datasets/{0}_images_{1}x{2}.npy'.format(segmentName,sizeH, sizeV), segment_validation_images)
# numpy.save('numpy_validation_datasets/{0}_images_{1}x{2}.npy'.format(segmentName,sizeH, sizeV), segment_validation_labels)

# Loading Load validation set from numpy array
#
# eimg = numpy.load('numpy_validation_datasets/{0}_images_{1}x{2}.npy'.format(segmentName,sizeH, sizeV))
# labels = numpy.load('numpy_validation_datasets/{0}_images_{1}x{2}.npy'.format(segmentName,sizeH, sizeV))

evaluate(segment_train_images, segment_validation_images,segment_train_labels, segment_validation_labels ,0)
'''
#-----------------------------------------------------------------------------------------------------------------------------
#k-fold(10)

kf = KFold(n_splits=10, random_state=42,shuffle=True)
# kf.get_n_splits(segment_training_set)
tot=0
count=0
accs=[]
accs2=[]

val_labels=[]
pred_labels=[]
for train_index, test_index in kf.split(segment_training_set):

    # print(segment_traininglabels[train_index])
    # print(segment_traininglabels[test_index])
    print(test_index)
    val_acc, val_label, pred_label = evaluate(segment_training_set[train_index], segment_training_set[test_index],
                                              segment_traininglabels[train_index], segment_traininglabels[test_index],
                                              test_index)
    tot += val_acc
    val_labels.extend(val_label)
    pred_labels.extend(pred_label)
    accs.append(val_acc)
    accs2.append(segment_traininglabels[test_index])
    count+=1
    print("------------------------------------------------------------------------")
    print("validation acc:",val_acc)
    print("------------------------------------------------------------------------")
print("accuracy: ",accuracy_score(val_labels, pred_labels))
cfm = confusion_matrix(val_labels, pred_labels)
# tp_and_fn = sum(cfm.sum(1))
# tp_and_fp = sum(cfm.sum(0))
# tp = sum(cfm.diagonal())
print("cfm: \n",cfm)
# print("tp_and_fn: ",tp_and_fn)
# print("tp_and_fp: ",tp_and_fp)
# print("tp: ",tp)
#
# precision = tp / tp_and_fp
# recall = tp / tp_and_fn
# print("precision: ",precision)
# print("recall: ",recall)
# print("F1-score: ",f1_score(val_labels,pred_labels,average="macro"))
print("F1-score: ",f1_score(val_labels,pred_labels,average="weighted"))

#---------------------------------------------------------------------------------------------------
# write to results

results=open("../TempResults.txt",'a')
results.write("---------------------------\n")
full_path = os.path.realpath(__file__)
results.write(str(os.path.dirname(full_path))+" LOOCV\n")
results.write("---------------------------\n")
results.write("accuracy: "+str(accuracy_score(val_labels, pred_labels))+"\n")
results.write("F1-score: "+str(f1_score(val_labels,pred_labels,average="weighted"))+"\n")
'''