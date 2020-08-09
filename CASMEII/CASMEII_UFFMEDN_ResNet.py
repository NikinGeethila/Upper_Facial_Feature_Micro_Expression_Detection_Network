import numpy
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score,f1_score
from keras.models import Sequential, Model
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.layers.convolutional import Convolution3D, MaxPooling3D, ZeroPadding3D
from keras.layers import LeakyReLU ,PReLU,Input,Add,BatchNormalization
from keras.callbacks import ModelCheckpoint,EarlyStopping,ReduceLROnPlateau,Callback
from sklearn.model_selection import train_test_split,LeaveOneOut,KFold
from keras import backend as K
from keras.optimizers import Adam,SGD
import os

class myCallback(Callback):
    def on_epoch_end(self, epoch, logs={}):
        if(logs.get('val_acc') >= 1.0):
            print("\nReached %2.2f%% accuracy, so stopping training!!" %(1.0*100))
            self.model.stop_training = True

def evaluate(segment_train_images, segment_validation_images, segment_train_labels, segment_validation_labels,test_index ):


    input = Input(shape=(1, sizeH, sizeV, sizeD))
    conv1 = Convolution3D(32, (20, 20, 9), strides=(10, 10, 3), padding='Same')(input)
    bn1=BatchNormalization()(conv1)
    ract_1 = PReLU()(bn1)
    conv2 = Convolution3D(32, (3, 3, 3), strides=1, padding='Same')(ract_1)
    bn2=BatchNormalization()(conv2)
    ract_2 = PReLU()(bn2)
    conv3 = Convolution3D(32, (3, 3, 3), strides=1, padding='Same')(ract_2)
    bn3=BatchNormalization()(conv3)
    add1= Add() ([bn3,bn1])
    conv4 = Convolution3D(32, (3, 3, 3), strides=1, padding='Same')(add1)
    bn4=BatchNormalization()(conv4)
    ract_3 = PReLU()(bn4)
    conv5 = Convolution3D(32, (3, 3, 3), strides=1, padding='Same')(ract_3)
    bn5=BatchNormalization()(conv5)
    add2 = Add()([bn5,add1])
    flatten_1 = Flatten()(add2)
    dense_1 = Dense(5, init='normal')(flatten_1)
    drop1 = Dropout(0.5)(dense_1)
    activation = Activation('softmax')(drop1)
    opt = SGD(lr=0.01)
    model = Model(inputs=[input], outputs=activation)
    model.compile(loss='categorical_crossentropy', optimizer=opt, metrics=['accuracy'])

    model.summary()

    filepath="weights_CASMEII/weights-improvement"+str(test_index)+"-{epoch:02d}-{val_acc:.2f}.hdf5"
    checkpoint = ModelCheckpoint(filepath, monitor='val_acc', verbose=1, save_best_only=True, mode='max')
    EarlyStop = EarlyStopping(monitor='val_acc', min_delta=0, patience=50, restore_best_weights=True, verbose=1, mode='max')
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

    return accuracy_score(validation_labels, predictions_labels),validation_labels,predictions_labels




#-----------------------------------------------------------------------------------------------------------------
#LOOCV
def loocv():
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
        count += 1
        print("------------------------------------------------------------------------")
        print("validation acc:", val_acc)
        print("------------------------------------------------------------------------")
    print("accuracy: ", accuracy_score(val_labels, pred_labels))
    cfm = confusion_matrix(val_labels, pred_labels)
    # tp_and_fn = sum(cfm.sum(1))
    # tp_and_fp = sum(cfm.sum(0))
    # tp = sum(cfm.diagonal())
    print("cfm: \n", cfm)
    # print("tp_and_fn: ",tp_and_fn)
    # print("tp_and_fp: ",tp_and_fp)
    # print("tp: ",tp)
    #
    # precision = tp / tp_and_fp
    # recall = tp / tp_and_fn
    # print("precision: ",precision)
    # print("recall: ",recall)
    # print("F1-score: ",f1_score(val_labels,pred_labels,average="macro"))
    print("F1-score: ", f1_score(val_labels, pred_labels, average="weighted"))
    return val_labels, pred_labels

#-----------------------------------------------------------------------------------------------------------------
#Test train split

def split():
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

    _,val_labels, pred_labels=evaluate(segment_train_images, segment_validation_images,segment_train_labels, segment_validation_labels ,0)
    return val_labels, pred_labels
#-----------------------------------------------------------------------------------------------------------------------------
#k-fold(10)

def kfold():
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
    return val_labels, pred_labels




####################################
#edit params
K.set_image_dim_ordering('th')

segmentName='UpperFace'
sizeH=64
sizeV=64
sizeD=30
testtype="kfold"
####################################

# Load training images and labels that are stored in numpy array

segment_training_set = numpy.load('numpy_training_datasets/{0}_images_{1}x{2}x{3}.npy'.format(segmentName,sizeH, sizeV,sizeD))
segment_traininglabels = numpy.load('numpy_training_datasets/{0}_labels_{1}x{2}x{3}.npy'.format(segmentName,sizeH, sizeV,sizeD))



if testtype=="kfold":
    val_labels, pred_labels=kfold()
elif testtype=="loocv":
    val_labels, pred_labels=loocv()
elif testtype=="split":
    val_labels, pred_labels=split()
else:
    print("error")


#---------------------------------------------------------------------------------------------------
# write to results

results=open("../TempResults.txt",'a')
results.write("---------------------------\n")
full_path = os.path.realpath(__file__)
results.write(str(os.path.dirname(full_path))+" {0}_{1}_{2}x{3}x{4}\n".format(testtype,segmentName,sizeH, sizeV,sizeD))
results.write("---------------------------\n")
results.write("accuracy: "+str(accuracy_score(val_labels, pred_labels))+"\n")
results.write("F1-score: "+str(f1_score(val_labels,pred_labels,average="weighted"))+"\n")