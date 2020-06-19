import numpy
import os

segmentName='UpperFace'
sizeH=32
sizeV=32

segment_traininglabels = numpy.load('numpy_training_datasets/{0}_labels_{1}x{2}.npy'.format(segmentName,sizeH, sizeV))
cat=[0]*8
for item in segment_traininglabels:
    for c in range(len(cat)):
        if item[c]==1:
            cat[c]+=1

print(cat)


angerpath = '../../../Datasets/SAMM_categorical/Anger/'
sadnesspath = '../../../Datasets/SAMM_categorical/Sadness/'
happinesspath = '../../../Datasets/SAMM_categorical/Happiness/'
disgustpath = '../../../Datasets/SAMM_categorical/Disgust/'
fearpath = '../../../Datasets/SAMM_categorical/Fear/'
surprisepath = '../../../Datasets/SAMM_categorical/Surprise/'
contemptpath = '../../../Datasets/SAMM_categorical/Contempt/'
otherpath = '../../../Datasets/SAMM_categorical/Other/'
paths=[angerpath, sadnesspath, happinesspath,disgustpath,fearpath,surprisepath,contemptpath,otherpath]
cat=[0]*8
dir=0
for typepath in (paths):
    directorylisting = os.listdir(typepath)
    for video in directorylisting:
        cat[dir]+=1
    dir+=1
print(cat)
print(sum(cat))