import numpy

segmentName='UpperFace'
sizeH=32
sizeV=32

segment_traininglabels = numpy.load('numpy_training_datasets/{0}_labels_{1}x{2}.npy'.format(segmentName,sizeH, sizeV))
n=0
p=0
s=0
for item in segment_traininglabels:
    if item[0]==1:
        n+=1
    if item[1]==1:
        p+=1
    if item[2]==1:
        s+=1
print(n,p,s)