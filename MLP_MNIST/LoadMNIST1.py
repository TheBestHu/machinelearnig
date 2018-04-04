import os
import struct
import numpy as np
from numpy import zeros
from numpy import array
import scipy as sp
def load_mnist(dataset="training", digits=np.arange(10), path=".", size = 60000):
    if dataset == "training":
        fname_img = os.path.join(path, 'train-images.idx3-ubyte')
        fname_lbl = os.path.join(path, 'train-labels.idx1-ubyte')
    elif dataset == "testing":
        fname_img = os.path.join(path, 't10k-images.idx3-ubyte')
        fname_lbl = os.path.join(path, 't10k-labels.idx1-ubyte')
    
    else:
        raise ValueError("dataset must be 'testing' or 'training'")
    # load things in numpy arrays
    with open(fname_lbl,'rb') as flbl:
        magic,num = struct.unpack(">II",flbl.read(8))
        lbl = np.fromfile(flbl,dtype = np.int8)

    with open(fname_img,'rb') as fimg:
        magic,num,rows,cols = struct.unpack(">IIII",fimg.read(16))
        img = np.fromfile(fimg,dtype = np.uint8).reshape(len(lbl),rows,cols)

    ind = [ k for k in range(size) if lbl[k] in digits ]
    N = size #int(len(ind) * size/100.)
    images = zeros((N, rows, cols), dtype='uint8')
    labels = zeros((N, 1), dtype='int8')
    for i in range(N): #int(len(ind) * size/100.)):
        images[i] = array(img[ ind[i]*rows*cols : (ind[i]+1)*rows*cols ])\
            .reshape((rows, cols))
        labels[i] = lbl[ind[i]]
    labels = [label[0] for label in labels]
    return images, labels


load_mnist('training',np.arange(10),'.',60000)
