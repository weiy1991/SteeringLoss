####
#modified by Yuanwi 2017-12-19
####

import scipy.misc
import random
import numpy as np
import os
import h5py
import cv2


xs = []
ys = []

xs_ = []
ys_ = []

#points to the end of the last batch
train_batch_pointer = 0
val_batch_pointer = 0

#modified by Yuanwei 20171224
def read_csv(filename):
    with open(filename, 'r') as f:
        lines_all = [ln.strip().split(",")[:] for ln in f.readlines()]
        #del(lines_all[0]) # remove the head of the csv
        lines_all = map(lambda x: (x[5], np.float128(x[6])), lines_all)
        return lines_all

def getTrainingData(filename):
    lines_all = read_csv(filename)
    for ln in lines_all:
        if ln[0].find('center')  != -1:
            xs_.append("/home/weiy/dataset/udacity-output/"+ln[0])
            ys_.append(ln[1])

getTrainingData("/home/weiy/dataset/udacity-output/interpolated_train_shuffle.csv")
xs = xs_
ys = ys_
#print("xs_:",xs_)
#print("ys_[0]:",ys_[0])
#print("ys_[1]:",ys_[1])
#print("ys_[2]:",ys_[2])
#end by Yuaniwei 20171224

#get number of images
num_images = len(xs)

#shuffle list of images
c = list(zip(xs, ys))
random.seed(0)
random.shuffle(c)
xs, ys = zip(*c)

print("first:", ys[0])

train_xs = xs[:int(len(xs) * 0.8)]
train_ys = ys[:int(len(xs) * 0.8)]

val_xs = xs[-int(len(xs) * 0.2):]
val_ys = ys[-int(len(xs) * 0.2):]

num_train_images = len(train_xs)
num_val_images = len(val_xs)

def LoadTrainBatch(batch_size):
    global train_batch_pointer
    x_out = []
    y_out = []
    for i in range(0, batch_size):
        #modified by Yuanwei 20171224
        #x_out.append(scipy.misc.imresize(scipy.misc.imread(train_xs[(train_batch_pointer + i) % num_train_images])[-150:], [66, 200]) / 255.0)
        x_out.append(scipy.misc.imresize(scipy.misc.imread(train_xs[(train_batch_pointer + i) % num_train_images]), [66, 200]) / 255.0)
        #scipy.misc.imshow(x_out[i])
        #x_out.append(scipy.misc.imread(train_xs[(train_batch_pointer + i) % num_train_images]) / 255.0)
        #end by Yuanwei 20171224
        y_out.append([train_ys[(train_batch_pointer + i) % num_train_images]])
        #print(y_out[i])
    train_batch_pointer += batch_size
    return x_out, y_out

def LoadValBatch(batch_size):
    global val_batch_pointer
    x_out = []
    y_out = []
    for i in range(0, batch_size):
        #modified by Yuanwei 20171224
        #x_out.append(scipy.misc.imresize(scipy.misc.imread(val_xs[(val_batch_pointer + i) % num_val_images])[-150:], [66, 200]) / 255.0)
        x_out.append(scipy.misc.imresize(scipy.misc.imread(val_xs[(val_batch_pointer + i) % num_val_images]), [66, 200]) / 255.0)
        #x_out.append(scipy.misc.imread(val_xs[(val_batch_pointer + i) % num_val_images]) / 255.0)
        #end by Yuanwei 20171224
        y_out.append([val_ys[(val_batch_pointer + i) % num_val_images]])
    val_batch_pointer += batch_size
    return x_out, y_out

