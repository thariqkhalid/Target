#author: Thariq Khalid

from __future__ import print_function

import keras
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.optimizers import RMSprop

from scipy.misc import imread
import matplotlib.pyplot as plt
import cv2
import scipy.ndimage as ndimage
from scipy.misc import imresize
from scipy.misc import imrotate
from skimage import img_as_ubyte
from skimage.filters import threshold_otsu
import numpy as np
from collections import defaultdict

batch_size = 128
num_classes = 10
epochs = 20

def blurnDenoise(bw,sz=(5,5),sigma=2,thresh=0.3):
    I = bw.astype(float)
    fg = cv2.GaussianBlur(I,sz,sigma)
    m1 = fg > thresh
    labels,num_items = ndimage.label(m1,np.ones((3,3),'int'))
    pixel_sum = ndimage.sum(m1, labels, range(num_items+1)).astype(np.int)
    mask_size = pixel_sum < 260
    remove_pixel = mask_size[labels]
    m1[remove_pixel] = 0
    return m1

def get_digit_segments(patch, model):
    im_patch = patch.copy()
    patch = blurnDenoise(patch)
    label_im, nb_label = ndimage.label(patch,np.ones((3,3),'int'))
    
    numbers = []
    digits_im = []
    offsets = []
    for i in range(1,nb_label+1):
        slice_x, slice_y = ndimage.find_objects(label_im==i)[0]
        digit = label_im[slice_x, slice_y]
        offsets.append(int(slice_y.start))
        digit = np.pad(digit,(10,10),'constant',constant_values = (0,0))
        digit = imresize(digit,(28,28),interp='nearest').astype('float64')/255.
        digit = imrotate(digit,-25)
        digits_im.append(digit)
        digit = digit.flatten()
        numbers.append(digit)
    sorted_numbers = [x for (y,x) in sorted(zip(offsets,numbers))]
    sorted_digits_im = [x for (y,x) in sorted(zip(offsets,digits_im))]
    sorted_numbers = np.asarray(sorted_numbers)
    predictions = model.predict(sorted_numbers)
    ax = plt.subplot(121)
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
    plt.suptitle("Recognized values")
    pred_number = np.argmax(predictions,axis=1).tolist()
    final_number = [str(i) for i in pred_number]
    disp_number = "".join(final_number)
    plt.title(disp_number)
    plt.imshow(patch,'gray')
    plt.show()
    plt.figure(figsize=(20, 8))
    for i in range(nb_label):
        ax = plt.subplot(1,nb_label,i+1)
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)
        plt.suptitle("Segmented digits")
        plt.title(np.argmax(predictions[i]))
        plt.imshow(sorted_digits_im[i])
        plt.gray()
    plt.show()

def ocr_bill(model):
	bill = imread("878355.TIF")
	#get the portion of the bill 878355.tif that has the handwritten digits
	#to get the exact portion of the bill that has handwritten digits is yet another recognition problem in ML compared to OCR
	clip_handwriting = bill[960:2450,1650:1850].astype('float32')
	bw = clip_handwriting < threshold_otsu(clip_handwriting)

	I, J = np.nonzero(bw)
	rows = defaultdict(int)
	for i in I:
	    rows[i] += 1
	row_values = rows.values()
	patches = []
	row_ids = rows.keys()

	start_id = row_ids[0]
	for v, rv in enumerate(row_values):
	    patch = []
	    if rv > 120:
	        if row_ids[v] - start_id > 50:
	            end_id = row_ids[v]
	            patch = bw[start_id:end_id,]
	            get_digit_segments(patch, model)
	            start_id = end_id
	        else:
	            start_id = row_ids[v]

def train_model_mnist():
	# the data, shuffled and split between train and test sets
	(x_train, y_train), (x_test, y_test) = mnist.load_data()

	x_train = x_train.reshape(60000, 784)
	x_test = x_test.reshape(10000, 784)
	x_train = x_train.astype('float32')
	x_test = x_test.astype('float32')
	x_train /= 255
	x_test /= 255
	print(x_train.shape[0], 'train samples')
	print(x_test.shape[0], 'test samples')

	# convert class vectors to binary class matrices
	y_train = keras.utils.to_categorical(y_train, num_classes)
	y_test = keras.utils.to_categorical(y_test, num_classes)

	model = Sequential()
	model.add(Dense(512, activation='relu', input_shape=(784,)))
	model.add(Dropout(0.2))
	model.add(Dense(512, activation='relu'))
	model.add(Dropout(0.2))
	model.add(Dense(10, activation='softmax'))

	model.summary()

	model.compile(loss='categorical_crossentropy',
	              optimizer=RMSprop(),
	              metrics=['accuracy'])

	history = model.fit(x_train, y_train,
	                    batch_size=batch_size,
	                    epochs=epochs,
	                    verbose=1,
	                    validation_data=(x_test, y_test))
	score = model.evaluate(x_test, y_test, verbose=0)
	print('Test loss:', score[0])
	print('Test accuracy:', score[1])
	return model

if __name__ == '__main__':
	model = train_model_mnist()
	ocr_bill(model)
