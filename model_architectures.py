import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Convolution2D, Activation, Flatten, MaxPooling2D,Input,Dropout,GlobalAveragePooling2D
from tensorflow.keras.layers import BatchNormalization
from tensorflow.python.framework import ops
from memristor_utils import *

batch_norm_eps=1e-4
batch_norm_momentum=0.9

def get_model(dataset, batch_size, group_idx=None, is_regularized=True, log_dir_full_path=None):
	if dataset=='MNIST':
		model=Sequential()
		model.add(memristor_dense(n_in=784, n_out=25, group_idx=group_idx, is_regularized=is_regularized, log_dir_full_path=log_dir_full_path, input_shape=[784]))
        # We will try to introduce non-linearities using dense layers.
		model.add(Activation('sigmoid'))
		model.add(memristor_dense(n_in=int(model.output.get_shape()[1]),n_out=10, group_idx=group_idx, log_dir_full_path=log_dir_full_path, is_regularized=is_regularized))
		model.add(Activation('softmax'))
	else:
		raise("Dataset {} is not recognised!".format(dataset))
	return model
