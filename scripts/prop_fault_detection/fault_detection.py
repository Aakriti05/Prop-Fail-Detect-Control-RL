import os, sys, argparse, re
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, CuDNNLSTM, Dense
from tensorflow.keras.regularizers import L1L2
from tensorflow.keras.optimizers import Adam, SGD
import numpy as np


class fault_detection():
	def __init__(self):
		# Load weights of 4-3 fault detection model
		wf43 = os.path.abspath("/home/rohit/Documents/raisim_stuff/"+ \
			"prop_loss_final/fault_detection_weights/4-3_fault_detection_weights/weights.08-0.99.hdf5")
		input_tensor = Input(shape = (100,18))
		outputs = self._init_model_43(input_tensor)
		model = Model(inputs=input_tensor, outputs=outputs) 
		# Dummy optimizer, loss and metrics
		model.compile(optimizer = SGD(), loss='categorical_crossentropy', metrics=['accuracy'])
		model.load_weights(wf43)
		print('Weights loaded for 4-3 fault detection network.')
		self.model43 = model
		
		# Load weights of 3-2 fault detection model
		wf32 = os.path.abspath("/home/rohit/Documents/raisim_stuff/"+ \
			"prop_loss_final/fault_detection_weights/3-2_fault_detection_weights/weights.416-0.92.hdf5")
		input_tensor = Input(shape = (200,18))
		outputs = self._init_model_32(input_tensor)
		model = Model(inputs=input_tensor, outputs=outputs)
		# Dummy optimizer, loss and metrics
		model.compile(optimizer = Adam(), loss='categorical_crossentropy', metrics=['accuracy'])
		model.load_weights(wf32)
		print('Weights loaded for 3-2 fault detection network.')
		self.model32 = model

	def _init_model_43(self, inputs):
		# 4 to 3 network
		x = inputs
		x = CuDNNLSTM(96, return_sequences = True)(x)
		x = CuDNNLSTM(64, return_sequences = False)(x)
		x = Dense(5, activation = 'sigmoid')(x)
		return x
		
	def _init_model_32(self, inputs):
		# 3 to 2 network
		x = inputs
		x = CuDNNLSTM(96, kernel_regularizer=L1L2(l1=0.01, l2=0.01), \
				recurrent_regularizer=L1L2(l1=0.01, l2=0.01), return_sequences = True)(x)
		x = CuDNNLSTM(32, kernel_regularizer=L1L2(l1=0.01, l2=0.01), \
				recurrent_regularizer=L1L2(l1=0.01, l2=0.01), return_sequences = False)(x)
		x = Dense(2, activation='softmax')(x)
		return x

	def predict_43(self, inputs):
		return self.model43.predict(inputs, batch_size=1)
	
	def predict_32(self, inputs):
		return self.model32.predict(inputs, batch_size=1)

if __name__ == "__main__":
	pass