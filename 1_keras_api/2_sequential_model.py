"""
Running the tiny sample model is faster on the CPU: Batch loading 
from RAM to GPU is slower at the start of each operation. Forward/backward 
computations are very quick in tiny networks so it's rational to use CPU.
You can also try using model.fit_generator instead of plain fit, so that 
CPU thread which loads minibatches works in parallel. 

At the time there is no way I am aware of to preload the whole dataset 
on GPU with Keras.
"""

# Hiding a GPU
import os
os.environ["CUDA_VISIBLE_DEVICES"] = '-1'

# Importing tf and numpy
import tensorflow as tf
from tensorflow import keras
import numpy as np

# Defining and running the model
model = keras.Sequential([keras.layers.Dense(units=1, input_shape=[1])])
model.compile(optimizer='sgd', loss='mean_squared_error')

xs = np.array([-1.0, 0.0, 1.0, 2.0, 3.0, 4.0], dtype=float)
ys = np.array([-3.0, -1.0, 1.0, 3.0, 5.0, 7.0], dtype=float)

model.fit(xs, ys, epochs=500)