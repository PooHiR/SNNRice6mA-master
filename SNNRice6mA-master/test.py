# from tensorflow.keras import  losses
import  tensorflow as tf
import numpy as np

y_true = tf.convert_to_tensor([2.3, 2.5, 1., 1.,10,6], np.float32)
y_pred = tf.convert_to_tensor([1., 1., 1., 2.4,10,3.0], np.float32)
bce = tf.keras.losses.BinaryCrossentropy()
loss = bce(y_true,y_pred)
print(y_true)
print(y_pred)
print('Loss: ', loss.numpy())