import  tensorflow as tf
import numpy



# a = 1.2
# aa = tf.constant(1.2)
# print(type(a),type(aa),tf.is_tensor(aa),tf.is_tensor(a))
# print(tf.shape(aa))
# x = tf.constant([1,2.,3.3])
# y = tf.constant(1.2)
# print(type(x),x.numpy().shape,x.shape,type(y),tf.is_tensor(y),x,y)

# a = tf.constant([[1,2],[3,4]])
# print(a,a.shape,a.numpy)
# b = tf.constant([[[1,2],[3,4]],[[5,6],[7,8]]])
# print(b)

# a = tf.constant('Hello Deep learning')
# print(tf.strings.lower(a))
# a = tf.constant(True)
# print(a)
# a = tf.constant([True,False,True,False])
# print(a)
# a = tf.constant(True)
# print(a == True)

# a = numpy.pi
# print(a,tf.constant(a,dtype=tf.float32),tf.constant(a,dtype=tf.float64))
# b = tf.constant(a)
# c = tf.convert_to_tensor([1,2,3])
# print(c)
# a = tf.zeros([3,3])
# b = tf.ones([3,3])
# print(tf.zeros_like(b),tf.ones_like(a))


from tensorflow import keras
from tensorflow.keras import layers,Sequential,losses,optimizers,datasets

# model = keras.Sequential([
#     layers.Dense(256,activation = 'relu'),
#     layers.Dense(256,activation = 'relu'),
#     layers.Dense(256,activation = 'relu'),
#     layers.Dense(10)
# ])
# model.build(input_shape=(4,784))
# model.summary()
# gpus = tf.config.experimental.list_physical_devices("GPU")
# print(gpus)


# x = tf.random.normal([2,5,5,3])
# w = tf.random.normal([3,3,3,4])
# out = tf.nn.conv2d(x,w,strides=1,padding=[[0,0],[0,0],[0,0],[0,0]])
# print(out.shape)
#
# x = tf.random.normal([2,5,5,3])
# w = tf.random.normal([3,3,3,4])
# out = tf.nn.conv2d(x,w,strides=1,padding=[[0,0],[1,1],[1,1],[0,0]])
# print(out.shape)
#
# x = tf.random.normal([2,5,5,3])
# w = tf.random.normal([3,3,3,4])
# out = tf.nn.conv2d(x,w,strides=1,padding='SAME')
# print(out.shape)

# x = tf.random.normal([2,5,5,3])
# w = tf.random.normal([3,3,3,4])
# out = tf.nn.conv2d(x,w,strides=1,padding='SAME')
# print(out.shape)

x = tf.random.normal([2,5,5,3])
layer = layers.Conv2D(5,kernel_size = 3,strides = 1,padding = 'SAME')
print(x,layer(x))








