import tensorflow as tf
import theano
theano.config.device = ‘gpu0’

theano.config.floatX = ‘float32’
print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))