import keras
import tensorflow as tf
from keras.layers import Lambda
import numpy as np
import keras.backend as K


keras.backend.set_image_data_format("channels_last")

print(K.image_data_format())

def SubpixelConv2D(input_shape, scale):
	## Кастомный слой для нейронной сети.
    def subpixel_shape(input_shape):
        dims = (input_shape[0],input_shape[1] * scale,
                                        input_shape[2] * scale,
                                        input_shape[3] // (scale * scale) )
        output_shape = tuple(dims)
        return output_shape

    return Lambda(lambda x: tf.depth_to_space(x, scale), output_shape=subpixel_shape)