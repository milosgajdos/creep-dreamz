import os
import numpy as np
import tensorflow as tf

from keras.preprocessing.image import load_img, img_to_array
from keras import backend as K

def preprocess_VGG16(img):
    """
    Preprocess image for use with VGG16

    Parameters
    ----------
    img : ndarray
        Raw image as a numpy array

    Returns
    -------
    img : ndarray
        Preprocessed image as a numpy array
    """
    from keras.applications.vgg16 import VGG16, preprocess_input
    return preprocess_input(img)

def preprocess_VGG19(img):
    """
    Preprocess image for use with VGG19

    Parameters
    ----------
    img : ndarray
        Raw image as a numpy array

    Returns
    -------
    img : ndarray
        Preprocessed image as a numpy array
    """
    from keras.applications.vgg19 import VGG19, preprocess_input
    return preprocess_input(img)

def preprocess_Resnet50(img):
    """
    Preprocess image for use with Resnet50

    Parameters
    ----------
    img : ndarray
        Raw image as a numpy array

    Returns
    -------
    img : ndarray
        Preprocessed image as a numpy array
    """
    from keras.applications.resnet50 import ResNet50, preprocess_input
    return preprocess_input(img)

def preprocess_Xception(img):
    """
    Preprocess image for use with Xception

    Parameters
    ----------
    img : ndarray
        Raw image as a numpy array

    Returns
    -------
    img : ndarray
        Preprocessed image as a numpy array
    """
    from keras.applications.xception import Xception, preprocess_input
    return preprocess_input(img)

def preprocess_InceptionV3(img):
    """
    Preprocess image for use with InceptionV3

    Parameters
    ----------
    img : ndarray
        Raw image as a numpy array

    Returns
    -------
    img : ndarray
        Preprocessed image as a numpy array
    """
    from keras.applications.inception_v3 import InceptionV3, preprocess_input
    return preprocess_input(img)

def preprocess(path, model_name):
    """
    Preprocesses an image

    Loads an image, resizes it to and preprocesses it
    for use with particular CNN model

    Parameters
    ----------
    path : str
        Path to image on the filesystem
    model : str
        Keras model name (InceptionV3, Xception, Resnet50, VGG19, VGG16)

    Returns
    -------
    img : ndarray
        Preprocessed image as a numpy array
    """
    img = load_img(path)
    img = img_to_array(img)
    img = np.expand_dims(img, axis=0)
    if model_name == 'VGG16':
        img = preprocess_VGG16(img)
    elif model_name == 'VGG19':
        img = preprocess_VGG10(img)
    elif model_name == 'Resnet50':
        img = preprocess_Resnet50(img)
    elif model_name == 'Xception':
        img = preprocess_Xception(img)
    elif model_name == 'InceptionV3':
        img = preprocess_InceptionV3(img)
    else:
        raise ValueError('Unsupported model: {}'.format(model_name))
    return img

def resize(img, size):
    """
    Resizes an image

    The size of the returned image is supplied as tuple

    Parameters
    ----------
    img: ndarray
        Image as numpy array
    size: array_like
        New image size

    Returns
    -------
    img : tf.Tensor
        Reized image tensor of shape `[new_height, new_width, channels]`
    """
    img = tf.convert_to_tensor(img, dtype=tf.float32)
    img = tf.image.resize_images(img, size)
    return img

def deprocess(img):
    """
    Converts a preprocessed image into original image

    Parameters
    ----------
    img : tf.Tensor
        Image tensor

    Returns
    -------
    img : tf.Tensor
        Image tensor
    """
    if K.image_data_format() == 'channels_first':
        _, _, w, h = img.get_shape().as_list()
        img = tf.reshape(img, [3, w, h])
        img = tf.transpose(img, [1, 2, 0])
    else:
        _, w, h, _ = img.get_shape().as_list()
        img = tf.reshape(img, [w, h, 3])
    img = tf.divide(img, 2.)
    img = tf.add(img, 0.5)
    img = tf.multiply(img, 255.)
    img = tf.cast(tf.clip_by_value(img, 0, 255), tf.uint8)
    return img

def encode(img, fname):
    """
    Encodes an image into chosen image format

    Image format is inferred from image extension

    Parameters
    ----------
    img : tf.Tensor
        Image tensor
    fname : str
        File name

    Returns
    -------
    img : tf.Tensor
        Image tensor encoded in requested image format
    """
    _, ext = os.path.splitext(fname)
    if ext == '.png':
        return tf.image.encode_png(img)
    elif ext == 'bmp':
        return tf.image.encode_bmp(img)
    elif ext == '.jpeg' or ext == '.jpg':
        return tf.image.encode_jpeg(img)
    else:
        raise ValueError('Unsupported extension: {}'.format(ext))

def save(img, fname):
    """
    Saves image on filesystem

    Parameters
    ----------
    img : tf.Tensor
        Image tensor
    fname : str
        Filesystem path
    """
    out_img = deprocess(img)
    out_img = encode(out_img, fname)
    fname = tf.constant(fname)
    K.get_session().run(tf.write_file(fname, out_img))
