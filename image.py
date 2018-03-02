import scipy
import numpy as np

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
    img : ndarray
        Reized image as numpy array
    """
    img = np.copy(img)
    if K.image_data_format() == 'channels_first':
        factors = (1, 1,
                   float(size[0]) / img.shape[2],
                   float(size[1]) / img.shape[3])
    else:
        factors = (1,
                   float(size[0]) / img.shape[1],
                   float(size[1]) / img.shape[2],
                   1)
    return scipy.ndimage.zoom(img, factors, order=1)

def deprocess(img):
    """
    Converts a preprocessed image into original image

    Parameters
    ----------
    img : ndarray
        Input image

    Returns
    -------
    img : ndarray
        Deprocessed image
    """
    if K.image_data_format() == 'channels_first':
        img = img.reshape((3, img.shape[2], img.shape[3]))
        img = img.transpose((1, 2, 0))
    else:
        img = img.reshape((img.shape[1], img.shape[2], 3))
    img /= 2.
    img += 0.5
    img *= 255.
    img = np.clip(img, 0, 255).astype('uint8')
    return img

def save(img, fname):
    """
    Saves image on filesystem

    Parameters
    ----------
    img : ndarray
        Image tensor
    fname : str
        Filesystem path
    """
    pil_img = deprocess(np.copy(img))
    scipy.misc.imsave(fname, pil_img)
