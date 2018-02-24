import os
import time
import argparse

import numpy as np
import tensorflow as tf

from keras.preprocessing.image import load_img, img_to_array
from keras.applications import inception_v3
from keras import backend as K

def preprocess_image(image_path):
    """
    Preprocess image

    Loads an image from image_path, resizes it to appropriate dimensions
    and preprocesses it for use with InceptionV3 model

    Parameters
    ----------
    image_path : str
        Path to image on the filesystem.

    Returns
    -------
    img : ndarray
        Preprocessed image
    """
    img = load_img(image_path)
    img = img_to_array(img)
    img = np.expand_dims(img, axis=0)
    img = inception_v3.preprocess_input(img)
    return img

def resize_img(img, size):
    """
    Resize image

    Parameters
    ----------
    img: ndarray
        Image as numpy array.
    size: sequence
        Requested resized image dimensions.

    Returns
    -------
    img : tf.Tensor
        Reized image tensor.
    """
    img = tf.convert_to_tensor(img, dtype=np.float32)
    img = tf.image.resize_images(img, size)
    return img

def deprocess_img(img):
    """
    Convert an image tensor into a numpy image

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

def encode_img(img, fname):
    """
    Encodes image to chosen image format

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

def save_img(img, fname):
    """
    Saves image on filesystem

    Parameters
    ----------
    img : tf.Tensor
        Image tensor
    fname : str
        Filesystem path
    """
    out_img = deprocess_img(img)
    out_img = encode_img(out_img, fname)
    fname = tf.constant(fname)
    K.get_session().run(tf.write_file(fname, out_img))

def build_loss(model, config):
    """
    Builds loss function for gradient ascent

    Parameters
    ----------
    model : keras.Model
        Keras DNN model used for creep dreaming
    config : dict
        Creep dream layer config dictionary

    Returns
    -------
    loss : keras.tensor.Op
        Loss function operation
    """
    # Get the symbolic outputs of each "key" layer (we gave them unique names).
    layer_dict = dict([(layer.name, layer) for layer in model.layers])
    # Define the loss.
    loss = K.variable(0.)
    for layer_name in config['features']:
        # Add the L2 norm of the features of a layer to the loss.
        assert layer_name in layer_dict.keys(), 'Layer ' + layer_name + ' not found in model.'
        # feature map weight
        w = config['features'][layer_name]
        x = layer_dict[layer_name].output
        # We avoid border artifacts by only involving non-border pixels in the loss.
        scaling = K.prod(K.cast(K.shape(x), 'float32'))
        if K.image_data_format() == 'channels_first':
            loss += w * K.sum(K.square(x[:, :, 2: -2, 2: -2])) / scaling
        else:
            loss += w * K.sum(K.square(x[:, 2: -2, 2: -2, :])) / scaling
    return loss

def build_gradients(model, loss):
    """
    Builds gradient tensor

    Parameters
    ----------
    model : keras.Model
        Neural network model used for creap dreaming
    loss : tf.Tensor
        Gradient ascent loss

    Returns
    -------
    grads : tf.Tensor
        Gradients tensor
    """
    # dream is created by flowing an input through model
    dream = model.input
    # Compute the gradients of the dream wrt the loss.
    grads = K.gradients(loss, dream)[0]
    # Normalize gradients.
    grads /= K.maximum(K.mean(K.abs(grads)), K.epsilon())
    return grads

def dream_shapes(img, octave, octave_scale):
    """
    Reads an image and generates a list of shapes based
    on supplied octave and octave scale parameters.

    Parameters
    ----------
    img : ndarray
        Input image
    octave : int
        Number of scales at which to run gradient ascent
    octavescale : float
        Size ratio between shapes

    Returns
    -------
    shapes : sequence
        Sequence of shape sorted in ascending order.
    """
    if K.image_data_format() == 'channels_first':
        orig_shape = img.shape[2:]
    else:
        orig_shape = img.shape[1:3]
    shapes = [orig_shape]
    for i in range(1, octave):
        shape = tuple([int(dim / (octave_scale ** i)) for dim in orig_shape])
        shapes.append(shape)
    shapes = shapes[::-1]
    return shapes

def gradient_ascent(x, loss_fn, iterations, step, max_loss=None):
    """
    Run gradient ascent

    Parameters
    ----------
    x : tf.Tensor
        Input data tensor
    loss_fn : function
        Function that evaluates and fetches loss and gradients
    iterations : int
        Number of gradient ascent iterations
    step : int
        Gradient ascent step size; this scales up the gradient size
    max_loss : float
        Maximum loss threshold

    Returns
    -------
    x : tf.Tensor
        Output data tensor after gradient ascent
    """
    for i in range(iterations):
        start = time.time()
        # evaluate loss and gradient for the supplied input data
        loss_value, grad_values = loss_fn([x.eval(session=K.get_session())])
        if max_loss is not None and loss_value > max_loss:
            break
        print('..Loss value at', i, ':', loss_value)
        end = time.time()
        print('gradient_ascent: ', end - start)
        # amplify gradient by step
        x = tf.add(x, step * grad_values)
    return x

def dream(img, model, iterations, step, max_loss, shapes, config):
    """
    Creep Dream:

    Runs gradient ascent. First it upscales provided image to particular
    scale then it reinjects the detail that was lost at upscaling time back.

    Parameters
    ----------
    img : ndarray
        Image as a numpy array
    model : keras.model
        Keras CNN model
    iterations : int
        Number of iterations
    step : int
        Gradient ascent step size
    max_loss : float
        Gradient descent max loss
    shapes : sequence
        Successive image shapes
    config : dict
        Model config dictionary

    Returns
    -------
    img : tf.Tensor
        Deep dream modified image
    """
    # build gradient ascent loss
    loss = build_loss(model, config)
    print('Loss built.')
    # build image gradients
    grads = build_gradients(model, loss)
    print('Gradients built.')
    # function that evaluates loss and gradients
    loss_fn = K.function([model.input], [loss, grads])

    orig_img = np.copy(img)
    shrunk_orig_img = resize_img(img, shapes[0])
    for shape in shapes:
    #for shape in [shapes[0]]:
        print('Processing image shape', shape)
        img = resize_img(img, shape)
        # shift image pixels by jitter pixels
        img = gradient_ascent(img,
                              loss_fn=loss_fn,
                              iterations=iterations,
                              step=step,
                              max_loss=max_loss)
        # upscale shrunk image: not in first iteration this will do nothing
        upscaled_shrunk_orig_img = resize_img(shrunk_orig_img, shape)
        # upscale original image: from the original size back to original
        same_size_orig_img = resize_img(orig_img, shape)
        lost_detail = tf.subtract(same_size_orig_img, upscaled_shrunk_orig_img)
        img = tf.add(img, lost_detail)
        shrunk_orig_img = resize_img(orig_img, shape)
    return img

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Creep Dreamz with Keras.')
    parser.add_argument('-i', '--input', type=str,
                        help='Path to the input data', required=True)
    parser.add_argument('-o', '--output', type=str,
                        help='Path to the output data', required=True)
    parser.add_argument('-oct', '--octave', type=int,
                        help='Number of scales at which to run gradient ascent', required=False)
    parser.add_argument('-ocs', '--octavescale', type=float,
                        help='Size ratio between scales', required=False)
    parser.add_argument('-s', '--step', type=float,
                        help='Gradient ascent step size', required=False)
    parser.add_argument('-iter', '--iterations', type=int,
                        help='Number of gradient ascent steps per scale', required=False)
    parser.add_argument('-mxl', '--maxloss', type=float,
                        help='Maximum gradient ascent loss', required=False)

    # These are the names of the InceptionV3 50 layers
    # for which we try to maximize activation,
    # as well as their weight in the loss # we try to maximize.
    # You can tweak these setting to obtain new visual effects.
    config = {
        'features': {
            'mixed2': 0.2,
            'mixed3': 0.5,
            'mixed4': 2.,
            'mixed5': 1.5,
        },
    }

    args = parser.parse_args()
    # set learning phase to test mode
    K.set_learning_phase(0)
    # Load ption model
    model = inception_v3.InceptionV3(weights='imagenet', include_top=False)
    print('Model loaded.')
    # preprocess input image
    img = preprocess_image(args.input)
    print('Preprocessed image', args.input)
    # generate creep dream shapes
    shapes = dream_shapes(img, args.octave, args.octavescale)
    # run creep dream and get the resulting image
    img = dream(img, model, args.iterations, args.step,
                args.maxloss, shapes, config)
    # save resulting image to hard drive
    save_img(img, fname=args.output)
