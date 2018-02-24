import argparse
import scipy
import numpy as np

from keras.preprocessing.image import load_img, img_to_array
from keras.applications import inception_v3
from keras import backend as K

def preprocess_img(image_path):
    """
    Preprocess image

    Loads an image, resizes it to and preprocesses it
    for use with InceptionV3 model

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
        Input image
    size: sequence
        New image size dimensions

    Returns
    -------
    img : ndarray
        Resized image
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

def deprocess_img(img):
    """
    Converts the preprocessed image into original

    Parameters
    ----------
    img : ndarray
        Image data

    Returns
    -------
    img : ndarray
        Image data
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

def save_img(img, fname):
    """
    Saves image on filesystem

    Parameters
    ----------
    img : ndarray
        Image data
    fname : str
        Filesystem path
    """
    pil_img = deprocess_img(np.copy(img))
    scipy.misc.imsave(fname, pil_img)

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
    Builds gradients keras operation

    Parameters
    ----------
    model : keras.Model
        Neural network model used for creap dreaming
    loss : keras.tensor.Op
        Gradient ascent loss

    Returns
    -------
    grads : keras.tensor.Op
        Gradients tensor operation
    """
    # dream is created by flowing an input through model
    dream = model.input
    # Compute the gradients of the dream wrt the loss.
    grads = K.gradients(loss, dream)[0]
    # Normalize gradients.
    grads /= K.maximum(K.mean(K.abs(grads)), K.epsilon())
    return grads

def dream_shapes(img, octave, octavescale):
    """
    Generates a list of image shapes

    Reads an image and generates a list of image shapes based
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
        Sequence of shape sorted in ascending order
    """
    if K.image_data_format() == 'channels_first':
        original_shape = img.shape[2:]
    else:
        original_shape = img.shape[1:3]
    successive_shapes = [original_shape]
    for i in range(1, octave):
        shape = tuple([int(dim / (octavescale ** i)) for dim in original_shape])
        successive_shapes.append(shape)
    successive_shapes = successive_shapes[::-1]
    return successive_shapes

def gradient_ascent(x, loss_fn, iterations, step, max_loss=None):
    """
    Runs gradient ascent

    Parameters
    ----------
    x : ndarray
        Input data tensor
    loss_fn : keras.function
        Function that evaluates and fetches loss and gradients
    iterations : int
        Number of gradient ascent iterations
    step : int
        Gradient ascent step size; this scales up the gradient size
    max_loss : float
        Maximum loss threshold

    Returns
    -------
    x : ndarray
        Output data
    """
    for i in range(iterations):
        # evaluate loss and gradient for the supplied input data
        loss_value, grad_values = loss_fn([x])
        if max_loss is not None and loss_value > max_loss:
            break
        print('..Loss value at', i, ':', loss_value)
        # amplify gradient by step
        x += step * grad_values
    return x

def dream(img, model, iters, step, max_loss, jitter, shapes, config):
    """
    Creep Dream:

    Generates Creep Dream image by running gradient ascent
    given an input image and loss function

    Parameters
    ----------
    img : ndarray
        Input image
    model : keras.model
        Keras CNN model
    iters : int
        Number of iterations
    step : int
        Gradient ascent step size
    max_loss : float
        Gradient descent max loss
    shapes : sequence
        Successive image shapes
    config : dict
        Model config dictionary
    jitter : int
        Pixel shift

    Returns
    -------
    img : ndarry
        Deep dream image
    """
    # build gradient ascent loss
    loss = build_loss(model, config)
    # build image gradients
    grads = build_gradients(model, loss)
    # function that evaluates loss and gradients
    loss_fn = K.function([model.input], [loss, grads])

    orig_img = np.copy(img)
    shrunk_orig_img = resize_img(img, shapes[0])
    for shape in shapes:
        img = resize_img(img, shape)
        ox, oy = np.random.randint(-jitter, jitter+1, 2)
        img = np.roll(np.roll(img, ox, -1), oy, -2)
        img = gradient_ascent(img,
                              loss_fn=loss_fn,
                              iterations=iters,
                              step=step,
                              max_loss=max_loss)
        img = np.roll(np.roll(img, -ox, -1), -oy, -2)
        # upscale shrunk image: not in first iteration this will do nothing
        upscaled_shrunk_orig_img = resize_img(shrunk_orig_img, shape)
        # upscale original image: from the original size back to original
        same_size_orig_img = resize_img(orig_img, shape)
        lost_detail = same_size_orig_img - upscaled_shrunk_orig_img
        img += lost_detail
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
    parser.add_argument('-j', '--jitter', type=int, default=32,
                        help='Pixel shift jitter', required=False)

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
    # preprocess input image
    img = preprocess_img(args.input)
    # generate creep dream shapes
    shapes = dream_shapes(img, args.octave, args.octavescale)
    # run creep dream and get the resulting image
    img = dream(img, model, args.iterations, args.step,
                args.maxloss, args.jitter, shapes, config)
    # save resulting image to hard drive
    save_img(img, fname=args.output)
