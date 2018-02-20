import argparse
import scipy
import numpy as np
import tensorflow as tf

from keras.preprocessing.image import load_img, img_to_array
from keras.applications import resnet50
from keras import backend as K

def preprocess_image(image_path):
    """
    Loads image from image_path, resizes it to appropriate tensor
    and preprocesses it for use with Resnet50 model
    :param image_path: path to image
    """
    img = load_img(image_path)
    img = img_to_array(img)
    img = np.expand_dims(img, axis=0)
    img = resnet50.preprocess_input(img)
    return img

def deprocess_image(x):
    """
    Coverts an image tensor into a numpy image
    :param x: image tensor
    """
    if K.image_data_format() == 'channels_first':
        x = x.reshape((3, x.shape[2], x.shape[3]))
        x = x.transpose((1, 2, 0))
    else:
        x = x.reshape((x.shape[1], x.shape[2], 3))
    x /= 2.
    x += 0.5
    x *= 255.
    x = np.clip(x, 0, 255).astype('uint8')
    return x

def resize_img(img, size):
    """
    Resize image to appropriate size
    :param img: image tensor
    :param size: image dimensions tuple
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
    # TODO: optimize using tensorflow functions
    # https://github.com/alexisbcook/ResNetCAM-keras/pull/2
    return scipy.ndimage.zoom(img, factors, order=1)

def save_img(img, fname):
    """
    Saves image on filesystem
    :params img: image as numpy array
    :params fname: filename
    """
    pil_img = deprocess_image(np.copy(img))
    scipy.misc.imsave(fname, pil_img)

def build_loss(model, config):
    """
    Builds loss function for gradient ascent
    :param model: DNN model used for creap dreaming
    :param config: creeep dream configuration
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
    :param model: DNN model used for creap dreaming
    :param loss: gradient ascent loss
    """
    # dream is created by flowing an input through model
    dream = model.input
    # Compute the gradients of the dream wrt the loss.
    grads = K.gradients(loss, dream)[0]
    # Normalize gradients.
    grads /= K.maximum(K.mean(K.abs(grads)), K.epsilon())
    return grads

def gradient_ascent(x, loss_fn, iterations, step, max_loss=None):
    """
    Runs gradient ascent for a given loss and input
    :params x: input data (image)
    :params loss_fn: function that fetches loss and gradients
    :params iterations: number of gradient ascent iterations
    :params step: gradient ascent step size; this scales up the gradient
    :params max_loss: maximum loss
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

def dream_shapes(img, octave, octavescale):
    """
    Reads an image and generates a list of shapes based
    on supplied octave and octave scale.
    Returned shapes are sorted in ascending order.
    :params img: input image as tensor
    :params octave: number of scales at which to run gradient ascent
    :params octavescale: size ratio between scales
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

def dream(img, loss_fn, shapes, iterations, step, max_loss):
    """
    Creep Dream:
        - Run gradient ascent
        - Upscale image to the next scale
        - Reinject the detail that was lost at upscaling time
    :param img: input image
    :params loss_fn: function that fetches loss and gradients
    :param shapes: successive image shapes
    :param iterations: number of iterations
    :param step: gradient ascent step size
    :param max_loss: gradient descent max loss
    """
    orig_img = np.copy(img)
    shrunk_orig_img = resize_img(img, shapes[0])
    for shape in shapes:
        print('Processing image shape', shape)
        img = resize_img(img, shape)
        img = gradient_ascent(img,
                              loss_fn=loss_fn,
                              iterations=iterations,
                              step=step,
                              max_loss=max_loss)
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

    # These are the names of the Resnet 50 layers
    # for which we try to maximize activation,
    # as well as their weight in the loss # we try to maximize.
    # You can tweak these setting to obtain new visual effects.
    config = {
        'features': {
            'add_3': 0.2,
            'add_5': 0.5,
            'add_9': 2.,
            'add_11': 1.5,
        },
    }

    args = parser.parse_args()
    # set learning phase to test mode
    K.set_learning_phase(0)
    # Load Resnet50 model
    model = resnet50.ResNet50(weights='imagenet', include_top=False)
    print('Model loaded.')
    # build gradient ascent loss
    loss = build_loss(model, config)
    print('Loss built.')
    # build image gradients
    grads = build_gradients(model, loss)
    print('Gradients built.')
    # function that evaluates loss and gradients
    loss_fn = K.function([model.input], [loss, grads])
    # preprocess input image
    img = preprocess_image(args.input)
    print('Preprocessed image', args.input)
    # generate creep dream shapes
    shapes = dream_shapes(img, args.octave, args.octavescale)
    # run creep dream and get the resulting image
    img = dream(img, loss_fn, shapes, args.iterations, args.step, args.maxloss)
    # save resulting image to hard drive
    save_img(img, fname=args.output)
