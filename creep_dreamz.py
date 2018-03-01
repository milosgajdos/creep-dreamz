import numpy as np

from keras import backend as K
from image import resize

class CreepDream:
    """CreepDream implements Deep Dream in Tensorflow """
    def __init__(self, model_name, layer_config):
        """
        Initializes CreepDream

        Parameters
        ----------
        model_name : str
            Name of a Keras model
        layer_config : dict
            Model layer configuration dictionary
            `layer_config` is a dictionary as follows `{'layer_name' : weight}`
            `layer_name` must be a name of an existing `model` layer, `weight` is a
            float number representing the particular layer effect in resulting dream
        """
        if model_name == 'VGG16':
            from keras.applications.vgg16 import VGG16
            model = VGG16(weights='imagenet', include_top=False)
        elif model_name == 'VGG19':
            from keras.applications.vgg19 import VGG19
            model = VGG19(weights='imagenet', include_top=False)
        elif model_name == 'Resnet50':
            from keras.applications.resnet50 import ResNet50
            model = ResNet50(weights='imagenet', include_top=False)
        elif model_name == 'Xception':
            from keras.applications.xception import Xception
            model = Xception(weights='imagenet', include_top=False)
        elif model_name == 'InceptionV3':
            from keras.applications.inception_v3 import InceptionV3
            model = InceptionV3(weights='imagenet', include_top=False)
        else:
            raise ValueError('Unsupported model: {}'.format(model_name))

        self.model = model
        self.layer_config = layer_config

    def _build_loss(self):
        """
        Builds gradient ascent loss

        Returns
        -------
        loss : keras.backend.ops.Tensor
            Loss tensor
        """
        # Get the symbolic outputs of each "key" layer (we gave them unique names).
        layer_dict = dict([(layer.name, layer) for layer in self.model.layers])
        # Define the loss.
        loss = K.variable(0.)
        for layer_name in self.layer_config:
            # Add the L2 norm of the features of a layer to the loss.
            assert layer_name in layer_dict.keys(), 'Layer ' + layer_name + ' not found in model.'
            # feature map weight
            w = self.layer_config[layer_name]
            x = layer_dict[layer_name].output
            # We avoid border artifacts by only involving non-border pixels in the loss.
            scaling = K.prod(K.cast(K.shape(x), 'float32'))
            if K.image_data_format() == 'channels_first':
                loss += w * K.sum(K.square(x[:, :, 2: -2, 2: -2])) / scaling
            else:
                loss += w * K.sum(K.square(x[:, 2: -2, 2: -2, :])) / scaling
        return loss

    def _build_gradients(self, loss):
        """
        Builds gradients keras operation

        Parameters
        ----------
        loss : keras.backend.ops.Tensor
            Gradient ascent loss operator

        Returns
        -------
        grads : keras.backend.ops.Tensor
            Gradients tensor operation
        """
        # dream is created by flowing an input through model
        dream = self.model.input
        # Compute the gradients of the dream wrt the loss.
        grads = K.gradients(loss, dream)[0]
        # Normalize gradients.
        grads /= K.maximum(K.mean(K.abs(grads)), K.epsilon())
        return grads

    def compile(self):
        """
        Builds CreepDream object

        Returns
        -------
        CreepDream object with built TensorFlow graph
        """
        K.set_learning_phase(0)
        # build gradient ascent loss
        loss = self._build_loss()
        # build image gradients
        grads = self._build_gradients(loss)
        # function that evaluates loss and gradients
        self.loss_fn = K.function([self.model.input], [loss, grads])
        return self

    def _gradient_ascent(self, img, iterations, step_size, max_loss=None):
        """
        Run gradient ascent

        Parameters
        ----------
        img : ndarray
            Input image as numpy array
        iterations : int
            Number of gradient ascent iterations
        step_size : int
            Gradient ascent step size; this scales up the gradient size
        max_loss : float
            Maximum loss threshold

        Returns
        -------
        img : ndarray
            Image after gradient ascent
        """
        for i in range(iterations):
            # evaluate loss and gradient for the supplied input data
            loss_value, grad_values = self.loss_fn([img])
            if max_loss is not None and loss_value > max_loss:
                break
            print('..Loss value at', i, ':', loss_value)
            # amplify gradient by step
            img += step_size * grad_values
        return img

    def _shapes(self, img, octave, octave_scale):
        """
        Generate list of image shapes

        Reads an image and generates a list of shapes based
        on supplied octave and octave scale parameters.

        Parameters
        ----------
        img : ndarray
            Input image
        octave : int
            Number of scales at which to run gradient ascent
        octave_scale : float
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

    def run(self, img, iterations, step_size, octave, octave_scale, max_loss=None):
        """
        Runs CreepDream

        Runs gradient ascent. First it upscales provided image to particular
        scale then it reinjects the detail that was lost at upscaling time back.

        Parameters
        ----------
        img : ndarray
            Image as a numpy array
        iterations : int
            Number of iterations
        step_size : int
            Gradient ascent step size
        octave : int
            Number of scales at which to run gradient ascent
        octave_scale : float
            Size ratio between shapes
        max_loss : float
            Gradient descent max loss

        Returns
        -------
        img : ndarray
            Deep dream modified image
        """
        # create image shapes of different sizes
        shapes = self._shapes(img, octave, octave_scale)
        # copy original image - we want to avoid modifying it
        orig_img = np.copy(img)
        shrunk_orig_img = resize(img, shapes[0])
        for shape in shapes:
            img = resize(img, shape)
            img = self._gradient_ascent(img,
                                       iterations=iterations,
                                       step_size=step_size,
                                       max_loss=max_loss)
            # upscale shrunk image: not in first iteration this will do nothing
            upscaled_shrunk_orig_img = resize(shrunk_orig_img, shape)
            # upscale original image: from the original size back to original
            same_size_orig_img = resize(orig_img, shape)
            lost_detail = same_size_orig_img - upscaled_shrunk_orig_img
            img += lost_detail
            shrunk_orig_img = resize(orig_img, shape)
        return img
