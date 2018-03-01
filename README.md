# creep-dreamz

Deep Dream experiments

This project is a "cleaned up" rewrite of [keras Deep Dream](https://github.com/keras-team/keras/blob/master/examples/deep_dream.py). It provides a `CreepDream` class which implements Deep Dream based on the supplied parameters.
The project no longer depends on `scipy` module for image manipulation that was replaced by `TensorFlow` native  [tf.image](https://www.tensorflow.org/api_guides/python/image) image manipulation module.

# Example Usage

You can find a simple example in `main.py` file which demonstrates how to use `CreepDream`:

```
usage: main.py [-h] -i INPUT -o OUTPUT -m MODEL [-iter ITERATIONS] [-s STEP]
               [-oct OCTAVE] [-ocs OCTAVESCALE] [-mxl MAXLOSS]

Creep Dreamz with Keras.

optional arguments:
  -h, --help            show this help message and exit
  -i INPUT, --input INPUT
                        Path to the input data
  -o OUTPUT, --output OUTPUT
                        Path to the output data
  -m MODEL, --model MODEL
                        Keras model name
  -iter ITERATIONS, --iterations ITERATIONS
                        Number of gradient ascent steps per scale
  -s STEP, --step STEP  Gradient ascent step size
  -oct OCTAVE, --octave OCTAVE
                        Number of scales at which to run gradient ascent
  -ocs OCTAVESCALE, --octavescale OCTAVESCALE
                        Size ratio between scales
  -mxl MAXLOSS, --maxloss MAXLOSS
                        Maximum gradient ascent loss
```

# Example run results

```
python3 main.py -i "random_man.jpg" -o "random_man_creep.png" -m "InceptionV3"  -oct 4 -ocs 1.4 -s 0.01 -iter 15 -mxl 10.0
```

Original image:

<img src="./random_man.jpg" alt="Random man" width="200">

Creep Dreamt image:

<img src="./random_man_creep.png" alt="Random man creep" width="200">

[1] Image source: [https://commons.wikimedia.org/wiki/File:Handsome-man-by-Willy-Volk-Creative-Commons.jpg](https://commons.wikimedia.org/wiki/File:Handsome-man-by-Willy-Volk-Creative-Commons.jpg)
