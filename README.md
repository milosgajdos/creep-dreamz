# creep-dreamz

Deep Dream experiments

This project is basically a cleaned up rewrite of [keras Deep Dream](https://github.com/keras-team/keras/blob/master/examples/deep_dream.py) with minor addition of image jitters

# Usage

```
usage: creep_dreamz.py [-h] -i INPUT -o OUTPUT [-oct OCTAVE]
                       [-ocs OCTAVESCALE] [-s STEP] [-iter ITERATIONS]
                       [-mxl MAXLOSS] [-j JITTER]

Creep Dreamz with Keras.

optional arguments:
  -h, --help            show this help message and exit
  -i INPUT, --input INPUT
                        Path to the input data
  -o OUTPUT, --output OUTPUT
                        Path to the output data
  -oct OCTAVE, --octave OCTAVE
                        Number of scales at which to run gradient ascent
  -ocs OCTAVESCALE, --octavescale OCTAVESCALE
                        Size ratio between scales
  -s STEP, --step STEP  Gradient ascent step size
  -iter ITERATIONS, --iterations ITERATIONS
                        Number of gradient ascent steps per scale
  -mxl MAXLOSS, --maxloss MAXLOSS
                        Maximum gradient ascent loss
  -j JITTER, --jitter JITTER
                        Pixel shift jitter
```

# Example run

```
creep_dreamz.py -i "input.jpg" -o "dream.png" -oct 4 -ocs 1.2 -s 0.01 -iter 5 -mxl 10.0
```
