# creep-dreamz

Deep Dream experiments

This project is a "cleaned up" rewrite of [keras Deep Dream](https://github.com/keras-team/keras/blob/master/examples/deep_dream.py).
The project no longer depends on `scipy` module for image manipulation that was replaced by `TensorFlow` native  [tf.image](https://www.tensorflow.org/api_guides/python/image) image manipulation module.

# How to use

Example run

```
creep_dreamz.py -i "input.jpg" -o "dream.png" -oct 4 -ocs 1.2 -s 0.01 -iter 5 -mxl 10.0
```
