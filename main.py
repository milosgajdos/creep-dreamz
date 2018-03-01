import argparse

from creep_dreamz import CreepDream
from image import save, preprocess

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Creep Dreamz with Keras.')
    parser.add_argument('-i', '--input', type=str,
                        help='Path to the input data', required=True)
    parser.add_argument('-o', '--output', type=str,
                        help='Path to the output data', required=True)
    parser.add_argument('-m', '--model', type=str,
                        help='Keras model name', required=True)
    parser.add_argument('-iter', '--iterations', type=int,
                        help='Number of gradient ascent steps per scale', required=False)
    parser.add_argument('-s', '--step', type=float,
                        help='Gradient ascent step size', required=False)
    parser.add_argument('-oct', '--octave', type=int,
                        help='Number of scales at which to run gradient ascent', required=False)
    parser.add_argument('-ocs', '--octavescale', type=float,
                        help='Size ratio between scales', required=False)
    parser.add_argument('-mxl', '--maxloss', type=float,
                        help='Maximum gradient ascent loss', required=False)

    # These are the names of the InceptionV3 50 layers
    # for which we try to maximize activation,
    # as well as their weight in the loss # we try to maximize.
    # You can tweak these setting to obtain new visual effects.
    config = {
        'mixed2': 0.2,
        'mixed3': 0.5,
        'mixed4': 2.,
        'mixed5': 1.5,
    }

    args = parser.parse_args()

    dream = CreepDream(args.model, config)
    dream = dream.compile()
    # preprocess image
    img = preprocess(args.input, args.model)
    # start creep dreaming
    img = dream.run(img, args.iterations, args.step, args.octave,
                    args.octavescale, args.maxloss)
    # save resulting image to hard drive
    save(img, fname=args.output)
