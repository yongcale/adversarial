import os

import numpy as np
import tensorflow as tf
from PIL import Image
import argparse

# Command Line Arguments
def ParseArguments():
    # Define arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--input_dir', help='Path to input file',
                        default='./input/images')
    parser.add_argument('-o', '--output_dir', help='Path to out file',
                        default='./output/adv_images')
    # Parse arguments and return
    args = vars(parser.parse_args())
    return args

def save_images(images, filenames, output_dir):
  for i, filename in enumerate(filenames):
    # Images for inception classifier are normalized to be in [-1, 1] interval,
    # so rescale them back to [0, 1].
    with tf.gfile.Open(os.path.join(output_dir, "perturb_"+filename), 'w') as f:
      img = (((images[i, :, :, :] + 1.0) * 0.5) * 255.0).astype(np.uint8)
      Image.fromarray(img).save(f, format='PNG')

def get_ratio():
    args = ParseArguments()
    numerator = 0.0
    denominator = 0.0
    for filepath in tf.gfile.Glob(os.path.join(args['input_dir'], '*.png')):
        print("image {}".format(os.path.basename(filepath)))
        image_input = np.array(Image.open(filepath).convert('RGB')).astype(np.float)
        image_output = np.array(
            Image.open(os.path.join(args['output_dir'], os.path.basename(filepath))).convert('RGB')).astype(np.float)
        denominator += np.sum(image_input) / 1000.0
        numerator += np.sum(np.abs(image_input - image_output)) / 1000.0
        with tf.gfile.Open(os.path.join(args['output_dir'], "perturb_" + os.path.basename(filepath)), 'w') as f:
            img = (image_output-image_input).astype(np.uint8)
            Image.fromarray(img).save(f, format='PNG')
    print(numerator)
    print(denominator)
    print(numerator / denominator)
    return numerator / denominator


if __name__ == '__main__':
    get_ratio()

