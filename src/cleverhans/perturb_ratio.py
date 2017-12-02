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
    print(numerator / denominator)
    return numerator / denominator


if __name__ == '__main__':
    get_ratio()

