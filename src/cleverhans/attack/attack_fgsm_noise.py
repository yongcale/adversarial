"""Implementation of sample attack."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os

import numpy as np
import tensorflow as tf
from PIL import Image
from tensorflow.contrib.slim.nets import inception
from tensorflow.contrib.slim.nets import resnet_v2
from tensorflow.contrib.slim.python.slim.nets import resnet_utils

from adversarial.src.cleverhans.lib.attacks import FastGradientMethod

resnet_arg_scope = resnet_utils.resnet_arg_scope

slim = tf.contrib.slim


tf.flags.DEFINE_string(
    'master', '', 'The address of the TensorFlow master to use.')

tf.flags.DEFINE_string(
    'checkpoint_path', '', 'Path to checkpoint for inception network.')

tf.flags.DEFINE_string(
    'input_dir', '', 'Input directory with images.')

tf.flags.DEFINE_string(
    'output_dir', '', 'Output directory with images.')

tf.flags.DEFINE_integer(
    'iterations', 10, 'how many iterations.')

tf.flags.DEFINE_float(
    'learning_rate', 8.0, 'learning rate of adversarial perturbation.')

tf.flags.DEFINE_float(
    'max_epsilon', 16.0, 'Maximum size of adversarial perturbation.')

tf.flags.DEFINE_integer(
    'image_width', 299, 'Width of each input images.')

tf.flags.DEFINE_integer(
    'image_height', 299, 'Height of each input images.')

tf.flags.DEFINE_integer(
    'batch_size', 16, 'How many images process at one time.')

FLAGS = tf.flags.FLAGS


def load_images(input_dir, batch_shape):
  """Read png images from input directory in batches.

  Args:
    input_dir: input directory
    batch_shape: shape of minibatch array, i.e. [batch_size, height, width, 3]

  Yields:
    filenames: list file names without path of each image
      Lenght of this list could be less than batch_size, in this case only
      first few images of the result are elements of the minibatch.
    images: array with all images from this batch
  """
  images = np.zeros(batch_shape)
  filenames = []
  idx = 0
  batch_size = batch_shape[0]
  for filepath in tf.gfile.Glob(os.path.join(input_dir, '*.png')):
    with tf.gfile.Open(filepath) as f:
      image = np.array(Image.open(f).convert('RGB')).astype(np.float) / 255.0
    # Images for inception classifier are normalized to be in [-1, 1] interval.
    images[idx, :, :, :] = image * 2.0 - 1.0
    filenames.append(os.path.basename(filepath))
    idx += 1
    if idx == batch_size:
      yield filenames, images
      filenames = []
      images = np.zeros(batch_shape)
      idx = 0
  if idx > 0:
    yield filenames, images


def save_images(images, filenames, output_dir):
  """Saves images to the output directory.

  Args:
    images: array with minibatch of images
    filenames: list of filenames without path
      If number of file names in this list less than number of images in
      the minibatch then only first len(filenames) images will be saved.
    output_dir: directory where to save images
  """
  for i, filename in enumerate(filenames):
    # Images for inception classifier are normalized to be in [-1, 1] interval,
    # so rescale them back to [0, 1].
    with tf.gfile.Open(os.path.join(output_dir, filename), 'w') as f:
      img = (((images[i, :, :, :] + 1.0) * 0.5) * 255.0).astype(np.uint8)
      Image.fromarray(img).save(f, format='PNG')


class ResNetModel(object):
  """Model class for CleverHans library."""

  def __init__(self, num_classes):
    self.num_classes = num_classes
    self.built = False

  def __call__(self, x_input):
    """Constructs model and return probabilities for given input."""
    reuse = True if self.built else None
    with slim.arg_scope(resnet_v2.resnet_utils.resnet_arg_scope()):
      _, end_points = resnet_v2.resnet_v2_50(
          x_input, num_classes=self.num_classes,reuse=reuse)
    self.built = True
    output = end_points['Predictions']
    # Strip off the extra reshape op at the output
    probs = output.op.inputs[0]
    return probs

class InceptionModel(object):
  """Model class for CleverHans library."""

  def __init__(self, num_classes):
    self.num_classes = num_classes
    self.built = False

  def __call__(self, x_input):
    """Constructs model and return probabilities for given input."""
    reuse = True if self.built else None
    with slim.arg_scope(inception.inception_v3_arg_scope()):
      _, end_points = inception.inception_v3(
          x_input, num_classes=self.num_classes, is_training=False,
          reuse=reuse)
    self.built = True
    output = end_points['Predictions']
    # Strip off the extra reshape op at the output
    probs = output.op.inputs[0]
    return probs


def main(_):
  # Images for inception classifier are normalized to be in [-1, 1] interval,
  # eps is a difference between pixels so it should be in [0, 2] interval.
  # Renormalizing epsilon from [0, 255] to [0, 2].

  eps = 2.0 * FLAGS.max_epsilon / 255.0
  learning_rate = 2.0 * FLAGS.learning_rate / 255.0
  iterations = FLAGS.iterations
  batch_shape = [FLAGS.batch_size, FLAGS.image_height, FLAGS.image_width, 3]
  num_classes = 1001

  tf.logging.set_verbosity(tf.logging.INFO)

  with tf.Graph().as_default():
    # Prepare graph
    x_input = tf.placeholder(tf.float32, shape=batch_shape)
    # Placeholder = Placeholder[dtype=DT_FLOAT, shape=[1,299,299,3]
    model = InceptionModel(num_classes)
    #model = ResNetModel(num_classes)  TODO

    fgsm = FastGradientMethod(model)

    # the ord with -1 means the gradient ascent with noise implementation
    x_adv = fgsm.generate(x_input, eps=learning_rate, ord=-1, clip_min=-1., clip_max=1.)

    # Run computation
    saver = tf.train.Saver(slim.get_model_variables())
    session_creator = tf.train.ChiefSessionCreator(
        scaffold=tf.train.Scaffold(saver=saver),
        checkpoint_filename_with_path=FLAGS.checkpoint_path,
        master=FLAGS.master)

    with slim.arg_scope(inception.inception_v3_arg_scope()):
      _, end_points = inception.inception_v3(
          x_input, num_classes=num_classes, is_training=False)
    predicted_labels = tf.argmax(end_points['Predictions'], 1)

    dict = {}
    label = -1
    # figure out the labels predicted by the target model
    # update the initial images and output for iteration

    for filenames, images in load_images(FLAGS.input_dir, batch_shape):
        print("to be processed image is {}".format(filenames[0]))
        x_adv = fgsm.generate(x_input, eps=learning_rate, ord=-1, clip_min=-1., clip_max=1.)
        with tf.train.MonitoredSession(session_creator=session_creator) as sess:
            label = sess.run(predicted_labels, feed_dict={x_input: images})
            adv_images = sess.run(x_adv, feed_dict={x_input: images})
            save_images(adv_images, filenames, FLAGS.output_dir)
        dict[filenames[0]] = label[0]
        if len(dict) == 50:
            break
    print(dict)

    # iteratively, greedy perturb images
    image_set = set()
    print("Note, iterative version does not support batch")
    for itr in range(iterations):
        print("-----  the {}th iteration  -----".format(itr+1))
        counter = 1
        for filenames, images in load_images(FLAGS.output_dir, batch_shape):
            print("the {}th image in iteration {}".format(counter, itr+1))
            counter += 1
            if filenames[0] in image_set:
                continue
            with tf.train.MonitoredSession(session_creator=session_creator) as sess:
                label = sess.run(predicted_labels, feed_dict={x_input: images})
            if dict.get(filenames[0]) != None and dict.get(filenames[0]) != label[0]:
                image_set.add(filenames[0])
                print("    image {} has been changed to {}".format(filenames[0], label[0]))
                continue
            print("    image {} still has the same label {}".format(filenames[0], label[0]))
            if (itr + 1) == iterations:
                continue
            x_adv = fgsm.generate(x_input, eps=learning_rate * (itr + 1), ord=-1, clip_min=-1., clip_max=1.)
            with tf.train.MonitoredSession(session_creator=session_creator) as sess:
                adv_images = sess.run(x_adv, feed_dict={x_input: images})
                save_images(adv_images, filenames, FLAGS.output_dir)
        print('  the {}th iteration performance: {} / {}'.format(itr+1, len(image_set), len(dict)))
    print('{} / {} images have been perturbed successfully'.format(len(image_set), len(dict)))


if __name__ == '__main__':
  tf.app.run()
