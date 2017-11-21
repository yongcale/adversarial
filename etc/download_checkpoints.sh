#!/bin/bash
#
# Scripts which download checkpoints for provided models.
#
# More models please refer to https://github.com/tensorflow/models/tree/master/research/slim

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

# Download inception v3 checkpoint for fgsm attack.
cd "${SCRIPT_DIR}/../src/cleverhans/fgsm/"
wget http://download.tensorflow.org/models/inception_v3_2016_08_28.tar.gz
wget http://download.tensorflow.org/models/resnet_v2_50_2017_04_14.tar.gz
tar -xvzf inception_v3_2016_08_28.tar.gz
tar -xvzf resnet_v2_50_2017_04_14.tar.gz
rm inception_v3_2016_08_28.tar.gz
rm resnet_v2_50_2017_04_14.tar.gz
