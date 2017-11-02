from theano.sandbox import cuda
%matplotlib inline
from imp import reload
import utils; reload(utils)
from utils import *
from __future__ import division, print_function

path = "/home/loopasam/yeast_colonies/test/"

## load and finetune vgg16 model

import vgg16; reload(vgg16)

batch_size=8
vgg = Vgg16()
batches = vgg.get_batches('/home/loopasam/yeast_colonies/train/multi3', batch_size=batch_size)
vgg.finetune(batches)

# load previously trained weights
vgg.model.load_weights('/home/loopasam/yeast_colonies/models/vgg_ft_3class_lrdecr.h5')

## predict labels for test data

test_batches, probs = vgg.test(path, batch_size=batch_size)
test_labels = test_batches.classes
test_filenames = test_batches.filenames