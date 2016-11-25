import os, time
import numpy as np
import logging
import pickle
import random
import datetime

from singa import device
from singa import tensor
from singa import metric
from singa import optimizer
from singa import utils
from singa import data
from singa import image_tool

import model


if __name__ == '__main__':
    import dicom
    def image_transform(img_path):
        new_imgs = []
        ary = np.asarray(dicom.read_file(img_path).pixel_array, dtype=dicom.read_file(img_path).pixel_array.dtype)
        new_imgs.append(ary)
        return new_imgs

    for i in range(1):
        train_iter = data.MImageBatchIter(img_list_file = 'meta.txt', image_folder='./', image_transform = image_transform, batch_size = 2, shuffle=True, capacity=10)
        stop = False
        try:
            train_iter.start()
            num_train_batches = train_iter.num_samples / 2
            for i in range(10):
                imgs, labels = train_iter.next()
                print "imgs[0].shape: ", imgs[0].shape
                print "imgs[1].shape: ", imgs[1].shape
        except Exception as e:
            print "except", e
            stop = True
        except:
            print 'interupt'
            stop = True
        finally:
            train_iter.end()
        if stop:
            break
