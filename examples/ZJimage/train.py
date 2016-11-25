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


def get_lr(epoch):
    if epoch < 100:
        return 0.05
    elif epoch < 120:
        return 0.001
    else:
        return 0.0001


class Conf():
    def __init__(self):
        self.num_aug = 10
        self.num_epoch = 300
        self.batch_size = 100
        self.use_cpu = False
        self.num_category=172
        self.input_folder="../data/ready_chinese_food"
        self.log_dir="./log"
        self.train_file="../data/SplitAndIngreLabel/train.txt"
        self.validate_file="../data/SplitAndIngreLabel/validation.txt"
        self.label_map_file="../data/label_map.txt"

        self.small_size = 112
        self.large_size = 192
        self.crop_size = 96
        self.lr = 0.1
        self.decay = 1e-4
        self.depth = 34
        self.net = 'vgg'
        self.output = self.net + '-model'


def gen_conf():
    conf = Conf()

    small_size = [96, 112, 128, 160, 192]
    large_size = [112, 128, 160, 192, 224]
    while True:
        conf.small_size = small_size[random.randint(0, len(small_size) - 1)]
        conf.large_size = large_size[random.randint(0, len(large_size) - 1)]
        if conf.small_size < conf.large_size and 2 * conf.small_size > conf.large_size:
            break

    crop_size = [96, 128, 160, 192]
    while True:
        conf.size = crop_size[random.randint(0, len(crop_size) -1)]
        if conf.size <= conf.small_size:
            break

    lr = [0.1, 0.05, 0.01, 0.005, 0.001, 0.0005, 0.0001]
    conf.lr = lr[random.randint(0, len(lr) - 1)]

    decay = [1e-3, 5e-4, 1e-4, 5e-5, 1e-5]
    conf.decay = decay[random.randint(0, len(decay) -1)]


    '''
    net_type = ['resnet', 'vgg']
    conf.net =  net_type[random.randint(0, 1)]
    if conf.net is 'resnet':
        depth = [18, 34, 50]
    else:
        depth = [13, 16, 19]
    conf.depth = depth[random.randint(0, len(depth) - 1)]
    '''

    return conf


def train(conf, train_iter, validate_iter):
    if conf.net is 'resnet':
        net = model.create_resnet(conf.num_category, conf.crop_size)
    else:
        assert conf.net is 'vgg', 'Wrong net name'
        net = model.create_vgg(conf.num_category, conf.crop_size)

    train_iter.start()
    validate_iter.start()

    if conf.use_cpu:
        dev = device.get_default_device()
    else:
        dev = device.create_cuda_gpu_on(0)
    net.to_device(dev)
    opt = optimizer.SGD(momentum = 0.9, weight_decay=conf.decay)

    tx = tensor.Tensor((conf.batch_size, 3, conf.crop_size, conf.crop_size), dev)
    ty = tensor.Tensor((conf.batch_size,), dev)

    best_acc = 0.0
    nb_epoch_for_best_acc = 0
    accuracy = metric.Accuracy()
    num_train_batches = train_iter.num_samples / conf.batch_size
    print num_train_batches, conf.num_epoch
    r, g, b = 153.185890, 129.307082, 98.43287
    rgb = np.asarray([r, g, b], dtype=np.float32)
    for epoch in range(conf.num_epoch):
        loss, acc = 0.0, 0.0
        print 'Epoch %d' % epoch
        for b in range(num_train_batches):
            t1 = time.time()
            x,y = train_iter.next()
            x -= rgb[np.newaxis, :, np.newaxis, np.newaxis]
            t2 = time.time()
            tx.copy_from_numpy(x)
            ty.copy_from_numpy(y)
            grads, (l, a) = net.train(tx, ty)
            loss += l
            acc += a
            for (s, p, g) in zip(net.param_specs(), net.param_values(), grads):
                opt.apply_with_lr(epoch, conf.lr, g, p, str(s.name))
            t3 = time.time()
            info = 'training accuracy = %f, loss = %f, load_time = %.4f, training_time= %.4f' % (a,l,t2-t1,t3-t2)
            # update progress bar
            utils.update_progress(b * 1.0 / num_train_batches, info)
        disp= 'Epoch %d, training accuracy = %f, loss = %f' \
                % (epoch, acc / num_train_batches, loss / num_train_batches)
        logging.info(disp)
        print disp

        if (epoch + 1) % 50 == 0:
            net.save(os.path.join(conf.output, 'model-%d.bin' % epoch))

        if epoch % 4 != 0:
            continue
        acc = 0.0
        num_augmentation = 10
        num_test_batches = validate_iter.num_samples / conf.batch_size * num_augmentation
        idx = np.arange(0, conf.batch_size, num_augmentation, dtype=np.int32)
        for b in range(num_test_batches):
            x,y = validate_iter.next()
            x -= rgb[np.newaxis, :, np.newaxis, np.newaxis]
            y = y[idx]
            tx.copy_from_numpy(x)
            xx = net.predict(tx)
            prob = tensor.softmax(xx)
            prob.to_host()
            prob = tensor.to_numpy(prob)
            prob = prob.reshape((conf.batch_size / num_augmentation, num_augmentation, -1))
            prob = np.average(prob, 1)
            acc += accuracy.evaluate(tensor.from_numpy(prob),
                                     tensor.from_numpy(y))

        acc /= num_test_batches
        disp = 'Epoch %d, test     accuracy = %f' % (epoch, acc)
        logging.info(disp)
        print disp

        if acc > best_acc + 0.005:
            best_acc = acc
            nb_epoch_for_best_acc = 0
        else:
            nb_epoch_for_best_acc += 1
            if nb_epoch_for_best_acc > 8:
                break
            elif nb_epoch_for_best_acc % 4 ==0:
                conf.lr /= 10
                logging.info("Decay the learning rate from %f to %f" %(conf.lr * 10, conf.lr))
        # if epoch > 0 and epoch % 50 == 0: net.save('model_%d.bin' % epoch)
    # net.save(os.path.join(config.output_folder, 'model_%d.bin' % config.num_epoch))
    net.save(os.path.join(conf.output, 'model.bin'))
    train_iter.end()
    validate_iter.end()
    return best_acc


if __name__ == '__main__':
    #train_tool = image_tool.ImageTool()
    #validate_tool = image_tool.ImageTool()
    #log_dir = os.path.join('log', datetime.datetime.now().strftime('%Y%m%d%H%M%S'))
    #os.makedirs(log_dir)
    #logging.basicConfig(filename=os.path.join(log_dir, 'log.txt'), format='%(message)s', level=logging.INFO)

    #best_acc = 0.0
    #best_idx = -1
    import dicom
    def image_transform(img_path):
        new_imgs = []
        ary = np.asarray(dicom.read_file(img_path).pixel_array, dtype=dicom.read_file(img_path).pixel_array.dtype)
        new_imgs.append(ary)
        return new_imgs

    for i in range(1):
        #conf = gen_conf()
        #conf = Conf()
        #with open(os.path.join(log_dir, '%d.conf' % i), 'w') as fd:
        #    pickle.dump(conf, fd)

        #logging.info('\n----------trial:%d, net:%s----------' % (i, conf.net))

        #def train_transform(path):
        #    global train_tool
        #    return train_tool.load(path).resize_by_range((conf.small_size, conf.large_size)).rotate_by_range((-10, 10)).random_crop((conf.crop_size, conf.crop_size)).flip().get()
        #def validate_transform(path):
        #    global validate_tool
        #    return train_tool.load(path).resize_by_list([(conf.small_size + conf.large_size) / 2]).crop5((conf.crop_size, conf.crop_size), 5).enhance(0.1).flip(2).get()
        train_iter = data.MImageBatchIter(img_list_file = 'meta.txt', image_folder='./', image_transform = image_transform, batch_size = 2, shuffle=True, capacity=10)
        #validate_iter = data.ImageBatchIter(conf.validate_file, conf.batch_size, validate_transform, shuffle=False, image_folder = conf.input_folder, capacity=50)

        stop = False
        try:
        #    acc = train(conf, train_iter, validate_iter)
        #    logging.info('The best test accuracy for %d-th trial is %f' % (i, acc))
        #    if best_acc < acc:
        #        best_acc = acc
        #        best_idx = i
        #    logging.info('The best test accuracy so far is %f, from the %d-th conf' % (best_acc, best_idx))
            train_iter.start()
            num_train_batches = train_iter.num_samples / 2
            for i in range(10):
                imgs, labels = train_iter.next()
                print "imgs[0].shape: ", imgs[0].shape
                print "imgs[1].shape: ", imgs[1].shape
            # print "imgs[2].shape: ", imgs[2].shape
            # print "imgs[3].shape: ", imgs[3].shape
            # print "imgs[4].shape: ", imgs[4].shape
            # print "train_iter num_train_batches: ", num_train_batches
        except Exception as e:
            print "except", e
            stop = True
        except:
            print 'interupt'
            stop = True
        finally:
            train_iter.end()
        #    validate_iter.end()
        if stop:
            break
