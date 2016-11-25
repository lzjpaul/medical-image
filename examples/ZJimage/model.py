# Licensed to the Apache Software Foundation (ASF) under one
# or more contributor license agreements.  See the NOTICE file
# distributed with this work for additional information
# regarding copyright ownership.  The ASF licenses this file
# to you under the Apache License, Version 2.0 (the
# "License"); you may not use this file except in compliance
# with the License.  You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# =============================================================================
from singa import layer
from singa import initializer
from singa import metric
from singa import loss
from singa import net as ffnet


def ConvBnReLU(net, name, nb_filers, sample_shape=None):
    net.add(layer.Conv2D(name + '_1', nb_filers, 3, 1, pad=1,
                         input_sample_shape=sample_shape))
    net.add(layer.BatchNormalization(name + '_2'))
    net.add(layer.Activation(name + '_3'))


def create_vgg(num_output, img_size, use_cpu=False):
    if use_cpu:
        layer.engine = 'singacpp'
    net = ffnet.FeedForwardNet(loss.SoftmaxCrossEntropy(), metric.Accuracy())
    ConvBnReLU(net, 'conv1_1', 64, (3, img_size, img_size))
    net.add(layer.Dropout('drop1', 0.3))
    ConvBnReLU(net, 'conv1_2', 64)
    net.add(layer.MaxPooling2D('pool1', 2, 2, border_mode='valid'))
    ConvBnReLU(net, 'conv2_1', 128)
    net.add(layer.Dropout('drop2_1', 0.4))
    ConvBnReLU(net, 'conv2_2', 128)
    net.add(layer.MaxPooling2D('pool2', 2, 2, border_mode='valid'))
    ConvBnReLU(net, 'conv3_1', 256)
    net.add(layer.Dropout('drop3_1', 0.4))
    ConvBnReLU(net, 'conv3_2', 256)
    net.add(layer.Dropout('drop3_2', 0.4))
    ConvBnReLU(net, 'conv3_3', 256)
    net.add(layer.MaxPooling2D('pool3', 2, 2, border_mode='valid'))
    ConvBnReLU(net, 'conv4_1', 512)
    net.add(layer.Dropout('drop4_1', 0.4))
    ConvBnReLU(net, 'conv4_2', 512)
    net.add(layer.Dropout('drop4_2', 0.4))
    ConvBnReLU(net, 'conv4_3', 512)
    net.add(layer.MaxPooling2D('pool4', 2, 2, border_mode='valid'))
    ConvBnReLU(net, 'conv5_1', 512)
    net.add(layer.Dropout('drop5_1', 0.4))
    ConvBnReLU(net, 'conv5_2', 512)
    net.add(layer.Dropout('drop5_2', 0.4))
    ConvBnReLU(net, 'conv5_3', 512)
    net.add(layer.MaxPooling2D('pool5', 2, 2, border_mode='valid'))
    net.add(layer.Flatten('flat'))
    net.add(layer.Dropout('drop_flat', 0.5))
    net.add(layer.Dense('ip1', 512))
    net.add(layer.BatchNormalization('batchnorm_ip1'))
    net.add(layer.Activation('relu_ip1'))
    net.add(layer.Dropout('drop_ip2', 0.5))
    net.add(layer.Dense('ip2', num_output))
    print 'Start intialization............'
    for (p, name) in zip(net.param_values(), net.param_names()):
        print name, p.shape
        if 'mean' in name or 'beta' in name:
            p.set_value(0.0)
        elif 'var' in name:
            p.set_value(1.0)
        elif 'gamma' in name:
            initializer.uniform(p, 0, 1)
        elif len(p.shape) > 1:
            if 'conv' in name:
                initializer.gaussian(p, 0, 3 * 3 * p.shape[0])
            else:
                p.gaussian(0, 0.02)
        else:
            p.set_value(0)
        print name, p.l1()

    return net


def Block(net, name, nb_filters, stride):
    split = net.add(layer.Split(name + "-split", 2))
    if stride > 1:
        net.add(layer.Conv2D(name + "-br2-conv", nb_filters, 1, stride, pad=0), split)
        br2bn = net.add(layer.BatchNormalization(name + "-br2-bn"))
    net.add(layer.Conv2D(name + "-br1-conv1", nb_filters, 3, stride, pad=1), split)
    net.add(layer.BatchNormalization(name + "-br1-bn1"))
    net.add(layer.Activation(name + "-br1-relu"))
    net.add(layer.Conv2D(name + "-br1-conv2", nb_filters, 3, 1, pad=1))
    br1bn2 = net.add(layer.BatchNormalization(name + "-br1-bn2"))
    if stride > 1:
        net.add(layer.Merge(name + "-merge"), [br1bn2, br2bn])
    else:
        net.add(layer.Merge(name + "-merge"), [br1bn2, split])


def create_resnet(num_output, img_size, use_cpu=False):
    if use_cpu:
        layer.engine = 'singacpp'

    net = ffnet.FeedForwardNet(loss.SoftmaxCrossEntropy(), metric.Accuracy())
    net.add(layer.Conv2D("conv1", 32, 3, 1, pad=1, input_sample_shape=(3, img_size, img_size)))
    net.add(layer.BatchNormalization("bn1"))
    net.add(layer.Activation("relu1"))
    net.add(layer.MaxPooling2D("pool1", 3, 2, border_mode='valid'))

    Block(net, "2a", 32, 1)
    Block(net, "2b", 32, 1)
    Block(net, "2c", 32, 1)

    Block(net, "3a", 64, 2)
    Block(net, "3b", 64, 1)
    Block(net, "3c", 64, 1)

    Block(net, "4a", 128, 2)
    Block(net, "4b", 128, 1)
    Block(net, "4c", 128, 1)

    Block(net, "5a", 256, 2)
    Block(net, "5b", 256, 1)
    Block(net, "5c", 256, 1)
#
    net.add(layer.AvgPooling2D("pool4", 6, 1, pad=0))
    net.add(layer.Flatten('flat'))
    net.add(layer.Dense('ip5', num_output))
    print 'Start intialization............'
    for (p, name) in zip(net.param_values(), net.param_names()):
        # print name, p.shape
        if 'mean' in name or 'beta' in name:
            p.set_value(0.0)
        elif 'var' in name:
            p.set_value(1.0)
        elif 'gamma' in name:
            initializer.uniform(p, 0, 1)
        elif len(p.shape) > 1:
            if 'conv' in name:
                # initializer.gaussian(p, 0, math.sqrt(2.0/p.shape[1]))
                initializer.gaussian(p, 0, 9.0 * p.shape[0])
            else:
                initializer.uniform(p, p.shape[0], p.shape[1])
        else:
            p.set_value(0)
        # print name, p.l1()

    return net
