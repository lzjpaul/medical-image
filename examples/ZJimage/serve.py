import numpy as np
import threading

from singa import device
from singa import tensor
from singa import data
from singa import image_tool
from singa import metric
from rafiki import agent
import model

rafiki = agent.Agent()

tool = image_tool.ImageTool()
small_size = 112
big_size = 192
crop_size = 96
num_augmentation = 10

r, g, b = 153.185890, 129.307082, 98.43287
rgb = np.asarray([r, g, b], dtype=np.float32)

glob_prob = [0] * 10


def image_transform(image):
    '''Input an image path and return a set of augmented images (type Image)'''
    global tool
    return tool.load(image).resize_by_list([(small_size + big_size)/2]).crop5(
            (crop_size, crop_size), 5).flip(2).get()


def evaluate_per_net(idx, net, images, num=10):
    '''predict probability distribution for one net.

    Args:
        idx(int): net ID
        net: neural net (vgg or resnet)
        images: a batch of augmented images (type numpy)
        num: num of augmentations
    '''
    xx = net.predict(images)
    prob = tensor.softmax(xx)
    prob.to_host()
    prob = tensor.to_numpy(prob)
    prob = prob.reshape((images.shape[0] / num, num, -1))
    prob = np.average(prob, 1)
    glob_prob[idx] = prob


def evaluate(nets, meta_file, image_folder, devs, batch_size=100):
    '''Evaluate the prediction (top1) using multiple nets.

    nets: a list of nets
    meta_file: image file list, each line is (image path label_id)
    image_folder: the absolute image path = image_folder/image_path
    devs: a list of devices, one per net
    '''

    validate_iter = data.ImageBatchIter(meta_file, batch_size, image_transform,
            shuffle=False, image_folder = image_folder, capacity=50)
    validate_iter.start()

    tx = []
    for dev in devs:
        tx.append(tensor.Tensor((batch_size, 3, crop_size, crop_size), dev))

    accuracy = metric.Accuracy()
    idx = np.arange(0, batch_size, num_augmentation, dtype=np.int32)
    acc = 0.0
    num_batches = validate_iter.num_samples / batch_size * num_augmentation
    for b in range(num_batches):
        x, y = validate_iter.next()
        x -= rgb[np.newaxis, :, np.newaxis, np.newaxis]
        y = y[idx]
        th = []
        th_idx = 0
        for (net, images) in zip(nets, tx):
            images.copy_from_numpy(x)
            th.append(threading.Thread(evaluate_per_net(th_idx, net, images, num_augmentation)))
            th[th_idx].start()
            th_idx += 1
        for t in th:
            t.join()
        prob = np.average(np.array(glob_prob[0:len(nets)]), 0)
        acc += accuracy.evaluate(tensor.from_numpy(prob), tensor.from_numpy(y))
    validate_iter.end()
    acc /= num_batches
    return acc


def predict_per_net(idx, net, images):
    '''Predict the probability distribution over all labels.

    Args:
        idx(int): net ID
        net: vgg or resnet
        images: a numpy array of augmented images
    '''
    xx = net.predict(images)
    prob = tensor.softmax(xx)
    prob.to_host()
    prob = tensor.to_numpy(prob)
    prob = prob.reshape((num_augmentation, -1))
    prob = np.average(prob, 0)
    glob_prob[idx] = prob


def predict(nets, inputs, topk=5):
    '''Do prediction using multiple nets.

    Args:
        nets: a list of nets
        inputs: a numpy array of augmented images for a single original image

    Returns:
        the index of food names whose probabilities are in the topk
    '''
    th = []
    th_idx = 0
    for (net, images) in zip(nets, inputs):
        th.append(threading.Thread(predict_per_net(th_idx, net, images)))
        th[-1].start()
    for t in th:
        t.join()
    avg_prob = np.average(np.array(glob_prob[0:len(nets)]), 0)
    labels = np.flipud(np.argsort(avg_prob))
    return labels[0:topk]


def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1] in \
        ["PNG", "png", "jpg", "JPG", "JPEG", "jpeg"]


def serve(nets, label_map, devs, topk=5):
    '''Serve to predict image labels.

    It prints the topk food names for each image.

    Args:
        nets: a list of nets
        label_map: a list of food names, corresponding to the index in meta_file
        devs: a list of devices, one per net

    '''

    tx = []
    for dev in devs:
        tx.append(tensor.Tensor((num_augmentation, 3, crop_size, crop_size), dev))
    while True:
        try:
            key, val = rafiki.Pull()
            if key is agent.STOP:
                break
            image = val.files['image']
            if not image:
                rafiki.PushStatus(agent.ERROR, 'no image found')
            if not allowed_file(image.filename):
                rafiki.PushStatus(agent.ERROR, 'only jpg/png image is allowed')
                images = []
                for im in image_transform(image):
                    dat = np.array(image.convert('RGB'), dtype=np.float32)
                    images.append(dat.transpose(2, 0, 1))
                images = np.array(images)
                images -= rgb[np.newaxis, :, np.newaxis, np.newaxis]
                for t in tx:
                    t.copy_from_numpy(images)
                labels = [label_map[idx] for idx in predict(nets, tx, topk)]
                rafiki.PushResponse('labels', ' '.join(labels))
        except Exception as e:
            rafiki.PushStatus(agent.ERROR, str(e))

    rafiki.PushStatus(agent.SUCCESS, 'Stopped the serving job')


if __name__ == '__main__':
    vgg = model.create_vgg(172, crop_size)
    vgg.load('vgg-model/model.bin')
    resnet = model.create_resnet(172, crop_size)
    resnet.load('resnet-model/model.bin')
    devs = device.create_cuda_gpus(2)
    vgg.to_device(devs[0])
    resnet.to_device(devs[1])
    with open('FoodList.txt', 'r') as fd:
        label_map = [fname.strip() for fname in fd.readlines()]
    # acc = evaluate([vgg, resnet], '../data/SplitAndIngreLabel/test.txt', '../data/ready_chinese_food', devs)
    #print acc
    serve([vgg, resnet], label_map, devs)
