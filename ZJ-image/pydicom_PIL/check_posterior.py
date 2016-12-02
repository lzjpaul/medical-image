# pydicom_PIL.py
"""View DICOM images using Python image Library (PIL)

Usage:
>>> import pydicom
>>> from pydicom.contrib.pydicom_PIL import show_PIL
>>> ds = pydicom.read_file("filename")
>>> show_PIL(ds)

Requires Numpy:  http://numpy.scipy.org/
and Python Imaging Library:   http://www.pythonware.com/products/pil/

"""
# Copyright (c) 2009 Darcy Mason, Adit Panchal
# This file is part of pydicom, relased under an MIT license.
#    See the file license.txt included with this distribution, also
#    available at https://github.com/darcymason/pydicom

# Based on image.py from pydicom version 0.9.3,
#    LUT code added by Adit Panchal
# Tested on Python 2.5.4 (32-bit) on Mac OS X 10.6
#    using numpy 1.3.0 and PIL 1.1.7b1
import dicom
import sys
import os
import argparse
have_PIL = True
try:
    import PIL.Image
except ImportError:
    have_PIL = False

have_numpy = True
try:
    import numpy as np
except ImportError:
    have_numpy = False


def get_LUT_value(data, window, level):
    """Apply the RGB Look-Up Table for the given data and window/level value."""
    if not have_numpy:
        raise ImportError("Numpy is not available. See http://numpy.scipy.org/"
                          " to download and install")
    print "window: ", window
    print "level: ", level
    return np.piecewise(data,
                        [data <= (level - 0.5 - (window - 1) / 2),
                         data > (level - 0.5 + (window - 1) / 2)],
                        [0, 255, lambda data: ((data - (level - 0.5)) /
                         (window - 1) + 0.5) * (255 - 0)])


def get_PIL_image(dataset):
    """Get Image object from Python Imaging Library(PIL)"""
    if not have_PIL:
        raise ImportError("Python Imaging Library is not available. "
                          "See http://www.pythonware.com/products/pil/ "
                          "to download and install")
    if ('PixelData' not in dataset):
        raise TypeError("Cannot show image -- DICOM dataset does not have "
                        "pixel data")
    # can only apply LUT if these window info exists
    print 'WindowWidth' not in dataset
    print 'WindowCenter' not in dataset
    if ('WindowWidth' not in dataset) or ('WindowCenter' not in dataset):
        print "attention: not LUT !!!"
        bits = dataset.BitsAllocated
        samples = dataset.SamplesPerPixel
        if bits == 8 and samples == 1:
            mode = "L"
        elif bits == 8 and samples == 3:
            mode = "RGB"
        elif bits == 16:
            # not sure about this -- PIL source says is 'experimental'
            # and no documentation. Also, should bytes swap depending
            # on endian of file and system??
            mode = "I;16"
        else:
            raise TypeError("Don't know PIL mode for %d BitsAllocated "
                            "and %d SamplesPerPixel" % (bits, samples))

        # PIL size = (width, height)
        size = (dataset.Columns, dataset.Rows)

        # Recommended to specify all details
        # by http://www.pythonware.com/library/pil/handbook/image.htm
        im = PIL.Image.frombuffer(mode, size, dataset.PixelData,
                                  "raw", mode, 0, 1)

    else:
        print dataset.WindowWidth
        print dataset.WindowCenter
        print type(dataset.WindowWidth)
        if type(dataset.WindowWidth) is dicom.multival.MultiValue:
            image = get_LUT_value(dataset.pixel_array, int(dataset.WindowWidth[0]),
                              int(dataset.WindowCenter[0]))
        else:
            image = get_LUT_value(dataset.pixel_array, int(dataset.WindowWidth),
                              int(dataset.WindowCenter))
        # Convert mode to L since LUT has only 256 values:
        #   http://www.pythonware.com/library/pil/handbook/image.htm
        im = PIL.Image.fromarray(np.uint8(image)).convert('L')
        
    return im


def show_PIL(dataset):
    """Display an image using the Python Imaging Library (PIL)"""
    im = get_PIL_image(dataset)
    im.show()
    im.save("test2.png")

#if __name__ == '__main__':
#    ds = dicom.read_file(sys.argv[1])
#    show_PIL(ds)
#    print ds.pixel_array
#    print ds.pixel_array.shape
# other operations
#    im = get_PIL_image(ds)
#    print "print Im.mode,Im.size,Im.format: ", im.mode,im.size,im.format
#    pixels = list(im.getdata())
#    print "pixels shape: ", len(pixels)
#    print "pixels asarray shape: ", np.asarray(pixels).shape
#    width, height = im.size
#    pixels = [pixels[i * width:(i + 1) * width] for i in xrange(height)]
#    print "pixels shape: ", len(pixels)
#    print "pixels[0]: ", len(pixels[0])
#    print "pixels asarray shape: ", np.asarray(pixels).shape
#    im = im.resize((128,128), PIL.Image.NEAREST)
#    im.save("resize128.png")
#    pixels = list(im.getdata())
#    print "pixels shape: ", len(pixels)
#    print "pixels asarray shape: ", np.asarray(pixels).shape
#    width, height = im.size
#    pixels = [pixels[i * width:(i + 1) * width] for i in xrange(height)]
#    print "pixels shape: ", len(pixels)
#    print "pixels[0]: ", len(pixels[0])
#    print "pixels asarray shape: ", np.asarray(pixels)[0] 

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Resize dcm image')
    parser.add_argument('rawdatadir', type=str, help='raw data directory')
    # parser.add_argument('outputdir', type=str, help='outputdir')
    parser.add_argument('img_list_file', type=str, help='special image (i.e. PA) list file')
    parser.add_argument('resize', type=int, help='resize size')
    
    # parser.add_argument('--use_gpu', action='store_true')
    args = parser.parse_args()

    PathDicom = args.rawdatadir
    lstFilesDCM = []  # create an empty list
    lstDirs = []
    lstFileName = []
    for dirName, subdirList, fileList in os.walk(PathDicom):
        for filename in fileList:
            if ".dcm" in filename.lower():  # check whether the file's DICOM
	#	print dirName
	#	print filename
                lstFilesDCM.append(os.path.join(dirName,filename))
                lstDirs.append(dirName)
                lstFileName.append(filename)

    resize_size = args.resize
    mean_vector = np.zeros([resize_size, resize_size])

    line_no = 0
    imglist = dict()
    for line in open(args.img_list_file, 'r'):
        item = line.split(' ')
        img_path = item[0]
        imglist["./raw/" + img_path] = line_no
        print "dict ./raw/ + img_path: ", "./raw/" + img_path
        print "line_no: ", line_no
        line_no = line_no + 1
    print "imglist: ", imglist
    
    PA_number = 0

    for i in range(len(lstFilesDCM)):
        ds = dicom.read_file(lstFilesDCM[i])
        print "processing: ", lstFilesDCM[i]
        im = get_PIL_image(ds)
        width, height = im.size
        print "origin im.size: ", im.size
        im = im.resize((resize_size,resize_size), PIL.Image.NEAREST)
        pixels = list(im.getdata())
        width, height = im.size
        print "im.size: ", im.size
        pixels = [pixels[index * width:(index + 1) * width] for index in xrange(height)]
        print "pixels asarray shape: ", np.asarray(pixels).shape
        pixels_array = np.asarray(pixels)
        print "i: ", i
        print "lstDirs[i]: ", lstDirs[i]
        transform_dir = lstDirs[i].replace("raw", "resize" + str(resize_size))
        print "transform_dir: ", transform_dir
        if not os.path.exists(transform_dir):
            os.makedirs(transform_dir)
        if pixels_array.dtype == np.int64:
            np.savetxt(transform_dir + "/" + lstFileName[i][: -4] + ".csv", pixels_array, "%d", ",")
            print "save dir: ", transform_dir + "/" + lstFileName[i][: -4] + ".csv"
        else:
            print "not int64"
        # im.save(transform_dir + "/" + lstFileName[i][: -4] + ".png")
        # print "save dir: ", transform_dir + "/" + lstFileName[i][: -4] + ".png"
        if lstFilesDCM[i] in imglist:
            PA_number = PA_number + 1
            mean_vector = mean_vector + pixels_array
            print "PA image: ", lstFilesDCM[i]

    print "imglist.keys: ", imglist.keys()
    print "PA_number: ", PA_number
    mean_vector = mean_vector / float(PA_number)
    np.savetxt(args.rawdatadir.replace("raw", "resize" + str(resize_size)) + "/mean.csv", mean_vector, "%5f", ",")
    print "save dir: ", args.rawdatadir.replace("raw", "resize" + str(resize_size)) + "/mean.csv"
