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
        image = get_LUT_value(dataset.pixel_array, int(dataset.WindowWidth[0]),
                              int(dataset.WindowCenter[0]))
        # Convert mode to L since LUT has only 256 values:
        #   http://www.pythonware.com/library/pil/handbook/image.htm
        im = PIL.Image.fromarray(np.uint8(image)).convert('L')
        
    return im


def show_PIL(dataset):
    """Display an image using the Python Imaging Library (PIL)"""
    im = get_PIL_image(dataset)
    im.show()
    im.save("test2.png")

if __name__ == '__main__':
    ds = dicom.read_file(sys.argv[1])
    show_PIL(ds)
    print ds.pixel_array
    print ds.pixel_array.shape
# other operations
    im = get_PIL_image(ds)
    print "print Im.mode,Im.size,Im.format: ", im.mode,im.size,im.format
    pixels = list(im.getdata())
    print "pixels shape: ", len(pixels)
    print "pixels asarray shape: ", np.asarray(pixels).shape
    width, height = im.size
    pixels = [pixels[i * width:(i + 1) * width] for i in xrange(height)]
    print "pixels shape: ", len(pixels)
    print "pixels[0]: ", len(pixels[0])
    print "pixels asarray shape: ", np.asarray(pixels).shape
    im = im.resize((128,128), PIL.Image.NEAREST)
    im.save("resize128.png")
    pixels = list(im.getdata())
    print "pixels shape: ", len(pixels)
    print "pixels asarray shape: ", np.asarray(pixels).shape
    width, height = im.size
    pixels = [pixels[i * width:(i + 1) * width] for i in xrange(height)]
    print "pixels shape: ", len(pixels)
    print "pixels[0]: ", len(pixels[0])
    print "pixels asarray shape: ", np.asarray(pixels)[0] 
