import dicom
import os
import numpy
from matplotlib import pyplot, cm
import Tkinter

PathDicom = "./"
lstFilesDCM = []  # create an empty list
for dirName, subdirList, fileList in os.walk(PathDicom):
    for filename in fileList:
        if ".dcm" in filename.lower():  # check whether the file's DICOM
            lstFilesDCM.append(os.path.join(dirName,filename))

# Get ref file
RefDs = dicom.read_file("./dcm_subimage/I00000009316.dcm")

f1 = open('testfile', 'w+')

print >> f1, "RefDs: \n", RefDs
# Load dimensions based on the number of rows, columns, and slices (along the Z axis)
ConstPixelDims = (int(RefDs.Rows), int(RefDs.Columns), len(lstFilesDCM))
print "\n !!!!!!!!!!!!should write ouput to file!!!!!!!!!!\n"
# Load spacing values (in mm)
#ConstPixelSpacing = (float(RefDs.PixelSpacing[0]), float(RefDs.PixelSpacing[1]), float(RefDs.SliceThickness))

#x = numpy.arange(0.0, (ConstPixelDims[0]+1)*ConstPixelSpacing[0], ConstPixelSpacing[0])
#y = numpy.arange(0.0, (ConstPixelDims[1]+1)*ConstPixelSpacing[1], ConstPixelSpacing[1])
#z = numpy.arange(0.0, (ConstPixelDims[2]+1)*ConstPixelSpacing[2], ConstPixelSpacing[2])

# The array is sized based on 'ConstPixelDims'
ArrayDicom = numpy.zeros(ConstPixelDims, dtype=RefDs.pixel_array.dtype)

ArrayDicom[:, :, 0] = RefDs.pixel_array
print "RefDs.pixel_array shape: \n", RefDs.pixel_array.shape
# loop through all the DICOM files
#for filenameDCM in lstFilesDCM:
    # read the file
#    print "filenameDCM = \n", filenameDCM
#    ds = dicom.read_file(filenameDCM)
    # store the raw image data
#    ArrayDicom[:, :, lstFilesDCM.index(filenameDCM)] = ds.pixel_array  


pyplot.figure(dpi=300)
pyplot.axes().set_aspect('equal', 'datalim')
pyplot.set_cmap(pyplot.gray())
pyplot.pcolormesh(x, y, numpy.flipud(ArrayDicom[:, :, 80]))
