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

for i in range(len(lstFilesDCM)):
    # Get ref file
    RefDs = dicom.read_file(lstFilesDCM[i])
    print "filename: ", lstFilesDCM[i]
    print "View position: ", RefDs.ViewPosition
    print "AcquisitionDeviceProcessingDescription: ", RefDs.AcquisitionDeviceProcessingDescription
    print RefDs.ViewPosition == 'PA'
    print 'postero' in RefDs.AcquisitionDeviceProcessingDescription
    print "\n\n"
