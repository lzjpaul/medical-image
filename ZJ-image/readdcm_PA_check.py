import dicom
import os
import numpy
from matplotlib import pyplot, cm
import Tkinter
import sys

def find_positive(PathDicom):
    lstFilesDCM = []  # create an empty list
    for dirName, subdirList, fileList in os.walk(PathDicom):
        for filename in fileList:
            if ".dcm" in filename.lower():  # check whether the file's DICOM
                lstFilesDCM.append(os.path.join(dirName,filename))

    for i in range(len(lstFilesDCM)):
        # Get ref file
        RefDs = dicom.read_file(lstFilesDCM[i])
        if RefDs.ViewPosition == 'PA' or 'postero' in RefDs.AcquisitionDeviceProcessingDescription:
            return True
    return False

print find_positive(sys.argv[1])
