import dicom
import os
import numpy as np
# from matplotlib import pyplot, cm
# import Tkinter
import sys

# PathDicom = "./"
# lstFilesDCM = []  # create an empty list
# for dirName, subdirList, fileList in os.walk(PathDicom):
#    for filename in fileList:
#        if ".dcm" in filename.lower():  # check whether the file's DICOM
#            lstFilesDCM.append(os.path.join(dirName,filename))

# Get ref file
img_list = []
for line in open(sys.argv[1], 'r'): #get img_list_file
    item = line.split(' ')
    img_path = item[0]
    img_label = int(item[1])
    img_list.append((img_label, img_path))

img_size = []
for i in range(len(img_list)):
    img_label, img_path = img_list[i]
    print "img_path: ", img_list[i][1]
    print "os.path.join(self.image_folder, img_path): ", os.path.join('./', img_path)
    RefDs = dicom.read_file(os.path.join('./', img_path))
    ImgRows = int(RefDs.Rows)
    img_size.append(ImgRows)
    print "img_size: ", img_size[i]
print "img_size: ", img_size
print "img_size numpy: ", np.asarray(img_size)
print "img_size unique values: ", np.unique(np.asarray(img_size), return_index = True, return_inverse=True)
