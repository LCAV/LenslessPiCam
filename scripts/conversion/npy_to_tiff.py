# This script is used to export the .mat paf from https://github.com/Waller-Lab/DiffuserCam/tree/master/example_data
# The output consists of the usable .npy file as well as tiff images for user visualisation

import os
import sys
import numpy as np
import cv2

if len(sys.argv) < 2:
    print("Error : no filename provided. Aborting.")
    sys.exit()

filename = sys.argv[1]
if not filename.endswith(".npy"):
    print("Error : file is not a .npy file. Aborting")
    sys.exit()

out_path = os.path.splitext(filename)[0]  # removing the file extension from the path, if any

data = np.load(filename).astype(np.float32)
data_shape = data.shape
l = len(data_shape)

print("\nInput shape :", data_shape)

if l == 2:
    print("As the shape has length 2, it will be interpreted as a single-layer grayscale image.")
    grayscale = True
    single_depth = True

elif l == 3:
    print("As the shape has length 3, could either be a multi-layer grayscale image or a single-layer rgb image.")
    if data_shape[2] == 3:
        print("As the third dimension of the data is 3, it will be interpreted as a single-layer rgb image"
              "with data corresponding to (width, height, color channel).")
        grayscale = False
        single_depth = True
    else:
        print("As the third dimension of the data is not, it will be interpreted as a multi-layer grayscale image "
              "with data corresponding to (depth, width, height).")
        grayscale = True
        single_depth = False

elif l == 4:
    print("As the shape has length 4, it will be interpreted as a multi-layer rgb image.")
    grayscale = False
    single_depth = False

else :
    print("Error : data shape has invalid length :", l, ", but should be 2, 3, or 4")
    grayscale = None
    single_depth = None
    quit()

if single_depth:
    if grayscale:
        if cv2.imwrite(out_path + "-out.tiff", data):
            print("Data exported succesfully in the", out_path + "-out.tiff file.")
        else :
            print("Error while exporting data in the", out_path + "-out.tiff file.")
    else:
        if cv2.imwrite(out_path + "-out.tiff", cv2.cvtColor(data.astype(np.uint8), cv2.COLOR_RGB2BGR)):
            print("Data exported succesfully in the", out_path + "-out.tiff file.")
        else:
            print("Error while exporting data in the", out_path + "-out.tiff file.")

else :
    print("As the data has several depth layers, it will be stored in the", out_path +"-out directory.")

    if os.path.exists(out_path + "-out/."):
        print("Directory already existing. The files inside will be replaced.")
    else:
        print("Directory not existing yet, it will be created.")
        os.mkdir(out_path + "-out/")

    for i in range(data_shape[0]):
        path = out_path + "-out/layer" + ("0" if i < 10 else "") + str(i) + ".tiff"
        if grayscale:
            if cv2.imwrite(path, data[i]):
                print("Data exported succesfully in the", path, "file.")
            else:
                print("Error while exporting data in the", path, "file.")
        else:
            if cv2.imwrite(path, cv2.cvtColor(data[i].astype(np.uint8), cv2.COLOR_RGB2BGR)):
                print("Data exported succesfully in the", path, "file.")
            else:
                print("Error while exporting data in the", path, "file.")



