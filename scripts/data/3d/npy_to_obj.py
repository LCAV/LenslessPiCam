import sys
import numpy as np
from lensless.util import resize3d

if len(sys.argv) < 2:
    print("Error : no filename provided. Aborting.")
    sys.exit()

filename = sys.argv[1]
if not filename.endswith(".npy"):
    print("Error : file is not a .npy file. Aborting")
    sys.exit()

data = np.load(filename)

#sum color channels for now
if len(data.shape) == 4:
    data = np.sum(data, axis=3)

factor = 1.0/float(sys.argv[2]) if len(sys.argv) > 2 else 1
data = resize3d(data, factor)


if np.max(data) > 0:
    data = data / np.max(data)
else :
    print("Error : data has no positive value. Aborting.")
    sys.exit()

# default value of mean^0.5 is an heuristic
threshold = pow(np.mean(data), 0.5) if len(sys.argv) == 2 else sys.argv[2]
print("threshold : ", threshold)

data_shape = data.shape

output_file = open(filename.replace(".npy", ".obj"), "w")

i = 0
for z in range(data_shape[0]):
    print("converting depth layer ", z+1, "/", data_shape[0])
    for x in range(data_shape[1]):
        for y in range(data_shape[2]):
            v = data[z, x, y]
            if v >= threshold:
                v = v/2

                output_file.write(
                    "v " + str(x) + " " + str(y) + " " + str(z-v) + "\n" +
                    "v " + str(x) + " " + str(y) + " " + str(z+v) + "\n" +
                    "v " + str(x) + " " + str(y-v) + " " + str(z) + "\n" +
                    "v " + str(x) + " " + str(y+v) + " " + str(z) + "\n" +
                    "v " + str(x-v) + " " + str(y) + " " + str(z) + "\n" +
                    "v " + str(x+v) + " " + str(y) + " " + str(z) + "\n" +
                    "\n" +
                    "f " + str(i+1) + " " + str(i+3) + " " + str(i+5) + "\n" +
                    "f " + str(i+1) + " " + str(i+3) + " " + str(i+6) + "\n" +
                    "f " + str(i+1) + " " + str(i+4) + " " + str(i+5) + "\n" +
                    "f " + str(i+1) + " " + str(i+4) + " " + str(i+6) + "\n" +
                    "f " + str(i+2) + " " + str(i+3) + " " + str(i+5) + "\n" +
                    "f " + str(i+2) + " " + str(i+3) + " " + str(i+6) + "\n" +
                    "f " + str(i+2) + " " + str(i+4) + " " + str(i+5) + "\n" +
                    "f " + str(i+2) + " " + str(i+4) + " " + str(i+6) + "\n" +
                    "\n\n"
                )
                i += 6


output_file.close()


