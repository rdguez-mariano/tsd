import glob
import numpy as np
paths = ['/home/rdguez-mariano/workspace/github/cloud-detection/datasets/multipleImage/combined', 
        '/home/rdguez-mariano/workspace/github/cloud-detection/datasets/singleImage/combined']

tiles = []
for path in paths:
    for img in glob.glob(path+"/*/*.jpg"):
        tiles.append( img.split("/")[-1].split("_")[0] )

tiles = np.unique(tiles)

np.save("tiles.npy",tiles)

tiles2 = np.load("tiles.npy")


print(len(tiles),tiles[0:10])
print(len(tiles2),tiles2[0:10])
        