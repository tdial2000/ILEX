from ilex.data import *
from time import time


data = np.load("/fred/oz002/tdial/HTR_paper_data/230708/230708_calib_I.npy", mmap_mode = "r")

t1 = time()
data1 = pslice(data, *[0.4, 0.5], axis = 1)
data2 = pslice(data1, *[0.1, 0.9], axis = 0)

print(f"Time taken: {time() - t1} s")