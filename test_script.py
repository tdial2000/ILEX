from ilex.frb import FRB
from ilex.data import *
from ilex.plot import _PLOT
import matplotlib.pyplot as plt
import numpy as np
import sys


frb = FRB(yaml_file = "/fred/oz002/tdial/HTR_paper_data/230708/230708_updated.yaml")

# data1 = frb.get_data(['tI'], get = True, apply_tW = False, t_crop = [1731.29, 1757.73])
data2 = frb.get_data(['tI'], get = True, t_crop = [1731.292, 1734.014], apply_tW = False)
# data3 = frb.get_data(['tI'], get = True, t_crop = [1731.292, 1734.014], apply_tW = False, tN = 30)
# data2 = frb.get_data(['tI'], t_crop = [1737.291, 1744.345], get = True, apply_tW = False)
# data3 = frb.get_data(['tI'], t_crop = [1741.432452151, 1746.5642356235], get = True, apply_tW = False)

# print(data2['time'])
# print(data3['time'])

# print(frb)

# plt.figure(figsize = (10,10))

# plt.plot(data1['time'], data1['tI'], alpha = 0.3)
# plt.plot(data2['time'], data2['tI'], alpha = 0.3)
# plt.plot(data3['time'], data3['tI'], alpha = 0.3)

# print(data1['time'])
print(data2['time'])
# print(data3['time'])

# plt.show()
