# imports

from ilex.frb import FRB
import matplotlib.pyplot as plt 


# create FRB instance
frb = FRB(yaml_file = "/fred/oz002/tdial/HTR_paper_data/230708/analysis/230708FULL.yaml")

# things to test
# 1. Test cropping and data plotting
frb.plot_data("tI")


# 2. Test stokes plotting in time and frequency
frb.plot_stokes(stk_type = 't')
frb.plot_stokes(stk_type = 'f')

# 2.5 Test plotting stokes t with ratios
frb.plot_stokes(stk_type = "t", stk_ratio = True, sigma = 10.0)


# 3. Fit RM
frb.fit_RM(method = "RMsynth", verbose = True)


# 4. fit scintillation
frb.fit_scintband(method = "least squares", t_crop = [2124.552, 2127.274])


# 5. plot PA
frb.plot_PA(plot_L = True)


# 6. Print out information about frb
print(frb)

