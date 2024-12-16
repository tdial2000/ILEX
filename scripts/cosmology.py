##################################################
# Author:   Tyson Dial                           #
# Email:    tdial@swin.edu.au                    #
# Date (created):     15/12/2024                 #
# Date (updated):     15/12/2024                 #
##################################################
# library for basic cosmology calculations such  #          
# as distance and travel time                    #
# Allows for radiation density, curvature        #
# different forms of dark energy including       #
# cosmological constant, W not -1 and a handful  #
# of quintessence parameteric models.            #
##################################################

# imports
import numpy as np
from math import sin, sinh

def integrate_midpoint(func, start, stop, N):
    """
    Integrate a function using the midpoint algorithm

    """

    x = np.linspace(start, stop, N)

    y = func(x)

    dx = x[1] - x[0]

    return np.sum(y) * dx



# DE functions
def de_eos_constant_w(a, w, omega_vac):

    return omega_vac * a**(-3 * (1 + w))


# Linder, E.V., Exploring the expansion history of the universe. Physical Review Letters, 2003. 
# 90(9): p. 091301.
def de_eos_CPL(a, w, wa, omega_vac):

    return omega_vac * a**(-3*(1 + w + wa)) * np.exp(3*wa*(a - 1))


# Tripathi, A., A. Sangwan, and H. Jassal, Dark energy equation of state parameter and its 
# evolution at low redshift. Journal of Cosmology and Astroparticle Physics, 2017. 2017(06): p. 
# 012.
def de_eos_JBP(a, w, wa, omega_vac):

    return omega_vac * a**(-3*(1+w)) * np.exp(1.5*wa*(a - 1)**2)

def de_eos_log(a, w, wa, omega_vac):

    return omega_vac * a**(-3*(1+w)) * np.exp(1.5*wa*np.log(a)**2)






# cosmology class

class cosmology:

    c = 299_792_458             # m/s
    pc2m = 3.08567758128e16
    Mpc2m = pc2m * 1e6
    pars = ['omega_m', 'omega_vac', 'K', 'omega_r', 'H0', 'w', 
            'wa', 'N', 'de_eos', 'z']    
    H0_units = 3.24078725e-20


    def __init__(self, **kwargs):

        # define variables
        self.omega_m =      0.3     # matter density
        self.omega_vac =    0.7     # DE density
        self.K =      0.0           # curvature density
        self.omega_r =      0.0     # radiation density
        self.H0 =           67.4    # Hubbles Constant [km/s/Mpc]

        # redshift
        self.z =            0.1

        # Dark Energy parameters
        self.w =            -1.0    # DE Equation of State (EOS) w parameter
        self.wa =            0.0    # First order EOS w parameter 
        self.de_eos =        "constant"

        # numerical intergration parameters
        self.N = 1_000_000
        
        # set params
        self.set(**kwargs)


    def set(self, **kwargs):

        for key in kwargs.keys():
            if key in self.pars:
                setattr(self, key, kwargs[key])
        
    
        return



    @property
    def omega_k(self):

        return -self.K * self.c**2/(self.H0 * self.H0_units)**2


    
    @property
    def r(self):

        # calculate DE density as a function of scale factor a
        if self.de_eos == "constant":
            omega_de = lambda a : de_eos_constant_w(a = a, w = self.w, omega_vac = self.omega_vac)
        elif self.de_eos == "CPL":
            omega_de = lambda a : de_eos_CPL(a = a, w = self.w, wa = self.wa, 
                                    omega_vac = self.omega_vac)
        elif self.de_eos == "JBP":
            omega_de = lambda a : de_eos_CPL(a = a, w = self.w, wa = self.wa,
                                    omega_vac = self.omega_vac)
        elif self.de_eos == "log":
            omega_de = lambda a : de_eos_log(a = a, w = self.w, wa = self.wa,
                                    omega_vac = self.omega_vac)
        else:
            raise ValueError(f"DE EOS [{self.de_eos}] not supported.")
        
        # co-moving (before curvature correction)
        r_func = lambda a : (self.c/(self.H0 * self.H0_units) / 
                            np.sqrt(self.omega_m * a + self.omega_k * a**2 + omega_de(a) * a**4 + self.omega_r))

        # get co-moving r (before curvature) and proper time
        return integrate_midpoint(r_func, 1/(1+self.z), 1, self.N)


    @property
    def t(self):
        # calculate DE density as a function of scale factor a
        if self.de_eos == "constant":
            omega_de = lambda a : de_eos_constant_w(a = a, w = self.w, omega_vac = self.omega_vac)
        elif self.de_eos == "CPL":
            omega_de = lambda a : de_eos_CPL(a = a, w = self.w, wa = self.wa, 
                                    omega_vac = self.omega_vac)
        elif self.de_eos == "JBP":
            omega_de = lambda a : de_eos_CPL(a = a, w = self.w, wa = self.wa,
                                    omega_vac = self.omega_vac)
        elif self.de_eos == "log":
            omega_de = lambda a : de_eos_log(a = a, w = self.w, wa = self.wa,
                                    omega_vac = self.omega_vac)
        else:
            raise ValueError(f"DE EOS [{self.de_eos}] not supported.") 


        # proper time
        t_func = lambda a : (1/(self.H0 * self.H0_units) / 
                            np.sqrt(self.omega_m/a + self.omega_k + omega_de(a) * a**2 + self.omega_r/a**2))

        return integrate_midpoint(t_func, 1/(1+self.z), 1, self.N)


    @property
    def lumin_D(self):

        # correct with scale factor
        r_scaled = self.r * (1+self.z)

        # correct with curvature
        if self.K == 0.0:
            return r_scaled
        elif self.K > 0.0:
            return 1/self.K**0.5 * sin(self.K**0.5 * r_scaled)
        else:
            return 1/(-self.K)**0.5 * sinh((-self.K)**0.5 * r_scaled)

    