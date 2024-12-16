# imports
import numpy as np 
from cosmology import cosmology
import argparse


def get_args():
    """
    Get arguments for calculating cosmological distance
    """

    parser = argparse.ArgumentParser(description = "Calculate distance and light travel time of a certain cosmological history")

    parser.add_argument("--omega_m", help = "Matter density", type = float, default = 0.315)
    parser.add_argument("--omega_vac", help = "Vaccum DE density", type = float, default = 0.685)
    parser.add_argument("--H0", help = "Hubble Constant [km/s/Mpc]", type = float, default = 67.4)
    parser.add_argument("--K", help = "Curvature parameter", type = float, default = 0.0)
    parser.add_argument("--omega_r", help = "Radiation density", type = float, default = 0.0)
    parser.add_argument("--w", help = "DE EOS w parameter (zeroth order)", type = float, default = -1.0)
    parser.add_argument("--wa", help = "DE EOS w(t) parameter (first order)", type = float, default = 0.0)
    parser.add_argument("--de_eos", help = "DE EOS quintessence model, default is constant w", type = str, default = "constant")

    # other
    parser.add_argument("-z", help = "redshift", type = float, required = True)
    parser.add_argument("-N", help = "Number of samples, for numerical integration", type = int, default = 1_000_000)

    return parser.parse_args()







def calculate_cosmo_history(args):

    # 
    cosmo = cosmology(omega_m = args.omega_m, omega_vac = args.omega_vac, H0 = args.H0,
                      K = args.K, omega_r = args.omega_r, w = args.w, wa = args.wa,
                      de_eos = args.de_eos, z = args.z, N = args.N)

    # calculate parameters
    dL = cosmo.lumin_D
    r = cosmo.r 
    t = cosmo.t 

    print(f"At a redshift of z = {args.z:.4f}")
    print(f"Luminosity distance: {dL / cosmo.Mpc2m:.6f} Mpc")
    print(f"Co-moving radial distance: {r / cosmo.Mpc2m:.6f} Mpc")
    print(f"Light travel time: {t / (1e9*31557600):.6f} Gyr")





if __name__ == "__main__":
    # Main block of code

    # get args
    args = get_args()

    # history!
    calculate_cosmo_history(args)