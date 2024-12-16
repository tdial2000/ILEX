# imports
from cosmology import cosmology
import numpy as np
import argparse




def get_args():
    """
    Get arguments for frb energetics script

    """

    parser = argparse.ArgumentParser(description = "Calculate FRB burst energetics assuming Lambda-CDM cosmology (flat). Default cosmological parameters from Plank 2018 results.")

    # FRB arguments
    parser.add_argument('-z', help = "Redshift of FRB", type = float, default = 0.0)
    parser.add_argument('--fluence', help = "FRB fluence [Jy ms]", type = float, required=True)
    parser.add_argument('--bw', help = "Bandwidth [MHz]", type = float, default = 336)
    parser.add_argument('--width', help = "width of FRB [ms]", type = float, default = 1.0)
    parser.add_argument("--lumin_D", help = "Luminosity distance [Mpc] (in case of near objects - galactic), will overide redshift (z)", type = float, default = None)
    
    # cosmology arguments
    parser.add_argument('--omega_m', help = "Matter density parameter", type = float, default = 0.315)
    parser.add_argument('--omega_vac', help = "Dark Energy density", type = float, default = 0.685)
    parser.add_argument('--H0', help = "Hubbles Constant [km/s/Mpc]", type = float, default = 67.4)

    return parser.parse_args()





def frb_energetics(args):
    """
    Get FRB energetics

    """

    # get Luminosity distance
    
    lambda_CDM = cosmology(omega_m = args.omega_m, omega_vac = args.omega_vac,
                           H0 = args.H0, z = args.z)

    if args.lumin_D is None:
        lumin_D = lambda_CDM.lumin_D        # in [m]
    else:
        lumin_D = args.lumin_D * lambda_CDM.Mpc2m

    # calculate spectral density
    spec_density = args.fluence * 1e-19 * 4 * np.pi * lumin_D**2 / 1000

    # calculate Spectral luminosity       in [ergs/s/Hz]
    spec_lumin = spec_density / (args.width / 1000)

    # calculate luminosity
    lumin = spec_lumin * (args.bw * 1e6)

    # calculate Total Energy
    energy = lumin * (args.width / 1000)

    # Now print everything out!

    print(f"\nFRB parameters:")
    print("="*16 + "\n")

    print("Fluence:".ljust(30) + f"{args.fluence:.4f}".ljust(15) + "[Jy ms]")
    print("Bandwidth:".ljust(30) + f"{args.bw:.4f}".ljust(15) + "[MHz]")
    print("Width:".ljust(30) + f"{args.width:.4f}".ljust(15) + "[ms]")
    print("Redshift (z):".ljust(30) + f"{args.z:.4f}\n")

    print(f"FRB Energetics: Assuming Lambda-CDM Cosmology with Omega_m = {args.omega_m:.3f}, Omega_vac = {args.omega_vac:.3f}, H0 = {args.H0:.3f} [km/s/Mpc]")
    print("="*110 + "\n")

    print("Luminosity Distance:".ljust(30) + f"{lumin_D / lambda_CDM.Mpc2m:.4f}".ljust(15) + "[Mpc]")
    print("Spectral Luminosity:".ljust(30) + f"{spec_lumin:.4E}".ljust(15) + "[ergs/s/Hz]")
    print("Spectral Density:".ljust(30) + f"{spec_density:.4E}".ljust(15) + "[ergs/Hz]")
    print("Luminosity:".ljust(30) + f"{lumin:.4E}".ljust(15) + "[ergs/s]")
    print("Total Energy:".ljust(30) + f"{energy:.4E}".ljust(15) + "[ergs]\n")









if __name__ == "__main__":

    # main block of code

    # get arguments
    args = get_args()

    if args.z == 0.0 and args.lumin_D is None:
        print("Must provide a non-zero redshift or a luminosity Distance")

    # calculate FRB energetics
    frb_energetics(args)
