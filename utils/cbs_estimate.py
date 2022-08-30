import numpy as np
import math

def erf_func(a, b, x):
    """
    Defines error function used to approximate pairwise BSSE
    args: a, b, (floats, parameters dependent on basis set), x (float, pairwise distance in angstrom)
    return: y (float, estimated BSSE correction for pair of water molecules in kcal/mol)
    """
    y = a*(1+math.erf(-b*x))
    return y

def get_mp2_cbs_est(energy, coords, basis_set='aVTZ'):
    """
    Takes in uncorrected MP2 energy, estimates BSSE-correction, and uses both to get MP2/CBS estimate
    args: uncorrected MP2 binding energy in kcal/mol (cluster_e-n*monomer_e),
    numpy array of coordinates (n_atomsx3), basis set used for calculations
    return: MP2/CBS estimate of the energy in kcal/mol
    """
    bsse_corr = 0
    if basis_set == 'aVDZ':
        a = 13.133
        b = 0.4542
        o_coords = coords[::3]
        for i in range(0, np.shape(o_coords)[0]-1):
            for j in range(i+1, np.shape(o_coords)[0]):
                x = np.linalg.norm(o_coords[i, :]-o_coords[j, :])
                y = erf_func(a, b, x)
                bsse_corr += y
        mp2_cbs = energy + 1/3*bsse_corr
        
    elif basis_set == 'aVTZ':
        a = 9.444
        b = 0.4887
        o_coords = coords[::3]
        for i in range(0, np.shape(o_coords)[0]-1):
            for j in range(i+1, np.shape(o_coords)[0]):
                x = np.linalg.norm(o_coords[i, :]-o_coords[j, :])
                y = erf_func(a, b, x)
                bsse_corr += y
        mp2_cbs = energy + 1/2*bsse_corr
    else:
        raise ValueError('Basis set not implemented')

    return mp2_cbs
