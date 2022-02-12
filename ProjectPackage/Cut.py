import awkward as ak
import numpy as np


def z_cut(events: ak.Array, z_lim=10):
    return events[np.abs(events.zVtx) < z_lim]


def radial_coord(x, y, z):
    """
    Compute the distance to the origin
    """
    return np.sqrt(x**2 + y**2 + z**2)


def p_fc(P, thetaAbs: float):
    """
    Return the momentum at the first chamber touched of the spectrometer i.e without correction of the absorption
    :param P: 3D vector (numpy array)
    :param thetaAbs: angle in degrees
    :return: 3D vector (numpy array)
    """
    corr = -3 if thetaAbs < 3 else -2.4   # average correction due to MSCs

    return P + corr


def pxDCA(P, DCA):

    return P * DCA


def sigma_abs(thetaAbs):
    return 80. if thetaAbs < 3 else 54.


def sigma_p(P, thetaAbs, N=1,  delta_p=0.0004):
    if N > 10 or N < 0:
        print("Wrong value of N")
    a = N * delta_p * P
    den = 1 - (a / (1 + a))
    return sigma_abs(thetaAbs) / den


def sigma_theta(P, delta_theta=0.0005):
    return 535 * delta_theta * P


def sigma_pxDCA(P: float, thetaAbs: float, N=1) -> float:

    return np.sqrt(sigma_p(P, thetaAbs, N)**2 + sigma_theta(P, thetaAbs)**2)


def DCA(x, y, z):
    return np.sqrt(x**2 + y**2, +z**2)

