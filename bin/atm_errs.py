"""This script calculates expected atmospheric errors."""
import galsim
import numpy as np
import pandas as pd
from lsst.ts.phosim.utils.ConvertZernikesToPsfWidth import convertZernikesToPsfWidth
from lsst.ts.wep.cwfs.instrument import Instrument
from scipy.spatial.distance import cdist
from scipy.special import gamma, kv

# first set the observational parameters

# first determine the mean Fried parameter of my simulations
obs_table = pd.read_parquet(
    "/astro/store/epyc/users/jfc20/data/aos_sims/observations.parquet"
)
wavelens = {
    band: galsim.Bandpass(f"LSST_{band}.dat", wave_type="nm").effective_wavelength / 1e9
    for band in "ugrizy"
}
psf = obs_table["seeingFwhm500"] * obs_table["airmass"] ** (3 / 5)  # arcsec
r0 = 1.22 * obs_table["lsstFilter"].replace(wavelens) / np.deg2rad(psf / 3600)
r0 = r0.to_numpy()

# now determine mean outer scale of von Karman turbulence (m)
# L0 distribution taken from wfsim.atm
rng = np.random.default_rng(0)
L0 = np.exp(rng.normal(size=10_000_000) * 0.6 + np.log(25))
L0 = L0[(L0 > 10) & (L0 < 100)]
L0 = L0[: len(r0)]

# finally, determine the number of atmosphere realizations to average over
# v distribution taken from wfsim.atm
T = 15  # exposure time, seconds
v = rng.uniform(0, 20, size=len(r0))  # wind velocity in dominant layer, m/s
t0 = 0.31 * r0 / v
N = np.clip(T / t0, 1, None)  # number of independent atmosphere realizations


# get the instrument object for LsstCam
inst = Instrument()
inst.configFromFile(dimOfDonutImgOnSensor=125, camType=1)

# get geometry info
R = inst.apertureDiameter / 2
eps = 0.61  # M1 obscuration ratio
area = np.pi * R**2 * (1 - eps**2)
xSensor, ySensor = inst.getSensorCoor()
dOmega = (inst.apertureDiameter * inst.getSensorFactor() / inst.dimOfDonutImg) ** 2

# calculate Zernikes
jmax = 22  # maximum number of Zernikes
zk = galsim.zernike.zernikeBasis(jmax, xSensor, ySensor, R_inner=eps)[4:]

# set everything outside pupil = 0
mask = (np.sqrt(xSensor**2 + ySensor**2) > eps) & (
    np.sqrt(xSensor**2 + ySensor**2) < 1
)
zk = zk * mask


# calculate the distance matrix, r
x = R * xSensor
y = R * ySensor
points = np.hstack([x.reshape(-1, 1), y.reshape(-1, 1)])
r = cdist(points, points).reshape((*x.shape, *x.shape))


# store unique values of r
r_unique, idx_unique = np.unique(r, return_inverse=True)


# this wrapper function takes the function f, applies it to each unique value in
# r, and then uses those values as a dictionary to fill in the rest of the output
def apply_unique(f):
    return np.array([f(ru) for ru in r_unique])[idx_unique].reshape(r.shape)


def kolmogorov(r0):
    return apply_unique(lambda r: 6.88 * (r / r0) ** (5 / 3))


def vonKarman(r0, L0):
    with np.errstate(invalid="ignore"):
        dvk = apply_unique(
            lambda r: np.nan_to_num(
                6.88
                * (r / r0) ** (5 / 3)
                / (np.pi * r / L0) ** (5 / 3)
                / gamma(-5 / 6)
                * (
                    2 * (np.pi * r / L0) ** (5 / 6) * kv(5 / 6, 2 * np.pi * r / L0)
                    - gamma(5 / 6)
                ),
                nan=0,
                posinf=0,
                neginf=0,
            )
        )
    return dvk


def calculate_errors(r0=0.20, L0=None, N=1):
    if L0 is None:
        D = kolmogorov(r0)
    else:
        D = vonKarman(r0, L0)

    atm_cov = -np.einsum(
        "jab,kcd,abcd->jk",
        zk * dOmega,
        zk * dOmega,
        D,
        optimize="optimal",
    ) / (2 * area**2 * N)

    return convertZernikesToPsfWidth(np.sqrt(np.diag(atm_cov)))


def calculate_total_error(r0=0.20, L0=None, N=1):
    errs = calculate_errors(r0=r0, L0=L0, N=N)
    return np.sqrt(np.sum(errs**2))


err_per_zk = []
rng = np.random.default_rng(42)
for i in range(10_000):
    err_per_zk.append(
        calculate_errors(r0=rng.choice(r0), L0=rng.choice(L0), N=rng.choice(N))
    )

err_per_zk = np.array(err_per_zk)  # type: ignore
np.save("../data/atm_errs.npy", err_per_zk)
