"""Class for calculating atmosphere statistics."""
from copy import deepcopy
from typing import Optional, Union

import galsim
import numpy as np
from scipy import optimize
from scipy.integrate import simps
from scipy.spatial.distance import cdist
from scipy.special import gamma, kv


def cp_profile(
    n_gl: int = 10,
    gl_quality: str = "typical",
    fa_quality: str = "typical",
    gl_hmin: float = 30,
    gl_hmax: float = 400,
) -> np.ndarray:
    """Generate integrated Cn2 profile for Cerro Pachon.

    Generates a discrete layer turbulence profile with a variable number of ground
    layers. These are chosen to match the models from Tokovinin 2006:
    https://academic.oup.com/mnras/article/365/4/1235/992742

    Parameters
    ----------
    n_gl: int, default=10
        The number of ground layers.
    gl_quality: str, default="typical"
        Quality of seeing in the ground layer. Can be "typical", "good", or "bad".
    fa_quality: str, default="typical"
        Quality of seeing in the free air above the ground layer.
        Can be "typical", "good", or "bad".
    gl_hmin: float, default=30
        The minimum height of the ground layer. The default 30m is about the height
        of the dome.
    gl_hmax: float, default=400
        The maximum height of the ground layer. The default 400m is set to be just
        under the free-air layers, which start at 500m.

    Returns
    -------
    np.ndarray
        2D array where the first row is altitudes in km, and the second is the
        integrated Cn2 value for each layer, in units of 1e-13 m^(1/3).
    """
    # set parameters for the ground layer model
    if gl_quality == "good":
        A = 70
        h0 = 15
        B = 0.4
        h1 = 700
    elif gl_quality == "typical":
        A = 70
        h0 = 20
        B = 1.4
        h1 = 900
    elif gl_quality == "bad":
        A = 60
        h0 = 100
        B = 2.0
        h1 = 1500

    # setup bins for integration
    gl_h_bins = np.linspace(gl_hmin, gl_hmax, n_gl + 1)
    gl_h = (gl_h_bins[:-1] + gl_h_bins[1:]) / 2

    # analytic integration
    gl_cn2_edges = h0 * A * np.exp(-gl_h_bins / h0) + h1 * B * np.exp(-gl_h_bins / h1)
    gl_cn2 = 1e-3 * (gl_cn2_edges[:-1] - gl_cn2_edges[1:])

    # now the free air
    fa_h = 1e3 * np.array([0.5, 1, 2, 4, 8, 16])
    if fa_quality == "good":
        fa_cn2 = np.array([0.2, 0.03, 0.02, 0.2, 0.15, 0.25])
    elif fa_quality == "typical":
        fa_cn2 = np.array([0.4, 0.1, 0.1, 0.4, 0.2, 0.3])
    elif fa_quality == "bad":
        fa_cn2 = np.array([0.7, 0.2, 0.4, 0.6, 0.3, 0.3])

    # combine ground layers and free air into same profile
    h = np.append(gl_h, fa_h) / 1e3
    cn2 = np.append(gl_cn2, fa_cn2)

    return np.vstack((h, cn2))


class AtmStat:
    """Object that computes atmosphere statistics."""

    def __init__(
        self,
        seeing: float = 0.67,
        seeing_type: str = "fwhm",
        zenith: float = 30,
        wavelength: Union[float, str] = "r",
        L0: float = 30,
        v: float = 10,
        T: float = 15,
        theta: float = 0,
        Cn2: np.ndarray = cp_profile(10),
        D: float = 8.36,
        eps: float = 0.61,
        jmax: int = 22,
        pupil_N: int = 125,
    ) -> None:
        """
        Parameters
        ----------
        seeing: float, default=0.67
            The 500 nm seeing at zenith, in arcseconds. The meaning of this value
            is determined by the seeing_type parameter.
        seeing_type: str, default="fwhm"
            If "fwhm", the seeing parameter is interpreted as the delivered PSF FWHM
            at zenith. If "dimm", the seeing parameter is interpreted as the angle
            corresponding to the Fried parameter measured by a DIMM, assuming
            Kolmogorov turbulence. In other words, if the seeing value is measured
            from images, you should use "fwhm", and if it's measured using a DIMM,
            you should use "dimm".
        zenith: float, default=30
            Zenith angle, in degrees.
        wavelength: float or string, default="r"
            Effective wavelength of the observation, in meters. You can also supply
            a string specifying an LSST band, in which case the effective wavelength
            of that band will be used.
        L0: float, default=30
            Outer scale of turbulence, in meters. If infinite or non-positive,
            Kolmogorov turbulence is assumed. Otherwise, von Karman turbulence.
        v: float, default=10
            Wind velocity in the dominant layer, in meters per second.
        T: float, default=15
            Exposure time, in seconds.
        theta: float, default=0
            Separation angle between the two sources, in degrees.
        Cn2: np.ndarray, optional
            Turbulence profile in a 2D array where the first row is altitude in km,
            and the second is the integrated Cn^2 value for each layer, in units of
            1e-13 m^(1/3). The overall normalization doesn't matter, as the profile
            will be normalized to sum to 1. Note these values are only relevant
            when theta != 0.
            Default values are taken from Tokovinin 2006:
            https://academic.oup.com/mnras/article/365/4/1235/992742
        D: float, default=8.36
            Diameter of the pupil, in meters.
        eps: float, default=0.61
            Fractional obscuration of the pupil.
        jmax: int, default=22
            The maximum Noll index for the Zernikes.
        pupil_N: int, default=125
            The pupil is discretized into a grid of pupil_N x pupil_N points.
            Increasing this number increases accuracy at the expense of computation
            time. The standard deviation of Zernikes 4-28 has mostly converged by
            125. Lower numbers may be desired for faster calculation, for a small
            sacrifice in accuracy.
        """
        self._params: dict = {}
        self.set_params(
            seeing=seeing,
            seeing_type=seeing_type,
            zenith=zenith,
            wavelength=wavelength,
            L0=L0,
            v=v,
            T=T,
            theta=theta,
            Cn2=Cn2,
            D=D,
            eps=eps,
            jmax=jmax,
            pupil_N=pupil_N,
        )

    def set_params(
        self,
        seeing: Optional[float] = None,
        seeing_type: Optional[str] = None,
        zenith: Optional[float] = None,
        wavelength: Optional[Union[float, str]] = None,
        L0: Optional[float] = None,
        v: Optional[float] = None,
        T: Optional[float] = None,
        theta: Optional[float] = None,
        Cn2: Optional[np.ndarray] = None,
        D: Optional[float] = None,
        eps: Optional[float] = None,
        jmax: Optional[int] = None,
        pupil_N: Optional[int] = None,
    ) -> None:
        """Set the observation and atmospheric parameters.

        For default values, look at the docstring for __init__.

        Parameters
        ----------
        seeing: float, optional
            The 500 nm PSF FWHM at zenith, in arcseconds.
        seeing_type: str, optional
            If "fwhm", the seeing parameter is interpreted as the delivered PSF FWHM
            at zenith. If "dimm", the seeing parameter is interpreted as the angle
            corresponding to the Fried parameter measured by a DIMM, assuming
            Kolmogorov turbulence. In other words, if the seeing value is measured
            from images, you should use "fwhm", and if it's measured using a DIMM,
            you should use "dimm".
        zenith: float, optional
            Zenith angle, in degrees.
        wavelength: float or string, default="r"
            Effective wavelength of the observation, in meters. You can also supply
            a string specifying an LSST band, in which case the effective wavelength
            of that band will be used.
        L0: float, optional
            Outer scale of turbulence, in meters. If infinite or non-positive,
            Kolmogorov turbulence is assumed. Otherwise, von Karman turbulence.
        v: float, optional
            Wind velocity in the dominant layer, in meters per second.
        T: float, optional
            Exposure time, in seconds.
        theta: float, optional
            Separation angle between the two sources, in degrees.
        Cn2: np.ndarray, optional
            Turbulence profile in a 2D array where the first row is altitude in km,
            and the second is the integrated Cn^2 value for each layer, in units of
            1e-13 m^(1/3). The overall normalization doesn't matter, as the profile
            will be normalized to sum to 1. Note these values are only relevant
            when theta != 0.
        D: float, optional
            Diameter of the pupil, in meters.
        eps: float, optional
            Fractional obscuration of the pupil.
        jmax: int, optional
            The maximum Noll index for the Zernikes.
        pupil_N: int, optional
            The pupil is discretized into a grid of pupil_N x pupil_N points.
            Increasing this number increases accuracy at the expense of computation
            time. The standard deviation of Zernikes 4-28 has mostly converged by
            125. Lower numbers may be desired for faster calculation, for a small
            sacrifice in accuracy.
        """
        # save the parameters
        if seeing is not None:
            self._params["seeing"] = seeing
        if seeing_type is not None:
            self._params["seeing_type"] = seeing_type
        if zenith is not None:
            self._params["zenith"] = zenith
        if wavelength is not None:
            if isinstance(wavelength, str):
                wavelength = (
                    galsim.Bandpass(
                        f"LSST_{wavelength}.dat",
                        wave_type="nm",
                    ).effective_wavelength
                    * 1e-9
                )
            self._params["wavelength"] = wavelength
        if L0 is not None:
            L0 = np.inf if L0 < 0 else L0
            self._params["L0"] = L0
        if v is not None:
            self._params["v"] = v
        if T is not None:
            self._params["T"] = T
        if theta is not None:
            self._params["theta"] = theta
        if Cn2 is not None:
            h, cn2 = np.array(Cn2)
            cn2 /= cn2.sum()
            self._params["Cn2"] = np.array([h, cn2])
        if D is not None:
            self._params["D"] = D
        if eps is not None:
            self._params["eps"] = eps
        if jmax is not None:
            self._params["jmax"] = jmax
        if pupil_N is not None:
            self._params["pupil_N"] = pupil_N

    @property
    def params(self) -> dict:
        """Return the parameter dictionary."""
        return deepcopy(self._params)

    @property
    def airmass(self) -> float:
        """Return the air mass."""
        return 1 / np.cos(np.deg2rad(self.params["zenith"]))

    @property
    def r0_ref(self) -> float:
        """Return the reference Fried parameter for 500nm at zenith."""
        # calculate r0_ref for Kolmogorov turbulence
        r0k = 0.976 * 500e-9 / np.deg2rad(self.params["seeing"] / 3600)

        # if the seeing comes from a DIMM, this is the corresponding Fried parameter
        if self.params["seeing_type"] == "dimm":
            r0vk = r0k

        # if the seeing comes from the image FWHM,
        # we need to invert the formula from Tokovinin 2002
        elif self.params["seeing_type"] == "fwhm":
            r0vk = optimize.root(
                lambda r0vk: (r0vk / r0k) ** 2
                + 2.183 * (r0vk / self.params["L0"]) ** 0.356
                - 1,
                r0k,
            ).x[0]

        return r0vk

    @property
    def r0(self) -> float:
        """Return the Fried parameter at the target airmass and wavelength."""
        return (
            self.r0_ref
            * (self.params["wavelength"] / 500e-9) ** 1.2
            / self.airmass**0.6
        )

    @property
    def psf_fwhm(self) -> float:
        """Return the target PSF FWHM in arcseconds."""
        fwhm_rad = 0.976 * self.params["wavelength"] / self.r0
        fwhm_rad *= np.sqrt(1 - 2.183 * (self.r0 / self.params["L0"]) ** 0.356)

        return 3600 * np.rad2deg(fwhm_rad)

    @property
    def t0(self) -> float:
        """Return the coherence time of the atmosphere, in seconds."""
        return 0.31 * self.r0 / self.params["v"]

    @property
    def N(self) -> float:
        """Return the effective number of independent atmosphere realizations."""
        return np.clip(self.params["T"] / self.t0, 1, None)

    def correlation(self, rho: Union[np.ndarray, float]) -> Union[np.ndarray, float]:
        """Calculate the projected correlation function for pupil distance rho.

        Note this includes the offset from a non-zero difference in field angle,
        corresponding to self.params["theta"].

        Note that the correlation function for Kolmogorov turbulence is formally
        infinite, so for Kolmogorov turbulence, this returns only the finite
        piece of the variance, which is negative.

        Parameters
        ----------
        rho: np.ndarray or float
            The pupil distance in meters.

        Returns
        -------
        np.ndarray or float
            Values of the correlation function at distances rho.
        """
        # pull out the atmosphere structure constants
        h, cn2 = self.params["Cn2"]
        h = 1e3 * h  # km -> m

        # calculate the argument
        rho = np.atleast_1d(rho)
        r = (
            rho[..., None]
            + np.deg2rad(self.params["theta"]) * h[None, :] * self.airmass
        )

        # no turbulence
        if np.isclose(self.params["L0"], 0):
            integrand = 0 * r

        # Kolmogorov turbulence
        elif self.params["L0"] == np.inf:
            # calculate the integrand
            integrand = (r / self.r0) ** (5 / 3)

            # multiply in the constants
            integrand *= -6.88 / 2

        # von Karman turbulence
        else:
            # scale the distances
            r *= 2 * np.pi / self.params["L0"]

            # to avoid problems with kv(5/6, 0), we will fill an array with lim_r->0 B(r)
            # and then for non-zero values of r, replace with B(r)
            integrand = np.full_like(r, gamma(5 / 6) / 2 ** (1 / 6))
            non_zero = np.nonzero(r)
            integrand[non_zero] = r[non_zero] ** (5 / 6) * kv(5 / 6, r[non_zero])

            # multiply in the constants
            integrand *= 0.0858 * (self.params["L0"] / self.r0) ** (5 / 3)

        # evaluate the integral to calculate the projected correlation function
        B = np.sum(cn2 * integrand, axis=-1)

        return B.squeeze()

    def structure(self, rho: Union[np.ndarray, float]) -> Union[np.ndarray, float]:
        """Calculate the projected structure function for pupil distance rho.

        Note this includes the offset from a non-zero difference in field angle,
        corresponding to self.params["theta"].

        Parameters
        ----------
        rho: np.ndarray or float
            The pupil distance in meters.

        Returns
        -------
        np.ndarray float
            Values of the structure function at distances rho.
        """
        return 2 * (self.correlation(0) - self.correlation(rho))

    def zk_cov(self) -> np.ndarray:
        """Calculate the covariance of the Zernike coefficients.

        Returns
        -------
        np.ndarray
            The covariance matrix for the Zernikes, in wavelengths^2.
        """
        # create the pupil grid
        N = self.params["pupil_N"]
        yPupil, xPupil = np.mgrid[-1 : 1 : 1j * N, -1 : 1 : 1j * N]

        # create the Zernike basis
        zk = galsim.zernike.zernikeBasis(
            self.params["jmax"],
            xPupil,
            yPupil,
            R_inner=self.params["eps"],
        )[4:]

        # mask outside the pupil
        rPupil = np.sqrt(xPupil**2 + yPupil**2)
        zk *= (rPupil > self.params["eps"]) & (rPupil < 1)

        # normalize the Zernikes
        zk /= np.diag(np.einsum("jab,kab->jk", zk, zk))[:, None, None]
        self._zk = zk

        # now create a grid that is twice as large as the pupil
        x, y = np.mgrid[-2 : 2 : 1j * (2 * N - 1), -2 : 2 : 1j * (2 * N - 1)]

        # calculate distance from center of grid
        rho = cdist([[0, 0]], np.vstack((x.flatten(), y.flatten())).T).reshape(x.shape)
        rho *= self.params["D"] / 2  # scale by mirror radius

        # calculate correlation function on this grid
        Bphi = self.correlation(rho)

        # Create a sliding window over the gridded correlation function.
        # The result is an (N x N x N x N) tensor, which contains the
        # correlation function between every pair of points on the pupil
        Bphi = np.lib.stride_tricks.sliding_window_view(Bphi, (N, N))

        # but the numpy function doesn't actually return these windows in the order
        # we want, so we need to reverse the order of the first two dimensions
        Bphi = Bphi[::-1, ::-1, ...]  # type: ignore

        # trace over the pixels and calculate covariance
        cov = (
            np.einsum(
                "jab,kcd,abcd->jk",
                zk,
                zk,
                Bphi,
                optimize="optimal",
            )
            / self.N
        )

        return cov

    def zk_std(self) -> np.ndarray:
        """Calculate the standard deviation of the Zernike coefficients.

        Returns
        -------
        np.ndarray
            Standard deviations for Zernike coefficients, in wavelengths.
        """
        cov = self.zk_cov()
        return np.sqrt(np.diag(cov))
