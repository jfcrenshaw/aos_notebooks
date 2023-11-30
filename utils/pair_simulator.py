"""Set of classes for quickly simulating donut pairs."""
from typing import Optional, Union, no_type_check

import batoid
import galsim
import numpy as np

import wfsim


class Star:
    """A Star object, which contains metadata and images."""

    def __init__(
        self,
        chip: Optional[str] = None,
        angle: Optional[tuple] = None,
        centroid: Optional[tuple] = None,
        flux: Optional[int] = None,
        T: Optional[float] = None,
        image: Optional[np.ndarray] = None,
        stamp_size: int = 160,
    ) -> None:
        """
        chip: str
            Name of the chip where the star was imaged, in the format "{corner}_{focal}",
            where corner is one of "R00", "R40", "R44", or "R04", and focal is one of
            "SW0" (for extrafocal) or "SW1" (for intrafocal).
        angle: tuple
            The field angle of the star in degrees.
        centroid: tuple
            The centroid of the star, in pixels. This is in the global coordinate
            system of the focal plane, not the rotated coordinate systems of the
            individual chips.
        flux: int
            The true flux of the star in photons.
        T: float
            The temperature of the star in K, which was used to determine the
            black body spectrum.
        image: np.ndarray
            The full image of the chip.
        stamp_size: int
            The side length of the square postage stamp that will be cut out when
            you access the stamp attribute.
        """
        self.chip = chip
        self.angle = angle
        self.centroid = centroid
        self.flux = flux
        self.T = T
        self.image = image
        self.stamp_size = stamp_size

    @property
    @no_type_check
    def stamp(self) -> np.ndarray:
        w0 = self.stamp_size // 2
        w1 = self.stamp_size - w0
        x, y = self.centroid
        return self.image[y - w0 : y + w1, x - w0 : x + w1].copy()


class Pair:
    """A Pair object, which stores an intra and extrafocal star,
    plus the metadata common to both."""

    def __init__(
        self,
        corner: Optional[str] = None,
        background: Optional[float] = None,
        band: Optional[str] = None,
        opd: Optional[np.ndarray] = None,
        wf_dev: Optional[np.ndarray] = None,
        intra: Star = Star(),
        extra: Star = Star(),
    ) -> None:
        """
        Parameters
        ----------
        corner: str
            The corner where the donuts live.
            Can be one of "R00", "R40", "R44", or "R04".
        background: float
            Standard deviation of the background noise, in photons.
        band: str
            The LSST band the stars are observed in.
        opd: np.ndarray
            Zernike amplitudes in microns, for Noll indices 4-22 (inclusive),
            corresponding to the OPD of the simulated telescope, at the center
            of the CWFS named above.
        wf_dev: np.ndarray
            Zernike amplitudes in microns, for Noll indices 4-22 (inclusive),
            corresponding to the wavefront deviation of the simulated telescope,
            at the center of the CWFS named above.
            The wavefront deviation is the OPD - intrinsic Zernikes.
        intra: Star
            The intrafocal star.
        extra: Star
            The extrafocal star.
        """
        self.corner = corner
        self.background = background
        self.band = band
        self.opd = opd
        self.wf_dev = wf_dev
        self.intra = intra
        self.extra = extra


class PairSimulator:
    """Object that simulates a pair of donuts for the AOS."""

    corner_locations = {
        # bottom left
        "R00": (-0.02075, -0.02075),
        # top left
        "R40": (-0.02075, +0.02075),
        # bottom right
        "R04": (+0.02075, -0.02075),
        # top right
        "R44": (+0.02075, +0.02075),
    }

    def __init__(
        self,
        band: str = "r",
        obs: Optional[dict] = None,
        atm_kwargs: Optional[dict] = None,
        atm: bool = True,
        dcr: bool = True,
        stamp_size: int = 160,
        seed: int = 42,
    ) -> None:
        """
        Parameters
        ----------
        band: str, default="r"
            The LSST band that the stars are observed in.
        obs: dict, optional
            The observation keyword arguments used by wfsim.
            Any values provided will update default arguments,
            which can be seen in the __init__ method of PairSimulator.
        atm_kwargs: dict, optional
            The atmosphere keyword arguments used by wfsim.
            Any values provided will update default arguments,
            which can be seen in the __init__ method of PairSimulator.
        atm: bool, default=True
            Whether to create an atmosphere for the simulator.
        dcr: bool, default=True
            Whether to simulate DCR.
        stamp_size: int, default=160
            Default size of the stamps associated with simulated stars. The stamp
            sizes can be changed ex post facto by changing the corresponding attribute
            on the Star objects.
        seed: int, default=42
            Random seed.
        """
        # load the bandpass and fiducial telescope
        self.band = band
        self.bandpass = galsim.Bandpass(f"LSST_{self.band}.dat", wave_type="nm")
        self.telescope0 = batoid.Optic.fromYaml(f"LSST_{self.band}.yaml")
        self.telescope = self.telescope0

        # set some observational parameters
        obs0 = {
            "zenith": 30 * galsim.degrees,
            "raw_seeing": 0.7 * galsim.arcsec,  # zenith 500nm seeing
            "wavelength": self.bandpass.effective_wavelength,
            "exptime": 15.0,  # seconds
            "temperature": 293.0,  # Kelvin
            "pressure": 69.0,  # kPa
            "H2O_pressure": 1.0,  # kPa
        }
        obs = {} if obs is None else obs
        self.obs = obs0 | obs

        # set atmospheric parameters
        atm_kwargs0 = {
            "screen_size": 819.2,
            "screen_scale": 0.1,
            "nproc": 6,  # create screens in parallel using this many CPUs
        }
        atm_kwargs = {} if atm_kwargs is None else atm_kwargs
        self.atm_kwargs = atm_kwargs0 | atm_kwargs

        # set the seed
        self.set_seed(seed)

        # create the simulator
        self.simulator = wfsim.SimpleSimulator(  # type: ignore
            self.obs,
            self.atm_kwargs,
            self.telescope0,
            self.bandpass,
            atm=atm,
            dcr=dcr,
            name="R00_SW0",
            rng=self.rng,
        )

        # save the default stamp size
        self.stamp_size = stamp_size

    def set_seed(self, seed: int) -> None:
        """Set the random seed

        Parameters
        ----------
        seed: int
            Random seed.
        """
        self.rng = np.random.default_rng(seed)

    def get_intrinsic_zernikes(self, corner: str) -> np.ndarray:
        """Get the intrinsic Zernikes of the telescope.

        Parameters
        ----------
        corner: str
            The name of the corner to get the OPD for.
            Can be "R00", "R40", "R44", or "R04".

        Returns
        -------
        np.ndarray
            Zernike amplitudes in microns for Noll indices 4-22, inclusive.
        """
        return batoid.zernike(
            self.telescope0,
            *self.corner_locations[corner],
            1e-6,  # reference wavelength = 1 micron
            jmax=22,
            eps=self.telescope0.pupilObscuration,
        )[4:]

    def set_wf_dev(self, wf_dev: np.ndarray) -> None:
        """Set the wavefront deviation, which is the OPD - intrinsic zernikes.

        Parameters
        ----------
        dev: np.ndarray
            Amplitudes of Zernikes in microns, for Noll indices 4-22 (inclusive).
        """
        # create the phase screen
        R_outer = self.telescope0.pupilSize / 2
        R_inner = R_outer * self.telescope0.pupilObscuration
        phase = batoid.Zernike(
            -np.pad(wf_dev, pad_width=(4, 0), mode="constant") * 1e-6,
            R_outer=R_outer,
            R_inner=R_inner,
        )

        # Add phase screen to telescope
        self.telescope = batoid.CompoundOptic(
            (
                batoid.optic.OPDScreen(
                    batoid.Plane(),
                    phase,
                    name="PhaseScreen",
                    obscuration=batoid.ObscNegation(batoid.ObscCircle(5.0)),
                    coordSys=self.telescope0.stopSurface.coordSys,
                ),
                *self.telescope0.items,
            ),
            name="PerturbedLSST",
            backDist=self.telescope0.backDist,
            pupilSize=self.telescope0.pupilSize,
            inMedium=self.telescope0.inMedium,
            stopSurface=self.telescope0.stopSurface,
            sphereRadius=self.telescope0.sphereRadius,
            pupilObscuration=self.telescope0.pupilObscuration,
        )

    def set_opd(self, opd: np.ndarray, corner: str) -> None:
        """Directly set the OPD of the telescope.

        Parameters
        ----------
        opd: np.ndarray
            Amplitudes of Zernikes in microns, for Noll
            indices 4-22 (inclusive).
        corner: str
            The name of the corner to get the OPD for.
            Can be "R00", "R40", "R44", or "R04".
        """
        wf_dev = opd - self.get_intrinsic_zernikes(corner)
        self.set_wf_dev(wf_dev)

    def set_dof(self, dof: np.ndarray) -> None:
        """Perturb the degrees of freedom of the telescope

        Parameters
        ----------
        dof: np.ndarray
            Perturbations to degrees of freedom of the telescope.
            Contains 50 entries, with definitions and units listed below:
            - 0: M2 dz (microns)
            - 1: M2 dx (microns)
            - 2: M2 dy (microns)
            - 3: M2 x-axis rotation (arcsecs)
            - 4: M2 y-axis rotation (arcsecs)
            - 5: Camera dz (microns)
            - 6: Camera dx (microns)
            - 7: Camera dy (microns)
            - 8: Camera x-axis rotation (arcsecs)
            - 9: Camera y-axis rotation (arcsecs)
            - 10-29: M1M3 bending modes (microns)
            - 30-49: M2 bending modes (microns)

        """
        # create perturbed telescope from DOF
        self.telescope = wfsim.SSTFactory(self.telescope0).get_telescope(dof=dof)  # type: ignore

    def get_opd(self, corner: str) -> np.ndarray:
        """Get the OPD of the telescope at the center of the given corner CWFS.

        Parameters
        ----------
        corner: str
            The name of the corner to get the OPD for.
            Can be "R00", "R40", "R44", or "R04".

        Returns
        -------
        np.ndarray
            Zernike amplitudes in microns for Noll indices 4-22, inclusive.
        """
        # get the OPD from batoid
        opd = batoid.zernike(
            self.telescope,
            *self.corner_locations[corner],
            1e-6,  # reference wavelength = 1 micron
            jmax=22,
            eps=self.telescope.pupilObscuration,
        )[4:]

        return opd

    def get_wf_dev(self, corner: str = "R00") -> np.ndarray:
        """Get the wavefront deviation at the center of the corner CWFS.

        The wavefront deviation is the OPD - intrinsic Zernikes.

        Parameters
        ----------
        corner: str
            The name of the corner to get the wavefront deviation for.
            Can be "R00", "R40", "R44", or "R04".

        Returns
        -------
        np.ndarray
            Zernike amplitudes in microns for Noll indices 4-22, inclusive.
        """
        wf_dev = self.get_opd(corner) - self.get_intrinsic_zernikes(corner)

        return wf_dev

    def _simulate_star(
        self,
        angle: Union[list, str, None],
        flux: int,
        T: float,
        background: float,
        corner: str,
        intra: bool,
    ) -> Star:
        """Private method for simulating a single donut."""
        # create the star
        star = Star()
        star.flux = int(flux)
        star.T = float(T)

        # create the star SED
        sed = wfsim.BBSED(star.T)  # type: ignore

        # calculate pixel variance of background
        variance = background**2

        # set the telescope
        offset = -0.0015 if intra else +0.0015
        self.simulator.telescope = self.telescope.withGloballyShiftedOptic(
            "Detector", [0, 0, offset]
        )

        # set the CCD to simulate
        chip = f"{corner}_SW{int(intra)}"
        star.chip = chip
        self.simulator.set_name(chip)

        # if no angle provided, random sample position
        if angle is None:
            # calculate the bounds on the x and y angles
            bounds = self.simulator.image.bounds
            bounds = 0.9 * np.array(
                [[bounds.xmin, bounds.ymin], [bounds.xmax, bounds.ymax]]
            )
            xmin, ymin = self.simulator.wcs.xyToradec(*bounds[0], galsim.degrees)
            xmax, ymax = self.simulator.wcs.xyToradec(*bounds[1], galsim.degrees)
            x = self.rng.uniform(xmin, xmax)
            y = self.rng.uniform(ymin, ymax)
            star.angle = np.array([x, y])  # type: ignore
        elif angle == "center":
            star.angle = np.rad2deg(self.simulator.get_bounds().mean(axis=1))
        else:
            star.angle = np.array(angle)  # type: ignore

        # simulate the star
        self.simulator.add_star(*np.deg2rad(star.angle), sed, star.flux, self.rng)  # type: ignore

        # add background
        self.simulator.add_background(variance, self.rng)

        # save the full image
        star.image = self.simulator.image.array.copy()

        # save the donut centroid
        x, y = self.simulator.wcs.radecToxy(*star.angle, galsim.degrees)
        x = int(x - self.simulator.image.bounds.xmin)  # x in image coords
        y = int(y - self.simulator.image.bounds.ymin)  # y in image coords
        star.centroid = np.array([x, y])  # type: ignore

        # and the default stamp size
        star.stamp_size = self.stamp_size

        return star

    def simulate(
        self,
        intra_angle: Union[list, str, None] = "center",
        extra_angle: Union[list, str, None] = "center",
        intra_flux: int = 1_000_000,
        extra_flux: int = 1_000_000,
        intra_T: float = 8_000,
        extra_T: float = 8_000,
        background: float = 10,
        corner: str = "R00",
    ) -> Pair:
        """Simulate and return a donut pair.

        Parameters
        ----------
        intra_angle: list, optional
            List of [field_x, field_y] angles, in degrees, for
            the intrafocal donut. If "center", the center of
            the chip is used. If None, random angle drawn.
        extra_angle: list, optional
            List of [field_x, field_y] angles, in degrees, for
            the extrafocal donut. If "center", the center of
            the chip is used. If None, random angle drawn.
        intra_flux: int, default=1_000_000
            Number of photons for intrafocal donut.
        extra_flux: int, default=1_000_000
            Number of photons for extrafocal donut.
        intra_T: float, default=8_000
            Temperature, in Kelvin, which is used to generate a
            blackbody spectrum for the intrafocal donut.
        extra_T: float, default=8_000
            Temperature, in Kelvin, which is used to generate a
            blackbody spectrum for the extrafocal donut.
        background: float, default=10
            Standard deviation of the background noise, in photons.
        corner: str, default="R00"
            The corner for which the donuts are simulated. Can be
            one of "R00", "R40", "R44", or "R04".

        Returns
        -------
        Pair
            A pair object with metadata, and an intra and
            extrafocal donut.
        """
        # save the pair metadata
        pair = Pair()
        pair.corner = corner
        pair.band = self.band
        pair.opd = self.get_opd(corner)
        pair.wf_dev = self.get_wf_dev(corner)
        pair.background = float(background)

        # simulate the intrafocal star
        pair.intra = self._simulate_star(
            intra_angle,
            intra_flux,
            intra_T,
            pair.background,
            pair.corner,
            intra=True,
        )

        # simulate the extrafocal star
        pair.extra = self._simulate_star(
            extra_angle,
            extra_flux,
            extra_T,
            pair.background,
            pair.corner,
            intra=False,
        )

        return pair
