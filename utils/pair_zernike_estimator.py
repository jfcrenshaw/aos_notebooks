"""Class that estimates Zernikes from a star pair."""

import numpy as np
from lsst.ts.wep.utility import DefocalType, FilterType, getConfigDir
from lsst.ts.wep.wfEstimator import WfEstimator

from .pair_simulator import Pair


class PairZernikeEstimator:
    def __init__(
        self,
        mlFile: str = "/phys/users/jfc20/mlaos2/ml-aos/models/v22_2023-05-23_19:32:30.pt",
        stamp_size: int = 160,
        units: str = "um",
    ) -> None:
        """
        Parameters
        ----------
        mlFile : str, optional
           File of the trained neural net.
           default="/astro/store/epyc/users/jfc20/ml-aos/models/v0_2023-06-19_09:41:19.pt"
        stamp_size : int, default=170
            Size of the donut stamp cutouts.
        units : str, default="um"
            Units to return zernikes in. Can be "um", "nm", "arcseconds".
        """
        self.ml_estimator = WfEstimator(f"{getConfigDir()}/cwfs/algo")
        self.ml_estimator.config(
            sizeInPix=stamp_size,
            units=units,
            algo="ml",
            mlFile=mlFile,
        )

        self.exp_estimator = WfEstimator(f"{getConfigDir()}/cwfs/algo")
        self.exp_estimator.config(
            sizeInPix=stamp_size,
            units=units,
            algo="exp",
        )

        self._stamp_size = stamp_size

    def estimate(self, pair: Pair, algo: str) -> np.ndarray:
        """Estimate Zernikes on the pair, using the requested algorithm.

        Parameters
        ----------
        pair : Pair
            The Pair object containing the donut pair.
        algo : str
            Name of the algorithm to use. Can be "ml" or "exp".

        Returns
        -------
        np.ndarray
            Array of Zernikes
        """
        if algo == "ml":
            wfEst = self.ml_estimator
        elif algo == "exp":
            wfEst = self.exp_estimator

        # make sure the stamp size matches
        pair.intra.stamp_size = self._stamp_size
        pair.extra.stamp_size = self._stamp_size

        wfEst.reset()
        wfEst.setImg(
            pair.intra.angle,
            DefocalType.Intra,
            filterLabel=FilterType[f"LSST_{pair.band.capitalize()}"],
            image=pair.intra.stamp,
        )
        wfEst.setImg(
            pair.extra.angle,
            DefocalType.Extra,
            filterLabel=FilterType[f"LSST_{pair.band.capitalize()}"],
            image=pair.extra.stamp,
        )

        return wfEst.calWfsErr()
