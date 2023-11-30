import pandas as pd
import numpy as np

from lsst.ts.wep.wfEstimator import WfEstimator
from lsst.ts.wep.utility import DefocalType, getConfigDir, FilterType

# load the dataframe of data
df = pd.read_pickle("../data/test_dataframe.pkl")

# loop through every corner for every pointing
# and randomly select intra/extra pairs
pairs = []
for (pntId, corner), group in df.groupby(["pntId", "corner"]):
    intra = group[group["intra"] == True].sort_values("mag")
    extra = group[group["intra"] == False].sort_values("mag")
    idx_pairs = list(zip(intra.index, extra.index))
    pairs.extend(idx_pairs)

wfEstExp = WfEstimator(f"{getConfigDir()}/cwfs/algo")
wfEstExp.config(sizeInPix=160, units="um", algo="exp")

wfEstML = WfEstimator(f"{getConfigDir()}/cwfs/algo")
wfEstML.config(
    sizeInPix=160,
    units="um",
    algo="ml",
    mlFile="/astro/store/epyc/users/jfc20/ml-aos/models/v4_2023-06-22_17:30:32.pt",
    mlReshape=None,
)

# loop through everything and calculate zernikes
zkExpList = []
zkMlIntraList = []
zkMlExtraList = []

for i, (intra, extra) in enumerate(pairs):
    # ML: estimate using intra *only*
    wfEstML.reset()
    donut = df.loc[intra]
    wfEstML.setImg(
        (donut["fx"], donut["fy"]),
        DefocalType.Intra if donut["intra"] else DefocalType.Extra,
        filterLabel=FilterType("ugrizy".index(donut["filter"]) + 1),
        blendOffsets=donut["brightBlendOffsets"],
        image=donut["image"],
    )
    zkMlIntra = wfEstML.calWfsErr()

    # ML: estimate using extra *only*
    wfEstML.reset()
    donut = df.loc[extra]
    wfEstML.setImg(
        (donut["fx"], donut["fy"]),
        DefocalType.Intra if donut["intra"] else DefocalType.Extra,
        filterLabel=FilterType("ugrizy".index(donut["filter"]) + 1),
        blendOffsets=donut["brightBlendOffsets"],
        image=donut["image"],
    )
    zkMlExtra = wfEstML.calWfsErr()

    # Exp: estimate using both images
    wfEstExp.reset()

    donut = df.loc[intra]
    wfEstExp.setImg(
        (donut["fx"], donut["fy"]),
        DefocalType.Intra if donut["intra"] else DefocalType.Extra,
        filterLabel=FilterType("ugrizy".index(donut["filter"]) + 1),
        blendOffsets=donut["brightBlendOffsets"],
        image=donut["image"],
    )

    donut = df.loc[extra]
    wfEstExp.setImg(
        (donut["fx"], donut["fy"]),
        DefocalType.Intra if donut["intra"] else DefocalType.Extra,
        filterLabel=FilterType("ugrizy".index(donut["filter"]) + 1),
        blendOffsets=donut["brightBlendOffsets"],
        image=donut["image"],
    )

    try:
        zkExp = wfEstExp.calWfsErr()
    except:
        zkExp = np.full(19, np.nan)

    zkExpList.append(zkExp)
    zkMlIntraList.append(zkMlIntra)
    zkMlExtraList.append(zkMlExtra)

    if i % 1000 == 0 and i > 0:
        pairList = np.array(pairs[: len(zkExpList)])
        np.savez(
            f"../data/zk_predictions{i}.npz",
            pairs=pairList,
            truth=df["zernikes"].loc[pairList[:, 0]].to_numpy(),
            exp=np.array(zkExpList),
            mlIntra=np.array(zkMlIntraList),
            mlExtra=np.array(zkMlExtraList),
        )

pairList = np.array(pairs[: len(zkExpList)])
np.savez(
    f"../data/zk_predictions.npz",
    pairs=pairList,
    truth=df["zernikes"].loc[pairList[:, 0]].to_numpy(),
    exp=np.array(zkExpList),
    mlIntra=np.array(zkMlIntraList),
    mlExtra=np.array(zkMlExtraList),
)
